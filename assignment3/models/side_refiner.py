#!/usr/bin/env python
"""Side-Tuning PG-FNO Refiner.

The base PGL-Net-T runs 100% frozen and untouched; it produces J_0 = PGL(I).
Read-only forward hooks expose multi-scale features {F_1(H/2), F_2(H/4), F_3(H/8)}
without back-propagating into the base. An independent refiner corrects the
residual:

    input  = [I, J_0, I-J_0, t_0] (10ch) + projected read-only F_2, F_3
    Module A: Fourier DC & low-frequency atmospheric calibrator (learnable
              complex scale/bias on the leading 4x4 modes, gated by global
              context) -- removes residual low-frequency colour/exposure bias,
              the dominant PSNR bottleneck on RW2AH.
    Module B: multi-scale amplitude-phase FNO (modes=16) + depthwise dilated conv
              at H/4 (and an H/2 branch) -- corrects mid/high-frequency structure.
    output = clamp(J_0 + tanh(delta) * 0.25, 0, 1), head init 1e-3.

Because the base is outside the gradient path, the refiner cannot collapse the
base features (the gamma->0 failure mode of internal injection is eliminated).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .pg_fno_block import PGFNOBlock

__all__ = ["SideRefiner", "DCLowFreqCalibrator", "dark_channel_transmission"]


def dark_channel_transmission(img: torch.Tensor, patch: int = 7) -> torch.Tensor:
    """Rough transmission proxy t_0 = 1 - dark_channel(img), 1 channel."""
    dark = img.min(dim=1, keepdim=True).values
    pad = patch // 2
    dark = -F.max_pool2d(-dark, kernel_size=patch, stride=1, padding=pad)
    return (1.0 - dark).clamp(0.0, 1.0)


class DCLowFreqCalibrator(nn.Module):
    """Learnable complex scale/bias on the leading m x m Fourier modes, gated by
    global context. Targets DC (mean brightness) and near-DC (colour/exposure
    gradients) residuals."""

    def __init__(self, channels: int, m: int = 4):
        super().__init__()
        self.m = m
        self.scale = nn.Parameter(torch.zeros(1, channels, m, m))      # -> 1+tanh
        self.bias_real = nn.Parameter(torch.zeros(1, channels, m, m))
        self.bias_imag = nn.Parameter(torch.zeros(1, channels, m, m))
        reduction = max(channels // 4, 8)
        self.ctx = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, reduction, 1),
            nn.GELU(),
            nn.Conv2d(reduction, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = torch.fft.rfft2(x.float(), norm="ortho")
        m = min(self.m, xf.shape[-2], xf.shape[-1])
        low = xf[:, :, :m, :m]
        gate = self.ctx(x.float())                          # (B,C,1,1)
        scale = 1.0 + torch.tanh(self.scale[:, :, :m, :m]) * gate
        bias = torch.complex(self.bias_real[:, :, :m, :m], self.bias_imag[:, :, :m, :m])
        correction = low * (scale - 1.0) + bias             # change applied to low modes
        pad_h = xf.shape[-2] - m
        pad_w = xf.shape[-1] - m
        correction_full = F.pad(correction, (0, pad_w, 0, pad_h))  # zero elsewhere
        xf = xf + correction_full                           # non-inplace
        return torch.fft.irfft2(xf, s=x.shape[-2:], norm="ortho").to(x.dtype)


class SideRefiner(nn.Module):
    def __init__(self, base_channels=(96, 192), work_channels: int = 64,
                 modes: int = 16, delta_scale: float = 0.25):
        super().__init__()
        self.delta_scale = delta_scale
        c2, c3 = base_channels
        # project read-only features to compact channels (trainable, refiner-side)
        self.proj_f2 = nn.Conv2d(c2, 32, 1)
        self.proj_f3 = nn.Conv2d(c3, 32, 1)
        # input: 10 image-level ch + 32 (F2) + 32 (F3 upsampled) = 74
        self.stem = nn.Sequential(nn.Conv2d(74, work_channels, 3, padding=1), nn.GELU())
        # Module A: DC / low-frequency atmospheric calibrator
        self.dc_calib = DCLowFreqCalibrator(work_channels, m=4)
        # Module B: amplitude-phase FNO + local dilated conv at H/4
        self.fno_h4 = PGFNOBlock(work_channels, modes=modes)
        self.local_h4 = nn.Sequential(
            nn.Conv2d(work_channels, work_channels, 3, padding=2, dilation=2, groups=work_channels),
            nn.Conv2d(work_channels, work_channels, 1), nn.GELU(),
        )
        # H/2 branch (multi-scale amplitude-phase FNO)
        self.fno_h2 = PGFNOBlock(work_channels, modes=modes)
        self.head = nn.Conv2d(work_channels, 3, 3, padding=1)
        nn.init.normal_(self.head.weight, std=1e-3)
        nn.init.zeros_(self.head.bias)

    def forward(self, I: torch.Tensor, J0: torch.Tensor,
                F2: torch.Tensor, F3: torch.Tensor) -> torch.Tensor:
        J0 = J0.clamp(0.0, 1.0)
        H, W = I.shape[-2:]
        h4, w4 = H // 4, W // 4
        t0 = dark_channel_transmission(I)
        img_in = torch.cat([I, J0, I - J0, t0], dim=1)
        img_in = F.interpolate(img_in, size=(h4, w4), mode="bilinear", align_corners=False)
        f2 = self.proj_f2(F2)
        f3 = self.proj_f3(F.interpolate(F3, size=(h4, w4), mode="bilinear", align_corners=False))
        x = self.stem(torch.cat([img_in, f2, f3], dim=1))
        x = self.dc_calib(x)                       # Module A: DC / low-freq calibrator
        x = self.fno_h4(x) + self.local_h4(x)      # Module B at H/4
        xh2 = F.interpolate(x, size=(H // 2, W // 2), mode="bilinear", align_corners=False)
        xh2 = self.fno_h2(xh2)                     # Module B at H/2
        x = x + F.interpolate(xh2, size=(h4, w4), mode="bilinear", align_corners=False)
        delta = self.head(x)
        delta = F.interpolate(delta, size=(H, W), mode="bilinear", align_corners=False)
        return (J0 + torch.tanh(delta) * self.delta_scale).clamp(0.0, 1.0)
