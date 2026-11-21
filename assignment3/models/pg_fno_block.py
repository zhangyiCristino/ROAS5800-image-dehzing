#!/usr/bin/env python
"""PGFNOBlock: physics-guided amplitude-phase disentangled Fourier operator.

Pure ``torch.fft`` implementation. Three components per the SOP:

  1. Physics-Guided Scattering Gate -- a dark/bright-channel prior. Per-pixel
     ``min`` (dark channel), ``max`` (bright channel) and ``mean`` over channels
     are concatenated, passed through a 1x1 conv + Sigmoid to give a spatial
     transmission proxy ``t_hat``, plus a GAP + 2-layer MLP gives the channel-wise
     atmospheric light ``A_hat``. The feature is pre-calibrated with
     ``F_phys = (F - A*(1-t)) / max(t, eps)``.
  2. Amplitude-Phase Disentangled FNO -- ``rfft2`` in float32 (forced with
     ``autocast(enabled=False)``), split into amplitude and phase; low-frequency
     amplitude modes are reweighted by a transmission-frequency gate and mixed by
     a grouped 1x1 channel transform; the phase gets a small 1x1 residual
     (``0.1 * tanh``); reconstructed via ``torch.polar`` + ``irfft2``.
  3. Local-Spectral Gated Fusion -- a dilated depthwise + pointwise local branch
     is fused with the spectral branch through two learnable per-channel gates
     ``gamma_global`` / ``gamma_local``, both initialised to 0.1 (never 0).
"""
from __future__ import annotations

from contextlib import nullcontext

import torch
from torch import nn

__all__ = ["PGFNOBlock", "attach_pg_fno", "pg_fno_parameter_count"]


class PGFNOBlock(nn.Module):
    def __init__(self, channels: int, modes: int = 16, groups: int = 4,
                 gamma_init: float = 0.1, eps: float = 0.05):
        super().__init__()
        if channels <= 0 or modes <= 0:
            raise ValueError("channels and modes must be positive")
        self.channels = channels
        self.modes = modes
        self.eps = eps

        # non-zero soft gates (never zero-init: keeps gradient alive from step 1)
        self.gamma_global = nn.Parameter(torch.ones(1, channels, 1, 1) * gamma_init)
        self.gamma_local = nn.Parameter(torch.ones(1, channels, 1, 1) * gamma_init)
        self.phase_scale = nn.Parameter(torch.tensor(0.1))

        # --- 1. physics scattering gate (dark/bright channel prior) -------- #
        self.t_conv = nn.Conv2d(3, 1, 1)                       # [dark,bright,mean] -> t_hat
        self.a_pool = nn.AdaptiveAvgPool2d(1)
        reduction = max(channels // 8, 4)
        self.a_mlp = nn.Sequential(                            # A_hat (B,C,1,1)
            nn.Conv2d(channels, reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduction, channels, 1),
        )

        # --- 2. amplitude-phase FNO ---------------------------------------- #
        self.mode_mix = nn.Conv2d(channels, channels, 1, groups=groups)   # low-freq mode mix
        self.pha_mix = nn.Conv2d(channels, channels, 1, groups=groups)    # phase residual
        self.gate_conv = nn.Conv2d(1, 1, 1)                               # t_hat freq projection
        # small init so the spectral mixing starts gentle (near-identity)
        for conv in (self.mode_mix, self.pha_mix, self.gate_conv):
            nn.init.normal_(conv.weight, std=1e-2)
            nn.init.zeros_(conv.bias)

        # --- 3. local dilated conv branch ----------------------------------- #
        self.local = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=2, dilation=2, groups=channels),
            nn.Conv2d(channels, channels, 1),
            nn.GELU(),
        )

        # --- output projections: small (near-identity) init so the block starts
        # as an identity residual and never destroys the frozen-base output --- #
        self.proj_global = nn.Conv2d(channels, channels, 1, groups=groups)
        self.proj_local = nn.Conv2d(channels, channels, 1, groups=groups)
        for proj in (self.proj_global, self.proj_local):
            nn.init.normal_(proj.weight, std=1e-2)
            nn.init.zeros_(proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The whole block runs in float32 with autocast disabled: the FFT
        # requires fp32 and the mixed bf16/fp32 conv path is dtype-fragile.
        orig_dtype = x.dtype
        ctx = torch.amp.autocast("cuda", enabled=False) if x.is_cuda else nullcontext()
        with ctx:
            x = x.float()

            # 1. physics scattering gate
            dark = x.min(dim=1, keepdim=True).values              # (B,1,H,W)
            bright = x.max(dim=1, keepdim=True).values            # (B,1,H,W)
            mean = x.mean(dim=1, keepdim=True)                    # (B,1,H,W)
            prior = torch.cat([dark, bright, mean], dim=1)        # (B,3,H,W)
            t = torch.sigmoid(self.t_conv(prior))                 # (B,1,H,W)
            a = self.a_mlp(self.a_pool(x))                        # (B,C,1,1)
            # Bounded scattering pre-calibration. The atmospheric model
            # I=J*t+A*(1-t) is defined on [0,1] intensities; backbone features
            # span ~[-10,5], so dividing by a small t would explode (x_phys up
            # to ~66). Use a safe transmission floor (<=2x gain) and clamp the
            # calibrated feature to the BN feature scale to keep the FNO stable.
            t_safe = torch.clamp(t, min=0.5)
            x_phys = (x - a * (1.0 - t)) / t_safe
            x_phys = torch.clamp(x_phys, min=-6.0, max=6.0)

            # 2. amplitude-phase FNO (float32 FFT)
            xf = torch.fft.rfft2(x_phys, norm="ortho")            # (B,C,H,W//2+1)
            amp = xf.abs() + 1e-8
            pha = xf.angle()

            m = min(self.modes, amp.shape[-2], amp.shape[-1])
            amp_low = amp[:, :, :m, :m]                            # (B,C,m,m)

            # transmission-frequency gate: G_freq from |rfft2(t_hat)|
            t_freq = torch.fft.rfft2(t, norm="ortho").abs()        # (B,1,H,W//2+1)
            g_freq = torch.tanh(self.gate_conv(t_freq[:, :, :m, :m]))  # (B,1,m,m)
            amp_low = amp_low * (1.0 + g_freq) + self.mode_mix(amp_low)

            # phase-preserving residual
            pha = pha + self.phase_scale * torch.tanh(self.pha_mix(pha))

            amp_out = amp.clone()
            amp_out[:, :, :m, :m] = amp_low
            xf_out = torch.polar(amp_out, pha)
            fno_out = torch.fft.irfft2(xf_out, s=x_phys.shape[-2:], norm="ortho")

            # 3. local branch + gated fusion (near-identity start)
            local_out = self.local(x)
            out = x + self.gamma_global * self.proj_global(fno_out) \
                + self.gamma_local * self.proj_local(local_out)
        return out.to(orig_dtype)


def attach_pg_fno(model: nn.Module, bottleneck_modes: int = 12, h4_modes: int = 16) -> nn.Module:
    """Inject two PGFNOBlock instances into a PGLNet.

    ``model.layers[3]`` is the H/8 bottleneck (C=192 at 32x32 for 256 crops);
    ``model.layers[4]`` is the H/4 decoder stage (C=96 at 64x64). Both are
    wrapped with ``nn.Sequential``, so the injection is non-invasive.
    """
    bottleneck_ch = model.layers[3].dim
    h4_ch = model.layers[4].dim
    model.layers[3] = nn.Sequential(model.layers[3], PGFNOBlock(bottleneck_ch, modes=bottleneck_modes))
    model.layers[4] = nn.Sequential(PGFNOBlock(h4_ch, modes=h4_modes), model.layers[4])
    return model


def pg_fno_parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for m in model.modules() if isinstance(m, PGFNOBlock) for p in m.parameters())


class ConvAdapterBlock(nn.Module):
    """Param-matched spatial-only control (ablation): multi-scale dilated convs.

    Same injection points, same near-identity output projection and gamma=0.1,
    but NO physics gate and NO spectral branch -- isolates the contribution of
    the Fourier operator itself.
    """

    def __init__(self, channels: int, groups: int = 4, gamma_init: float = 0.1):
        super().__init__()
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))
        self.stage1 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.Conv2d(channels, channels, 1),
            nn.GELU(),
        )
        self.stage2 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=2, dilation=2, groups=channels),
            nn.Conv2d(channels, channels, 1),
            nn.GELU(),
        )
        self.proj = nn.Conv2d(channels, channels, 1, groups=groups)
        nn.init.normal_(self.proj.weight, std=1e-2)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        ctx = torch.amp.autocast("cuda", enabled=False) if x.is_cuda else nullcontext()
        with ctx:
            x = x.float()
            out = self.stage2(self.stage1(x))
            out = x + self.gamma * self.proj(out)
        return out.to(orig_dtype)


class VanillaFNOBlock(nn.Module):
    """Standard complex FNO mode mixing (ablation): no physics gate, no
    amplitude/phase disentanglement, no local branch."""

    def __init__(self, channels: int, modes: int = 16, groups: int = 4,
                 gamma_init: float = 0.1):
        super().__init__()
        self.modes = modes
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))
        self.mix_real = nn.Conv2d(channels, channels, 1, groups=groups)
        self.mix_imag = nn.Conv2d(channels, channels, 1, groups=groups)
        self.proj = nn.Conv2d(channels, channels, 1, groups=groups)
        nn.init.normal_(self.proj.weight, std=1e-2)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        ctx = torch.amp.autocast("cuda", enabled=False) if x.is_cuda else nullcontext()
        with ctx:
            x = x.float()
            xf = torch.fft.rfft2(x, norm="ortho")
            m = min(self.modes, xf.shape[-2], xf.shape[-1])
            real, imag = xf.real, xf.imag
            real_low = real[:, :, :m, :m]
            imag_low = imag[:, :, :m, :m]
            real[:, :, :m, :m] = self.mix_real(real_low)
            imag[:, :, :m, :m] = self.mix_imag(imag_low)
            xf = torch.complex(real, imag)
            out = torch.fft.irfft2(xf, s=x.shape[-2:], norm="ortho")
            out = x + self.gamma * self.proj(out)
        return out.to(orig_dtype)


def attach_conv_adapter(model: nn.Module) -> nn.Module:
    """Inject two param-matched ConvAdapterBlock instances (spatial control)."""
    bottleneck_ch = model.layers[3].dim
    h4_ch = model.layers[4].dim
    model.layers[3] = nn.Sequential(model.layers[3], ConvAdapterBlock(bottleneck_ch))
    model.layers[4] = nn.Sequential(ConvAdapterBlock(h4_ch), model.layers[4])
    return model


def attach_vanilla_fno(model: nn.Module, bottleneck_modes: int = 12, h4_modes: int = 16) -> nn.Module:
    """Inject two VanillaFNOBlock instances (spectral control, no physics)."""
    bottleneck_ch = model.layers[3].dim
    h4_ch = model.layers[4].dim
    model.layers[3] = nn.Sequential(model.layers[3], VanillaFNOBlock(bottleneck_ch, modes=bottleneck_modes))
    model.layers[4] = nn.Sequential(VanillaFNOBlock(h4_ch, modes=h4_modes), model.layers[4])
    return model


def operator_parameter_count(model: nn.Module) -> int:
    """Params of any injected operator block (PGFNO / Conv / Vanilla)."""
    kinds = (PGFNOBlock, ConvAdapterBlock, VanillaFNOBlock)
    return sum(p.numel() for m in model.modules() if isinstance(m, kinds) for p in m.parameters())
