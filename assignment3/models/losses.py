#!/usr/bin/env python
"""Training losses for PGL-PG-FNO (zero extra VRAM).

L_total = L_Charbonnier(pred, gt)
        + 0.10 * || |rfft2(pred)| - |rfft2(gt)| ||_1              (amplitude)
        + 0.05 * || angle(rfft2(pred)) - angle(rfft2(gt)) ||_1   (phase)
        + 0.15 * (1 - SSIM(pred, gt))                             (structure)

SSIM is a pure convolution with a fixed Gaussian window (11x11, sigma 1.5,
matching the evaluation convention); no VGG/LPIPS/perceptual network is loaded.
FFT runs in float32.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

__all__ = ["charbonnier", "fft_loss", "ssim", "ms_ssim", "PG_FNO_Loss", "SideRefinerLoss"]


def charbonnier(pred: torch.Tensor, gt: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    return torch.sqrt(((pred - gt) ** 2) + eps**2).mean()


def fft_loss(pred: torch.Tensor, gt: torch.Tensor,
             lambda_amp: float = 0.10, lambda_phase: float = 0.05) -> torch.Tensor:
    """Amplitude + phase alignment in the frequency domain (float32).

    The phase term uses the *smooth complex-spectrum* L1 rather than a raw
    ``angle()`` L1: ``angle(z)`` has gradient 1/|z| which diverges at the many
    low-amplitude high-frequency bins, producing exploding noisy gradients that
    destabilise training. ``|pred_f - gt_f|`` is smooth everywhere and still
    penalises phase misalignment (it is the chord distance on the complex
    plane), so the amplitude/phase decoupling is preserved without the
    singularity.
    """
    pred_f = torch.fft.rfft2(pred.float(), norm="ortho")
    gt_f = torch.fft.rfft2(gt.float(), norm="ortho")
    amp = (pred_f.abs() - gt_f.abs()).abs().mean()
    pha = (pred_f - gt_f).abs().mean()
    return lambda_amp * amp + lambda_phase * pha


def _gaussian_window(size: int, sigma: float, channels: int, device) -> torch.Tensor:
    coords = torch.arange(size, device=device, dtype=torch.float32) - (size - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    kernel = torch.outer(g, g).reshape(1, 1, size, size).expand(channels, 1, size, size).contiguous()
    return kernel


def ssim(pred: torch.Tensor, gt: torch.Tensor, data_range: float = 1.0,
         size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    """Differentiable SSIM (Gaussian window), mean over batch/channels/spatial.

    Training-stable variant: local variances are clamped to >= 0 (numerical
    error can make them slightly negative) and a stability epsilon is added to
    the variance denominator. Without it, uniform haze/sky regions have
    sigma^2 ~ 0 so the denominator ~ c2 = 9e-4 and the gradient ~ 1/c2 ~ 1e3
    explodes, which was observed to diverge training (grad_norm > 400).
    """
    channels = pred.shape[1]
    win = _gaussian_window(size, sigma, channels, pred.device).to(pred.dtype)
    padding = size // 2
    mu_pred = F.conv2d(pred, win, padding=padding, groups=channels)
    mu_gt = F.conv2d(gt, win, padding=padding, groups=channels)
    mu_pred_sq, mu_gt_sq, mu_pred_gt = mu_pred ** 2, mu_gt ** 2, mu_pred * mu_gt
    sigma_pred_sq = (F.conv2d(pred * pred, win, padding=padding, groups=channels) - mu_pred_sq).clamp(min=0.0)
    sigma_gt_sq = (F.conv2d(gt * gt, win, padding=padding, groups=channels) - mu_gt_sq).clamp(min=0.0)
    sigma_pred_gt = F.conv2d(pred * gt, win, padding=padding, groups=channels) - mu_pred_gt

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    stab = 1e-2  # bounds the variance-denominator gradient in flat regions
    ssim_map = ((2 * mu_pred_gt + c1) * (2 * sigma_pred_gt + c2)) / (
        (mu_pred_sq + mu_gt_sq + c1) * (sigma_pred_sq + sigma_gt_sq + c2 + stab))
    return ssim_map.mean()


class PG_FNO_Loss(nn.Module):
    def __init__(self, lambda_amp: float = 0.10, lambda_phase: float = 0.05,
                 lambda_ssim: float = 0.15, charb_eps: float = 1e-3):
        super().__init__()
        self.lambda_amp = lambda_amp
        self.lambda_phase = lambda_phase
        self.lambda_ssim = lambda_ssim
        self.charb_eps = charb_eps

    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> dict[str, torch.Tensor]:
        l_charb = charbonnier(pred, gt, self.charb_eps)
        l_fft = fft_loss(pred, gt, self.lambda_amp, self.lambda_phase)
        l_ssim = self.lambda_ssim * (1.0 - ssim(pred, gt))
        total = l_charb + l_fft + l_ssim
        return {"total": total, "charb": l_charb, "fft": l_fft, "ssim": l_ssim}


def ms_ssim(pred: torch.Tensor, gt: torch.Tensor, data_range: float = 1.0,
            scales: int = 3) -> torch.Tensor:
    """Multi-scale SSIM (average of SSIM over progressively downsampled pairs)."""
    vals = []
    p, g = pred, gt
    for _ in range(scales):
        vals.append(ssim(p, g, data_range=data_range))
        if min(p.shape[-2:]) <= 16:
            break
        p = F.avg_pool2d(p, 2)
        g = F.avg_pool2d(g, 2)
    return torch.stack(vals).mean()


class SideRefinerLoss(nn.Module):
    """Charbonnier + 0.1 FFT + 0.05 MS-SSIM (per the Side-Refiner recipe)."""

    def __init__(self, lambda_fft: float = 0.10, lambda_msssim: float = 0.05,
                 charb_eps: float = 1e-3):
        super().__init__()
        self.lambda_fft = lambda_fft
        self.lambda_msssim = lambda_msssim
        self.charb_eps = charb_eps

    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> dict[str, torch.Tensor]:
        l_charb = charbonnier(pred, gt, self.charb_eps)
        l_fft = fft_loss(pred, gt, self.lambda_fft, 0.0)
        l_ms = self.lambda_msssim * (1.0 - ms_ssim(pred, gt))
        total = l_charb + l_fft + l_ms
        return {"total": total, "charb": l_charb, "fft": l_fft, "ms_ssim": l_ms}
