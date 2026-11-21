#!/usr/bin/env python
"""Unified metric functions for the HW3 baseline evaluation.

Every method is scored with the *same* functions on the *same* quantised uint8
RGB output, so no baseline can use a different yardstick (SOP rule 0.1.4).

Conventions
  * images are uint8 HWC RGB at native resolution
  * PSNR / SSIM are computed on RGB (3-channel mean) and additionally on the
    BT.601 luma channel (Y), data_range = 255
  * LPIPS uses the Alex net with inputs mapped to [-1, 1]
"""
from __future__ import annotations

import numpy as np
from skimage.metrics import structural_similarity


def to_luma_u8(rgb: np.ndarray) -> np.ndarray:
    """BT.601 luma of a uint8 RGB image, returned as float64 in [0, 255]."""
    return (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]).astype(np.float64)


def psnr(pred_u8: np.ndarray, ref_u8: np.ndarray) -> float:
    """PSNR on the RGB mean squared error, data_range = 255 (uint8)."""
    if pred_u8.shape != ref_u8.shape or pred_u8.dtype != np.uint8 or ref_u8.dtype != np.uint8:
        raise ValueError("PSNR requires identically shaped uint8 RGB arrays")
    error = pred_u8.astype(np.float64) - ref_u8.astype(np.float64)
    mse = float(np.mean(error ** 2))
    if mse == 0.0:
        return float("inf")
    return float(10.0 * np.log10(255.0 ** 2 / mse))


def psnr_y(pred_u8: np.ndarray, ref_u8: np.ndarray) -> float:
    mse = float(np.mean((to_luma_u8(pred_u8) - to_luma_u8(ref_u8)) ** 2))
    if mse == 0.0:
        return float("inf")
    return float(10.0 * np.log10(255.0 ** 2 / mse))


def ssim(pred_u8: np.ndarray, ref_u8: np.ndarray) -> float:
    """skimage RGB SSIM, Gaussian window, data_range = 255."""
    if pred_u8.shape != ref_u8.shape or pred_u8.dtype != np.uint8 or ref_u8.dtype != np.uint8:
        raise ValueError("SSIM requires identically shaped uint8 RGB arrays")
    return float(structural_similarity(
        ref_u8, pred_u8, channel_axis=-1, data_range=255,
        gaussian_weights=True, sigma=1.5, win_size=11, use_sample_covariance=False,
    ))


def ssim_y(pred_u8: np.ndarray, ref_u8: np.ndarray) -> float:
    return float(structural_similarity(
        to_luma_u8(ref_u8), to_luma_u8(pred_u8), data_range=255,
        gaussian_weights=True, sigma=1.5, win_size=11, use_sample_covariance=False,
    ))


class Lpips:
    """Thin wrapper around lpips.LPIPS(net='alex') with a lazy, cached load."""

    def __init__(self, device: str = "cuda"):
        self.device = device
        self._model = None

    def _ensure(self):
        if self._model is None:
            import torch
            import lpips
            self._model = lpips.LPIPS(net="alex")
            self._model = self._model.to(self.device).eval()
        return self._model

    def score(self, pred_u8: np.ndarray, ref_u8: np.ndarray) -> float:
        """LPIPS between two uint8 RGB images, mapped to [-1, 1]."""
        import torch
        model = self._ensure()
        pred = torch.from_numpy(pred_u8.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
        ref = torch.from_numpy(ref_u8.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
        pred = (pred * 2 - 1).to(self.device)
        ref = (ref * 2 - 1).to(self.device)
        with torch.inference_mode():
            value = model(pred, ref).item()
        del pred, ref
        return float(value)

    def close(self):
        import torch
        self._model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
