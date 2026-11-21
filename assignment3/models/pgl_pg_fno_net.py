#!/usr/bin/env python
"""PGL-Net-T + PG-FNO main model (HW3 Stage 2).

Loads the released RW2AH PGL-Net-T checkpoint, injects the two PGFNOBlock
instances at H/8 and H/4, and exposes helpers for the two-phase differential
learning-rate training, checkpoint save/load and EMA.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
HW1 = ROOT.parent / "hw1"

from .pg_fno_block import PGFNOBlock, attach_pg_fno, pg_fno_parameter_count  # noqa: E402

__all__ = ["build_model", "build_base_model", "configure_trainable", "load_checkpoint",
           "save_checkpoint", "ModelEMA", "PGL_PG_FNO_CONFIG"]

PGL_PG_FNO_CONFIG = {
    "base": "PGL-Net-T (RW2AH released checkpoint rw2ah_pglnet_t.pk)",
    "base_params": 780283,
    "injection": {
        "bottleneck": {"layer": "layers.3", "channels": 192, "resolution": "H/8 (32x32 @256 crop)", "modes": 12},
        "h4_decoder": {"layer": "layers.4", "channels": 96, "resolution": "H/4 (64x64 @256 crop)", "modes": 16},
    },
    "block": "PGFNOBlock (physics gate + amplitude-phase FNO + local dilated conv, gamma init 0.1)",
}


def _pglnet_src_path() -> Path:
    """Prefer a local vendored copy (for the TA package); fall back to hw1."""
    local = ROOT / "third_party" / "pgl_net" / "PGLNet.py"
    if local.is_file():
        return local
    return HW1 / "third_party" / "pgl_net" / "PGLNet.py"


def _safe_load_pglnet_state():
    """Load the PGL-Net-T state dict from the hw1 vendored checkpoint."""
    sys.path.insert(0, str(_pglnet_src_path().parent))
    import PGLNet

    vendored = ROOT / "third_party" / "pgl_net" / "rw2ah_pglnet_t.pk"
    path = vendored if vendored.is_file() else (HW1 / "checkpoints" / "pgl_net" / "rw2ah_pglnet_t.pk")
    if not path.is_file():
        raise FileNotFoundError(f"missing checkpoint: {path}")
    import hashlib
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    expected = "1ea04c0d993cc2e5aa6a36a6c77f5ac7719d4faa1fede551c3bc8db23f89526e"
    if sha != expected:
        raise RuntimeError(f"checkpoint SHA-256 mismatch: {path}")

    state = _safe_torch_load(path)
    return PGLNet, state["model"]


def _safe_torch_load(path: Path) -> dict:
    allowed = [np.core.multiarray.scalar, np.dtype, np.dtypes.Float64DType]
    with torch.serialization.safe_globals(allowed):
        return torch.load(path, map_location="cpu", weights_only=True)


def build_model(device: str | torch.device = "cuda") -> nn.Module:
    """Build PGL-Net-T + PG-FNO with released weights, on ``device``."""
    PGLNet, state = _safe_load_pglnet_state()
    model = PGLNet.pglnet_t()
    model.load_state_dict(state, strict=True)
    attach_pg_fno(model)
    return model.to(device).eval()


def build_model_arch(device: str | torch.device = "cuda") -> nn.Module:
    """Build the PGL-Net-T + PG-FNO architecture with random weights.

    Self-contained: needs only a vendored PGLNet.py (local third_party copy,
    or the hw1 fallback) and models/pg_fno_block.py -- no checkpoint reads.
    Used by ``load_checkpoint`` and the TA reproduction package.
    """
    sys.path.insert(0, str(_pglnet_src_path().parent))
    import PGLNet

    model = PGLNet.pglnet_t()
    attach_pg_fno(model)
    return model.to(device).eval()


def build_base_model(device: str | torch.device = "cuda") -> nn.Module:
    """Build the plain PGL-Net-T (no PG-FNO) with released weights."""
    PGLNet, state = _safe_load_pglnet_state()
    model = PGLNet.pglnet_t()
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def configure_trainable(model: nn.Module, phase: str) -> None:
    """Set ``requires_grad`` for the training schedule.

    phase 'A': only the PG-FNO blocks train (frozen backbone).
    phase 'B': PG-FNO blocks + the decoder half train at a lower differential LR.
    phase 'full': every parameter trains (end-to-end differential-LR fine-tune);
               the operator gets a high LR and the whole backbone a low LR so
               BatchNorm and downstream layers co-adapt to the operator from
               step 1 (avoids the frozen-BN distribution mismatch that collapses
               val PSNR under a frozen Phase A).
    """
    if phase not in ("A", "B", "full"):
        raise ValueError("phase must be 'A', 'B' or 'full'")
    for name, param in model.named_parameters():
        is_pgfno = name.startswith("layers.3.1.") or name.startswith("layers.4.0.")
        is_decoder = (
            name.startswith("ups.")
            or name.startswith("fusions.")
            or name.startswith("outconv.")
            or name.startswith("layers.4.1.")
            or name.startswith("layers.5.")
            or name.startswith("layers.6.")
        )
        if phase == "full":
            param.requires_grad = True
        else:
            param.requires_grad = is_pgfno or (phase == "B" and is_decoder)


def parameter_groups(model: nn.Module, phase: str, lr_operator: float, lr_backbone: float):
    """Return two parameter groups: the PG-FNO operator and the backbone."""
    op_params, backbone_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_pgfno = name.startswith("layers.3.1.") or name.startswith("layers.4.0.")
        (op_params if is_pgfno else backbone_params).append(param)
    groups = [{"params": op_params, "lr": lr_operator}]
    if phase in ("B", "full") and backbone_params:
        groups.append({"params": backbone_params, "lr": lr_backbone})
    return groups


def load_checkpoint(path: Path, device: str | torch.device = "cuda") -> nn.Module:
    model = build_model_arch(device)
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state["model"], strict=True)
    return model


def save_checkpoint(model: nn.Module, path: Path, extra: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model": model.state_dict(), "config": PGL_PG_FNO_CONFIG}
    if extra:
        payload["extra"] = extra
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


class ModelEMA:
    """Exponential moving average over the full model state dict."""

    def __init__(self, model: nn.Module, decay: float = 0.995):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module):
        for name, param in model.state_dict().items():
            if param.dtype.is_floating_point or param.dtype.is_complex:
                self.shadow[name].mul_(self.decay).add_(param.detach(), alpha=1.0 - self.decay)
            else:
                self.shadow[name] = param.detach().clone()

    @torch.no_grad()
    def copy_to(self, model: nn.Module):
        model.load_state_dict(self.shadow)
