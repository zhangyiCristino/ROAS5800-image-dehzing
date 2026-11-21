#!/usr/bin/env python
"""Run the PG-FNO Side-Refiner demo or the 352-pair RW2AH test.

Run prepare.py before inference. Both entry points use native RGB input and
round their predictions to uint8 before evaluating PSNR and SSIM.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402

from models.pgl_pg_fno_net import load_checkpoint, build_base_model  # noqa: E402
from models.side_refiner import SideRefiner  # noqa: E402
from tools.unified_metrics import psnr, ssim  # noqa: E402

PAD = 64
TEST_COUNT = 352
DEFAULT_CHECKPOINT = ROOT / "checkpoints" / "side_refiner_track2" / "best_val_ema.pth"


class SideModel(torch.nn.Module):
    """Frozen base + Side-Refiner; extracts J0/F2/F3 on the fly via read-only hooks."""

    def __init__(self, refiner: SideRefiner, device):
        super().__init__()
        self.base = build_base_model(device)
        self.base.eval()
        self.refiner = refiner
        self.refiner.eval()
        self._feats = {}
        mods = dict(self.base.named_modules())
        mods["layers.2"].register_forward_hook(lambda m, i, o: self._feats.update(F2=o.detach()))
        mods["layers.3"].register_forward_hook(lambda m, i, o: self._feats.update(F3=o.detach()))

    def forward(self, x):
        h, w = x.shape[-2:]
        self._feats.clear()
        j0 = self.base(x)[..., :h, :w].clamp(0, 1)
        f2 = self._feats["F2"][..., :h // 4, :w // 4]
        f3 = self._feats["F3"][..., :h // 8, :w // 8]
        return self.refiner(x[..., :h, :w], j0, f2, f3)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="run_ta_test.py",
        description="HW3 TA one-click reproduction: PGL-Net-T + PG-FNO dehazing.",
    )
    parser.add_argument(
        "--mode", required=True, choices=["demo", "test"],
        help="demo: dehaze sample_hazy_5imgs and score vs bundled GT; "
             "test: full 352-image official split from data/manifests/disjoint/test.jsonl",
    )
    parser.add_argument(
        "--ckpt", default=str(DEFAULT_CHECKPOINT),
        help="path to checkpoint (defaults to the included Track 2 refiner weights)",
    )
    parser.add_argument(
        "--input_dir", default="./sample_hazy_5imgs",
        help="demo only: input dir containing hazy/ and gt/ subdirs",
    )
    parser.add_argument(
        "--output_dir", default="./ta_demo_out",
        help="demo only: directory where dehazed PNGs are written",
    )
    parser.add_argument(
        "--device", default="cuda",
        help="torch device, default 'cuda' (falls back to CPU when CUDA is unavailable)",
    )
    parser.add_argument(
        "--model", default="side", choices=["side", "base", "pgfno"],
        help="side: frozen base + PG-FNO Side-Refiner (default); base: frozen PGL-Net-T; "
             "pgfno: legacy internal-injection adapter",
    )
    return parser.parse_args(argv)


def resolve_device(requested: str) -> torch.device:
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("NOTE: CUDA is not available; falling back to CPU.", file=sys.stderr)
        return torch.device("cpu")
    return device


def read_rgb(path: Path) -> np.ndarray:
    """Load a PNG as uint8 HWC RGB."""
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB")).copy()


def dehaze(model: torch.nn.Module, device: torch.device, img_u8: np.ndarray) -> np.ndarray:
    """Run one uint8 HWC RGB image through the model, returning uint8 HWC RGB."""
    h, w = img_u8.shape[:2]
    x = torch.from_numpy(img_u8.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    pad_h = (-h) % PAD
    pad_w = (-w) % PAD
    x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
    with torch.inference_mode():
        y = model(x.to(device))
    y = y[:, :, :h, :w]
    y = torch.clamp(y, 0.0, 1.0)
    out = y.squeeze(0).permute(1, 2, 0).mul(255.0).cpu().numpy()
    return np.round(out).astype(np.uint8)


def peak_vram_mib(device: torch.device) -> float | None:
    """Peak allocated CUDA memory in MiB, or None when CUDA is unavailable."""
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / (1024.0 ** 2)
    return None


def load_model(ckpt: Path, device: torch.device, model_type: str = "side") -> torch.nn.Module:
    """Load the checkpoint or exit(2) with a clear error when it is missing."""
    if model_type == "base":
        model = build_base_model(device)
        model.eval()
        return model
    if not ckpt.is_file():
        print(f"ERROR: checkpoint not found: {ckpt}", file=sys.stderr)
        print("       train the model first or pass --ckpt <path to best_val_ema.pth>",
              file=sys.stderr)
        sys.exit(2)
    if model_type == "side":
        state = torch.load(ckpt, map_location="cpu", weights_only=True)
        refiner = SideRefiner().to(device)
        refiner.load_state_dict(state["model"], strict=True)
        model = SideModel(refiner, device)
        model.eval()
        return model
    model = load_checkpoint(ckpt, device)
    model.eval()
    return model


def run_demo(args: argparse.Namespace) -> int:
    ckpt = Path(args.ckpt).expanduser().resolve()
    input_dir = Path(args.input_dir).expanduser().resolve()
    hazy_dir = input_dir / "hazy"
    gt_dir = input_dir / "gt"
    if not hazy_dir.is_dir():
        print(f"ERROR: hazy image directory not found: {hazy_dir}", file=sys.stderr)
        return 1
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    print(f"Loading checkpoint: {ckpt}")
    model = load_model(ckpt, device, args.model)

    files = sorted(hazy_dir.glob("*.png"))
    if not files:
        print(f"ERROR: no *.png images found in {hazy_dir}", file=sys.stderr)
        return 1

    psnr_vals: list[float] = []
    ssim_vals: list[float] = []
    missing_gt = 0
    for path in files:
        hazy = read_rgb(path)
        if min(hazy.shape[:2]) < PAD:
            print(f"ERROR: {path.name} must be at least {PAD} pixels on each side", file=sys.stderr)
            return 1
        out = dehaze(model, device, hazy)
        out_path = output_dir / path.name
        Image.fromarray(out).save(out_path)
        gt_path = gt_dir / path.name
        if not gt_path.is_file():
            print(f"[{path.name}] WARN: GT not found at {gt_path}; metrics skipped")
            missing_gt += 1
            continue
        gt = read_rgb(gt_path)
        p_val = psnr(out, gt)
        s_val = ssim(out, gt)
        psnr_vals.append(p_val)
        ssim_vals.append(s_val)
        print(f"[{path.name}] PSNR={p_val:.4f} dB  SSIM={s_val:.4f}  -> {out_path}")

    count = len(psnr_vals)
    print("--- demo summary ---")
    print(f"images processed: {len(files)}; scored: {count}")
    if count:
        print(f"mean PSNR = {float(np.mean(psnr_vals)):.4f} dB")
        print(f"mean SSIM = {float(np.mean(ssim_vals)):.4f}")
    else:
        print("WARN: no image was scored (missing GT); means are undefined", file=sys.stderr)
    if missing_gt:
        print(f"WARN: {missing_gt} image(s) had no GT and were excluded from the means")
    peak = peak_vram_mib(device)
    if peak is not None:
        print(f"peak CUDA memory allocated: {peak:.1f} MiB")
    return 0


def run_test(args: argparse.Namespace) -> int:
    ckpt = Path(args.ckpt).expanduser().resolve()
    manifest = ROOT / "data" / "manifests" / "disjoint" / "test.jsonl"
    if not manifest.is_file():
        print(f"ERROR: test manifest not found: {manifest}", file=sys.stderr)
        return 1

    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != TEST_COUNT or len({row["id"] for row in rows}) != TEST_COUNT:
        print(f"ERROR: expected {TEST_COUNT} distinct test pairs. Rerun prepare.py.", file=sys.stderr)
        return 1

    pairs = []
    for row in rows:
        hazy_path = ROOT / row["hazy"]
        gt_path = ROOT / row["gt"]
        if not hazy_path.is_file() or not gt_path.is_file():
            print(
                f"ERROR: missing test pair {row['id']}. Run prepare.py --test-data with the "
                "official dataset directory. See README.md.",
                file=sys.stderr,
            )
            return 1
        pairs.append((row["id"], hazy_path, gt_path))

    device = resolve_device(args.device)
    print(f"Loading checkpoint: {ckpt}")
    model = load_model(ckpt, device, args.model)
    psnr_vals: list[float] = []
    ssim_vals: list[float] = []
    for image_id, hazy_path, gt_path in pairs:
        hazy = read_rgb(hazy_path)
        out = dehaze(model, device, hazy)
        gt = read_rgb(gt_path)
        p_val = psnr(out, gt)
        s_val = ssim(out, gt)
        psnr_vals.append(p_val)
        ssim_vals.append(s_val)
        print(f"[{image_id}] PSNR={p_val:.4f} dB  SSIM={s_val:.4f}")

    print("--- test summary ---")
    print(f"manifest rows: {len(rows)}; evaluated: {len(psnr_vals)}; skipped: 0")
    print(f"mean PSNR = {float(np.mean(psnr_vals)):.4f} dB")
    print(f"mean SSIM = {float(np.mean(ssim_vals)):.4f}")
    peak = peak_vram_mib(device)
    if peak is not None:
        print(f"peak CUDA memory allocated: {peak:.1f} MiB")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.mode == "demo":
        return run_demo(args)
    return run_test(args)


if __name__ == "__main__":
    sys.exit(main())
