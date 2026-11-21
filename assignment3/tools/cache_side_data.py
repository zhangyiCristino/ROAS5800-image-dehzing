#!/usr/bin/env python
"""Pre-cache the frozen base output J_0 and read-only multi-scale features
(F_2 at H/4, F_3 at H/8) for every image in a manifest, as fp16 .pt files.

This lets the Side-Refiner train without any base forward pass (10-15 min on
RTX 4060). The base is never modified; hooks only read features.

Run:
  python tools/cache_side_data.py --manifest data/manifests/disjoint/train.jsonl --cache cache/side905
  python tools/cache_side_data.py --manifest data/manifests/train_1406.jsonl --cache cache/side1406
  python tools/cache_side_data.py --manifest data/manifests/disjoint/test.jsonl --cache cache/side_test
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.pgl_pg_fno_net import build_base_model  # noqa: E402


def resolve(rel):
    p = Path(rel)
    return p if p.is_absolute() else (ROOT / p).resolve()


def load_jsonl(path):
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    args.cache.mkdir(parents=True, exist_ok=True)
    rows = load_jsonl(args.manifest)
    model = build_base_model(args.device).eval()
    mods = dict(model.named_modules())
    feats = {}
    mods["layers.2"].register_forward_hook(lambda m, i, o: feats.update(F2=o.detach()))
    mods["layers.3"].register_forward_hook(lambda m, i, o: feats.update(F3=o.detach()))

    t0 = time.perf_counter()
    done = 0
    with torch.inference_mode():
        for r in rows:
            cid = r["id"]
            out_path = args.cache / f"{cid}.pt"
            if out_path.exists():
                done += 1
                continue
            hazy = np.asarray(Image.open(resolve(r["hazy"])).convert("RGB")).copy()
            x = torch.from_numpy(hazy).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            h, w = x.shape[-2:]
            xp = F.pad(x, (0, (-w) % 64, 0, (-h) % 64), mode="reflect").to(args.device)
            feats.clear()
            j0 = model(xp)[..., :h, :w].clamp(0, 1)
            f2 = feats["F2"][..., :h // 4, :w // 4]
            f3 = feats["F3"][..., :h // 8, :w // 8]
            torch.save({"J0": j0[0].half().cpu(), "F2": f2[0].half().cpu(),
                        "F3": f3[0].half().cpu(), "h": h, "w": w}, out_path)
            done += 1
            if done % 100 == 0:
                print(f"cached {done}/{len(rows)}  elapsed={time.perf_counter()-t0:.0f}s", flush=True)
    print(f"DONE cached {done}/{len(rows)} -> {args.cache}  ({time.perf_counter()-t0:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
