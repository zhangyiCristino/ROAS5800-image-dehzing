#!/usr/bin/env python
"""Train the Side-Tuning PG-FNO Refiner on pre-cached base outputs.

The base PGL-Net-T is frozen and never run here; training uses cached J_0/F_2/F_3.
Dual-track: --train-cache cache/side905 (Track 1) or cache/side1406 (Track 2).
Full-image training, geometric augmentation (applied to images AND features),
Charbonnier + 0.1 FFT + 0.05 MS-SSIM, lr 3e-4 cosine, AdamW, grad clip, EMA.
Validates on val_102, keeps best_val, evaluates test-352 (standard + TTA) at end.

Run:
  python tools/train_side_refiner.py --track 2 --train-cache cache/side1406 \
      --train-manifest data/manifests/train_1406.jsonl --epochs 60
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

from models.side_refiner import SideRefiner  # noqa: E402
from models.losses import SideRefinerLoss  # noqa: E402
from models.pgl_pg_fno_net import ModelEMA  # noqa: E402
from tools.unified_metrics import Lpips, psnr, ssim  # noqa: E402


def resolve(rel):
    p = Path(rel)
    return p if p.is_absolute() else (ROOT / p).resolve()


def load_jsonl(path):
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def read_rgb(path):
    return np.asarray(Image.open(path).convert("RGB")).copy()


def preload_cache(rows, cache_dir):
    """Load I, gt (uint8) and cached J0/F2/F3 (fp16 CPU tensors) into RAM."""
    data = []
    for r in rows:
        cp = Path(cache_dir) / f"{r['id']}.pt"
        if not cp.exists():
            continue
        c = torch.load(cp, map_location="cpu", weights_only=True)
        data.append({"id": r["id"],
                     "I": read_rgb(resolve(r["hazy"])),
                     "gt": read_rgb(resolve(r["gt"])),
                     "J0": c["J0"], "F2": c["F2"], "F3": c["F3"]})
    return data


def augment(sample, rng):
    """Geometric augmentation applied consistently to images and features."""
    I = torch.from_numpy(sample["I"]).permute(2, 0, 1).float() / 255.0
    gt = torch.from_numpy(sample["gt"]).permute(2, 0, 1).float() / 255.0
    J0, F2, F3 = sample["J0"].float(), sample["F2"].float(), sample["F3"].float()
    if rng.random() < 0.5:
        I, gt, J0 = I.flip(-1), gt.flip(-1), J0.flip(-1)
        F2, F3 = F2.flip(-1), F3.flip(-1)
    if rng.random() < 0.5:
        I, gt, J0 = I.flip(-2), gt.flip(-2), J0.flip(-2)
        F2, F3 = F2.flip(-2), F3.flip(-2)
    k = int(rng.integers(0, 4))
    if k:
        I, gt, J0 = torch.rot90(I, k, [-2, -1]), torch.rot90(gt, k, [-2, -1]), torch.rot90(J0, k, [-2, -1])
        F2, F3 = torch.rot90(F2, k, [-2, -1]), torch.rot90(F3, k, [-2, -1])
    return I, gt, J0, F2, F3


def pad64(x):
    h, w = x.shape[-2:]
    return F.pad(x, (0, (-w) % 64, 0, (-h) % 64), mode="reflect"), h, w


def crop_batch(samples, crop, rng, device):
    """Crop a batch of cached samples (image + features aligned), with augmentation."""
    Is, gts, J0s, F2s, F3s = [], [], [], [], []
    for s in samples:
        I = torch.from_numpy(s["I"]).permute(2, 0, 1).float() / 255.0
        gt = torch.from_numpy(s["gt"]).permute(2, 0, 1).float() / 255.0
        J0, F2, F3 = s["J0"].float(), s["F2"].float(), s["F3"].float()
        H, W = I.shape[-2:]
        if H < crop or W < crop:
            ph, pw = max(0, crop - H), max(0, crop - W)
            I = F.pad(I, (0, pw, 0, ph), mode="reflect"); gt = F.pad(gt, (0, pw, 0, ph), mode="reflect")
            J0 = F.pad(J0, (0, pw, 0, ph), mode="reflect")
            F2 = F.pad(F2, (0, pw // 4, 0, ph // 4), mode="reflect")
            F3 = F.pad(F3, (0, pw // 8, 0, ph // 8), mode="reflect")
            H, W = I.shape[-2:]
        top = int(rng.integers(0, (H - crop) // 8 + 1)) * 8
        left = int(rng.integers(0, (W - crop) // 8 + 1)) * 8
        Ic = I[:, top:top + crop, left:left + crop]
        gc = gt[:, top:top + crop, left:left + crop]
        Jc = J0[:, top:top + crop, left:left + crop]
        F2c = F2[:, top // 4:(top + crop) // 4, left // 4:(left + crop) // 4]
        F3c = F3[:, top // 8:(top + crop) // 8, left // 8:(left + crop) // 8]
        if rng.random() < 0.5:
            Ic, gc, Jc, F2c, F3c = [t.flip(-1) for t in (Ic, gc, Jc, F2c, F3c)]
        if rng.random() < 0.5:
            Ic, gc, Jc, F2c, F3c = [t.flip(-2) for t in (Ic, gc, Jc, F2c, F3c)]
        k = int(rng.integers(0, 4))
        if k:
            Ic, gc, Jc, F2c, F3c = [torch.rot90(t, k, [-2, -1]) for t in (Ic, gc, Jc, F2c, F3c)]
        Is.append(Ic); gts.append(gc); J0s.append(Jc); F2s.append(F2c); F3s.append(F3c)
    return (torch.stack(Is).to(device), torch.stack(gts).to(device), torch.stack(J0s).to(device),
            torch.stack(F2s).to(device), torch.stack(F3s).to(device))


@torch.no_grad()
def infer_refiner(model, I, J0, F2, F3, device, tta=False):
    I, J0 = I.to(device), J0.to(device)
    F2, F3 = F2.to(device), F3.to(device)
    out = model(I.unsqueeze(0), J0.unsqueeze(0), F2.unsqueeze(0), F3.unsqueeze(0))[0]
    if tta:
        out = (out + model(I.flip(-1).unsqueeze(0), J0.flip(-1).unsqueeze(0),
                           F2.flip(-1).unsqueeze(0), F3.flip(-1).unsqueeze(0))[0].flip(-1)) / 2.0
    return (out.clamp(0, 1).cpu().permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)


def evaluate(model, data, device, tta=False, lpips=None):
    rows = []
    for s in data:
        I = torch.from_numpy(s["I"]).permute(2, 0, 1).float() / 255.0
        gt = s["gt"]
        pred = infer_refiner(model, I, s["J0"].float(), s["F2"].float(), s["F3"].float(), device, tta)
        rec = {"id": s["id"], "psnr": psnr(pred, gt), "ssim": ssim(pred, gt)}
        if lpips is not None:
            rec["lpips"] = lpips.score(pred, gt)
        rows.append(rec)
    out = {"count": len(rows),
           "psnr": round(float(np.mean([r["psnr"] for r in rows])), 6),
           "ssim": round(float(np.mean([r["ssim"] for r in rows])), 6)}
    if lpips is not None:
        out["lpips"] = round(float(np.mean([r["lpips"] for r in rows])), 6)
    return out, rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", type=int, default=2)
    ap.add_argument("--train-cache", type=Path, required=True)
    ap.add_argument("--train-manifest", type=Path, required=True)
    ap.add_argument("--val-cache", type=Path, default=ROOT / "cache" / "side_val")
    ap.add_argument("--val-manifest", type=Path, default=ROOT / "data" / "manifests" / "disjoint" / "val.jsonl")
    ap.add_argument("--test-cache", type=Path, default=ROOT / "cache" / "side_test")
    ap.add_argument("--test-manifest", type=Path, default=ROOT / "data" / "manifests" / "disjoint" / "test.jsonl")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--crop", type=int, default=0, help="crop size for fast batch training (0=full image)")
    ap.add_argument("--batch", type=int, default=4, help="batch size when --crop>0")
    ap.add_argument("--val-every", type=int, default=5)
    ap.add_argument("--ema-decay", type=float, default=0.995)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--no-test", action="store_true")
    args = ap.parse_args()

    if args.out is None:
        args.out = ROOT / "checkpoints" / f"side_refiner_track{args.track}"
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = args.device

    print(f"[Track {args.track}] preloading train cache...", flush=True)
    train_manifest = load_jsonl(args.train_manifest)
    val_manifest = load_jsonl(args.val_manifest)
    train_data = preload_cache(train_manifest, args.train_cache)
    val_data = preload_cache(val_manifest, args.val_cache)
    if not train_data or not val_data or len(train_data) != len(train_manifest) or len(val_data) != len(val_manifest):
        raise RuntimeError("Training or validation cache is incomplete. Run the cache commands in README.md.")
    print(f"train={len(train_data)} val={len(val_data)}", flush=True)

    model = SideRefiner().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"refiner params={n_params:,}", flush=True)
    loss_fn = SideRefinerLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    total_steps = args.epochs * max(1, len(train_data) // args.accum)
    ema = ModelEMA(model, decay=args.ema_decay)
    eval_model = SideRefiner().to(device)
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    best_val = {"psnr": -1.0, "ssim": 0.0}
    val_history = []
    step = 0
    bs = args.batch if args.crop > 0 else 1
    total_steps = args.epochs * max(1, len(train_data) // (bs * args.accum))
    t0 = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = rng.permutation(len(train_data))
        opt.zero_grad(set_to_none=True)
        epoch_loss = 0.0; nb = 0; micro = 0
        for bi in range(0, len(order), bs):
            batch_idx = order[bi:bi + bs]
            if args.crop > 0:
                samples = [train_data[int(i)] for i in batch_idx]
                Ib, gtb, J0b, F2b, F3b = crop_batch(samples, args.crop, rng, device)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = model(Ib, J0b, F2b, F3b)
                    losses = loss_fn(out, gtb)
                    loss = losses["total"] / args.accum
            else:
                I, gt, J0, F2, F3 = augment(train_data[int(batch_idx[0])], rng)
                Ip, h, w = pad64(I)
                J0p, _, _ = pad64(J0)
                F2p = F.pad(F2, (0, (Ip.shape[-1] // 4 - F2.shape[-1]), 0, (Ip.shape[-2] // 4 - F2.shape[-2])), mode="reflect")
                F3p = F.pad(F3, (0, (Ip.shape[-1] // 8 - F3.shape[-1]), 0, (Ip.shape[-2] // 8 - F3.shape[-2])), mode="reflect")
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = model(Ip.unsqueeze(0).to(device), J0p.unsqueeze(0).to(device),
                                F2p.unsqueeze(0).to(device), F3p.unsqueeze(0).to(device))[..., :h, :w]
                    losses = loss_fn(out, gt.unsqueeze(0).to(device))
                    loss = losses["total"] / args.accum
            loss.backward()
            epoch_loss += float(losses["total"].detach()); nb += 1; micro += 1
            if micro % args.accum == 0:
                lr = args.lr * 0.5 * (1 + np.cos(np.pi * step / total_steps))
                opt.param_groups[0]["lr"] = lr
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); ema.update(model); opt.zero_grad(set_to_none=True)
                step += 1
        print(f"[epoch {epoch}/{args.epochs}] loss={epoch_loss/max(1,nb):.5f} "
              f"elapsed={time.perf_counter()-t0:.0f}s", flush=True)
        if epoch % args.val_every == 0 or epoch == args.epochs:
            eval_model.load_state_dict(ema.shadow)
            val, _ = evaluate(eval_model, val_data, device)
            print(f"[epoch {epoch}/{args.epochs}] loss={epoch_loss/max(1,nb):.5f} val psnr={val['psnr']} "
                  f"ssim={val['ssim']} best={best_val['psnr']} elapsed={time.perf_counter()-t0:.0f}s", flush=True)
            val_history.append({"epoch": epoch, **val})
            if val["psnr"] > best_val["psnr"]:
                best_val = {"psnr": val["psnr"], "ssim": val["ssim"], "epoch": epoch}
                args.out.mkdir(parents=True, exist_ok=True)
                tmp = args.out / "best_val_ema.pth.tmp"
                torch.save({"model": ema.shadow, "extra": {"ema_shadow": ema.shadow, "best_val": best_val,
                           "track": args.track, "epoch": epoch}}, tmp)
                tmp.replace(args.out / "best_val_ema.pth")
                print(f"[epoch {epoch}] saved best_val_ema.pth (psnr={best_val['psnr']})", flush=True)

    train_peak = round(torch.cuda.max_memory_allocated() / 2**20, 2) if device.startswith("cuda") else None
    print(f"training done. peak_vram={train_peak} MiB best_val={best_val}", flush=True)

    if not args.no_test and (args.out / "best_val_ema.pth").exists():
        ckpt = torch.load(args.out / "best_val_ema.pth", map_location="cpu", weights_only=True)
        test_model = SideRefiner().to(device)
        test_model.load_state_dict(ckpt["model"], strict=True)
        test_data = preload_cache(load_jsonl(args.test_manifest), args.test_cache)
        lpips = Lpips(device=device)
        std, std_rows = evaluate(test_model, test_data, device, tta=False, lpips=lpips)
        tta, tta_rows = evaluate(test_model, test_data, device, tta=True, lpips=lpips)
        lpips.close()
        result = {"method": f"PG-FNO-Side-Refiner (Track {args.track})", "track": args.track,
                  "train_split": "Official-1406" if args.track == 2 else "Clean-Disjoint-905",
                  "best_val": best_val, "val_history": val_history, "train_peak_vram_mib": train_peak,
                  "refiner_params": n_params, "test_standard": std, "test_tta": tta,
                  "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                  "sota_target": {"psnr": 22.943, "ssim": 0.6545},
                  "sota_pass_standard": bool(std["psnr"] >= 22.943 and std["ssim"] >= 0.6545),
                  "sota_pass_tta": bool(tta["psnr"] >= 22.943 and tta["ssim"] >= 0.6545)}
        out_json = ROOT / "results" / f"side_refiner_track{args.track}.json"
        out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        (ROOT / "results" / f"side_refiner_track{args.track}_per_image.json").write_text(
            json.dumps({"standard": std_rows, "tta": tta_rows}, indent=2), encoding="utf-8")
        print(f"\n== TEST Track {args.track} ==\n  standard: {std}\n  TTA     : {tta}", flush=True)
        print(f"  SOTA pass: standard={result['sota_pass_standard']} TTA={result['sota_pass_tta']}", flush=True)
        print(f"  -> {out_json}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
