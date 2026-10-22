"""A common, traceable small-sample dehazing evaluation and image export.

Run from hw1: python -m dehaze.evaluation --manifest data/manifest.json
              --output results --max-side 512 --runs 3
The helpers are also used by the Gradio UI; importing them does not load models.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import time
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METHODS = ("PGL-Net-T", "HNDiff-FocalNet", "PixRestore-S")
UNTREATED = "Untreated"
PROTOCOL = {
    "scope": "Small fixed held-out test subset; not a full benchmark or a reproduction of paper scores",
    "color_space": "RGB, all three channels; no Y conversion, histogram matching, or post-hoc alignment",
    "preprocessing": "Input and reference must have identical original dimensions. Downsample both once with PIL BICUBIC to the same aspect-preserving size, max side 512 by default, round(W*scale), round(H*scale), minimum 1; never upscale or crop.",
    "model_input": "The same prepared RGB uint8 input is passed to every model with max_side=None. The model wrapper reflect-pads bottom/right to multiples of 64, then removes that padding. No geometric crop remains in metrics.",
    "psnr": "skimage PSNR on RGB float64 uint8/255, data_range=1, full image, no border shave",
    "ssim": "skimage RGB SSIM, channel_axis=-1, data_range=1, gaussian_weights=True, sigma=1.5, win_size=11, use_sample_covariance=False; requires both axes >=11. Standard skimage SSIM averages the valid interior of the Gaussian window.",
    "mae": "Mean absolute RGB error on float64 uint8/255; range [0,1]",
    "output_quantization": "Metrics use the same clipped, rounded uint8 RGB model output that is saved and displayed in the UI",
    "aggregation": "Equal-weight arithmetic mean of per-image PSNR, SSIM, and MAE; no pooled-MSE PSNR",
    "randomness": "Same explicit seed for every call. Repeated runs measure timing; the first result is the quality output. Output SHA-256 is recorded and repeated-output equality is checked.",
    "timing": "Per-image median of --runs complete forward GPU timings (manager gpu_ms); loading, warmup, transfers and image IO are excluded from gpu_ms. Wall elapsed_ms, loading and warmup remain separately recorded. Sequential batch 1; the manager holds one model at a time.",
    "precision": "Released inference configuration per model: PGL-Net-T and HNDiff-FocalNet FP32, PixRestore-S BF16 autocast when supported. This is a deployment comparison, not a controlled equal-precision architecture experiment.",
    "weights": "PGL-Net-T and HNDiff use RW2AH-specific released weights; PixRestore-S uses the released general restoration checkpoint. Training data/compute are not equalized.",
    "limitations": "The deterministic first-N selection may overrepresent adjacent scene groups. Resizing can change apparent haze/noise and all absolute scores. Do not claim full-dataset SOTA or robot-task gains from these results. No paper-table values are mixed with local metrics.",
    "nonfinite_json": "Perfect-match PSNR is +infinity in compute_metrics; strict JSON exports null for nonfinite numbers and preserves perfect_match=true. CSV writes inf.",
}


def _rgb_uint8(image: np.ndarray, name: str) -> np.ndarray:
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
        raise TypeError(f"{name} must be a numpy uint8 HWC RGB array")
    if image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 1:
        raise ValueError(f"{name} must have nonempty shape (height, width, 3)")
    return image


def prepare_pair(hazy_rgb: np.ndarray, clear_rgb: np.ndarray | None,
                 max_side: int | None = 512) -> tuple[np.ndarray, np.ndarray | None]:
    """Validate an aligned pair and resize both identically, without upsampling.

    A mismatched reference is an error: silently resizing different image content
    would invalidate full-reference metrics. Model padding is separate and removed.
    """
    hazy_rgb = _rgb_uint8(hazy_rgb, "Hazy image")
    if clear_rgb is not None:
        clear_rgb = _rgb_uint8(clear_rgb, "Clear reference")
        if hazy_rgb.shape != clear_rgb.shape:
            raise ValueError("Hazy image and clear reference must have identical original dimensions; choose an aligned pair.")
    if max_side is not None and (isinstance(max_side, bool) or not isinstance(max_side, (int, np.integer)) or max_side < 1):
        raise ValueError("max_side must be a positive integer or None")
    height, width = hazy_rgb.shape[:2]
    scale = min(1.0, int(max_side) / max(height, width)) if max_side is not None else 1.0
    size = max(1, round(width * scale)), max(1, round(height * scale))

    def resize(image: np.ndarray | None):
        if image is None:
            return None
        if size == (width, height):
            return np.ascontiguousarray(image).copy()
        return np.array(Image.fromarray(image).resize(size, Image.Resampling.BICUBIC))

    return resize(hazy_rgb), resize(clear_rgb)


def compute_metrics(prediction_rgb: np.ndarray, reference_rgb: np.ndarray) -> dict[str, float]:
    """Full-reference RGB metrics, with an explicit Gaussian SSIM protocol."""
    prediction_rgb = _rgb_uint8(prediction_rgb, "Prediction")
    reference_rgb = _rgb_uint8(reference_rgb, "Reference")
    if prediction_rgb.shape != reference_rgb.shape:
        raise ValueError("Prediction and reference must have identical dimensions for PSNR/SSIM")
    if min(prediction_rgb.shape[:2]) < 11:
        raise ValueError("PSNR/SSIM comparison requires an image of at least 11 x 11 pixels for the fixed Gaussian SSIM window")
    pred = prediction_rgb.astype(np.float64) / 255.0
    ref = reference_rgb.astype(np.float64) / 255.0
    mae = float(np.mean(np.abs(pred - ref)))
    psnr = float("inf") if mae == 0 else float(peak_signal_noise_ratio(ref, pred, data_range=1.0))
    return {
        "psnr": psnr,
        "ssim": float(structural_similarity(ref, pred, channel_axis=-1, data_range=1.0,
                                             gaussian_weights=True, sigma=1.5, win_size=11,
                                             use_sample_covariance=False)),
        "mae": mae,
    }


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest(path: str | Path, verify_hashes: bool = True,
                  root: str | Path | None = None) -> dict:
    """Load local pairs, optionally validating all source bytes before inference."""
    path = Path(path).resolve()
    root = Path(root).resolve() if root is not None else ROOT
    manifest = json.loads(path.read_text(encoding="utf-8"))
    pairs = manifest.get("pairs")
    if not isinstance(pairs, list) or not pairs:
        raise ValueError("Manifest must contain a nonempty pairs list")
    declared_count = manifest.get("selection", {}).get("count")
    if declared_count is not None and declared_count != len(pairs):
        raise ValueError(f"Incomplete manifest: declares {declared_count} pairs, contains {len(pairs)}")
    if manifest.get("download_status", {}).get("complete") is False:
        raise ValueError("Dataset download is incomplete; finish preparing the manifest before evaluation")
    ids = set()
    for pair in pairs:
        if pair.get("id") in ids:
            raise ValueError(f"Duplicate manifest id: {pair.get('id')}")
        ids.add(pair.get("id"))
        for role in ("hazy", "clear"):
            candidate = (root / pair[role]).resolve()
            if not candidate.is_relative_to(root):
                raise ValueError(f"Dataset path must stay inside project: {pair[role]}")
            if not candidate.is_file():
                raise FileNotFoundError(candidate)
            expected = pair.get(role + "_source", {}).get("sha256")
            if verify_hashes and not expected:
                raise ValueError(f"Missing source hash for {pair['id']} {role}")
            if verify_hashes and _hash_file(candidate) != expected:
                raise ValueError(f"Dataset SHA-256 mismatch: {candidate}")
    return manifest


def _load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.array(image.convert("RGB"))


def _strict_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _strict_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_strict_json(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(_strict_json(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _mean(rows: list[dict], key: str) -> float | None:
    values = [row[key] for row in rows if row.get(key) is not None]
    return float(statistics.mean(values)) if values else None


def _median_available(values) -> float | None:
    present = [float(value) for value in values if value is not None]
    return float(statistics.median(present)) if present else None


def _max_available(values) -> float | None:
    present = [float(value) for value in values if value is not None]
    return max(present) if present else None


def _export_overview(output: Path, pairs: list[dict], methods: list[str], max_rows: int = 3) -> str:
    """Fixed first rows, for inspection only; every full output is also exported."""
    thumb_w, thumb_h, label_h, gap = 288, 200, 26, 10
    labels = [UNTREATED, "Clear reference", *methods]
    chosen = pairs[:max_rows]
    canvas = Image.new("RGB", (len(labels) * (thumb_w + gap) + gap,
                               len(chosen) * (thumb_h + label_h + gap) + gap), "#edf1f5")
    draw = ImageDraw.Draw(canvas)
    for row_index, pair in enumerate(chosen):
        safe_id = pair["safe_id"]
        paths = [output / "images" / "input" / f"{safe_id}.png",
                 output / "images" / "reference" / f"{safe_id}.png"]
        paths += [output / "images" / _safe_name(method) / f"{safe_id}.png" for method in methods]
        for column, (label, path) in enumerate(zip(labels, paths)):
            x, y = gap + column * (thumb_w + gap), gap + row_index * (thumb_h + label_h + gap)
            draw.text((x, y + 5), f"{pair['id']} | {label}", fill="#172b45")
            with Image.open(path) as image:
                thumbnail = image.convert("RGB")
                thumbnail.thumbnail((thumb_w, thumb_h), Image.Resampling.LANCZOS)
                canvas.paste(thumbnail, (x + (thumb_w - thumbnail.width) // 2,
                                         y + label_h + (thumb_h - thumbnail.height) // 2))
    name = "comparison_grid.png"
    canvas.save(output / name)
    return name


def _safe_name(value: str) -> str:
    import re
    result = re.sub(r"[^A-Za-z0-9_.-]", "_", value)
    if not result or result in {".", ".."}:
        raise ValueError("Invalid image/method identifier")
    return result


def evaluate(manager: Any, manifest_path: str | Path = ROOT / "data/manifest.json",
             output: str | Path = ROOT / "results", max_side: int | None = 512,
             runs: int = 3, seed: int = 0, methods: list[str] | None = None,
             verify_hashes: bool = True) -> dict:
    """Evaluate the fixed manifest using the public ModelManager.restore API.

    Manager is injected so this module's helpers and protocol can be checked on
    CPU without importing/loading any model or touching CUDA.
    """
    if isinstance(runs, bool) or not isinstance(runs, int) or runs < 1:
        raise ValueError("runs must be a positive integer")
    methods = list(DEFAULT_METHODS if methods is None else methods)
    if not methods or len(set(methods)) != len(methods) or any(method not in DEFAULT_METHODS for method in methods):
        raise ValueError(f"methods must be a nonempty unique subset of {DEFAULT_METHODS}")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(manifest_path).resolve()
    manifest = load_manifest(manifest_path, verify_hashes=verify_hashes)
    protocol = dict(PROTOCOL)
    if manifest.get("selection", {}).get("is_full_benchmark"):
        protocol["scope"] = "All paired images in the source release's held-out test split, using the common resized local protocol; not a reproduction of paper-table scores"
        protocol["limitations"] = "All source-release test pairs are included. Resizing can change apparent haze/noise and absolute scores. Released models have different training data, inference precision and wrappers. Do not claim a global SOTA ranking or robot-task gains from this experiment. No paper-table values are mixed with local metrics."
    protocol["max_side"] = max_side
    protocol["runs"] = runs
    protocol["seed"] = seed
    start = time.perf_counter()
    prepared, rows = [], []
    status = {"complete": False, "expected_pairs": len(manifest["pairs"]),
              "expected_model_outputs": len(manifest["pairs"]) * len(methods),
              "completed_model_outputs": 0, "completed_by_method": {method: 0 for method in methods},
              "failures": [], "current": "preparing common inputs"}
    _write_json(output / "evaluation_status.json", status)
    for pair in manifest["pairs"]:
        hazy, clear = prepare_pair(_load_rgb(ROOT / pair["hazy"]), _load_rgb(ROOT / pair["clear"]), max_side)
        # Fail before model loading for an invalid reference or SSIM image size.
        metrics = compute_metrics(hazy, clear)
        safe_id = _safe_name(str(pair["id"]))
        height, width = hazy.shape[:2]
        entry = {"id": pair["id"], "safe_id": safe_id, "input": hazy, "reference": clear,
                 "width": width, "height": height}
        prepared.append(entry)
        for role, image in (("input", hazy), ("reference", clear)):
            target = output / "images" / role / (safe_id + ".png")
            target.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(image).save(target)
        rows.append({"id": pair["id"], "method": UNTREATED, "width": width, "height": height,
                     **metrics, "perfect_match": metrics["mae"] == 0, "gpu_ms": 0.0,
                     "forward_ms": 0.0, "elapsed_ms": 0.0, "model_load_ms": 0.0, "warmup_ms": 0.0,
                     "peak_allocated_mib": 0.0, "parameters": 0, "precision": "uint8 input",
                     "seed": seed, "runs": 0, "repeat_outputs_identical": True,
                     "repeat_max_abs_difference_uint8": 0,
                     "output_path": f"images/input/{safe_id}.png", "output_pixel_sha256": hashlib.sha256(hazy.tobytes()).hexdigest(),
                     "runs_metadata": []})
    for method in methods:
        for index, pair in enumerate(prepared, 1):
            status["current"] = {"method": method, "image_id": pair["id"], "index": index}
            _write_json(output / "evaluation_status.json", status)
            predictions, metadata = [], []
            for _ in range(runs):
                try:
                    prediction, meta = manager.restore(pair["input"], method, max_side=None, seed=seed)
                except Exception as error:
                    status["failures"].append({"method": method, "image_id": pair["id"],
                                               "exception": type(error).__name__, "message": str(error)})
                    _write_json(output / "evaluation_status.json", status)
                    raise
                prediction = _rgb_uint8(prediction, "Model prediction")
                if prediction.shape != pair["reference"].shape:
                    raise ValueError(f"{method} output shape mismatch on {pair['id']}")
                predictions.append(prediction)
                metadata.append(dict(meta))
            prediction = predictions[0]
            metrics = compute_metrics(prediction, pair["reference"])
            relative = Path("images") / _safe_name(method) / (pair["safe_id"] + ".png")
            destination = output / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(prediction).save(destination)
            first = metadata[0]
            row = {"id": pair["id"], "method": method, "width": pair["width"], "height": pair["height"],
                   **metrics, "perfect_match": metrics["mae"] == 0,
                   "gpu_ms": _median_available(meta.get("gpu_ms") for meta in metadata),
                   "forward_ms": _median_available(meta.get("forward_ms") for meta in metadata),
                   "elapsed_ms": statistics.median(float(meta["elapsed_ms"]) for meta in metadata),
                   "model_load_ms": sum(float(meta.get("model_load_ms", 0)) for meta in metadata),
                   "warmup_ms": sum(float(meta.get("warmup_ms", 0)) for meta in metadata),
                   "peak_allocated_mib": _max_available(meta.get("peak_allocated_mib") for meta in metadata),
                   "parameters": first.get("parameters"), "precision": first.get("precision"),
                   "seed": seed, "runs": runs,
                   "repeat_outputs_identical": all(np.array_equal(prediction, other) for other in predictions[1:]),
                   "repeat_max_abs_difference_uint8": max((int(np.abs(prediction.astype(np.int16) - other.astype(np.int16)).max()) for other in predictions[1:]), default=0),
                   "output_path": relative.as_posix(), "output_pixel_sha256": hashlib.sha256(prediction.tobytes()).hexdigest(),
                   "runs_metadata": metadata}
            rows.append(row)
            status["completed_model_outputs"] += 1
            status["completed_by_method"][method] += 1
            _write_json(output / "evaluation_status.json", status)
            timing = f"GPU median={row['gpu_ms']:.2f} ms" if row["gpu_ms"] is not None else f"CPU forward median={row['forward_ms']:.2f} ms"
            print(f"{method} {index}/{len(prepared)} {pair['id']}: PSNR={metrics['psnr']:.3f}, SSIM={metrics['ssim']:.4f}, {timing}", flush=True)
    summary = []
    for method in [UNTREATED, *methods]:
        selected = [row for row in rows if row["method"] == method]
        summary.append({"method": method, "image_count": len(selected),
                        "psnr": _mean(selected, "psnr"), "ssim": _mean(selected, "ssim"), "mae": _mean(selected, "mae"),
                        "median_gpu_ms": _median_available(row["gpu_ms"] for row in selected),
                        "mean_gpu_ms": _mean(selected, "gpu_ms"),
                        "median_forward_ms": _median_available(row["forward_ms"] for row in selected),
                        "max_peak_allocated_mib": _max_available(row["peak_allocated_mib"] for row in selected),
                        "parameters": selected[0]["parameters"], "precision": selected[0]["precision"],
                        "perfect_match_count": sum(row["perfect_match"] for row in selected),
                        "all_repeats_identical": all(row["repeat_outputs_identical"] for row in selected),
                        "nonidentical_repeat_image_count": sum(not row["repeat_outputs_identical"] for row in selected),
                        "max_repeat_abs_difference_uint8": max(row["repeat_max_abs_difference_uint8"] for row in selected)})
    overview = _export_overview(output, prepared, methods)
    status["complete"] = True
    status["current"] = "finished"
    _write_json(output / "evaluation_status.json", status)
    result = {"schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "dataset": manifest["dataset"], "split": manifest.get("split"),
              "selection": manifest.get("selection"), "sources": manifest.get("sources"),
              "manifest_sha256": _hash_file(manifest_path), "data_hashes_verified": verify_hashes,
              "protocol": protocol, "device": str(getattr(manager, "device", "unknown")),
              "evaluation_wall_seconds": time.perf_counter() - start,
              "completion_status": status,
              "summary": summary, "per_image": rows, "comparison_grid": overview}
    _write_json(output / "evaluation.json", result)
    _write_json(output / "summary.json", {key: value for key, value in result.items() if key != "per_image"})
    columns = [key for key in rows[0] if key != "runs_metadata"]
    with (output / "per_image.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    with (output / "summary.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/manifest.json"))
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--max-side", type=int, default=512)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--methods", nargs="+", choices=DEFAULT_METHODS, default=list(DEFAULT_METHODS))
    parser.add_argument("--verify-only", action="store_true", help="Verify manifest hashes and image pairs without loading models")
    args = parser.parse_args()
    manifest_path = args.manifest if args.manifest.is_absolute() else ROOT / args.manifest
    if args.verify_only:
        manifest = load_manifest(manifest_path)
        for pair in manifest["pairs"]:
            hazy, clear = prepare_pair(_load_rgb(ROOT / pair["hazy"]), _load_rgb(ROOT / pair["clear"]), args.max_side)
            compute_metrics(hazy, clear)
        print(f"Verified {len(manifest['pairs'])} test pairs, {2 * len(manifest['pairs'])} source hashes, matching image dimensions and metric compatibility.")
        return
    from .models import ModelManager
    manager = ModelManager(device=args.device)
    try:
        output = args.output if args.output.is_absolute() else ROOT / args.output
        result = evaluate(manager, manifest_path, output, args.max_side, args.runs, args.seed, args.methods)
        print(json.dumps(_strict_json(result["summary"]), indent=2, allow_nan=False))
        print(f"Results saved: {output}")
    finally:
        manager.close()


if __name__ == "__main__":
    main()
