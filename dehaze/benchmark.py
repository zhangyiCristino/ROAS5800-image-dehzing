"""Read completed local evaluations for the UI without importing any model."""
from __future__ import annotations

from datetime import datetime
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import statistics

METHODS = ("PGL-Net-T", "HNDiff-FocalNet", "PixRestore-S")
ORDER = ("Untreated", *METHODS)
SUMMARY_HEADERS = ["Method", "Count", "Mean PSNR (dB)", "Mean SSIM", "PSNR gain (dB)", "Median GPU (ms)", "Peak memory (MiB)"]
IMAGE_HEADERS = ["Method", "Image size", "PSNR (dB)", "SSIM", "Median GPU (ms)"]


class BenchmarkUnavailable(ValueError):
    pass


def _number(value, digits=2):
    return "—" if value is None else f"{value:.{digits}f}"


def _asset(folder: Path, relative: str) -> str:
    path = (folder / relative).resolve()
    if not path.is_relative_to((folder / "images").resolve()) or not path.is_file():
        raise BenchmarkUnavailable("Some result images are missing. Complete the evaluation, then refresh.")
    return str(path)


def _signature(root: Path):
    paths = [root / "results_full" / name for name in ("evaluation.json", "summary.json", "evaluation_status.json")]
    paths.append(root / "data/manifest_full.json")
    return tuple((str(p), p.stat().st_mtime_ns, p.stat().st_size) for p in paths)


def load_benchmark(root: Path):
    try:
        root = root.resolve()
        signature = _signature(root)
        snapshot = _load_snapshot(str(root), signature)
        if _signature(root) != signature:
            raise BenchmarkUnavailable("Results are being updated. Please refresh shortly.")
        return snapshot
    except BenchmarkUnavailable:
        raise
    except (OSError, ValueError, KeyError, TypeError, AssertionError) as error:
        raise BenchmarkUnavailable("Full results are not ready. Complete the evaluation, then click Refresh full results.") from error


@lru_cache(maxsize=1)
def _load_snapshot(root_string, signature):
    root = Path(root_string)
    folder = root / "results_full"
    status = json.loads((folder / "evaluation_status.json").read_text(encoding="utf-8"))
    if not status.get("complete") or status.get("failures"):
        done = status.get("completed_model_outputs", 0)
        expected = status.get("expected_model_outputs", "?")
        raise BenchmarkUnavailable(f"Full evaluation is incomplete ({done}/{expected} outputs). Refresh when it finishes.")
    data = json.loads((folder / "evaluation.json").read_text(encoding="utf-8"))
    summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    manifest_bytes = (root / "data/manifest_full.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    ids = [str(p["id"]) for p in manifest["pairs"]]
    count = len(ids)
    assert count > 0 and len(set(ids)) == count
    assert manifest["split"] == "test" and all(p["split"] == "test" for p in manifest["pairs"])
    assert manifest["selection"]["is_full_benchmark"] is True
    assert manifest["selection"]["count"] == count
    assert data["selection"] == summary["selection"] == manifest["selection"]
    assert data["manifest_sha256"] == summary["manifest_sha256"] == hashlib.sha256(manifest_bytes).hexdigest()
    assert data["data_hashes_verified"] is True
    assert data["completion_status"] == summary["completion_status"] == status
    assert status["expected_pairs"] == count
    assert status["expected_model_outputs"] == status["completed_model_outputs"] == count * len(METHODS)
    assert status["completed_by_method"] == {method: count for method in METHODS}
    assert data["summary"] == summary["summary"] and data["protocol"] == summary["protocol"]
    indexed = {(str(r["id"]), r["method"]): r for r in data["per_image"]}
    assert len(indexed) == len(data["per_image"]) == count * len(ORDER)
    assert set(indexed) == {(identifier, method) for identifier in ids for method in ORDER}
    by_method = {r["method"]: r for r in data["summary"]}
    assert len(by_method) == len(data["summary"]) == len(ORDER) and set(by_method) == set(ORDER)
    for method in ORDER:
        rows = [indexed[identifier, method] for identifier in ids]
        assert by_method[method]["image_count"] == count
        for key in ("psnr", "ssim", "mae"):
            assert all(isinstance(r[key], (int, float)) and math.isfinite(r[key]) for r in rows)
            assert math.isclose(statistics.mean(r[key] for r in rows), by_method[method][key], abs_tol=1e-9)
        for row in rows:
            _asset(folder, row["output_path"])
    for identifier in ids:
        input_path = Path(indexed[identifier, "Untreated"]["output_path"])
        _asset(folder, str(input_path.parent.parent / "reference" / input_path.name))
    winner = max(METHODS, key=lambda m: by_method[m]["psnr"])
    ranked = sorted(ids, key=lambda identifier: indexed[identifier, winner]["psnr"] - indexed[identifier, "Untreated"]["psnr"])
    return {"folder": folder, "data": data, "ids": ids, "index": indexed,
            "summary": by_method, "winner": winner, "typical": ranked[len(ranked) // 2], "hard": ranked[0]}


def overview(snapshot):
    data, rows = snapshot["data"], snapshot["summary"]
    count = len(snapshot["ids"])
    protocol = data["protocol"]
    stamp = datetime.fromisoformat(data["created_at_utc"]).astimezone().strftime("%Y-%m-%d %H:%M")
    heading = (f"### Full RW2AH test set · {count} image pairs\n"
               f"All three methods are complete, with {count * len(METHODS):,} model outputs. "
               f"Results generated: {stamp} (local time).")
    baseline = rows["Untreated"]
    table = []
    for method in ORDER:
        row = rows[method]
        table.append(["Hazy input" if method == "Untreated" else method, str(count),
                      _number(row["psnr"]), _number(row["ssim"], 4),
                      "—" if method == "Untreated" else f"{row['psnr'] - baseline['psnr']:+.2f}",
                      "—" if method == "Untreated" else _number(row["median_gpu_ms"]),
                      "—" if method == "Untreated" else _number(row["max_peak_allocated_mib"])])
    best_psnr = max(METHODS, key=lambda m: rows[m]["psnr"])
    best_ssim = max(METHODS, key=lambda m: rows[m]["ssim"])
    timed = [m for m in METHODS if rows[m].get("median_gpu_ms") is not None]
    insight = f"**Findings:** {best_psnr} has the highest mean PSNR; {best_ssim} has the highest mean SSIM."
    if timed:
        fastest = min(timed, key=lambda m: rows[m]["median_gpu_ms"])
        insight += f" {fastest} has the lowest median GPU forward time."
    worse = [m for m in METHODS if rows[m]["psnr"] < baseline["psnr"] and rows[m]["ssim"] < baseline["ssim"]]
    if worse:
        insight += f"\n\n{', '.join(worse)} scored below the hazy input on both mean quality metrics. These measured results are retained."
    description = (f"- Hazy inputs and clear references are resized together with their aspect ratio preserved, to a longest side of at most **{protocol['max_side']}** pixels. Small images are not enlarged.\n"
                   f"- RGB PSNR and Gaussian SSIM are each averaged equally across images. GPU time is the median of **{protocol['runs']} runs** per image, followed by the median across images.\n"
                   "- GPU time excludes loading, warm-up, data transfers, file I/O, and UI interactions. Memory is the peak PyTorch tensor allocation.\n"
                   "- PGL / HNDiff use FP32; PixRestore uses BF16. Training data and inference settings differ, so these results compare the released systems under this local setup.\n"
                   "- The result browser reads saved images. Switching images does not run inference again.")
    repeats = rows["HNDiff-FocalNet"]
    if repeats.get("nonidentical_repeat_image_count", 0):
        description += (f"\n- Repeated HNDiff outputs were not byte-identical for {repeats['nonidentical_repeat_image_count']}/{count} images. "
                        f"The maximum difference was {repeats['max_repeat_abs_difference_uint8']} on the uint8 intensity scale (0-255). Quality is measured on the first output.")
    downloads = [str(snapshot["folder"] / name) for name in ("summary.csv", "per_image.csv", "evaluation.json") if (snapshot["folder"] / name).is_file()]
    return heading, table, insight, description, downloads


def image_result(snapshot, identifier):
    if identifier not in snapshot["ids"]:
        raise BenchmarkUnavailable("This image is not in the current full results. Refresh and select it again.")
    folder, indexed = snapshot["folder"], snapshot["index"]
    source = indexed[identifier, "Untreated"]
    input_path = _asset(folder, source["output_path"])
    source_path = Path(source["output_path"])
    reference = _asset(folder, str(source_path.parent.parent / "reference" / source_path.name))
    sliders = [(input_path, _asset(folder, indexed[identifier, method]["output_path"])) for method in METHODS]
    table = []
    for method in ORDER:
        row = indexed[identifier, method]
        table.append(["Hazy input" if method == "Untreated" else method, f"{row['width']} × {row['height']}",
                      _number(row["psnr"]), _number(row["ssim"], 4), "—" if method == "Untreated" else _number(row["gpu_ms"])])
    caption = f"**Image {identifier}** · The table below shows measurements for this image. The summary above reports means across all {len(snapshot['ids'])} images."
    return caption, input_path, reference, *sliders, table
