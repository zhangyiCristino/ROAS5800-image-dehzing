"""Verify completed evaluation files against data, saved pixels and summaries."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dehaze.evaluation import compute_metrics, load_manifest

METHODS = {"Untreated", "PGL-Net-T", "HNDiff-FocalNet", "PixRestore-S"}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/manifest_full.json")
    parser.add_argument("--results", default="results_full")
    parser.add_argument("--expected-count", type=int, default=352)
    args = parser.parse_args()
    manifest_path = ROOT / args.manifest
    output = ROOT / args.results
    manifest = load_manifest(manifest_path, verify_hashes=True)
    assert len(manifest["pairs"]) == args.expected_count, "Dataset pair count differs from expected"
    assert manifest["selection"]["is_full_benchmark"] is True, "Manifest is not the full source test split"
    assert all(pair["split"] == "test" for pair in manifest["pairs"]), "Unexpected data split"
    evaluation = json.loads((output / "evaluation.json").read_text(encoding="utf-8"))
    assert evaluation["manifest_sha256"] == digest(manifest_path), "Result uses a different manifest"
    status = evaluation["completion_status"]
    assert status["complete"] is True and not status["failures"], "Evaluation has not completed successfully"
    assert status["completed_model_outputs"] == 3 * args.expected_count, "Incorrect model output count"
    assert status["completed_by_method"] == {method: args.expected_count for method in METHODS - {"Untreated"}}
    assert evaluation["data_hashes_verified"] is True
    assert evaluation["protocol"]["max_side"] == 512 and evaluation["protocol"]["runs"] == 3
    assert evaluation["protocol"]["seed"] == 0
    rows = evaluation["per_image"]
    ids = [str(pair["id"]) for pair in manifest["pairs"]]
    expected = {(identifier, method) for identifier in ids for method in METHODS}
    actual = {(str(row["id"]), row["method"]) for row in rows}
    assert actual == expected and len(rows) == len(expected), "Missing, duplicate or unexpected result rows"
    verified = 0
    for row in rows:
        path = (output / row["output_path"]).resolve()
        assert path.is_relative_to(output.resolve()) and path.is_file(), "Invalid/missing output path"
        with Image.open(path) as image:
            pixels = np.array(image.convert("RGB"))
        assert pixels.shape == (row["height"], row["width"], 3), "Output dimensions differ"
        assert hashlib.sha256(pixels.tobytes()).hexdigest() == row["output_pixel_sha256"], "Saved pixels differ from measured pixels"
        assert all(math.isfinite(row[key]) for key in ("psnr", "ssim", "mae")), "Invalid metric value"
        if row["method"] != "Untreated":
            assert row["runs"] == 3 and len(row["runs_metadata"]) == 3, "Incomplete repeated measurements"
        verified += 1
        if verified % 250 == 0:
            print(f"Verified {verified}/{len(rows)} saved images", flush=True)

    chosen = sorted({0, len(ids) // 4, len(ids) // 2, 3 * len(ids) // 4, len(ids) - 1})
    quality_checks = 0
    for index in chosen:
        identifier = ids[index]
        for row in (row for row in rows if str(row["id"]) == identifier):
            with Image.open(output / row["output_path"]) as image:
                prediction = np.array(image.convert("RGB"))
            with Image.open(output / "images" / "reference" / f"{identifier}.png") as image:
                reference = np.array(image.convert("RGB"))
            recalculated = compute_metrics(prediction, reference)
            for key in ("psnr", "ssim", "mae"):
                assert math.isclose(recalculated[key], row[key], abs_tol=1e-10), f"Recomputed {key} differs"
            quality_checks += 1

    for summary in evaluation["summary"]:
        selected = [row for row in rows if row["method"] == summary["method"]]
        assert summary["image_count"] == len(selected) == args.expected_count
        for key in ("psnr", "ssim", "mae"):
            assert math.isclose(statistics.mean(row[key] for row in selected), summary[key], abs_tol=1e-10)

    provenance = json.loads((ROOT / "assets.json").read_text(encoding="utf-8"))
    source_checks = 0
    for entry in provenance["files"]:
        assert digest(ROOT / entry["path"]) == entry["sha256"], f"Source/checkpoint changed: {entry['path']}"
        source_checks += 1
    report = {"status": "passed", "dataset_pairs": len(ids), "source_images_hash_checked": 2 * len(ids),
              "saved_result_images_checked": verified, "recomputed_quality_records": quality_checks,
              "source_and_checkpoint_files_checked": source_checks,
              "all_methods_complete": sorted(METHODS),
              "scope": "Checks source/data integrity, complete output coverage, saved-pixel hashes, sample metric recomputation and full aggregate consistency. Does not imply paper-score reproduction."}
    (output / "validation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
