"""CPU-only checks for loading complete saved evaluations into the UI."""
from copy import deepcopy
import base64
import hashlib
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dehaze.benchmark import BenchmarkUnavailable, image_result, load_benchmark, overview


METHODS = ("PGL-Net-T", "HNDiff-FocalNet", "PixRestore-S")
PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jFioAAAAASUVORK5CYII="
)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def saved_evaluation(root):
    """One complete synthetic image, sufficient to exercise rejection paths."""
    selection = {"count": 1, "is_full_benchmark": True}
    manifest = {"split": "test", "selection": selection,
                "pairs": [{"id": "sample", "split": "test"}]}
    manifest_path = root / "data/manifest_full.json"
    write_json(manifest_path, manifest)
    status = {"complete": True, "expected_pairs": 1,
              "expected_model_outputs": 3, "completed_model_outputs": 3,
              "completed_by_method": {method: 1 for method in METHODS},
              "failures": [], "current": "finished"}
    rows, summaries = [], []
    for index, method in enumerate(("Untreated", *METHODS)):
        relative = f"images/{'input' if index == 0 else method}/sample.png"
        path = root / "results_full" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(PNG)
        rows.append({"id": "sample", "method": method, "output_path": relative,
                     "width": 1, "height": 1, "psnr": 20.0 + index,
                     "ssim": 0.5 + index / 10, "mae": 0.1,
                     "gpu_ms": float(index)})
        summaries.append({"method": method, "image_count": 1,
                          "psnr": rows[-1]["psnr"], "ssim": rows[-1]["ssim"],
                          "mae": 0.1, "median_gpu_ms": float(index),
                          "max_peak_allocated_mib": float(index)})
    reference = root / "results_full/images/reference/sample.png"
    reference.parent.mkdir(parents=True, exist_ok=True)
    reference.write_bytes(PNG)
    data = {"created_at_utc": "2026-10-19T10:00:00+00:00",
            "selection": selection,
            "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "data_hashes_verified": True, "completion_status": status,
            "protocol": {"max_side": 512, "runs": 3},
            "summary": summaries, "per_image": rows}
    summary = {key: deepcopy(value) for key, value in data.items() if key != "per_image"}
    write_json(root / "results_full/evaluation.json", data)
    write_json(root / "results_full/summary.json", summary)
    write_json(root / "results_full/evaluation_status.json", status)
    return data, summary, status


class BenchmarkTests(unittest.TestCase):
    @unittest.skipUnless((ROOT / "results_full/evaluation.json").is_file(),
                         "The local full evaluation is not available")
    def test_existing_full_evaluation_is_presented_without_inference(self):
        snapshot = load_benchmark(ROOT)
        heading, table, insight, description, downloads = overview(snapshot)
        self.assertEqual(len(snapshot["ids"]), 352)
        self.assertEqual(snapshot["data"]["completion_status"]["completed_model_outputs"], 1056)
        self.assertIn("352", heading)
        self.assertEqual(len(table), 4)
        self.assertTrue(all(row[1] == "352" for row in table))
        self.assertTrue(downloads)
        result = image_result(snapshot, snapshot["ids"][0])
        self.assertIn(snapshot["ids"][0], result[0])
        self.assertEqual(len(result[-1]), 4)
        for path in (result[1], result[2], *(pair[1] for pair in result[3:6])):
            self.assertTrue(Path(path).is_file())

    def test_incomplete_results_cannot_appear_as_a_complete_benchmark(self):
        with TemporaryDirectory(prefix="benchmark-test-", dir=ROOT / "tests") as directory:
            root = Path(directory)
            _, _, status = saved_evaluation(root)
            status.update(complete=False, completed_model_outputs=2)
            write_json(root / "results_full/evaluation_status.json", status)
            with self.assertRaises(BenchmarkUnavailable):
                load_benchmark(root)

    def test_agreeing_summaries_still_must_match_saved_image_metrics(self):
        with TemporaryDirectory(prefix="benchmark-test-", dir=ROOT / "tests") as directory:
            root = Path(directory)
            data, summary, _ = saved_evaluation(root)
            data["summary"][1]["psnr"] += 1
            summary["summary"][1]["psnr"] += 1
            write_json(root / "results_full/evaluation.json", data)
            write_json(root / "results_full/summary.json", summary)
            with self.assertRaises(BenchmarkUnavailable):
                load_benchmark(root)

    def test_unknown_image_id_is_rejected(self):
        with TemporaryDirectory(prefix="benchmark-test-", dir=ROOT / "tests") as directory:
            root = Path(directory)
            saved_evaluation(root)
            snapshot = load_benchmark(root)
            with self.assertRaises(BenchmarkUnavailable):
                image_result(snapshot, "../../outside.png")

    def test_existing_output_outside_result_images_is_rejected(self):
        for absolute in (False, True):
            with self.subTest(absolute=absolute), TemporaryDirectory(prefix="benchmark-test-", dir=ROOT / "tests") as directory:
                root = Path(directory)
                data, _, _ = saved_evaluation(root)
                outside = root / "outside.png"
                outside.write_bytes(PNG)
                data["per_image"][1]["output_path"] = str(outside) if absolute else "../outside.png"
                write_json(root / "results_full/evaluation.json", data)
                with self.assertRaises(BenchmarkUnavailable):
                    load_benchmark(root)


if __name__ == "__main__":
    unittest.main()
