"""Asset integrity/path/network contracts; synthetic fixtures, no GPU or network."""
import hashlib
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location("prepare_assets", Path(__file__).resolve().parents[1] / "scripts/prepare_assets.py")
assets = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(assets)


def record(raw=b"source\n", path="third_party/example/model.py"):
    return {"path": path, "kind": "source", "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "url": "https://raw.githubusercontent.com/example/repo/pin/model.py"}


class AssetTests(unittest.TestCase):
    def test_paths_reject_escape_and_windows_aliases(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for relative in ("../outside", "/absolute", "checkpoints/../outside", "C:/secret",
                             "third_party\\secret", "third_party/a.", "third_party//a", "outputs/a",
                             "third_party/CON.py", "third_party/a?b"):
                with self.subTest(relative=relative), self.assertRaises(assets.AssetError):
                    assets.safe_path(root, relative)
            self.assertEqual(assets.safe_path(root, "checkpoints/model.pth"), root / "checkpoints/model.pth")

    def test_manifest_matches_expected_runtime_scope(self):
        manifest = assets.load_manifest()
        self.assertEqual(len(manifest), 34)
        self.assertEqual(sum(item["kind"] == "source" for item in manifest), 28)
        self.assertFalse(any("LOCAL_SOURCE_NOTES" in item["path"] for item in manifest))
        self.assertEqual(sum(bool(item.get("transform")) for item in manifest), 3)

    def test_exact_hndiff_adaptation(self):
        asset = record(path="third_party/hndiff/FocalNet/model/FocalNetPrior.py")
        asset["transform"] = "hndiff_unused_thop_comment_lf"
        self.assertEqual(assets.transform_source(b"from thop import profile\nx=1", asset),
                         assets.THOP_COMMENT + b"\nx=1\n")
        with self.assertRaises(assets.AssetError):
            assets.transform_source(b"x=1\n", asset)

    def test_verified_cache_reuse_and_verify_only_never_download(self):
        asset = record()
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            source = assets.safe_path(base / "cache", asset["path"])
            source.parent.mkdir(parents=True)
            source.write_bytes(b"source\n")
            with patch.object(assets, "_download", side_effect=AssertionError("network invoked")):
                self.assertEqual(assets.prepare([asset], base / "target", local_cache=base / "cache")["copied"], 1)
                self.assertEqual(assets.prepare([asset], base / "target", verify_only=True)["verified"], 1)
                with self.assertRaises(assets.AssetError):
                    assets.prepare([asset], base / "empty", verify_only=True)
            self.assertFalse((base / "empty").exists())

    def test_failed_download_preserves_existing_destination_and_cleans_partial(self):
        class Response(io.BytesIO):
            pass
        asset = record()
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "model.py"
            target.write_bytes(b"keep existing")
            with patch.object(assets, "_open_url", return_value=Response(b"invalid")):
                with self.assertRaises(assets.AssetError):
                    assets._download(asset, target, 1)
            self.assertEqual(target.read_bytes(), b"keep existing")
            self.assertEqual(list(Path(folder).glob("*.partial")), [])

    def test_source_only_skips_checkpoint_and_does_not_touch_network(self):
        asset = record()
        asset["kind"] = "checkpoint"
        with tempfile.TemporaryDirectory() as folder, patch.object(assets, "_download", side_effect=AssertionError("network invoked")):
            self.assertEqual(assets.prepare([asset], Path(folder), sources_only=True),
                             {"verified": 0, "copied": 0, "downloaded": 0})


if __name__ == "__main__":
    unittest.main()
