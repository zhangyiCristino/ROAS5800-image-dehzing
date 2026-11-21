"""Prepare official backbone assets, demo inputs and local RW2AH manifests."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
SOURCE = "https://raw.githubusercontent.com/sc-30-bit/PGL-Net/dfb418adf6fd2dd55ad6f4bec72128f618d50eeb/models/PGLNet.py"
SOURCE_SHA = "85b4f94967555ee55d8536408109a77f7565c6ac2ada4bc7dd8992b04a5ef4d7"
BACKBONE_SHA = "1ea04c0d993cc2e5aa6a36a6c77f5ac7719d4faa1fede551c3bc8db23f89526e"
CHECKPOINT_URL = "https://drive.google.com/file/d/1Sl7cCSK3peAeOzEXWvv2_3m9cJFvrvJx/view"
SCENES = ("12-0002-05", "1-0010-02", "2-0019-03", "4-0005-07", "8-0016-01")


def install(data: bytes, expected: str, destination: Path) -> None:
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError(f"SHA-256 mismatch for {destination.name}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(data)


def prepare_backbone(checkpoint: Path | None, source: Path | None) -> None:
    vendor = ROOT / "third_party" / "pgl_net"
    target_checkpoint = vendor / "rw2ah_pglnet_t.pk"
    checkpoint = checkpoint or target_checkpoint
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"Download the original checkpoint from {CHECKPOINT_URL} and pass --checkpoint PATH."
        )
    install(checkpoint.read_bytes(), BACKBONE_SHA, target_checkpoint)
    target_source = vendor / "PGLNet.py"
    if source:
        data = source.read_bytes()
    elif target_source.is_file():
        data = target_source.read_bytes()
    else:
        with urlopen(SOURCE, timeout=30) as response:
            data = response.read(32768)
    install(data, SOURCE_SHA, target_source)


def prepare_demo() -> None:
    for scene in SCENES:
        existing = REPO / "site" / "assignment2" / "samples" / scene
        for source_name, destination_name in (("input.png", "hazy"), ("reference.png", "gt")):
            folder = ROOT / "sample_hazy_5imgs" / destination_name
            folder.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(existing / source_name, folder / f"{scene}.png")


def write_manifest(ids: list[str], data: Path, clear_folder: str, output: Path) -> None:
    rows = []
    for image_id in ids:
        filename = image_id + ".png"
        rows.append({
            "id": image_id,
            "hazy": str(data / "hazy" / filename),
            "gt": str(data / clear_folder / filename),
        })
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, help="Original rw2ah_pglnet_t.pk from PGL-Net")
    parser.add_argument("--backbone-source", type=Path, help="Optional local copy of upstream PGLNet.py")
    parser.add_argument("--test-data", type=Path, default=REPO / "data" / "rw2ah_test352")
    parser.add_argument("--train-data", type=Path, default=ROOT / "data" / "rw2ah_train")
    args = parser.parse_args()

    prepare_backbone(args.checkpoint, args.backbone_source)
    prepare_demo()
    splits = json.loads((ROOT / "split_ids.json").read_text(encoding="utf-8"))
    manifests = ROOT / "data" / "manifests"
    write_manifest(splits["test_352"], args.test_data.resolve(), "clear", manifests / "disjoint" / "test.jsonl")
    write_manifest(splits["val_102"], args.train_data.resolve(), "gt", manifests / "disjoint" / "val.jsonl")
    write_manifest(splits["train_1304"], args.train_data.resolve(), "gt", manifests / "train_1304.jsonl")
    write_manifest(splits["train_905"], args.train_data.resolve(), "gt", manifests / "disjoint" / "train.jsonl")
    print("Prepared verified backbone assets, five demo pairs and four local manifests.")


if __name__ == "__main__":
    main()
