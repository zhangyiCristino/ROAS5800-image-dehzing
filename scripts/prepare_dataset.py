"""Download a deterministic, bounded RW2AH *test* subset from the authors' link.

This reads ZIP members over HTTP Range and never downloads the full 1.89 GB ZIP.
Use --inspect to print the selection and byte estimate without saving images.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import html
import io
import json
from pathlib import Path, PurePosixPath
import re
import struct
import threading
import time
import urllib.parse
import urllib.request
import zipfile
import zlib

from PIL import Image

HW_ROOT = Path(__file__).resolve().parents[1]
FILE_ID = "1RmqMw-105j0Berybl8EAdRwVGlnfG3mT"
FOLDER_URL = "https://drive.google.com/drive/folders/1iZZq8k6Jb7CtFT9uYi96F8TZht2d3b3O"
ARCHIVE_URL = f"https://drive.google.com/file/d/{FILE_ID}/view"
OFFICIAL_REPO = "https://github.com/sc-30-bit/PGL-Net"
DATASET_REPO = "https://github.com/fiwy0527/AAAI25_SGDN"
PAPER = "https://arxiv.org/abs/2412.17496"


class RangeReader(io.RawIOBase):
    """Seekable read-only remote file, with a strict total transfer limit."""

    def __init__(self, url: str, max_download_bytes: int):
        self.url, self.pos = url, 0
        self.downloaded_bytes, self.max_bytes = 0, max_download_bytes
        self.requested_bytes = 0
        self._budget_lock = threading.Lock()
        self.length = 0
        self._cached: list[tuple[int, bytes]] = []
        # A one-byte response confirms that Range is actually honored.
        self._request(0, 0)

    def _request(self, start: int, end: int) -> bytes:
        if end < start:
            return b""
        wanted = end - start + 1
        request = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{end}"})
        for attempt in range(3):
            # Reserve the entire requested range, also for retries. Counting the
            # full attempted range bounds transfer even if a response is partial.
            with self._budget_lock:
                if self.requested_bytes + wanted > self.max_bytes:
                    raise RuntimeError("Download safety budget exceeded; no full ZIP download attempted")
                self.requested_bytes += wanted
            try:
                with urllib.request.urlopen(request, timeout=60) as response:
                    if response.status != 206:
                        raise RuntimeError(f"Server ignored byte Range (status {response.status}); stopped")
                    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
                    if not match or int(match[1]) != start or int(match[2]) != end:
                        raise RuntimeError("Invalid Content-Range; stopped")
                    self.length = int(match[3])
                    body = response.read(wanted + 1)
                    with self._budget_lock:
                        self.downloaded_bytes += len(body)
                    if len(body) != wanted:
                        raise RuntimeError(f"Incomplete range response: {len(body)} / {wanted}")
                    return body
            except (TimeoutError, urllib.error.URLError):
                if attempt == 2:
                    raise
                time.sleep(1 + attempt)
        raise AssertionError("unreachable")

    def read_at(self, start: int, size: int, cache: bool = False) -> bytes:
        for offset, body in self._cached:
            if offset <= start and start + size <= offset + len(body):
                return body[start - offset:start - offset + size]
        body = self._request(start, min(start + size, self.length) - 1)
        if cache:
            self._cached.append((start, body))
        return body

    def read(self, size: int = -1) -> bytes:
        size = self.length - self.pos if size < 0 else min(size, self.length - self.pos)
        data = self.read_at(self.pos, size, cache=size > 1000)
        self.pos += len(data)
        return data

    def seek(self, offset: int, whence: int = 0) -> int:
        self.pos = offset if whence == 0 else self.pos + offset if whence == 1 else self.length + offset
        if self.pos < 0:
            raise ValueError("negative seek")
        return self.pos

    def tell(self) -> int:
        return self.pos

    def seekable(self) -> bool:
        return True


def confirmed_download_url() -> str:
    url = f"https://drive.google.com/uc?export=download&id={FILE_ID}"
    with urllib.request.urlopen(url, timeout=30) as response:
        if "text/html" not in response.headers.get("Content-Type", ""):
            return response.geturl()
        page = response.read(100_000).decode("utf-8")
    fields = re.findall(r'<input[^>]+name="([^"]+)"[^>]+value="([^"]*)"', page)
    values = {html.unescape(key): html.unescape(value) for key, value in fields}
    if values.get("id") != FILE_ID or values.get("confirm") != "t":
        raise RuntimeError("Public Google Drive download unavailable; do not use unverified mirrors")
    return "https://drive.usercontent.google.com/download?" + urllib.parse.urlencode(values)


def select_pairs(infos: list[zipfile.ZipInfo], count: int) -> tuple[list[tuple[str, str]], dict]:
    """Only test paths, sorted by original hazy member name before inference."""
    image_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    hazy, clean = {}, {}
    for info in infos:
        parts = PurePosixPath(info.filename).parts
        lower = [part.casefold() for part in parts]
        if info.is_dir() or "test" not in lower or PurePosixPath(info.filename).suffix.lower() not in image_exts:
            continue
        if "__macosx" in lower or PurePosixPath(info.filename).name.startswith("."):
            continue
        split_index = lower.index("test")
        category = lower[split_index + 1:-1]
        target = hazy if any(part in {"hazy", "input", "haze"} for part in category) else clean if any(part in {"gt", "clean", "clear", "target"} for part in category) else None
        if target is not None:
            key = PurePosixPath(info.filename).stem
            if key in target:
                raise ValueError(f"Duplicate pairing key: {key}")
            target[key] = info.filename
    if not hazy or not clean:
        raise RuntimeError("Could not identify explicit test input/reference folders")
    missing = sorted(set(hazy) - set(clean))
    if missing:
        raise RuntimeError(f"Unpaired test files: {missing[:5]}")
    pairs = sorted(((name, clean[key]) for key, name in hazy.items()), key=lambda pair: pair[0])
    return pairs[:count], {"hazy_test_count": len(hazy), "clear_test_count": len(clean), "paired_test_count": len(pairs)}


def fetch_member(remote: RangeReader, info: zipfile.ZipInfo) -> bytes:
    # ZIP local header: signature, version, flags, compression, time/date, crc,
    # compressed/uncompressed length, filename length, extra length.
    # One bounded Range per image. The local extra field is at most 65535
    # bytes; fetching this upper bound avoids an additional network round trip.
    name_bytes = info.filename.encode("utf-8" if info.flag_bits & 0x800 else "cp437")
    local_record = remote.read_at(info.header_offset, 30 + len(name_bytes) + 65535 + info.compress_size)
    header = local_record[:30]
    if header[:4] != b"PK\x03\x04":
        raise RuntimeError("Bad ZIP local-file signature")
    name_len, extra_len = struct.unpack_from("<HH", header, 26)
    if info.file_size > 40 * 1024 * 1024 or info.compress_size > 40 * 1024 * 1024:
        raise RuntimeError("Unexpectedly large image member; stopped")
    block = local_record[30:30 + name_len + extra_len + info.compress_size]
    local_name = block[:name_len].decode("utf-8" if info.flag_bits & 0x800 else "cp437")
    if local_name != info.filename:
        raise RuntimeError("ZIP central/local filename mismatch")
    compressed = block[name_len + extra_len:]
    if info.compress_type == zipfile.ZIP_STORED:
        body = compressed
    elif info.compress_type == zipfile.ZIP_DEFLATED:
        body = zlib.decompress(compressed, -zlib.MAX_WBITS)
    else:
        raise RuntimeError(f"Unsupported ZIP compression {info.compress_type}")
    if len(body) != info.file_size or zlib.crc32(body) != info.CRC:
        raise RuntimeError(f"ZIP CRC/size validation failed: {info.filename}")
    return body


def _project_data_path(value: str | Path) -> Path:
    path = Path(value)
    path = (HW_ROOT / path).resolve() if not path.is_absolute() else path.resolve()
    if not path.is_relative_to((HW_ROOT / "data").resolve()):
        raise ValueError("Dataset output must stay inside this project's data directory")
    return path


def _atomic_json(path: Path, data: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def prepare(count: int, inspect: bool = False, budget_mb: int = 100,
            output_dir: str | Path = "data/rw2ah_test12",
            manifest_path: str | Path = "data/manifest.json", workers: int = 4) -> dict:
    if not 1 <= count <= 352 or not 1 <= budget_mb <= 700 or not 1 <= workers <= 4:
        raise ValueError("count must be 1..352, budget-mb 1..700, and workers 1..4")
    directory = _project_data_path(output_dir)
    output = _project_data_path(manifest_path)
    if count != 12 and (directory == (HW_ROOT / "data/rw2ah_test12").resolve() or output == (HW_ROOT / "data/manifest.json").resolve()):
        raise ValueError("A non-12 count requires separate --output-dir and --manifest, preserving the original subset")
    remote = RangeReader(confirmed_download_url(), budget_mb * 1024 * 1024)
    with zipfile.ZipFile(remote) as archive:
        infos = archive.infolist()
    by_name = {info.filename: info for info in infos}
    pairs, counts = select_pairs(infos, count)
    compressed_bytes = sum(by_name[name].compress_size for pair in pairs for name in pair)
    uncompressed_bytes = sum(by_name[name].file_size for pair in pairs for name in pair)
    estimate = sum(by_name[name].compress_size + 65565 + len(name.encode("utf-8")) for pair in pairs for name in pair)
    inspection = {"archive_bytes": remote.length, **counts, "selected_count": len(pairs),
                  "selected_compressed_member_bytes": compressed_bytes,
                  "selected_uncompressed_member_bytes": uncompressed_bytes,
                  "estimated_download_bytes_upper_bound_including_zip_headers": estimate,
                  "selected_pairs": pairs}
    print(json.dumps(inspection, indent=2), flush=True)
    if inspect:
        return inspection
    if remote.downloaded_bytes + estimate > remote.max_bytes:
        raise RuntimeError("Selection exceeds transfer budget; lower --count")
    manifest = {
        "schema_version": 1,
        "dataset": "RW2AH (Real-World Well-Aligned Haze), PGL-Net release",
        "split": "test",
        "selection": {"rule": "First N pairs sorted lexicographically by the original test hazy ZIP member name; selected before model inference", "count": len(pairs), **counts, "seed": None, "is_full_benchmark": len(pairs) == counts["paired_test_count"]},
        "sources": {"dataset_authors_repository": DATASET_REPO, "dataset_paper": PAPER, "download_link_provenance": OFFICIAL_REPO, "official_model_authors_data_folder": FOLDER_URL, "archive_url": ARCHIVE_URL, "archive_drive_file_id": FILE_ID, "archive_filename": "RW2AH.zip", "archive_size_bytes": remote.length},
        "license_note": "Dataset is publicly released for research by its creators and linked by the model authors. No separate image-data license was identified in the inspected download/README. Do not infer that the SGDN code Apache-2.0 license licenses third-party webcam imagery. Retain attribution and obtain permission before broader redistribution or commercial reuse.",
        "pairs": [], "download_status": {"complete": False, "expected_pairs": len(pairs), "completed_pairs": 0, "failures": []},
    }
    # Reuse verified bytes from either a previous full run or the untouched
    # 12-pair subset. Never infer membership from arbitrary local filenames.
    prior_by_member = {}
    for prior_path in {output, HW_ROOT / "data/manifest.json"}:
        if prior_path.is_file():
            prior = json.loads(prior_path.read_text(encoding="utf-8"))
            if prior.get("sources", {}).get("archive_drive_file_id") != FILE_ID:
                raise RuntimeError(f"Different source archive in existing manifest: {prior_path}")
            for pair in prior.get("pairs", []):
                for role in ("hazy", "clear"):
                    source = pair.get(role + "_source", {})
                    path = (HW_ROOT / pair[role]).resolve()
                    if not path.is_relative_to((HW_ROOT / "data").resolve()):
                        raise RuntimeError("Unsafe existing manifest path")
                    prior_by_member[source.get("archive_member")] = (path, source)
    directory.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(output, manifest)
    counters = {"downloaded_files": 0, "reused_files": 0}
    counter_lock = threading.Lock()

    def process_pair(names: tuple[str, str]) -> dict:
        hazy_name, clear_name = names
        entry = {"id": PurePosixPath(hazy_name).stem, "split": "test"}
        for role, original in (("hazy", hazy_name), ("clear", clear_name)):
            info = by_name[original]
            destination = directory / role / PurePosixPath(original).name
            relative = destination.relative_to(HW_ROOT)
            destination.parent.mkdir(parents=True, exist_ok=True)
            prior_file, prior_source = prior_by_member.get(original, (None, {}))
            body = None
            for candidate in (destination, prior_file):
                if candidate is None or not candidate.is_file():
                    continue
                candidate_body = candidate.read_bytes()
                if len(candidate_body) != info.file_size or zlib.crc32(candidate_body) != info.CRC:
                    continue
                known_hash = prior_source.get("sha256")
                if known_hash and hashlib.sha256(candidate_body).hexdigest() != known_hash:
                    continue
                body = candidate_body
                break
            reused = body is not None
            if body is None:
                body = fetch_member(remote, info)
            with counter_lock:
                counters["reused_files" if reused else "downloaded_files"] += 1
            with Image.open(io.BytesIO(body)) as image:
                image.load()
                size, mode = list(image.size), image.mode
            if not destination.is_file() or hashlib.sha256(destination.read_bytes()).digest() != hashlib.sha256(body).digest():
                temporary = destination.with_suffix(destination.suffix + ".tmp")
                temporary.write_bytes(body)
                temporary.replace(destination)
            entry[role] = relative.as_posix()
            entry[role + "_source"] = {"archive_member": original, "bytes": len(body), "sha256": hashlib.sha256(body).hexdigest(), "zip_crc32": f"{info.CRC:08x}", "size_wh": size, "mode": mode}
        if entry["hazy_source"]["size_wh"] != entry["clear_source"]["size_wh"]:
            raise RuntimeError(f"Pair dimensions differ: {entry['id']}")
        return entry

    completed, failures = {}, []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        pending = {executor.submit(process_pair, pair): pair for pair in pairs}
        for future in as_completed(pending):
            names = pending[future]
            try:
                entry = future.result()
                completed[names[0]] = entry
                print(f"Saved pair {len(completed)}/{len(pairs)}: {entry['id']}; transferred {remote.downloaded_bytes / 1024**2:.2f} MiB", flush=True)
            except Exception as error:
                failures.append({"hazy_archive_member": names[0], "error": str(error), "exception": type(error).__name__})
                print(f"FAILED: {names[0]}: {error}", flush=True)
            manifest["pairs"] = [completed[key] for key in sorted(completed)]
            manifest["downloaded_bytes_this_run"] = remote.downloaded_bytes
            manifest["reserved_range_bytes_this_run"] = remote.requested_bytes
            manifest["download_status"] = {"complete": len(completed) == len(pairs) and not failures,
                                           "expected_pairs": len(pairs), "completed_pairs": len(completed),
                                           **counters, "failures": failures}
            _atomic_json(output, manifest)
    print(f"Manifest: {output}; transferred {remote.downloaded_bytes / 1024**2:.2f} MiB", flush=True)
    if failures:
        raise RuntimeError(f"{len(failures)} pair downloads failed; progress saved, rerun safely to resume")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument("--budget-mb", type=int, default=100, help="Hard transfer bound including retries, max 700 MiB")
    parser.add_argument("--output-dir", type=Path, default=Path("data/rw2ah_test12"))
    parser.add_argument("--manifest", type=Path, default=Path("data/manifest.json"))
    parser.add_argument("--workers", type=int, default=4, help="Concurrent pair tasks, at most four Range requests at once")
    parser.add_argument("--inspect", action="store_true")
    args = parser.parse_args()
    prepare(args.count, args.inspect, args.budget_mb, args.output_dir, args.manifest, args.workers)
