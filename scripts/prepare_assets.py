"""Fetch the exact runtime sources and weights used for the recorded experiment.

Only standard-library modules are needed. Assets are fetched from their authors;
they are not redistributed in this repository. The final byte size and SHA-256
are authoritative, including the documented HNDiff import-only adaptation.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
CHUNK_BYTES = 1024 * 1024
THOP_COMMENT = b"# Profiling-only thop import removed for the inference feasibility wrapper."
THOP_PATHS = {
    f"third_party/hndiff/FocalNet/model/{name}.py"
    for name in ("FocalNetPrior", "HazeDiffusionUnet", "LatentEncoder")
}
HOSTS = {
    "raw.githubusercontent.com", "huggingface.co", "drive.google.com",
    "drive.usercontent.google.com", "dl.fbaipublicfiles.com",
}
WINDOWS_DEVICES = {"CON", "PRN", "AUX", "NUL", "CONIN$", "CONOUT$"} | {
    f"{prefix}{number}" for prefix in ("COM", "LPT") for number in range(1, 10)
}


class AssetError(RuntimeError):
    """A download or integrity check failed without installing partial data."""


def safe_path(root: Path, relative: str) -> Path:
    """Reject absolute paths, traversal, Windows aliases and symlink escapes."""
    if not isinstance(relative, str) or "\\" in relative or ":" in relative:
        raise AssetError("Asset paths must use plain relative POSIX paths")
    path = PurePosixPath(relative)
    if (path.is_absolute() or path.as_posix() != relative or not path.parts
            or path.parts[0] not in {"third_party", "checkpoints"}
            or any(part in {"", ".", ".."} or part.rstrip(" .") != part
                   or part.split(".", 1)[0].upper() in WINDOWS_DEVICES
                   or any(ord(character) < 32 or character in '<>"|?*' for character in part)
                   for part in path.parts)):
        raise AssetError(f"Unsafe asset path: {relative!r}")
    base = root.resolve()
    target = base.joinpath(*path.parts)
    if not target.resolve().is_relative_to(base):
        raise AssetError(f"Asset path escapes its root: {relative!r}")
    return target


def validate_url(url: str) -> None:
    parsed = urllib.parse.urlsplit(url)
    if (parsed.scheme != "https" or parsed.hostname not in HOSTS
            or parsed.username or parsed.password or parsed.port not in {None, 443}):
        raise AssetError("Asset URL is outside the approved HTTPS author hosts")


def load_manifest(path: Path = ROOT / "assets.json") -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") != 1 or not isinstance(data.get("files"), list):
        raise AssetError("Unsupported assets.json schema")
    seen = set()
    for asset in data["files"]:
        relative = asset["path"]
        safe_path(ROOT, relative)
        if relative in seen:
            raise AssetError(f"Duplicate asset path: {relative}")
        seen.add(relative)
        if (asset.get("kind") not in {"source", "checkpoint"}
                or not isinstance(asset.get("bytes"), int) or asset["bytes"] <= 0
                or not re.fullmatch(r"[0-9a-f]{64}", asset.get("sha256", ""))):
            raise AssetError(f"Invalid asset integrity record: {relative}")
        validate_url(asset["url"])
        transform = asset.get("transform")
        if transform is not None and not (
                transform == "hndiff_unused_thop_comment_lf" and relative in THOP_PATHS):
            raise AssetError(f"Unsupported source adaptation: {relative}")
    return data["files"]


def verify(path: Path, asset: dict) -> bool:
    if not path.is_file() or path.stat().st_size != asset["bytes"]:
        return False
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(CHUNK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest() == asset["sha256"]


def transform_source(raw: bytes, asset: dict) -> bytes:
    if asset.get("transform") is None:
        return raw
    if (asset["path"] not in THOP_PATHS
            or asset.get("transform") != "hndiff_unused_thop_comment_lf"):
        raise AssetError("Unexpected source adaptation")
    lines = raw.splitlines(keepends=True)
    matching = [i for i, line in enumerate(lines) if line.strip() == b"from thop import profile"]
    if len(matching) != 1:
        raise AssetError(f"Expected exactly one unused thop import in {asset['path']}")
    index = matching[0]
    newline = b"\r\n" if lines[index].endswith(b"\r\n") else b"\n"
    lines[index] = THOP_COMMENT + newline
    adapted = b"".join(lines)
    return adapted if adapted.endswith(b"\n") else adapted + b"\n"


def _open_url(asset: dict, timeout: float):
    """Handle Google's public-file confirmation without logging signed URLs."""
    url = asset["url"]
    request = urllib.request.Request(url, headers={"User-Agent": "dehazing-assets/1.0"})
    response = urllib.request.urlopen(request, timeout=timeout)
    if "text/html" not in response.headers.get("Content-Type", "").lower():
        return response
    with response:
        page = response.read(100_001)
    file_id = asset.get("drive_file_id")
    if not file_id or len(page) > 100_000:
        raise AssetError(f"Author host returned an HTML page for {asset['path']}")
    fields = re.findall(r'<input[^>]+name="([^"]+)"[^>]+value="([^"]*)"',
                        page.decode("utf-8", errors="replace"))
    values = {html.unescape(key): html.unescape(value) for key, value in fields}
    if values.get("id") != file_id or values.get("confirm") != "t":
        raise AssetError(f"Public Google Drive file unavailable for {asset['path']}; "
                         "check the authors' link or use a verified local cache")
    url = "https://drive.usercontent.google.com/download?" + urllib.parse.urlencode(values)
    request = urllib.request.Request(url, headers={"User-Agent": "dehazing-assets/1.0"})
    response = urllib.request.urlopen(request, timeout=timeout)
    if "text/html" in response.headers.get("Content-Type", "").lower():
        response.close()
        raise AssetError(f"Google Drive did not provide binary data for {asset['path']}")
    return response


def _download(asset: dict, target: Path, timeout: float) -> None:
    """Bound the stream, verify it, then atomically replace the destination."""
    temporary = None
    try:
        with _open_url(asset, timeout) as response:
            with tempfile.NamedTemporaryFile(dir=target.parent, suffix=".partial", delete=False) as output:
                temporary = Path(output.name)
                if asset.get("transform"):
                    raw = response.read(asset["bytes"] + 1025)
                    if len(raw) > asset["bytes"] + 1024:
                        raise AssetError(f"Oversized source response for {asset['path']}")
                    output.write(transform_source(raw, asset))
                else:
                    transferred = 0
                    while True:
                        block = response.read(min(CHUNK_BYTES, asset["bytes"] - transferred + 1))
                        if not block:
                            break
                        transferred += len(block)
                        if transferred > asset["bytes"]:
                            raise AssetError(f"Downloaded size exceeds manifest for {asset['path']}")
                        output.write(block)
                output.flush()
                os.fsync(output.fileno())
        if not verify(temporary, asset):
            raise AssetError(f"Size/SHA-256 mismatch for {asset['path']}; "
                             "destination unchanged; do not bypass integrity checks")
        temporary.replace(target)
    except urllib.error.HTTPError as error:
        raise AssetError(f"HTTP {error.code} downloading {asset['path']}; "
                         "the author may have changed access or imposed a quota") from None
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        # Do not interpolate request URLs or exception text containing signed URLs/tokens.
        raise AssetError(f"Network/file operation failed for {asset['path']} "
                         f"({type(error).__name__}); retry or use --local-cache") from None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _copy_cache(source: Path, target: Path, asset: dict) -> None:
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=target.parent, suffix=".partial", delete=False) as output:
            temporary = Path(output.name)
            with source.open("rb") as stream:
                shutil.copyfileobj(stream, output, CHUNK_BYTES)
            output.flush()
            os.fsync(output.fileno())
        if not verify(temporary, asset):
            raise AssetError(f"Local cache changed during copy: {asset['path']}")
        temporary.replace(target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def prepare(assets: list[dict], root: Path = ROOT, *, sources_only: bool = False,
            verify_only: bool = False, local_cache: Path | None = None,
            timeout: float = 60.0) -> dict:
    counts = {"verified": 0, "copied": 0, "downloaded": 0}
    failed = []
    for asset in assets:
        if sources_only and asset["kind"] != "source":
            continue
        target = safe_path(root, asset["path"])
        if verify(target, asset):
            counts["verified"] += 1
            print(f"verified  {asset['path']}")
            continue
        if verify_only:
            failed.append(asset["path"])
            print(f"missing/invalid  {asset['path']}")
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        cached = safe_path(local_cache, asset["path"]) if local_cache is not None else None
        if cached is not None and verify(cached, asset):
            _copy_cache(cached, target, asset)
            counts["copied"] += 1
            print(f"copied    {asset['path']}")
        else:
            _download(asset, target, timeout)
            counts["downloaded"] += 1
            print(f"downloaded {asset['path']}")
    if failed:
        raise AssetError(f"{len(failed)} asset(s) missing or invalid; "
                         "run without --verify-only to prepare them")
    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources-only", action="store_true", help="Prepare runtime Python sources and DINO LICENSE only")
    parser.add_argument("--verify-only", action="store_true", help="Check installed assets; no downloads, copies or writes")
    parser.add_argument("--local-cache", type=Path, help="Reuse exact assets from another project root (for example the original hw1)")
    parser.add_argument("--timeout", type=float, default=60, help="Network socket timeout in seconds (default: 60)")
    args = parser.parse_args(argv)
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    if args.local_cache is not None and not args.local_cache.is_dir():
        parser.error("--local-cache must be an existing project directory")
    try:
        counts = prepare(load_manifest(), sources_only=args.sources_only,
                         verify_only=args.verify_only, local_cache=args.local_cache,
                         timeout=args.timeout)
    except (AssetError, KeyError, ValueError, OSError) as error:
        print(f"Asset preparation failed: {error}", file=sys.stderr)
        return 1
    print("Assets ready: " + ", ".join(f"{key}={value}" for key, value in counts.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
