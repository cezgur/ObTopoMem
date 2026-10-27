#!/usr/bin/env python3
"""
Download ObTopoMem's pretrained models from their official sources.

Run from any directory: python scripts/download_models.py
No third-party Python packages are required.
"""

from __future__ import annotations
import argparse
import hashlib
import os
from pathlib import Path
import shutil
import tarfile
import tempfile
from urllib.request import Request, urlopen
import zipfile

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_ROOT = PROJECT_ROOT / "models"

# Sources are documented in the OmniGlue and Segment Anything READMEs.
MODELS = (
    # (group, label, destination relative to models/, URL, archive type, min file size)
    (
        "omniglue", "OmniGlue matcher", "omniglue/og_export",
        "https://storage.googleapis.com/omniglue/og_export.zip", "zip", 0,
    ),
    (
        "omniglue", "SuperPoint sp_v6", "omniglue/sp_v6",
        "https://raw.githubusercontent.com/rpautrat/SuperPoint/master/pretrained_models/sp_v6.tgz", "tgz", 0,
    ),
    (
        "omniglue", "DINOv2 ViT-B/14", "omniglue/dinov2_vitb14_pretrain.pth",
        "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth", "file", 10_000_000,
    ),
    (
        "sam", "SAM ViT-L", "sam/sam_vit_l_0b3195.pth",
        "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth", "file", 100_000_000,
    ),
)


def is_present(destination: Path, archive_type: str, minimum_size: int) -> bool:
    if archive_type == "file":
        return destination.is_file() and destination.stat().st_size >= minimum_size
    # Both exports are TensorFlow SavedModels.
    return destination.is_dir() and (destination / "saved_model.pb").is_file()


def download(url: str, target: Path) -> str:
    """Stream into a temporary file and return its SHA-256 (of the download)."""
    request = Request(url, headers={"User-Agent": "ObTopoMem-model-downloader/1.0"})
    sha256 = hashlib.sha256()
    with urlopen(request, timeout=60) as response, target.open("wb") as output:
        expected = response.headers.get("Content-Length")
        received = 0
        while True:
            block = response.read(1024 * 1024)
            if not block:
                break
            output.write(block)
            sha256.update(block)
            received += len(block)
            print(f"\r    Downloaded {received / (1024**2):,.1f} MiB", end="", flush=True)
        print()
        if expected is not None and received != int(expected):
            raise OSError(f"Incomplete download: got {received} of {expected} bytes")
        if not received:
            raise OSError("Empty download")
    return sha256.hexdigest()


def safe_archive_path(root: Path, name: str) -> Path:
    """Reject absolute paths and parent traversal in downloaded archives."""
    from pathlib import PurePosixPath
    parts = PurePosixPath(name).parts
    if not parts or name.startswith("/") or ".." in parts:
        raise ValueError(f"Unsafe archive entry: {name}")
    return root.joinpath(*parts)


def extract(archive: Path, archive_type: str, directory: Path) -> None:
    if archive_type == "zip":
        with zipfile.ZipFile(archive) as zf:
            for info in zf.infolist():
                destination = safe_archive_path(directory, info.filename)
                # Official export contains only regular files and directories.
                kind = (info.external_attr >> 16) & 0o170000
                if kind not in (0, 0o100000, 0o040000):
                    raise ValueError(f"Unexpected archive entry type: {info.filename}")
                if info.is_dir():
                    destination.mkdir(parents=True, exist_ok=True)
                else:
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    with zf.open(info) as src, destination.open("wb") as dst:
                        shutil.copyfileobj(src, dst)
    elif archive_type == "tgz":
        with tarfile.open(archive, "r:gz") as tf:
            for member in tf:
                destination = safe_archive_path(directory, member.name)
                if member.isdir():
                    destination.mkdir(parents=True, exist_ok=True)
                elif member.isfile():
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    stream = tf.extractfile(member)
                    if stream is None:
                        raise ValueError(f"Cannot extract: {member.name}")
                    with stream, destination.open("wb") as dst:
                        shutil.copyfileobj(stream, dst)
                else:
                    raise ValueError(f"Unexpected archive entry type: {member.name}")
    else:
        raise ValueError(f"Unknown archive type: {archive_type}")


def install_model(spec: tuple) -> None:
    group, label, relative_path, url, archive_type, minimum_size = spec
    destination = MODEL_ROOT / relative_path
    print(f"\n{label}: {destination.relative_to(PROJECT_ROOT)}")

    if is_present(destination, archive_type, minimum_size):
        print("  Already present; skipping.")
        return
    if destination.exists():
        raise RuntimeError(
            f"{destination} exists but failed the basic file/structure check. "
            "Inspect or remove it manually before retrying."
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".obtopomem-", dir=destination.parent) as temp:
        temp_path = Path(temp)
        downloaded = temp_path / "download"
        print(f"  Source: {url}")
        digest = download(url, downloaded)
        print(f"  SHA-256 (download): {digest}")

        if archive_type == "file":
            if downloaded.stat().st_size < minimum_size:
                raise RuntimeError(f"Downloaded checkpoint is unexpectedly small: {label}")
            os.replace(downloaded, destination)
        else:
            extracted = temp_path / "extracted"
            extracted.mkdir()
            extract(downloaded, archive_type, extracted)
            expected_dir = extracted / destination.name
            if not is_present(expected_dir, archive_type, 0):
                raise RuntimeError(
                    f"{label} archive did not contain expected {destination.name}/saved_model.pb"
                )
            os.replace(expected_dir, destination)
    print("  Installed successfully.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=("sam", "omniglue"),
                        help="Download just one model group (default: both).")
    parser.add_argument("--list", action="store_true",
                        help="Show download sources and paths without downloading.")
    args = parser.parse_args()
    selected = [m for m in MODELS if args.only is None or m[0] == args.only]
    print(f"Model directory: {MODEL_ROOT}")
    if args.list:
        for group, label, rel_path, url, kind, size in selected:
            print(f"  {label}: models/{rel_path}\n    {url}")
        return 0
    for model in selected:
        try:
            install_model(model)
        except (OSError, RuntimeError, ValueError, zipfile.BadZipFile, tarfile.TarError) as exc:
            print(f"\nERROR while installing {model[1]}: {exc}")
            return 1
    print("\nAll selected models are present. Note: no reference checksums are pinned yet.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
