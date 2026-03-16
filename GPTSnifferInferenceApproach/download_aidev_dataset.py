#!/usr/bin/env python3
"""Download files from the AIDev dataset on Hugging Face.

Examples:
    python download_aidev_dataset.py --list
    python download_aidev_dataset.py
    python download_aidev_dataset.py --files all_pull_request.parquet all_repository.parquet
    python download_aidev_dataset.py --output-dir data/aidev --force
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


DEFAULT_DATASET = "hao-li/AIDev"
HF_API_URL = "https://huggingface.co/api/datasets/{dataset}"
HF_DOWNLOAD_URL = "https://huggingface.co/datasets/{dataset}/resolve/main/{filename}"
DEFAULT_OUTPUT_DIR = "data/aidev"
CHUNK_SIZE = 1024 * 1024
KNOWN_DATA_SUFFIXES = {
    ".parquet",
    ".csv",
    ".tsv",
    ".json",
    ".jsonl",
    ".txt",
    ".zip",
    ".gz",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download files from the Hugging Face AIDev dataset."
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help=f"Hugging Face dataset repo id (default: {DEFAULT_DATASET})",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to save files into (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--files",
        nargs="+",
        help="Specific filenames to download. If omitted, downloads all detected data files.",
    )
    parser.add_argument(
        "--include-pattern",
        action="append",
        default=[],
        help="Only download files whose path contains this substring. May be passed multiple times.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available files and exit.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download files even if they already exist locally.",
    )
    return parser.parse_args()


def fetch_dataset_metadata(dataset: str) -> dict:
    url = HF_API_URL.format(dataset=urllib.parse.quote(dataset, safe="/"))
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "python-download-aidev-dataset"},
    )
    try:
        with urllib.request.urlopen(request) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            raise SystemExit(f"Dataset not found: {dataset}") from exc
        raise SystemExit(f"Failed to fetch dataset metadata: HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise SystemExit(f"Network error while fetching dataset metadata: {exc}") from exc


def list_dataset_files(dataset: str) -> list[str]:
    metadata = fetch_dataset_metadata(dataset)
    siblings = metadata.get("siblings", [])
    return sorted(
        entry["rfilename"]
        for entry in siblings
        if isinstance(entry, dict) and "rfilename" in entry
    )


def is_data_file(filename: str) -> bool:
    path = Path(filename)
    if path.name.lower() in {"readme.md", ".gitattributes"}:
        return False
    return path.suffix.lower() in KNOWN_DATA_SUFFIXES


def resolve_files(available_files: list[str], args: argparse.Namespace) -> list[str]:
    files = available_files

    if args.include_pattern:
        files = [
            path
            for path in files
            if all(pattern in path for pattern in args.include_pattern)
        ]

    if args.files:
        available_lookup = set(files)
        missing = [name for name in args.files if name not in available_lookup]
        if missing:
            raise SystemExit(
                "Requested files were not found in the dataset:\n"
                + "\n".join(f"  - {name}" for name in missing)
            )
        return args.files

    return [path for path in files if is_data_file(path)]


def format_bytes(num_bytes: int | None) -> str:
    if num_bytes is None:
        return "unknown size"

    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{num_bytes} B"


def build_download_url(dataset: str, filename: str) -> str:
    quoted_dataset = urllib.parse.quote(dataset, safe="/")
    quoted_filename = urllib.parse.quote(filename, safe="/")
    return HF_DOWNLOAD_URL.format(dataset=quoted_dataset, filename=quoted_filename)


def download_file(dataset: str, filename: str, output_dir: Path, force: bool) -> None:
    destination = output_dir / filename
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists() and not force:
        print(f"Skipping existing file: {destination}")
        return

    url = build_download_url(dataset, filename)
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "python-download-aidev-dataset"},
    )

    try:
        with urllib.request.urlopen(request) as response, destination.open("wb") as handle:
            total_size = response.headers.get("Content-Length")
            expected = int(total_size) if total_size else None
            downloaded = 0

            print(f"Downloading {filename} -> {destination} ({format_bytes(expected)})")
            while True:
                chunk = response.read(CHUNK_SIZE)
                if not chunk:
                    break
                handle.write(chunk)
                downloaded += len(chunk)
                if expected:
                    percent = downloaded / expected * 100
                    print(
                        f"  {downloaded:,}/{expected:,} bytes ({percent:5.1f}%)",
                        end="\r",
                        flush=True,
                    )

            if expected:
                print(" " * 80, end="\r")
            print(f"Finished {filename} ({format_bytes(downloaded)})")
    except urllib.error.HTTPError as exc:
        if destination.exists():
            destination.unlink()
        raise SystemExit(f"Failed to download {filename}: HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        if destination.exists():
            destination.unlink()
        raise SystemExit(f"Network error while downloading {filename}: {exc}") from exc


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()

    available_files = list_dataset_files(args.dataset)
    if not available_files:
        print("No files were found in the dataset repository.")
        return 1

    if args.list:
        print(f"Files in {args.dataset}:")
        for filename in available_files:
            print(f"  - {filename}")
        return 0

    selected_files = resolve_files(available_files, args)
    if not selected_files:
        print("No files matched your selection.")
        return 1

    print(f"Dataset: {args.dataset}")
    print(f"Output directory: {output_dir}")
    print(f"Files to download: {len(selected_files)}")

    for filename in selected_files:
        download_file(args.dataset, filename, output_dir, args.force)

    print("Download complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
