#!/usr/bin/env python
"""Download the public Zenodo WST arrays.

The files are large. This script intentionally downloads them into `data/`,
which is ignored by Git.
"""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

ZENODO_RECORD = "14298279"
BASE_URL = f"https://zenodo.org/records/{ZENODO_RECORD}/files"
FILES = [
    "Normal_Data_Training.npy",
    "Normal_Data_Validation.npy",
    "Anomalous_Data_Validation.npy",
    "Normal_Data_Test.npy",
    "Anomalous_Data_Test.npy",
]


def download_file(url: str, destination: Path, overwrite: bool = False) -> None:
    if destination.exists() and not overwrite:
        print(f"Already exists, skipping: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url}")
    with urllib.request.urlopen(url) as response, destination.open("wb") as handle:
        total = response.headers.get("Content-Length")
        total_int = int(total) if total else None
        downloaded = 0
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
            downloaded += len(chunk)
            if total_int:
                pct = downloaded / total_int * 100
                print(f"  {destination.name}: {pct:5.1f}%", end="\r")
        print(f"  saved: {destination}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data"), help="Output directory")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files")
    args = parser.parse_args()

    for filename in FILES:
        url = f"{BASE_URL}/{filename}?download=1"
        download_file(url, args.output / filename, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
