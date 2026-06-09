#!/usr/bin/env python
"""Check that local Zenodo `.npy` files are present and shaped as expected."""

from __future__ import annotations

import argparse
from pathlib import Path

from audio_anomaly.data import DATASET_FILES, EXPECTED_SHAPES, load_array, missing_dataset_files


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--strict", action="store_true", help="Enforce exact public sample counts")
    args = parser.parse_args()

    missing = missing_dataset_files(args.data_dir)
    if missing:
        print("Missing dataset files:")
        for path in missing:
            print(f"  - {path}")
        raise SystemExit(1)

    print("Dataset files found:")
    for key, filename in DATASET_FILES.items():
        arr = load_array(args.data_dir, key, mmap_mode="r")
        expected = EXPECTED_SHAPES[key]
        sample_shape_ok = arr.shape[1:3] == expected[1:3]
        sample_count_ok = arr.shape[0] == expected[0]
        status = "ok" if sample_shape_ok and (sample_count_ok or not args.strict) else "check"
        print(f"  {status:5s} {filename:35s} shape={arr.shape} expected={expected}")
        if not sample_shape_ok or (args.strict and not sample_count_ok):
            raise SystemExit(1)


if __name__ == "__main__":
    main()
