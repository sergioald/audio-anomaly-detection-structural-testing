#!/usr/bin/env python
"""Create paper-style WST `.npy` dataset files from labelled raw `.wav` folders.

This is the advanced workflow for new data. The main public reproduction path
uses the already processed Zenodo WST arrays. Use this script when you have new
raw audio and want to create the same five dataset files expected by the rest of
this repository.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from audio_anomaly.audio import compute_wst_batch_from_audio_dir, write_window_records_csv
from audio_anomaly.data import DATASET_FILES

SPLIT_TO_ARG = {
    "normal_train": "normal_train_dir",
    "normal_validation": "normal_validation_dir",
    "anomalous_validation": "anomalous_validation_dir",
    "normal_test": "normal_test_dir",
    "anomalous_test": "anomalous_test_dir",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("data_new"))
    parser.add_argument("--normal-train-dir", type=Path, required=True)
    parser.add_argument("--normal-validation-dir", type=Path, required=True)
    parser.add_argument("--anomalous-validation-dir", type=Path, required=True)
    parser.add_argument("--normal-test-dir", type=Path, required=True)
    parser.add_argument("--anomalous-test-dir", type=Path, required=True)
    parser.add_argument("--sample-rate", type=int, default=48_000)
    parser.add_argument("--sample-seconds", type=float, default=0.5)
    parser.add_argument("--overlap-seconds", type=float, default=0.1)
    parser.add_argument("--j", type=int, default=6)
    parser.add_argument("--q", type=int, default=16)
    parser.add_argument("--allow-non-paper-shape", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    shape_check = None if args.allow_non_paper_shape else (221, 375)

    for split_key, arg_name in SPLIT_TO_ARG.items():
        audio_dir = getattr(args, arg_name)
        print(f"\nProcessing {split_key}: {audio_dir}")
        batch = compute_wst_batch_from_audio_dir(
            audio_dir=audio_dir,
            target_sample_rate=args.sample_rate,
            sample_seconds=args.sample_seconds,
            overlap_seconds=args.overlap_seconds,
            j=args.j,
            q=args.q,
            shape_check=shape_check,
        )
        output_path = args.output_dir / DATASET_FILES[split_key]
        np.save(output_path, batch.features)
        write_window_records_csv(batch.records, args.output_dir / f"{split_key}_windows.csv")
        print(f"Saved {output_path} with shape {batch.features.shape}")


if __name__ == "__main__":
    main()
