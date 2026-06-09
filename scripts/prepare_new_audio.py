#!/usr/bin/env python
"""Compute WST feature maps from a folder of new `.wav` files.

Use this for inference on new audio when you already have a trained CAE,
reference maps and classifier from the paper-style workflow.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from audio_anomaly.audio import compute_wst_batch_from_audio_dir, write_window_records_csv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--output-features", type=Path, default=Path("outputs/new_audio/features.npy"))
    parser.add_argument("--output-windows", type=Path, default=Path("outputs/new_audio/windows.csv"))
    parser.add_argument("--sample-rate", type=int, default=48_000)
    parser.add_argument("--sample-seconds", type=float, default=0.5)
    parser.add_argument("--overlap-seconds", type=float, default=0.1)
    parser.add_argument("--j", type=int, default=6)
    parser.add_argument("--q", type=int, default=16)
    parser.add_argument("--allow-non-paper-shape", action="store_true")
    args = parser.parse_args()

    shape_check = None if args.allow_non_paper_shape else (221, 375)
    batch = compute_wst_batch_from_audio_dir(
        audio_dir=args.audio_dir,
        target_sample_rate=args.sample_rate,
        sample_seconds=args.sample_seconds,
        overlap_seconds=args.overlap_seconds,
        j=args.j,
        q=args.q,
        shape_check=shape_check,
    )
    args.output_features.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output_features, batch.features)
    write_window_records_csv(batch.records, args.output_windows)
    print(f"Saved features: {args.output_features} {batch.features.shape}")
    print(f"Saved windows:  {args.output_windows}")


if __name__ == "__main__":
    main()
