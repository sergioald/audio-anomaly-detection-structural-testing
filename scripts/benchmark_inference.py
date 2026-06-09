#!/usr/bin/env python
"""Benchmark feature-map inference and NCC scoring latency."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from audio_anomaly.data import load_array
from audio_anomaly.feature_maps import average_reference_maps, score_dataset_against_references
from audio_anomaly.model import find_feature_layer, load_autoencoder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--model", type=Path, default=Path("models/pretrained_cae_wst_latent24_structural_audio.h5"))
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--map-ids", type=int, nargs="+", default=[18, 19])
    args = parser.parse_args()

    model = load_autoencoder(args.model, compile_model=False)
    layer_name = find_feature_layer(model)
    normal_train = load_array(args.data_dir, "normal_train")[: args.n_samples]
    normal_test = load_array(args.data_dir, "normal_test")[: args.n_samples]

    refs = average_reference_maps(model, normal_train, layer_name=layer_name, map_ids=args.map_ids, batch_size=16)
    start = time.perf_counter()
    scores = score_dataset_against_references(
        model, normal_test, refs, layer_name=layer_name, batch_size=args.batch_size
    )
    elapsed = time.perf_counter() - start
    print(f"Scored {len(scores)} samples in {elapsed:.4f} s")
    print(f"Mean time per sample: {elapsed / len(scores):.6f} s")


if __name__ == "__main__":
    main()
