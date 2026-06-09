#!/usr/bin/env python
"""Evaluate reconstruction-error baselines for PCA and CAE.

This script is included for reproducibility of the negative result reported in
paper development: reconstruction error alone did not provide satisfactory
normal/anomalous separation for this dataset.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

from audio_anomaly.data import ensure_channel_axis, load_array
from audio_anomaly.metrics import mean_absolute_error, mean_squared_error, normalized_cross_correlation
from audio_anomaly.model import load_autoencoder


def reconstruction_metrics(original: np.ndarray, reconstructed: np.ndarray) -> dict[str, list[float]]:
    return {
        "mae": [mean_absolute_error(a, b) for a, b in zip(original, reconstructed, strict=True)],
        "mse": [mean_squared_error(a, b) for a, b in zip(original, reconstructed, strict=True)],
        "ncc": [normalized_cross_correlation(a, b) for a, b in zip(original, reconstructed, strict=True)],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--model", type=Path, default=None, help="Optional CAE model for reconstruction baseline")
    parser.add_argument("--pca-components", type=int, nargs="+", default=[24, 1347])
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/reconstruction_baselines"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train = load_array(args.data_dir, "normal_train")
    normal_validation = load_array(args.data_dir, "normal_validation")
    anomalous_validation = load_array(args.data_dir, "anomalous_validation")

    train_flat = train.reshape(len(train), -1)
    normal_flat = normal_validation.reshape(len(normal_validation), -1)
    anomalous_flat = anomalous_validation.reshape(len(anomalous_validation), -1)

    results: dict[str, dict] = {}
    for n_components in args.pca_components:
        pca = PCA(n_components=n_components)
        pca.fit(train_flat)
        normal_reconstructed = pca.inverse_transform(pca.transform(normal_flat))
        anomalous_reconstructed = pca.inverse_transform(pca.transform(anomalous_flat))
        results[f"pca_{n_components}"] = {
            "normal": reconstruction_metrics(normal_flat, normal_reconstructed),
            "anomalous": reconstruction_metrics(anomalous_flat, anomalous_reconstructed),
        }

    if args.model:
        model = load_autoencoder(args.model, compile_model=False)
        normal_input = ensure_channel_axis(normal_validation)
        anomalous_input = ensure_channel_axis(anomalous_validation)
        normal_reconstructed = model.predict(normal_input, verbose=1)[..., 0]
        anomalous_reconstructed = model.predict(anomalous_input, verbose=1)[..., 0]
        results["cae"] = {
            "normal": reconstruction_metrics(normal_validation, normal_reconstructed),
            "anomalous": reconstruction_metrics(anomalous_validation, anomalous_reconstructed),
        }

    output = args.output_dir / "reconstruction_metrics.json"
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
