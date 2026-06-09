#!/usr/bin/env python
"""Evaluate the final CAE feature-map/NCC classifier pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from audio_anomaly.classifiers import save_classifier, train_classifiers
from audio_anomaly.data import load_dataset_bundle
from audio_anomaly.evaluation import evaluate_feature_map_pipeline
from audio_anomaly.feature_maps import save_reference_maps
from audio_anomaly.model import find_feature_layer, load_autoencoder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--model", type=Path, default=Path("models/pretrained_cae_wst_latent24_structural_audio.h5"), help="Path to trained `.keras` or `.h5` CAE model")
    parser.add_argument("--layer-name", default=None, help="Feature layer name. If omitted, encoder_conv2 or the first 64-filter Conv2D layer is used.")
    parser.add_argument("--map-ids", type=int, nargs="+", default=[18, 19])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/evaluation"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset_bundle(args.data_dir, mmap_mode=None)
    model = load_autoencoder(args.model, compile_model=False)
    layer_name = args.layer_name or find_feature_layer(model)
    print(f"Using feature layer: {layer_name}")

    result = evaluate_feature_map_pipeline(
        model=model,
        dataset=dataset,
        layer_name=layer_name,
        map_ids=args.map_ids,
        batch_size=args.batch_size,
    )

    metrics_path = args.output_dir / "classifier_metrics.json"
    metrics_path.write_text(json.dumps(result.metrics, indent=2), encoding="utf-8")
    np.save(args.output_dir / "validation_scores.npy", result.validation_scores)
    np.save(args.output_dir / "validation_labels.npy", result.validation_labels)
    np.save(args.output_dir / "test_scores.npy", result.test_scores)
    np.save(args.output_dir / "test_labels.npy", result.test_labels)
    save_reference_maps(result.reference_maps, args.output_dir / "reference_maps.npz")

    classifiers = train_classifiers(result.validation_scores, result.validation_labels)
    best_classifier = classifiers[result.best_classifier_name]
    save_classifier(best_classifier, args.output_dir / "best_classifier.joblib")
    save_classifier(best_classifier, args.output_dir / f"{result.best_classifier_name}.joblib")

    print(f"Saved metrics: {metrics_path}")
    print(f"Best classifier: {result.best_classifier_name}")
    print(json.dumps(result.metrics[result.best_classifier_name], indent=2))


if __name__ == "__main__":
    main()
