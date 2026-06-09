#!/usr/bin/env python
"""Predict anomalies for new WST features or raw audio using saved pipeline assets."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from audio_anomaly.audio import AudioWindowRecord, compute_wst_batch_from_audio_dir
from audio_anomaly.classifiers import load_classifier
from audio_anomaly.feature_maps import load_reference_maps, score_dataset_against_references
from audio_anomaly.model import find_feature_layer, load_autoencoder


def _records_for_features(n: int) -> tuple[AudioWindowRecord, ...]:
    return tuple(AudioWindowRecord(source_file="features.npy", window_index=i, start_seconds=float("nan"), end_seconds=float("nan")) for i in range(n))


def main() -> None:
    parser = argparse.ArgumentParser()
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--features", type=Path, help="Precomputed WST feature array with shape (n, 221, 375)")
    input_group.add_argument("--audio-dir", type=Path, help="Folder of `.wav` files to convert to WST features before prediction")
    parser.add_argument("--model", type=Path, default=Path("models/pretrained_cae_wst_latent24_structural_audio.h5"), help="Trained `.keras` or `.h5` CAE")
    parser.add_argument("--classifier", type=Path, required=True, help="Saved joblib classifier from evaluate_feature_map_classifier.py")
    parser.add_argument("--reference-maps", type=Path, required=True, help="reference_maps.npz from evaluate_feature_map_classifier.py")
    parser.add_argument("--layer-name", default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/new_audio_predictions"))
    parser.add_argument("--sample-rate", type=int, default=48_000)
    parser.add_argument("--sample-seconds", type=float, default=0.5)
    parser.add_argument("--overlap-seconds", type=float, default=0.1)
    parser.add_argument("--j", type=int, default=6)
    parser.add_argument("--q", type=int, default=16)
    args = parser.parse_args()

    if args.audio_dir:
        batch = compute_wst_batch_from_audio_dir(
            args.audio_dir,
            target_sample_rate=args.sample_rate,
            sample_seconds=args.sample_seconds,
            overlap_seconds=args.overlap_seconds,
            j=args.j,
            q=args.q,
        )
        features = batch.features
        records = batch.records
    else:
        features = np.load(args.features)
        records = _records_for_features(len(features))

    model = load_autoencoder(args.model, compile_model=False)
    layer_name = args.layer_name or find_feature_layer(model)
    reference_maps = load_reference_maps(args.reference_maps)
    classifier = load_classifier(args.classifier)

    scores = score_dataset_against_references(
        model=model,
        data=features,
        reference_maps=reference_maps,
        layer_name=layer_name,
        batch_size=args.batch_size,
    )
    predictions = classifier.predict(scores)
    if hasattr(classifier, "predict_proba"):
        probabilities = classifier.predict_proba(scores)[:, 1]
    else:
        probabilities = np.full(len(predictions), np.nan)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "scores.npy", scores)
    np.save(args.output_dir / "predictions.npy", predictions)

    csv_path = args.output_dir / "predictions.csv"
    score_headers = [f"ncc_map_{map_id}" for map_id in reference_maps]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["source_file", "window_index", "start_seconds", "end_seconds", *score_headers, "prediction", "anomaly_probability"])
        for record, score_row, prediction, probability in zip(records, scores, predictions, probabilities, strict=True):
            writer.writerow([
                record.source_file,
                record.window_index,
                record.start_seconds,
                record.end_seconds,
                *[f"{value:.8f}" for value in score_row],
                int(prediction),
                "" if np.isnan(probability) else f"{probability:.8f}",
            ])
    print(f"Saved predictions: {csv_path}")
    print(f"Predicted anomalous windows: {int(np.sum(predictions == 1))}/{len(predictions)}")


if __name__ == "__main__":
    main()
