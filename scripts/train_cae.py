#!/usr/bin/env python
"""Train the latent-24 CAE on normal WST training data."""

from __future__ import annotations

import argparse
from pathlib import Path

from audio_anomaly.data import ensure_channel_axis, load_array
from audio_anomaly.model import build_cae_latent24, compile_autoencoder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-model", type=Path, default=Path("models/cae_wst_latent24_retrained.keras"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    args = parser.parse_args()

    data = ensure_channel_axis(load_array(args.data_dir, "normal_train"))
    split = int((1 - args.validation_fraction) * len(data))
    x_train, x_validation = data[:split], data[split:]

    model = compile_autoencoder(build_cae_latent24())
    args.output_model.parent.mkdir(parents=True, exist_ok=True)

    from tensorflow import keras

    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath=args.output_model,
        save_best_only=True,
        monitor="val_loss",
        mode="min",
        verbose=1,
    )

    model.fit(
        x_train,
        x_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        shuffle=True,
        validation_data=(x_validation, x_validation),
        callbacks=[checkpoint],
    )


if __name__ == "__main__":
    main()
