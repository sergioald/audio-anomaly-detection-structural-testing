#!/usr/bin/env python
"""Inspect legacy Keras `.h5` layer names without requiring TensorFlow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    args = parser.parse_args()

    with h5py.File(args.model, "r") as handle:
        config = handle.attrs.get("model_config")
        if isinstance(config, bytes):
            config = config.decode("utf-8")
        model_config = json.loads(config)

    print(f"Layers in {args.model}:")
    for index, layer in enumerate(model_config["config"]["layers"]):
        cfg = layer.get("config", {})
        print(
            f"{index:02d} {layer['class_name']:20s} "
            f"name={cfg.get('name')} filters={cfg.get('filters')} "
            f"kernel={cfg.get('kernel_size')} strides={cfg.get('strides')}"
        )


if __name__ == "__main__":
    main()
