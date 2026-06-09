"""Dataset loading and validation utilities.

The public data are not stored in this repository. They are expected as `.npy`
files downloaded from Zenodo into a local data directory.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

DATASET_FILES: dict[str, str] = {
    "normal_train": "Normal_Data_Training.npy",
    "normal_validation": "Normal_Data_Validation.npy",
    "anomalous_validation": "Anomalous_Data_Validation.npy",
    "normal_test": "Normal_Data_Test.npy",
    "anomalous_test": "Anomalous_Data_Test.npy",
}

EXPECTED_SHAPES: dict[str, tuple[int, int, int]] = {
    "normal_train": (1347, 221, 375),
    "normal_validation": (1347, 221, 375),
    "anomalous_validation": (80, 221, 375),
    "normal_test": (1347, 221, 375),
    "anomalous_test": (79, 221, 375),
}

LEGACY_TO_PUBLIC_FILENAMES: dict[str, str] = {
    "REVISIONS_combined_training_DS_scat.npy": "Normal_Data_Training.npy",
    "REVISIONS_combined_val_DS.npy": "Normal_Data_Validation.npy",
    "REVISIONS_anomalies_validation.npy": "Anomalous_Data_Validation.npy",
    "REVISIONS_combined_test_DS.npy": "Normal_Data_Test.npy",
    "REVISIONS_anomalies_test.npy": "Anomalous_Data_Test.npy",
}


@dataclass(frozen=True)
class DatasetBundle:
    """Container for the five public WST arrays."""

    normal_train: np.ndarray
    normal_validation: np.ndarray
    anomalous_validation: np.ndarray
    normal_test: np.ndarray
    anomalous_test: np.ndarray

    def as_dict(self) -> dict[str, np.ndarray]:
        return {
            "normal_train": self.normal_train,
            "normal_validation": self.normal_validation,
            "anomalous_validation": self.anomalous_validation,
            "normal_test": self.normal_test,
            "anomalous_test": self.anomalous_test,
        }


def dataset_path(data_dir: str | Path, key: str) -> Path:
    """Return the expected path for a dataset key."""
    try:
        filename = DATASET_FILES[key]
    except KeyError as exc:
        valid = ", ".join(DATASET_FILES)
        raise KeyError(f"Unknown dataset key {key!r}. Valid keys: {valid}") from exc
    return Path(data_dir) / filename


def validate_dataset_files(data_dir: str | Path) -> dict[str, bool]:
    """Check which expected dataset files are present."""
    data_dir = Path(data_dir)
    return {key: dataset_path(data_dir, key).exists() for key in DATASET_FILES}


def missing_dataset_files(data_dir: str | Path) -> list[Path]:
    """Return missing dataset file paths."""
    data_dir = Path(data_dir)
    return [dataset_path(data_dir, key) for key in DATASET_FILES if not dataset_path(data_dir, key).exists()]


def load_array(data_dir: str | Path, key: str, mmap_mode: str | None = None) -> np.ndarray:
    """Load one WST array by key."""
    path = dataset_path(data_dir, key)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing dataset file: {path}. Run scripts/download_data.py or place the Zenodo file in data/."
        )
    return np.load(path, mmap_mode=mmap_mode)


def load_dataset_bundle(data_dir: str | Path, mmap_mode: str | None = None) -> DatasetBundle:
    """Load all five public arrays."""
    arrays = {key: load_array(data_dir, key, mmap_mode=mmap_mode) for key in DATASET_FILES}
    return DatasetBundle(**arrays)


def ensure_channel_axis(array: np.ndarray) -> np.ndarray:
    """Return array with final channel dimension for Keras image input.

    The Zenodo data are expected as `(n_samples, 221, 375)`. Keras models in this
    repository expect `(n_samples, 221, 375, 1)`.
    """
    arr = np.asarray(array)
    if arr.ndim == 3:
        return arr[..., np.newaxis]
    if arr.ndim == 4 and arr.shape[-1] == 1:
        return arr
    raise ValueError(
        "Expected array with shape (n_samples, 221, 375) or (n_samples, 221, 375, 1); "
        f"got {arr.shape}."
    )


def validate_shapes(
    arrays: Mapping[str, np.ndarray],
    expected_shapes: Mapping[str, tuple[int, int, int]] = EXPECTED_SHAPES,
    strict_sample_count: bool = False,
) -> dict[str, str]:
    """Validate dataset shapes and return human-readable status messages.

    Parameters
    ----------
    arrays:
        Mapping of dataset key to numpy array.
    expected_shapes:
        Expected public shapes.
    strict_sample_count:
        If false, only the sample dimensions `(221, 375)` are enforced. This is
        useful for tests or smaller local subsets.
    """
    messages: dict[str, str] = {}
    for key, arr in arrays.items():
        expected = expected_shapes.get(key)
        if expected is None:
            messages[key] = "unknown key; skipped"
            continue
        if arr.ndim not in (3, 4):
            raise ValueError(f"{key}: expected 3D or 4D array, got shape {arr.shape}")
        sample_shape = arr.shape[1:3]
        if sample_shape != expected[1:3]:
            raise ValueError(f"{key}: expected sample shape {expected[1:3]}, got {sample_shape}")
        if strict_sample_count and arr.shape[0] != expected[0]:
            raise ValueError(f"{key}: expected {expected[0]} samples, got {arr.shape[0]}")
        messages[key] = f"ok: {arr.shape}"
    return messages


def make_binary_dataset(normal_scores: np.ndarray, anomalous_scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build feature matrix and binary labels from normal/anomalous score matrices.

    Normal samples are label `0`; anomalous samples are label `1`.
    """
    x = np.vstack([np.asarray(normal_scores), np.asarray(anomalous_scores)])
    y = np.concatenate([
        np.zeros(len(normal_scores), dtype=int),
        np.ones(len(anomalous_scores), dtype=int),
    ])
    return x, y
