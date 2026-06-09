from pathlib import Path

import numpy as np

from audio_anomaly.data import (
    DATASET_FILES,
    LEGACY_TO_PUBLIC_FILENAMES,
    ensure_channel_axis,
    validate_shapes,
)


def test_public_dataset_filenames_are_defined():
    assert DATASET_FILES["normal_train"] == "Normal_Data_Training.npy"
    assert DATASET_FILES["anomalous_test"] == "Anomalous_Data_Test.npy"


def test_legacy_filename_mapping():
    assert LEGACY_TO_PUBLIC_FILENAMES["REVISIONS_combined_training_DS_scat.npy"] == "Normal_Data_Training.npy"


def test_ensure_channel_axis_adds_singleton_channel():
    x = np.zeros((2, 221, 375))
    y = ensure_channel_axis(x)
    assert y.shape == (2, 221, 375, 1)


def test_validate_shapes_accepts_subset_sample_counts():
    arrays = {"normal_train": np.zeros((2, 221, 375))}
    messages = validate_shapes(arrays, strict_sample_count=False)
    assert "ok" in messages["normal_train"]
