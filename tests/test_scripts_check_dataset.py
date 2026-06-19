import subprocess
import sys

import numpy as np

from audio_anomaly.data import DATASET_FILES


def _write_minimal_dataset(data_dir, n_samples=2, sample_shape=(221, 375)):
    data_dir.mkdir(parents=True, exist_ok=True)
    for filename in DATASET_FILES.values():
        np.save(data_dir / filename, np.zeros((n_samples, *sample_shape), dtype=np.float32))


def test_check_dataset_script_passes_for_subset_when_not_strict(tmp_path):
    data_dir = tmp_path / "data"
    _write_minimal_dataset(data_dir, n_samples=2)

    result = subprocess.run(
        [sys.executable, "scripts/check_dataset.py", "--data-dir", str(data_dir)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "Dataset files found" in result.stdout


def test_check_dataset_script_fails_when_files_are_missing(tmp_path):
    result = subprocess.run(
        [sys.executable, "scripts/check_dataset.py", "--data-dir", str(tmp_path / "missing")],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "Missing dataset files" in result.stdout


def test_check_dataset_script_fails_strict_sample_count_for_subset(tmp_path):
    data_dir = tmp_path / "data"
    _write_minimal_dataset(data_dir, n_samples=2)

    result = subprocess.run(
        [sys.executable, "scripts/check_dataset.py", "--data-dir", str(data_dir), "--strict"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "check" in result.stdout
