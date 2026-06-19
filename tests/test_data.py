import numpy as np
import pytest

from audio_anomaly.data import (
    DATASET_FILES,
    EXPECTED_SHAPES,
    dataset_path,
    ensure_channel_axis,
    load_array,
    load_dataset_bundle,
    make_binary_dataset,
    missing_dataset_files,
    validate_dataset_files,
    validate_shapes,
)


def _write_minimal_dataset(data_dir, n_samples=2, sample_shape=(221, 375)):
    data_dir.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for key, filename in DATASET_FILES.items():
        array = np.full((n_samples, *sample_shape), fill_value=len(arrays), dtype=np.float32)
        np.save(data_dir / filename, array)
        arrays[key] = array
    return arrays


def test_dataset_path_returns_expected_filename(tmp_path):
    path = dataset_path(tmp_path, "normal_train")
    assert path == tmp_path / "Normal_Data_Training.npy"


def test_dataset_path_rejects_unknown_key(tmp_path):
    with pytest.raises(KeyError, match="Unknown dataset key"):
        dataset_path(tmp_path, "not_a_split")


def test_validate_and_missing_dataset_files(tmp_path):
    status = validate_dataset_files(tmp_path)
    assert set(status) == set(DATASET_FILES)
    assert not any(status.values())

    missing = missing_dataset_files(tmp_path)
    assert len(missing) == len(DATASET_FILES)
    assert all(path.parent == tmp_path for path in missing)

    _write_minimal_dataset(tmp_path)
    assert all(validate_dataset_files(tmp_path).values())
    assert missing_dataset_files(tmp_path) == []


def test_load_array_and_bundle_from_minimal_dataset(tmp_path):
    arrays = _write_minimal_dataset(tmp_path, n_samples=3)

    loaded = load_array(tmp_path, "normal_train")
    assert np.array_equal(loaded, arrays["normal_train"])

    bundle = load_dataset_bundle(tmp_path)
    assert np.array_equal(bundle.normal_train, arrays["normal_train"])
    assert np.array_equal(bundle.anomalous_test, arrays["anomalous_test"])
    assert set(bundle.as_dict()) == set(DATASET_FILES)


def test_load_array_reports_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="Missing dataset file"):
        load_array(tmp_path, "normal_train")


def test_ensure_channel_axis_adds_or_preserves_single_channel():
    array_3d = np.zeros((2, 221, 375), dtype=np.float32)
    expanded = ensure_channel_axis(array_3d)
    assert expanded.shape == (2, 221, 375, 1)
    assert np.array_equal(expanded[..., 0], array_3d)

    array_4d = np.zeros((2, 221, 375, 1), dtype=np.float32)
    preserved = ensure_channel_axis(array_4d)
    assert preserved.shape == array_4d.shape
    assert np.array_equal(preserved, array_4d)


@pytest.mark.parametrize(
    "bad_shape",
    [
        (221, 375),
        (2, 221, 375, 2),
    ],
)
def test_ensure_channel_axis_rejects_invalid_shapes(bad_shape):
    with pytest.raises(ValueError, match="Expected array"):
        ensure_channel_axis(np.zeros(bad_shape, dtype=np.float32))


def test_ensure_channel_axis_preserves_generic_single_channel_4d_arrays():
    array = np.zeros((2, 10, 10, 1), dtype=np.float32)

    preserved = ensure_channel_axis(array)

    assert preserved.shape == array.shape
    assert np.array_equal(preserved, array)


def test_validate_shapes_allows_subset_counts_when_not_strict():
    arrays = {
        key: np.zeros((2, EXPECTED_SHAPES[key][1], EXPECTED_SHAPES[key][2]), dtype=np.float32)
        for key in DATASET_FILES
    }

    messages = validate_shapes(arrays, strict_sample_count=False)

    assert set(messages) == set(DATASET_FILES)
    assert all(message.startswith("ok:") for message in messages.values())


def test_validate_shapes_rejects_bad_sample_shape():
    arrays = {"normal_train": np.zeros((2, 100, 375), dtype=np.float32)}

    with pytest.raises(ValueError, match="expected sample shape"):
        validate_shapes(arrays)


def test_validate_shapes_rejects_bad_sample_count_when_strict():
    arrays = {"normal_train": np.zeros((2, 221, 375), dtype=np.float32)}

    with pytest.raises(ValueError, match="expected 1347 samples"):
        validate_shapes(arrays, strict_sample_count=True)


def test_validate_shapes_skips_unknown_keys():
    messages = validate_shapes({"custom": np.zeros((2, 10, 10), dtype=np.float32)})
    assert messages == {"custom": "unknown key; skipped"}


def test_make_binary_dataset_stacks_scores_and_labels():
    normal_scores = np.array([[0.9, 0.8], [0.85, 0.75]])
    anomalous_scores = np.array([[0.1, 0.2]])

    x, y = make_binary_dataset(normal_scores, anomalous_scores)

    assert np.array_equal(x, np.vstack([normal_scores, anomalous_scores]))
    assert np.array_equal(y, np.array([0, 0, 1]))
