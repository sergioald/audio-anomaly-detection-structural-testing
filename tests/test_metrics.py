import numpy as np

from audio_anomaly.metrics import (
    histogram_overlap,
    mean_absolute_error,
    mean_squared_error,
    normalized_cross_correlation,
    normalized_cross_correlation_batch,
)


def test_ncc_identical_arrays_is_one():
    x = np.arange(9).reshape(3, 3)
    assert np.isclose(normalized_cross_correlation(x, x), 1.0)


def test_ncc_batch_matches_single():
    ref = np.arange(9).reshape(3, 3)
    samples = np.stack([ref, -ref])
    scores = normalized_cross_correlation_batch(samples, ref)
    assert np.isclose(scores[0], normalized_cross_correlation(ref, ref))
    assert np.isclose(scores[1], normalized_cross_correlation(-ref, ref))


def test_error_metrics():
    a = np.array([1, 2, 3])
    b = np.array([1, 2, 4])
    assert mean_absolute_error(a, b) == 1 / 3
    assert mean_squared_error(a, b) == 1 / 3


def test_histogram_overlap_lower_for_separated_data():
    a = np.array([0, 0, 0, 0])
    b = np.array([10, 10, 10, 10])
    c = np.array([0, 0, 10, 10])
    assert histogram_overlap(a, b, bins=2) < histogram_overlap(a, c, bins=2)
