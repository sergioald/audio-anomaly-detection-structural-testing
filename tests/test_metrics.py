import numpy as np
import pytest

from audio_anomaly.metrics import (
    histogram_overlap,
    mean_absolute_error,
    mean_squared_error,
    normalized_cross_correlation,
    normalized_cross_correlation_batch,
    peak_signal_to_noise_ratio,
)


def test_ncc_identical_arrays_is_one():
    x = np.arange(9).reshape(3, 3)
    assert np.isclose(normalized_cross_correlation(x, x), 1.0)


def test_ncc_opposite_arrays_is_minus_one():
    x = np.arange(9).reshape(3, 3)
    assert np.isclose(normalized_cross_correlation(x, -x), -1.0)


def test_ncc_constant_array_returns_zero():
    x = np.ones((3, 3))
    assert normalized_cross_correlation(x, x) == 0.0


def test_ncc_requires_equal_shapes():
    with pytest.raises(ValueError, match="equal shapes"):
        normalized_cross_correlation(np.zeros((2, 2)), np.zeros((2, 3)))


def test_ncc_batch_matches_single():
    ref = np.arange(9).reshape(3, 3)
    samples = np.stack([ref, -ref])
    scores = normalized_cross_correlation_batch(samples, ref)
    assert np.isclose(scores[0], normalized_cross_correlation(ref, ref))
    assert np.isclose(scores[1], normalized_cross_correlation(-ref, ref))


def test_ncc_batch_rejects_non_3d_samples():
    with pytest.raises(ValueError, match="Expected samples"):
        normalized_cross_correlation_batch(np.zeros((3, 3)), np.zeros((3, 3)))


def test_ncc_batch_rejects_reference_shape_mismatch():
    samples = np.zeros((2, 3, 3))
    reference = np.zeros((4, 3))
    with pytest.raises(ValueError, match="does not match"):
        normalized_cross_correlation_batch(samples, reference)


def test_ncc_batch_returns_zero_for_constant_reference():
    samples = np.stack([np.arange(9).reshape(3, 3), np.ones((3, 3))])
    reference = np.ones((3, 3))
    scores = normalized_cross_correlation_batch(samples, reference)
    assert np.array_equal(scores, np.zeros(2))


def test_error_metrics():
    a = np.array([1, 2, 3])
    b = np.array([1, 2, 4])
    assert np.isclose(mean_absolute_error(a, b), 1 / 3)
    assert np.isclose(mean_squared_error(a, b), 1 / 3)


def test_psnr_is_infinite_for_identical_arrays():
    a = np.array([0.0, 0.5, 1.0])
    assert peak_signal_to_noise_ratio(a, a) == float("inf")


def test_psnr_uses_supplied_data_range():
    a = np.array([0.0, 0.0])
    b = np.array([0.5, -0.5])
    assert np.isclose(peak_signal_to_noise_ratio(a, b, data_range=1.0), 10 * np.log10(1.0 / 0.25))


def test_histogram_overlap_lower_for_separated_data():
    a = np.array([0, 0, 0, 0])
    b = np.array([10, 10, 10, 10])
    c = np.array([0, 0, 10, 10])
    assert histogram_overlap(a, b, bins=2) < histogram_overlap(a, c, bins=2)


def test_histogram_overlap_identical_distributions_counts_all_samples():
    a = np.array([0, 0, 1, 1])
    assert histogram_overlap(a, a, bins=2) == len(a)


def test_histogram_overlap_constant_distributions_returns_shorter_length():
    a = np.ones(5)
    b = np.ones(3)
    assert histogram_overlap(a, b, bins=10) == 3
