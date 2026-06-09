"""Metrics used in the anomaly-detection workflow."""

from __future__ import annotations

import numpy as np


def normalized_cross_correlation(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    """Compute zero-mean normalised cross-correlation between two arrays."""
    a_arr = np.asarray(a, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)
    if a_arr.shape != b_arr.shape:
        raise ValueError(f"NCC requires equal shapes; got {a_arr.shape} and {b_arr.shape}")
    a_zero = a_arr - np.mean(a_arr)
    b_zero = b_arr - np.mean(b_arr)
    denominator = np.sqrt(np.sum(a_zero**2) * np.sum(b_zero**2))
    if denominator < eps:
        return 0.0
    return float(np.sum(a_zero * b_zero) / denominator)


def normalized_cross_correlation_batch(
    samples: np.ndarray,
    reference: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    """Compute NCC for many samples against one reference map.

    `samples` should have shape `(n_samples, height, width)` and `reference`
    should have shape `(height, width)`.
    """
    sample_arr = np.asarray(samples, dtype=np.float64)
    ref_arr = np.asarray(reference, dtype=np.float64)
    if sample_arr.ndim != 3:
        raise ValueError(f"Expected samples with shape (n, h, w), got {sample_arr.shape}")
    if sample_arr.shape[1:] != ref_arr.shape:
        raise ValueError(f"Reference shape {ref_arr.shape} does not match samples {sample_arr.shape[1:]}")

    sample_zero = sample_arr - np.mean(sample_arr, axis=(1, 2), keepdims=True)
    ref_zero = ref_arr - np.mean(ref_arr)
    numerator = np.sum(sample_zero * ref_zero, axis=(1, 2))
    denominator = np.sqrt(np.sum(sample_zero**2, axis=(1, 2)) * np.sum(ref_zero**2))
    return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > eps)


def mean_absolute_error(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))


def mean_squared_error(a: np.ndarray, b: np.ndarray) -> float:
    diff = np.asarray(a) - np.asarray(b)
    return float(np.mean(diff**2))


def peak_signal_to_noise_ratio(a: np.ndarray, b: np.ndarray, data_range: float | None = None) -> float:
    """Compute PSNR for two arrays."""
    a_arr = np.asarray(a)
    b_arr = np.asarray(b)
    mse = mean_squared_error(a_arr, b_arr)
    if mse == 0:
        return float("inf")
    if data_range is None:
        data_range = float(np.max(a_arr) - np.min(a_arr)) or 1.0
    return float(10 * np.log10((data_range**2) / mse))


def histogram_overlap(a: np.ndarray, b: np.ndarray, bins: int = 200) -> float:
    """Estimate overlap between two one-dimensional distributions.

    This follows the paper-script idea of using histogram overlap as a practical
    separation score. Lower values indicate better separation.
    """
    a_arr = np.asarray(a).ravel()
    b_arr = np.asarray(b).ravel()
    lower = min(float(np.min(a_arr)), float(np.min(b_arr)))
    upper = max(float(np.max(a_arr)), float(np.max(b_arr)))
    if lower == upper:
        return float(min(len(a_arr), len(b_arr)))
    hist_a, edges = np.histogram(a_arr, bins=bins, range=(lower, upper))
    hist_b, _ = np.histogram(b_arr, bins=edges)
    return float(np.minimum(hist_a, hist_b).sum())
