"""Plotting helpers for reports and exploratory analysis."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_ncc_histogram(
    normal_scores: np.ndarray,
    anomalous_scores: np.ndarray,
    output_path: str | Path | None = None,
    bins: int = 200,
):
    """Plot normal and anomalous NCC score histograms."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(normal_scores, bins=bins, alpha=0.7, label="Normal")
    ax.hist(anomalous_scores, bins=bins, alpha=0.7, label="Anomalous")
    ax.set_xlabel("NCC score")
    ax.set_ylabel("Number of samples")
    ax.legend()
    fig.tight_layout()
    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200)
    return fig


def plot_2d_scores(
    normal_scores: np.ndarray,
    anomalous_scores: np.ndarray,
    output_path: str | Path | None = None,
):
    """Plot two-dimensional feature-map score separation."""
    if normal_scores.shape[1] != 2 or anomalous_scores.shape[1] != 2:
        raise ValueError("2D score plot requires score matrices with exactly two columns")
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(normal_scores[:, 0], normal_scores[:, 1], s=8, label="Normal", alpha=0.7)
    ax.scatter(anomalous_scores[:, 0], anomalous_scores[:, 1], s=16, label="Anomalous", alpha=0.8)
    ax.set_xlabel("NCC for feature map 18")
    ax.set_ylabel("NCC for feature map 19")
    ax.legend()
    fig.tight_layout()
    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200)
    return fig
