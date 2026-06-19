import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from audio_anomaly.plotting import plot_2d_scores, plot_ncc_histogram


def test_plot_ncc_histogram_returns_figure_and_saves_file(tmp_path):
    output = tmp_path / "histogram.png"

    fig = plot_ncc_histogram(np.array([0.8, 0.9]), np.array([0.1, 0.2]), output_path=output, bins=4)

    assert output.exists()
    assert fig.axes[0].get_xlabel() == "NCC score"


def test_plot_2d_scores_returns_figure_and_saves_file(tmp_path):
    output = tmp_path / "scores.png"
    normal = np.array([[0.8, 0.9], [0.85, 0.95]])
    anomalous = np.array([[0.1, 0.2]])

    fig = plot_2d_scores(normal, anomalous, output_path=output)

    assert output.exists()
    assert fig.axes[0].get_xlabel() == "NCC for feature map 18"
    assert fig.axes[0].get_ylabel() == "NCC for feature map 19"


def test_plot_2d_scores_rejects_non_two_column_scores():
    with pytest.raises(ValueError, match="exactly two columns"):
        plot_2d_scores(np.zeros((2, 3)), np.zeros((2, 2)))
