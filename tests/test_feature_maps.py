import numpy as np
import pytest

from audio_anomaly.feature_maps import (
    load_reference_maps,
    rank_feature_maps_by_overlap,
    save_reference_maps,
    score_precomputed_feature_maps,
)


def test_score_precomputed_feature_maps_returns_one_column_per_reference():
    features = np.zeros((2, 2, 2, 3), dtype=float)
    features[0, ..., 0] = np.array([[0, 1], [2, 3]])
    features[1, ..., 0] = -features[0, ..., 0]
    features[..., 2] = np.array([[1, 2], [3, 4]])

    references = {
        0: np.array([[0, 1], [2, 3]], dtype=float),
        2: np.array([[1, 2], [3, 4]], dtype=float),
    }

    scores = score_precomputed_feature_maps(features, references)

    assert scores.shape == (2, 2)
    assert np.isclose(scores[0, 0], 1.0)
    assert np.isclose(scores[1, 0], -1.0)


def test_score_precomputed_feature_maps_rejects_non_4d_features():
    with pytest.raises(ValueError, match="Expected feature tensor"):
        score_precomputed_feature_maps(np.zeros((2, 2, 2)), {0: np.zeros((2, 2))})


def test_score_precomputed_feature_maps_rejects_bad_map_id():
    features = np.zeros((2, 2, 2, 1), dtype=float)
    with pytest.raises(IndexError):
        score_precomputed_feature_maps(features, {2: np.zeros((2, 2))})


def test_rank_feature_maps_by_overlap_orders_more_separating_maps_first():
    reference = np.zeros((2, 2, 2), dtype=float)
    reference[..., 0] = np.array([[0, 1], [2, 3]])
    reference[..., 1] = np.array([[0, 1], [0, 1]])

    normal = np.zeros((4, 2, 2, 2), dtype=float)
    anomalous = np.zeros((4, 2, 2, 2), dtype=float)

    # Map 0 separates clearly: normal matches reference, anomalous is inverted.
    normal[..., 0] = reference[..., 0]
    anomalous[..., 0] = -reference[..., 0]

    # Map 1 overlaps: both classes match the same reference.
    normal[..., 1] = reference[..., 1]
    anomalous[..., 1] = reference[..., 1]

    ranked = rank_feature_maps_by_overlap(normal, anomalous, reference)

    assert ranked[0][0] == 0
    assert ranked[0][1] <= ranked[1][1]


def test_rank_feature_maps_by_overlap_rejects_unknown_metric():
    features = np.zeros((2, 2, 2, 1), dtype=float)
    reference = np.zeros((2, 2, 1), dtype=float)

    with pytest.raises(NotImplementedError, match="Only NCC"):
        rank_feature_maps_by_overlap(features, features, reference, metric="mae")


def test_rank_feature_maps_by_overlap_requires_same_number_of_maps():
    normal = np.zeros((2, 2, 2, 1), dtype=float)
    anomalous = np.zeros((2, 2, 2, 2), dtype=float)
    reference = np.zeros((2, 2, 1), dtype=float)

    with pytest.raises(ValueError, match="same number of maps"):
        rank_feature_maps_by_overlap(normal, anomalous, reference)


def test_reference_maps_round_trip(tmp_path):
    reference_maps = {
        19: np.full((2, 2), 19.0),
        18: np.full((2, 2), 18.0),
    }

    path = save_reference_maps(reference_maps, tmp_path / "nested" / "reference_maps.npz")
    loaded = load_reference_maps(path)

    assert path.exists()
    assert list(loaded) == [18, 19]
    assert np.array_equal(loaded[18], reference_maps[18])
    assert np.array_equal(loaded[19], reference_maps[19])


def test_load_reference_maps_rejects_npz_without_map_entries(tmp_path):
    path = tmp_path / "bad_reference_maps.npz"
    np.savez_compressed(path, not_a_map=np.zeros((2, 2)))

    with pytest.raises(ValueError, match="No reference maps"):
        load_reference_maps(path)
