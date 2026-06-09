import numpy as np

from audio_anomaly.feature_maps import score_precomputed_feature_maps


def test_score_precomputed_feature_maps_shape():
    rng = np.random.default_rng(1)
    features = rng.normal(size=(5, 4, 4, 3))
    reference_maps = {
        0: features[..., 0].mean(axis=0),
        2: features[..., 2].mean(axis=0),
    }
    scores = score_precomputed_feature_maps(features, reference_maps)
    assert scores.shape == (5, 2)
