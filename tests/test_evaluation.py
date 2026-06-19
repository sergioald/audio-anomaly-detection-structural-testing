import numpy as np

from audio_anomaly.data import DatasetBundle
from audio_anomaly.evaluation import FeatureMapEvaluationResult, evaluate_feature_map_pipeline


def test_evaluate_feature_map_pipeline_with_monkeypatched_scoring(monkeypatch):
    dataset = DatasetBundle(
        normal_train=np.zeros((4, 2, 2)),
        normal_validation=np.zeros((6, 2, 2)),
        anomalous_validation=np.ones((6, 2, 2)),
        normal_test=np.zeros((6, 2, 2)),
        anomalous_test=np.ones((6, 2, 2)),
    )
    reference_maps = {0: np.zeros((2, 2)), 1: np.ones((2, 2))}

    def fake_average_reference_maps(model, normal_data, layer_name=None, map_ids=(18, 19), batch_size=16):
        assert model == "fake-model"
        assert normal_data.shape == (4, 2, 2)
        assert tuple(map_ids) == (0, 1)
        return reference_maps

    def fake_score_dataset_against_references(model, data, refs, layer_name=None, batch_size=16):
        assert refs is reference_maps
        if np.all(data == 0):
            return np.column_stack([np.ones(len(data)), np.zeros(len(data))])
        return np.column_stack([np.zeros(len(data)), np.ones(len(data))])

    monkeypatch.setattr("audio_anomaly.evaluation.average_reference_maps", fake_average_reference_maps)
    monkeypatch.setattr(
        "audio_anomaly.evaluation.score_dataset_against_references",
        fake_score_dataset_against_references,
    )

    result = evaluate_feature_map_pipeline(
        model="fake-model",
        dataset=dataset,
        layer_name="fake-layer",
        map_ids=(0, 1),
        batch_size=2,
    )

    assert isinstance(result, FeatureMapEvaluationResult)
    assert result.reference_map_ids == (0, 1)
    assert result.reference_maps is reference_maps
    assert result.validation_scores.shape == (12, 2)
    assert result.test_scores.shape == (12, 2)
    assert np.array_equal(result.validation_labels, np.array([0] * 6 + [1] * 6))
    assert np.array_equal(result.test_labels, np.array([0] * 6 + [1] * 6))
    assert result.best_classifier_name in result.metrics
    assert result.metrics[result.best_classifier_name]["accuracy"] == 1.0
