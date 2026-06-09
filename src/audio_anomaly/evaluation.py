"""End-to-end evaluation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .classifiers import evaluate_classifiers, select_best_classifier, train_classifiers
from .data import DatasetBundle, make_binary_dataset
from .feature_maps import average_reference_maps, score_dataset_against_references


@dataclass(frozen=True)
class FeatureMapEvaluationResult:
    reference_map_ids: tuple[int, ...]
    reference_maps: dict[int, np.ndarray]
    validation_scores: np.ndarray
    validation_labels: np.ndarray
    test_scores: np.ndarray
    test_labels: np.ndarray
    metrics: dict[str, Any]
    best_classifier_name: str


def evaluate_feature_map_pipeline(
    model,
    dataset: DatasetBundle,
    layer_name: str | None = None,
    map_ids: Sequence[int] = (18, 19),
    batch_size: int = 16,
    random_state: int = 42,
) -> FeatureMapEvaluationResult:
    """Run the final CAE-feature-map/NCC classifier pipeline."""
    reference_maps = average_reference_maps(
        model=model,
        normal_data=dataset.normal_train,
        layer_name=layer_name,
        map_ids=map_ids,
        batch_size=batch_size,
    )

    normal_validation_scores = score_dataset_against_references(
        model, dataset.normal_validation, reference_maps, layer_name=layer_name, batch_size=batch_size
    )
    anomalous_validation_scores = score_dataset_against_references(
        model, dataset.anomalous_validation, reference_maps, layer_name=layer_name, batch_size=batch_size
    )
    x_validation, y_validation = make_binary_dataset(normal_validation_scores, anomalous_validation_scores)

    normal_test_scores = score_dataset_against_references(
        model, dataset.normal_test, reference_maps, layer_name=layer_name, batch_size=batch_size
    )
    anomalous_test_scores = score_dataset_against_references(
        model, dataset.anomalous_test, reference_maps, layer_name=layer_name, batch_size=batch_size
    )
    x_test, y_test = make_binary_dataset(normal_test_scores, anomalous_test_scores)

    classifiers = train_classifiers(x_validation, y_validation, random_state=random_state)
    metrics = evaluate_classifiers(classifiers, x_test, y_test)
    best_name = select_best_classifier(metrics)

    return FeatureMapEvaluationResult(
        reference_map_ids=tuple(map_ids),
        reference_maps=reference_maps,
        validation_scores=x_validation,
        validation_labels=y_validation,
        test_scores=x_test,
        test_labels=y_test,
        metrics=metrics,
        best_classifier_name=best_name,
    )
