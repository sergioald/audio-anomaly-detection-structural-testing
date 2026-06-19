import numpy as np

from audio_anomaly.classifiers import (
    candidate_classifiers,
    evaluate_classifier,
    evaluate_classifiers,
    load_classifier,
    save_classifier,
    select_best_classifier,
    train_classifiers,
)


def test_candidate_classifiers_contains_expected_models():
    classifiers = candidate_classifiers()

    assert set(classifiers) == {
        "knn_k3",
        "knn_k5",
        "logistic_regression",
        "svm_linear",
        "svm_rbf",
        "decision_tree",
    }


def test_train_and_evaluate_classifiers_on_simple_data():
    x_normal = np.zeros((10, 2))
    x_anom = np.ones((10, 2))
    x = np.vstack([x_normal, x_anom])
    y = np.array([0] * 10 + [1] * 10)

    classifiers = train_classifiers(x, y)
    results = evaluate_classifiers(classifiers, x, y)

    assert "knn_k5" in results
    assert results["knn_k5"]["accuracy"] == 1.0
    assert select_best_classifier(results) in results


def test_evaluate_classifier_returns_serialisable_structure():
    x = np.array([[0, 0], [0, 0.1], [1, 1], [1, 0.9]])
    y = np.array([0, 0, 1, 1])
    classifier = candidate_classifiers()["knn_k3"]
    classifier.fit(x, y)

    metrics = evaluate_classifier(classifier, x, y)

    assert isinstance(metrics["accuracy"], float)
    assert isinstance(metrics["classification_report"], dict)
    assert isinstance(metrics["confusion_matrix"], list)
    assert metrics["confusion_matrix"] == [[2, 0], [0, 2]]


def test_select_best_classifier_prioritises_anomalous_recall_then_accuracy_then_name():
    results = {
        "model_a": {
            "accuracy": 0.95,
            "classification_report": {"1": {"recall": 0.80}},
        },
        "model_b": {
            "accuracy": 0.90,
            "classification_report": {"1": {"recall": 0.90}},
        },
        "model_c": {
            "accuracy": 0.92,
            "classification_report": {"1": {"recall": 0.90}},
        },
    }

    assert select_best_classifier(results) == "model_c"


def test_save_and_load_classifier_round_trip(tmp_path):
    x = np.array([[0, 0], [0, 0.1], [1, 1], [1, 0.9]])
    y = np.array([0, 0, 1, 1])
    classifier = candidate_classifiers()["decision_tree"]
    classifier.fit(x, y)

    path = save_classifier(classifier, tmp_path / "models" / "classifier.joblib")
    loaded = load_classifier(path)

    assert path.exists()
    assert np.array_equal(loaded.predict(x), classifier.predict(x))
