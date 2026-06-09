import numpy as np

from audio_anomaly.classifiers import evaluate_classifiers, select_best_classifier, train_classifiers


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
