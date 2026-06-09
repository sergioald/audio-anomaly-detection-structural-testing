"""Classifier training and evaluation utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def candidate_classifiers(random_state: int = 42) -> dict[str, Any]:
    """Return the candidate classifiers used in the paper workflow."""
    return {
        "knn_k3": KNeighborsClassifier(n_neighbors=3),
        "knn_k5": KNeighborsClassifier(n_neighbors=5),
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=random_state),
        "svm_linear": SVC(kernel="linear"),
        "svm_rbf": SVC(kernel="rbf"),
        "decision_tree": DecisionTreeClassifier(random_state=random_state),
    }


def train_classifiers(x_train: np.ndarray, y_train: np.ndarray, random_state: int = 42) -> dict[str, Any]:
    """Train all candidate classifiers."""
    trained = candidate_classifiers(random_state=random_state)
    for classifier in trained.values():
        classifier.fit(x_train, y_train)
    return trained


def evaluate_classifier(classifier: Any, x_test: np.ndarray, y_test: np.ndarray) -> dict[str, Any]:
    """Evaluate one classifier and return serialisable metrics."""
    y_pred = classifier.predict(x_test)
    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "classification_report": classification_report(y_test, y_pred, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }


def evaluate_classifiers(classifiers: dict[str, Any], x_test: np.ndarray, y_test: np.ndarray) -> dict[str, Any]:
    """Evaluate multiple classifiers."""
    return {name: evaluate_classifier(clf, x_test, y_test) for name, clf in classifiers.items()}


def select_best_classifier(results: dict[str, Any], positive_label: str = "1") -> str:
    """Select the best model, prioritising anomalous recall then accuracy."""
    def key(item):
        name, metrics = item
        report = metrics["classification_report"]
        anomaly_recall = report.get(positive_label, {}).get("recall", 0.0)
        return (anomaly_recall, metrics.get("accuracy", 0.0), name)

    return max(results.items(), key=key)[0]


def save_classifier(classifier: Any, path: str | Path) -> Path:
    """Save a fitted classifier with joblib."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(classifier, path)
    return path


def load_classifier(path: str | Path) -> Any:
    """Load a fitted joblib classifier."""
    return joblib.load(path)
