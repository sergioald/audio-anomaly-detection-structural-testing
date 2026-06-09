"""Small synthetic quickstart that does not require the Zenodo dataset or TensorFlow."""

from __future__ import annotations

import numpy as np

from audio_anomaly.classifiers import evaluate_classifiers, train_classifiers
from audio_anomaly.data import make_binary_dataset
from audio_anomaly.metrics import normalized_cross_correlation

rng = np.random.default_rng(42)
reference = np.ones((8, 8))
normal = reference + rng.normal(scale=0.05, size=(20, 8, 8))
anomalous = rng.normal(scale=1.0, size=(8, 8, 8))

normal_scores = np.array([[normalized_cross_correlation(sample, reference)] for sample in normal])
anomalous_scores = np.array([[normalized_cross_correlation(sample, reference)] for sample in anomalous])

x, y = make_binary_dataset(normal_scores, anomalous_scores)
classifiers = train_classifiers(x, y)
results = evaluate_classifiers(classifiers, x, y)

for name, metrics in results.items():
    print(f"{name:20s} accuracy={metrics['accuracy']:.3f}")
