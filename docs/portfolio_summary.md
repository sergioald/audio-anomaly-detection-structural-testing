# Portfolio summary

This document explains the repository from a portfolio and research-software perspective. It is written for reviewers, collaborators, hiring committees, and visitors who want to understand what the project demonstrates without reading every implementation detail.

## Project in one paragraph

This repository implements an audio-based anomaly-detection workflow for large-scale structural testing. It starts from public Wavelet Scattering Transform (WST) feature arrays derived from FastBlade operational audio, passes them through a convolutional autoencoder (CAE), compares selected hidden feature maps against normal-operation reference maps using normalised cross-correlation (NCC), and evaluates lightweight classifiers for normal/anomalous separation. The repository is structured as reusable Python research software rather than a one-off paper script.

## Why it is useful as a portfolio project

The project demonstrates applied AI for engineering monitoring rather than a generic machine-learning example. It connects structural testing, signal processing, deep-learning feature extraction, similarity scoring, classifier evaluation, reproducibility, and public/private data boundaries.

It is particularly relevant to work involving:

- digital twins and condition monitoring;
- structural health monitoring;
- sensor-data workflows;
- scientific machine learning;
- anomaly detection under constrained instrumentation;
- reusable research-software packaging;
- public reproduction of paper workflows without exposing confidential facility data.

## What the repository demonstrates technically

### 1. Research workflow translation

The original research workflow is converted into a public Python package with scripts, documentation, tests, and citation metadata. The public repository separates reusable implementation from confidential raw facility data.

### 2. Signal-to-feature processing

The primary public workflow uses precomputed WST arrays from Zenodo. Optional scripts support processing new local `.wav` files into WST-style feature maps when the user installs the optional audio dependencies.

### 3. Deep feature-map comparison

The CAE is used not only as a reconstruction model but also as a feature-map extractor. Selected hidden feature maps are compared with normal-operation reference maps through NCC. This creates a compact score space for downstream anomaly classification.

### 4. Lightweight classifier evaluation

The project includes utilities for training and evaluating several simple classifiers, including k-nearest neighbours, logistic regression, support vector machines, and decision trees. The best model is selected by prioritising anomalous recall and then accuracy.

### 5. Testing and quality control

The repository includes lightweight automated tests for core utilities, metrics, feature-map scoring, classifiers, audio helpers, plotting, package imports, selected scripts, and a synthetic integration path. CI runs these tests without downloading the full dataset or requiring TensorFlow in the core path. `ruff` and `pre-commit` provide automated code-quality checks.

### 6. Scientific boundary setting

The repository explicitly documents what the automated tests prove and what they do not prove. The tests verify software behaviour, not universal scientific validity. Full scientific reproduction requires downloading the Zenodo data and running the public-data evaluation workflow.

## What a reviewer can check quickly

A reviewer can inspect the repository at three levels:

| Level | What to check | Approximate effort |
|---|---|---:|
| Software hygiene | README, `pyproject.toml`, tests, CI, pre-commit, license, citation metadata | 5–10 min |
| Lightweight verification | `pre-commit run --all-files` and `pytest` | minutes |
| Public-data reproduction | Download Zenodo WST arrays and run the pre-trained evaluation script | longer; dataset/model dependent |

The fastest local checks are:

```bash
python -m pip install -e ".[dev]"
pre-commit run --all-files
pytest
```

The public-data reproduction path is:

```bash
python -m pip install -e ".[dev,deep-learning]"
python scripts/download_data.py --output data
python scripts/check_dataset.py --data-dir data --strict
python scripts/evaluate_feature_map_classifier.py --data-dir data --output-dir outputs/evaluation_pretrained
```

## Skills represented

This repository is intended to show evidence of:

- Python package organisation using a `src/` layout;
- scientific data validation and dataset-contract checks;
- numerical metric implementation and testing;
- deep-learning model loading and feature extraction;
- signal-processing workflow design;
- classifier benchmarking and model selection;
- script-based reproducibility;
- CI and pre-commit quality control;
- clear documentation for research reuse;
- responsible handling of confidential experimental data.

## Limitations and responsible interpretation

This repository should not be read as a claim that audio monitoring can replace dedicated structural instrumentation. It demonstrates a reproducible implementation of a specific research workflow using a public processed dataset.

The automated tests do not prove that:

- the CAE architecture is optimal;
- the workflow generalises to every structural facility;
- anomalies correspond to a unique physical failure mechanism;
- WST extraction from every raw-audio source will match the original experimental setup;
- the model is suitable for safety-critical automated decisions without further validation.

For details, see [`scientific_validation.md`](scientific_validation.md).

## Suggested GitHub topics

Useful repository topics include:

```text
audio
anomaly-detection
structural-health-monitoring
structural-testing
digital-twin
scientific-machine-learning
wavelet-scattering-transform
convolutional-autoencoder
sensor-data
research-software
python
fastblade
```

## Suggested short description

> Research-software workflow for audio-based anomaly detection in large-scale structural testing using WST features, CAE feature maps, NCC similarity, and lightweight classifiers.
