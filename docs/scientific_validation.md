# Scientific validation and testing scope

This document explains what the automated tests in this repository check, and what they do not prove scientifically.

The repository is a research-software implementation of an audio-based anomaly-detection workflow for large-scale structural testing. The core public workflow uses precomputed Wavelet Scattering Transform (WST) feature arrays, a convolutional autoencoder (CAE), feature-map similarity scores based on normalised cross-correlation (NCC), and lightweight classifiers.

The automated test suite is intentionally lightweight. It is designed to run quickly in continuous integration without downloading the full Zenodo dataset, training a neural network, or requiring TensorFlow/Keras, SciPy, or Kymatio in the base CI path.

## What the automated tests check

### Dataset contract utilities

The tests check that the expected public dataset filenames are mapped correctly, that missing files are reported, that small local `.npy` arrays can be loaded into a `DatasetBundle`, that channel axes are handled consistently, that shape validation distinguishes sample-count checks from per-sample shape checks, and that binary normal/anomalous labels are built correctly.

These tests reduce the risk of breaking the public data contract used by the reproduction scripts.

### Numerical metrics

The tests check basic and edge-case behaviour for the numerical metrics used by the workflow, including normalised cross-correlation, batched normalised cross-correlation, mean absolute error, mean squared error, peak signal-to-noise ratio, and histogram overlap.

These tests check implementation behaviour on controlled arrays. They do not prove that any metric is the best scientific choice for all structural-test anomaly-detection problems.

### Feature-map scoring helpers

The tests check TensorFlow-free feature-map utilities, including scoring precomputed feature tensors against reference maps, ranking feature maps by normal/anomalous histogram overlap, and saving/loading reference maps.

These tests protect the NCC feature-map scoring logic without requiring a trained CAE in CI.

### Classifier helpers

The tests check that candidate classifiers are exposed, that classifiers can train and evaluate on simple separable data, that evaluation outputs have serialisable structures, that best-classifier selection prioritises anomalous recall before accuracy, and that saved classifiers can be loaded again.

These tests verify software behaviour. They do not prove classifier generalisation on new facilities, sensors, operating regimes, or anomaly types.

### Raw-audio helper utilities

The tests check lightweight raw-audio helper behaviour such as PCM normalisation, window segmentation, min-max normalisation, file discovery, and CSV metadata writing.

The automated tests do not compute WST features from real audio in CI. That path depends on optional scientific/audio dependencies and is better treated as a manual or optional integration workflow.

### End-to-end synthetic pipeline

The tests include a synthetic integration test of the feature-map evaluation pipeline using monkeypatched feature-map scoring. This verifies that the orchestration of reference maps, validation scores, test scores, labels, classifier training, metrics, and best-model selection remains consistent.

This is an integration test of the code path, not a scientific reproduction of the paper results.

### Plotting and script smoke tests

The tests check that plotting helpers return figures, save output files, and reject invalid score dimensions. Script smoke tests cover selected lightweight script behaviours without running full model inference or downloading data.

## What the automated tests do not prove

The automated tests do not prove that:

- the CAE architecture is scientifically optimal;
- the included pre-trained model reproduces paper metrics on every TensorFlow/Keras version;
- the Zenodo dataset has been downloaded correctly in a given user environment;
- full WST extraction from raw audio matches the original experimental processing in every detail;
- the trained classifier will generalise to every structural testing facility or operational condition;
- detected anomalies correspond to a specific physical failure mechanism;
- microphone-based monitoring can replace dedicated structural instrumentation;
- the workflow is validated for safety-critical automated decision-making without further domain review.

## Manual validation recommended for scientific use

For scientific reproduction or serious reuse, run the automated test suite first:

```bash
pytest
```

Then run the public-data reproduction path:

```bash
python scripts/download_data.py --output data
python scripts/check_dataset.py --data-dir data --strict
python scripts/evaluate_feature_map_classifier.py \
  --data-dir data \
  --model models/pretrained_cae_wst_latent24_structural_audio.h5 \
  --output-dir outputs/evaluation_pretrained
```

The output metrics should be compared with the README, the companion paper, and any saved run artefacts. Differences can arise from TensorFlow/Keras model-loading behaviour, hardware, dependency versions, random seeds, and classifier implementation details.

For new raw audio, additional validation is required. Users should verify sampling rate, window length, overlap, WST parameters, feature-map shape, normal-reference data, classifier calibration, and the physical meaning of any detected anomaly.

## CI policy

The continuous-integration path should remain lightweight and deterministic. It should run linting/pre-commit checks and tests on small synthetic arrays. It should not download the full public dataset, train the CAE, or require GPU resources.

Heavy reproduction and raw-audio workflows should be documented and runnable manually, but they should not block routine development commits.
