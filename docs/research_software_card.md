# Research software card

## Name

Audio Anomaly Detection for Large-Scale Structural Testing

## Repository purpose

Provide a reproducible Python implementation of an audio-based anomaly-detection workflow for structural testing, using public WST feature arrays, a convolutional autoencoder, feature-map similarity scoring, and downstream classifiers.

## Intended users

- Researchers reviewing or reproducing the companion workflow.
- Engineers interested in acoustic monitoring for structural tests.
- Applied AI practitioners working with sensor, audio, or condition-monitoring data.
- Portfolio reviewers assessing research-software, ML, and engineering-data capability.

## Main inputs

- Public processed WST `.npy` arrays.
- Optional user-provided `.wav` files for local experimentation.
- Optional trained or retrained CAE model artefacts.

## Main outputs

- Dataset validation results.
- CAE model artefacts.
- Hidden feature-map reference maps.
- Normalised cross-correlation score arrays.
- Classifier metrics and saved classifier artefacts.
- Reproducibility plots and diagnostic figures.

## Core method

1. Load or prepare audio-derived features.
2. Pass features through a convolutional autoencoder.
3. Extract selected hidden feature maps.
4. Compare feature maps with normal-operation reference maps using normalised cross-correlation.
5. Train and evaluate lightweight classifiers on the resulting score space.

## Reuse boundary

The repository is suitable for reproducing the public workflow and for adapting the method to new local audio datasets. It is not a substitute for site-specific validation, instrumentation review, or safety-critical certification.

## Data boundary

The public repository uses public processed feature arrays and does not publish confidential raw facility audio, private structural-test logs, or proprietary control-system details.

## Validation boundary

The included tests check software behaviour, data contracts, metric utilities, scoring helpers, classifier workflows, and selected script behaviour. They do not prove that the method is universally valid for all structural-test facilities or all anomaly types.

## Maintenance notes

When extending the repository, prefer changes that preserve:

- clear separation between public and private data;
- script-level reproducibility;
- small synthetic tests for CI;
- documented assumptions and limitations;
- compatibility notes for TensorFlow/Keras model loading;
- lightweight examples that can run without the full dataset.
