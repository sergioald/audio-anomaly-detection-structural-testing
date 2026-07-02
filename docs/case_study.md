# Case study: audio-based anomaly detection for structural testing

## Context

Large-scale structural testing produces rich operational data, but conventional monitoring often depends on predefined sensors placed at known locations. Acoustic monitoring provides a complementary route: microphones can observe changes in system behaviour without being attached to every physical component.

This repository turns a published audio-anomaly-detection workflow into a reusable Python research-software project. The aim is not only to show a model result, but to make the workflow inspectable, reproducible, and suitable for adaptation to new local data.

## Challenge

The practical challenge is to distinguish normal and anomalous structural-test behaviour from audio-derived information while keeping the public repository safe to share. The repository therefore works with public processed Wavelet Scattering Transform feature arrays and avoids exposing confidential raw facility data or proprietary operational details.

Key constraints:

- the public workflow must be reproducible without private data;
- the implementation must separate data handling, model code, scoring, evaluation, and plotting;
- the method must be understandable to both engineering and machine-learning reviewers;
- the repository must make clear what the tests validate and what remains a scientific/operational assumption.

## Approach

The workflow uses precomputed WST feature arrays as the public input. These features are passed through a convolutional autoencoder. Instead of relying only on reconstruction error, selected hidden feature maps are compared with normal-operation reference maps using normalised cross-correlation. The resulting scores define a compact feature space for downstream classifiers.

The public workflow is organised around three use cases:

1. **Quick reproduction** using the public Zenodo arrays and the included pre-trained model.
2. **Full reproduction** by retraining the CAE from the public processed data.
3. **New local audio** by converting user-provided `.wav` files into feature arrays and applying the trained workflow.

## Repository design

The project is structured as research software rather than as a single notebook. It separates:

- reusable package code under `src/audio_anomaly/`;
- command-line scripts under `scripts/`;
- documentation under `docs/`;
- small examples under `examples/`;
- tests under `tests/`;
- model artefacts under `models/`;
- visual README assets under `docs/assets/`.

This structure makes the project easier to review, test, extend, and explain in a portfolio context.

## Outputs

Typical outputs include:

- dataset-contract checks;
- trained or loaded CAE model artefacts;
- feature-map reference maps;
- validation and test NCC scores;
- trained downstream classifier artefacts;
- classifier metrics;
- plots showing score distributions, score spaces, confusion matrices, and classifier comparison.

## What this demonstrates

This project demonstrates the ability to connect domain knowledge, signal processing, deep-learning feature extraction, validation, documentation, and safe data publication. It is especially relevant to:

- structural health monitoring;
- digital twins and condition monitoring;
- industrial anomaly detection;
- scientific machine learning;
- research software engineering;
- sensor and acoustic data QA/QC;
- reproducible engineering AI workflows.

## Limitations and responsible use

The repository should be treated as a reproducible research workflow, not as a certified safety system. Performance on the public processed dataset does not guarantee performance on a different facility, microphone position, material system, background noise condition, or operating regime. New applications require local validation, calibration, and domain review.

## Portfolio message

This repository is a strong portfolio project because it shows applied AI beyond model fitting. It demonstrates how to package a research method into a maintainable public software artefact while respecting confidentiality and explaining the assumptions, reproduction path, and validation boundary.
