# Paper summary

This repository accompanies the paper:

> Munko, M. J., Cuthill, F., Valdivia Camacho, M. A., Ó Bradaigh, C. M., & Lopez Dubon, S. (2025). *An audio-based framework for anomaly detection in large-scale structural testing*. Engineering Applications of Artificial Intelligence, 142, 109889. https://doi.org/10.1016/j.engappai.2024.109889

## Problem

FastBlade is a large-scale structural testing facility where uninterrupted or unmanned operation is desirable. Anomaly detection is required to identify system imbalance, third-party interactions, changed test parameters and data-logging faults.

## Signal and data

- Single-channel audio was recorded at 48 kHz.
- Samples are 0.5 seconds long, with 0.1 seconds overlap.
- WST was selected as the most suitable time-frequency feature representation.
- Public data are processed WST arrays rather than raw audio.

## Method

The final method is:

```text
WST arrays
→ CAE trained on normal operation data
→ hidden-layer feature maps
→ average normal feature maps
→ NCC score against selected maps
→ classifier-based anomaly detection
```

The reconstruction-error approach using CAE and PCA was tested but did not provide enough separation between normal and anomalous data because normal-operation samples had strong intraclass variability.

The successful final workflow uses feature maps 18 and 19 from the second convolutional layer, with normalised cross-correlation as the feature score.

## Classifiers

The candidate classifiers are:

- KNN, k = 3
- KNN, k = 5
- Logistic regression
- SVM with linear kernel
- SVM with RBF kernel
- Decision tree

The paper reports the strongest two-dimensional result with KNN, k = 5.

## Scope of this repository

This repository provides a clean implementation and reproducibility workflow. It does not commit the large WST arrays or raw facility audio.
