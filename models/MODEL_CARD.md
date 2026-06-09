# Model card: pre-trained CAE for WST structural-audio features

## File

```text
models/pretrained_cae_wst_latent24_structural_audio.h5
```

## Intended use

This model is provided for the quick reproduction workflow using the public Zenodo WST feature arrays. It is used to extract hidden-layer feature maps before NCC scoring and classifier-based anomaly detection.

## Inputs and outputs

- Expected input shape: `(221, 375, 1)` WST feature map per sample.
- Default feature maps used downstream: 18 and 19.
- Default feature layer selection: the helper `find_feature_layer()` detects the suitable 64-filter Conv2D layer in both cleaned and legacy Keras models.

## Scope

The model is intended as a reproducibility and demonstration asset for the published workflow. For data from a different acoustic environment or facility, evaluate it as a baseline and consider retraining or recalibrating the pipeline.

## File size

Approximately 6.0 MB.
