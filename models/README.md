# Models

This repository includes one publication-ready pre-trained CAE model for the quick reproduction workflow:

```text
models/pretrained_cae_wst_latent24_structural_audio.h5
```

The model is intended for the public WST dataset workflow described in the README. It uses the latent-24 CAE associated with the paper-style feature-map/NCC classifier pipeline.

Newly trained models and generated classifiers are ignored by Git by default. Use names such as:

```text
models/cae_wst_latent24_retrained.keras
outputs/evaluation_pretrained/best_classifier.joblib
```
