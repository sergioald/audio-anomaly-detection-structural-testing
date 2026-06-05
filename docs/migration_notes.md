# Migration notes

## Step 1: Create the public repository

Suggested name:

```text
audio-anomaly-detection-structural-testing
```

Suggested GitHub description:

```text
Reproducible audio-based anomaly-detection workflow for large-scale structural testing using WST features and CAE feature maps.
```

Suggested topics:

```text
anomaly-detection, audio, wavelet-scattering-transform, convolutional-autoencoder, structural-testing, fastblade, research-software, python, scientific-machine-learning
```

## Step 2: Add files from this cleaned draft

Do not add the original `All_code.zip` directly.

## Step 3: Add data locally only

Download the Zenodo `.npy` files into `data/`.

## Step 4: Decide how to publish trained models

Recommended choices:

- do not commit `.h5`/`.keras` models;
- upload the final model as a GitHub Release asset;
- document model provenance and training command.

## Step 5: Pin on profile

Once polished, pin this repository alongside:

1. `synthetic-hydraulic-digital-twin-demo`
2. `LDSFL_Meander`
3. this repository
