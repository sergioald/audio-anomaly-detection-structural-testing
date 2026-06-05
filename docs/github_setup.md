# GitHub setup guide

Use this guide when publishing the cleaned repository.

## 1. Repository name and description

Suggested repository name:

```text
audio-anomaly-detection-structural-testing
```

Suggested GitHub description:

```text
Research-software workflow for audio-based anomaly detection in large-scale structural testing using WST features, CAE feature maps and NCC classifiers.
```

Suggested topics:

```text
anomaly-detection, audio, wavelet-scattering-transform, convolutional-autoencoder,
structural-testing, fastblade, research-software, scientific-machine-learning,
python, condition-monitoring
```

## 2. First local commit

From the repository root:

```bash
git init
git add .
git commit -m "Initial public research-software release"
git branch -M main
```

Then create an empty GitHub repository and connect it:

```bash
git remote add origin https://github.com/sergioald/audio-anomaly-detection-structural-testing.git
git push -u origin main
```

## 3. Add model assets as a release

Do not commit `.h5` files. Create a GitHub release and upload:

```text
audio-model-assets.zip
```

Suggested tag:

```text
v0.1.0-model-assets
```

Suggested release note:

```text
Pre-trained CAE model assets for quick evaluation of the public WST dataset workflow.
The repository also supports training the CAE from scratch from the public Zenodo arrays.
```

## 4. Pin the repository

After publishing, pin this repository on the profile next to:

- `synthetic-hydraulic-digital-twin-demo`
- `LDSFL_Meander`
- `tdms-sync-checker`

## 5. Optional GitHub Project board

Create a GitHub Project named:

```text
Audio Anomaly Detection Portfolio Repo
```

Suggested columns/statuses:

- Backlog
- Ready
- In progress
- Done

Suggested first cards:

- Verify pre-trained model path on Windows
- Run full Zenodo evaluation locally
- Add example output plots to README
- Add release asset link for model ZIP
- Add raw-audio workflow validation using a short public/demo WAV
- Add benchmark table for CPU vs GPU inference
