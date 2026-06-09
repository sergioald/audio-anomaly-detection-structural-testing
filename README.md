# Audio Anomaly Detection for Large-Scale Structural Testing

[![Tests](https://github.com/sergioald/audio-anomaly-detection-structural-testing/actions/workflows/tests.yml/badge.svg)](https://github.com/sergioald/audio-anomaly-detection-structural-testing/actions/workflows/tests.yml)

Reproducible Python workflow for **audio-based anomaly detection in large-scale structural testing** using Wavelet Scattering Transform (WST) features, convolutional autoencoder (CAE) feature maps, normalised cross-correlation (NCC), and lightweight classifiers.

This repository is prepared as a public research-software companion to:

> Munko, M. J., Cuthill, F., Valdivia Camacho, M. A., Ó Bradaigh, C. M., & Lopez Dubon, S. (2025). *An audio-based framework for anomaly detection in large-scale structural testing*. Engineering Applications of Artificial Intelligence, 142, 109889.

The public dataset is hosted externally on Zenodo:

> Munko, M., Lopez Dubon, S., & Cuthill, F. (2024). *Normal and anomalous audio data processed with the wavelet scattering transform, collected during the operation of FastBlade, a site for regenerative fatigue testing*. Zenodo. https://doi.org/10.5281/zenodo.14298279

## Why this project exists

Large-scale structural testing facilities can benefit from low-cost, non-specific sensing for anomaly detection. Microphones can capture system-wide changes without requiring detailed instrumentation for every asset.

This repository turns the original research scripts into a cleaner public workflow:

```text
WST feature arrays
→ convolutional autoencoder (CAE)
→ hidden-layer feature maps
→ averaged normal reference maps
→ normalised cross-correlation (NCC)
→ classifier-based anomaly detection
```

## Supported workflows

The repository supports three workflows.

| Workflow | Data input | Model option | Purpose |
|---|---|---|---|
| A. Quick reproduction | public Zenodo WST `.npy` files | pre-trained CAE | fastest reviewer path |
| B. Full reproduction | public Zenodo WST `.npy` files | train CAE from scratch | reproducibility path |
| C. New raw audio | new `.wav` files | pre-trained or retrained CAE | advanced extension path |

The Zenodo arrays are downloaded locally. The repository includes one small pre-trained CAE model at `models/pretrained_cae_wst_latent24_structural_audio.h5` so the quick reproduction workflow can be run without first training a neural network.

## Installation

Create an environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

For deep-learning workflows:

```bash
python -m pip install -e ".[deep-learning]"
```

For raw-audio-to-WST workflows:

```bash
python -m pip install -e ".[wst]"
```

On Windows, use a short path such as `C:\Test\dth` to avoid TensorFlow path-length issues.

## Workflow A: quick reproduction with pre-trained CAE

Download the public WST data:

```bash
python scripts/download_data.py --output data/
python scripts/check_dataset.py --data-dir data/ --strict
```

Run the feature-map/NCC classifier pipeline using the included pre-trained CAE:

```bash
python scripts/evaluate_feature_map_classifier.py \
  --data-dir data/ \
  --model models/pretrained_cae_wst_latent24_structural_audio.h5 \
  --output-dir outputs/evaluation_pretrained
```

The `--model` argument can be omitted because this is the default pre-trained model path.

This writes:

```text
outputs/evaluation_pretrained/classifier_metrics.json
outputs/evaluation_pretrained/reference_maps.npz
outputs/evaluation_pretrained/best_classifier.joblib
outputs/evaluation_pretrained/validation_scores.npy
outputs/evaluation_pretrained/test_scores.npy
```

## Workflow B: train the CAE from scratch

```bash
python scripts/download_data.py --output data/
python scripts/check_dataset.py --data-dir data/ --strict

python scripts/train_cae.py \
  --data-dir data/ \
  --output-model models/cae_wst_latent24_retrained.keras \
  --epochs 100

python scripts/evaluate_feature_map_classifier.py \
  --data-dir data/ \
  --model models/cae_wst_latent24_retrained.keras \
  --output-dir outputs/evaluation_retrained
```

## Workflow C: use new raw audio

For a folder of new `.wav` files, compute WST features:

```bash
python -m pip install -e ".[wst,deep-learning]"

python scripts/prepare_new_audio.py \
  --audio-dir new_audio/ \
  --output-features outputs/new_audio/features.npy \
  --output-windows outputs/new_audio/windows.csv
```

Predict with an existing trained pipeline:

```bash
python scripts/predict_new_audio.py \
  --features outputs/new_audio/features.npy \
  --model models/pretrained_cae_wst_latent24_structural_audio.h5 \
  --classifier outputs/evaluation_pretrained/best_classifier.joblib \
  --reference-maps outputs/evaluation_pretrained/reference_maps.npz \
  --output-dir outputs/new_audio_predictions
```

For a complete new labelled dataset, organise raw audio into:

```text
new_raw_audio/
  normal_train/
  normal_validation/
  anomalous_validation/
  normal_test/
  anomalous_test/
```

Then create paper-style `.npy` files:

```bash
python scripts/prepare_raw_audio_dataset.py \
  --normal-train-dir new_raw_audio/normal_train \
  --normal-validation-dir new_raw_audio/normal_validation \
  --anomalous-validation-dir new_raw_audio/anomalous_validation \
  --normal-test-dir new_raw_audio/normal_test \
  --anomalous-test-dir new_raw_audio/anomalous_test \
  --output-dir data_new
```

Then run Workflow B using `--data-dir data_new`.

See [`docs/new_data_workflow.md`](docs/new_data_workflow.md).

## Dataset files expected locally

The code expects these public Zenodo filenames:

| File | Role |
|---|---|
| `Normal_Data_Training.npy` | normal data used to train the CAE and compute reference feature maps |
| `Normal_Data_Validation.npy` | normal validation data used to train/tune classifiers |
| `Anomalous_Data_Validation.npy` | anomalous validation data used to train/tune classifiers |
| `Normal_Data_Test.npy` | normal held-out test data |
| `Anomalous_Data_Test.npy` | anomalous held-out test data |

Place them in `data/`, or run `scripts/download_data.py`.

## Current status

This repository includes:

- public dataset contract for the Zenodo `.npy` files;
- CAE architecture matching the paper appendix;
- included pre-trained CAE model: `models/pretrained_cae_wst_latent24_structural_audio.h5`;
- support for legacy `.h5` models and clean `.keras` models;
- batched feature-map extraction to avoid storing large intermediate arrays;
- NCC scoring for feature maps 18 and 19;
- candidate classifiers: KNN, logistic regression, SVM, decision tree;
- raw `.wav` to WST feature generation for advanced new-data workflows;
- CLI scripts for data checking, training, evaluation, prediction and benchmarking;
- unit tests using small synthetic arrays, so CI does not download the 2.8 GB dataset;
- documentation for reproducibility, dataset boundaries, model use and new-data workflows.

## Repository structure

```text
audio-anomaly-detection-structural-testing/
  README.md
  pyproject.toml
  CITATION.cff
  LICENSE
  configs/
    default.yaml
  src/audio_anomaly/
    audio.py
    data.py
    model.py
    feature_maps.py
    metrics.py
    classifiers.py
    evaluation.py
    plotting.py
  scripts/
    download_data.py
    check_dataset.py
    train_cae.py
    evaluate_feature_map_classifier.py
    evaluate_reconstruction_baselines.py
    prepare_new_audio.py
    prepare_raw_audio_dataset.py
    predict_new_audio.py
    benchmark_inference.py
    inspect_h5_model.py
  docs/
    dataset.md
    paper_summary.md
    method_notes.md
    reproducibility.md
    new_data_workflow.md
    confidentiality_statement.md
  tests/
    test_data_contract.py
    test_metrics.py
    test_feature_map_pipeline.py
    test_classifiers.py
    test_audio_windows.py
```

## Tests

```bash
pytest
```

The CI test suite does not download large data or require TensorFlow.

## Confidentiality and data boundary

This repository does not contain raw FastBlade audio, private facility data, confidential operational records or proprietary control logic. It is designed to work with the public processed WST dataset hosted on Zenodo and with user-provided local audio files.

See [`docs/confidentiality_statement.md`](docs/confidentiality_statement.md).

## License

MIT License for the code in this repository. See [`LICENSE`](LICENSE).

The Zenodo dataset has its own license and citation requirements. Cite the dataset and the paper when using this repository.
