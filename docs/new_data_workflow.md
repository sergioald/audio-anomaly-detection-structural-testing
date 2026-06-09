# New raw-audio workflow

The main reproduction workflow starts from the public Zenodo WST `.npy` arrays.
This advanced workflow is for users who have new raw `.wav` recordings and want
to compute WST feature maps before using the CAE/NCC classifier pipeline.

## Assumptions

The paper workflow used:

- single-channel audio recorded at 48 kHz;
- 0.5-second windows;
- 0.1-second overlap;
- `Scattering1D(J=6, Q=16)`;
- separate min-max normalisation for 1st- and 2nd-order WST coefficients;
- concatenated WST feature maps with shape `(221, 375)`.

New data from a different facility or acoustic environment may need retraining
or recalibration. The pre-trained model is most defensible for data collected in
a similar setup.

## Path A: prepare one folder for inference

```bash
python -m pip install -e ".[wst,deep-learning]"

python scripts/prepare_new_audio.py \
  --audio-dir new_audio \
  --output-features outputs/new_audio/features.npy \
  --output-windows outputs/new_audio/windows.csv
```

Then predict with previously trained pipeline assets:

```bash
python scripts/predict_new_audio.py \
  --features outputs/new_audio/features.npy \
  --model models/pretrained_cae_wst_latent24_structural_audio.h5 \
  --classifier outputs/evaluation_pretrained/best_classifier.joblib \
  --reference-maps outputs/evaluation_pretrained/reference_maps.npz \
  --output-dir outputs/new_audio_predictions
```

You can also pass `--audio-dir new_audio` directly to `predict_new_audio.py` and
skip the intermediate feature file.

## Path B: create a complete new labelled dataset

Organise new `.wav` files into five folders:

```text
new_raw_audio/
  normal_train/
  normal_validation/
  anomalous_validation/
  normal_test/
  anomalous_test/
```

Convert them to the five `.npy` files expected by the rest of the repository:

```bash
python scripts/prepare_raw_audio_dataset.py \
  --normal-train-dir new_raw_audio/normal_train \
  --normal-validation-dir new_raw_audio/normal_validation \
  --anomalous-validation-dir new_raw_audio/anomalous_validation \
  --normal-test-dir new_raw_audio/normal_test \
  --anomalous-test-dir new_raw_audio/anomalous_test \
  --output-dir data_new
```

Then run the normal training and evaluation commands using `data_new`:

```bash
python scripts/train_cae.py \
  --data-dir data_new \
  --output-model models/cae_wst_latent24_new_data.keras \
  --epochs 100

python scripts/evaluate_feature_map_classifier.py \
  --data-dir data_new \
  --model models/cae_wst_latent24_new_data.keras \
  --output-dir outputs/new_data_evaluation
```

## Interpreting predictions

`prediction = 0` means normal and `prediction = 1` means anomalous. For new
audio, review the output CSV alongside the original audio/video/test logs. This
workflow is a decision-support tool, not a replacement for engineering judgement.
