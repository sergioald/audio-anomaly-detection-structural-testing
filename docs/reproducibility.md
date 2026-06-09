# Reproducibility

## Environment

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m pip install -e ".[deep-learning]"
```

TensorFlow installation can be platform-specific. Install the version that matches your CPU/GPU environment.

## Data

```bash
python scripts/download_data.py --output data/
python scripts/check_dataset.py --data-dir data/ --strict
```

## Quick evaluation with the included pre-trained CAE

```bash
python scripts/evaluate_feature_map_classifier.py \
  --data-dir data/ \
  --model models/pretrained_cae_wst_latent24_structural_audio.h5 \
  --map-ids 18 19 \
  --output-dir outputs/evaluation_pretrained
```

## Train the CAE from scratch

```bash
python scripts/train_cae.py \
  --data-dir data/ \
  --output-model models/cae_wst_latent24_retrained.keras \
  --epochs 100 \
  --batch-size 32
```

## Evaluate final classifier workflow

```bash
python scripts/evaluate_feature_map_classifier.py \
  --data-dir data/ \
  --model models/cae_wst_latent24_retrained.keras \
  --map-ids 18 19 \
  --output-dir outputs/evaluation
```

## Evaluate reconstruction baselines

```bash
python scripts/evaluate_reconstruction_baselines.py \
  --data-dir data/ \
  --model models/cae_wst_latent24_retrained.keras \
  --output-dir outputs/reconstruction_baselines
```

## Continuous integration

CI should run unit tests only. It should not download the full dataset or train the CAE.
