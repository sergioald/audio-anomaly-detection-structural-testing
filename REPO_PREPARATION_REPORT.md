# Repository preparation report

Prepared repository: `audio-anomaly-detection-structural-testing`

## Prepared scope

The repository is structured as a public research-software companion for the audio-based anomaly-detection paper and the public Zenodo WST dataset.

It supports three workflows:

1. **Quick reproduction** using public Zenodo WST arrays and a pre-trained CAE.
2. **Full reproduction** using public Zenodo WST arrays and training the CAE from scratch.
3. **Advanced new-data workflow** using raw `.wav` files, WST feature generation, and either the pre-trained or retrained pipeline.

## What was cleaned

- Split research/autosave logic into reusable package modules.
- Replaced legacy internal dataset filenames with public Zenodo filenames.
- Added batched feature-map extraction and NCC scoring utilities.
- Added pre-trained model asset installation helper.
- Added raw-audio WST preparation scripts.
- Added prediction script for new audio/features.
- Added docs for model assets, new data, GitHub setup, release checklist and project-board cards.
- Added issue templates and pull-request template.
- Added tests that do not require TensorFlow or the large dataset.

## Main files

```text
src/audio_anomaly/audio.py
src/audio_anomaly/data.py
src/audio_anomaly/model.py
src/audio_anomaly/feature_maps.py
src/audio_anomaly/metrics.py
src/audio_anomaly/classifiers.py
src/audio_anomaly/evaluation.py
scripts/download_data.py
scripts/install_model_assets.py
scripts/train_cae.py
scripts/evaluate_feature_map_classifier.py
scripts/prepare_new_audio.py
scripts/prepare_raw_audio_dataset.py
scripts/predict_new_audio.py
```

## Tests run

```text
13 passed
```

## Not run here

The full dataset/model workflow was not run in this environment because:

- the public WST arrays are not included locally;
- the data are large;
- TensorFlow is not installed in the test environment.

Run the quick pre-trained workflow locally after extracting the model assets.

## Recommended publishing sequence

1. Create an empty GitHub repository.
2. Push this cleaned repository.
3. Upload `audio-model-assets.zip` as a release asset.
4. Add GitHub topics.
5. Run the quick workflow locally and paste metrics into README.
6. Pin the repository on the GitHub profile.
