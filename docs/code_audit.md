# Code audit of the original ZIP

The original `All_code.zip` contains useful research code, trained models and historical exploratory scripts. It was not directly ready for GitHub because it mixed production logic, paper-figure generation, local paths and intermediate outputs.

## Files used as source material

| Original file | Clean destination / role |
|---|---|
| `old_autosave/CAE_Aug_24_LS24.py` | `src/audio_anomaly/model.py`, `scripts/train_cae.py` |
| `old_autosave/CAE_Aug_24_LS6.py` | archived alternative latent-space experiment |
| `old_autosave/examine_why_fails.py` | `feature_maps.py`, `metrics.py`, `classifiers.py`, `evaluate_feature_map_classifier.py` |
| `old_autosave/Rescue_PCA_and_CAE.py` | `evaluate_reconstruction_baselines.py` |
| `old_autosave/Paper_revisions_Aug_24.py` | documentation and future figure-reproduction script |
| `old_autosave/benchmark_complete_processing.py` | `benchmark_inference.py` |
| `Models/my_paper_ae_24.h5` | optional local/release model asset, not committed by default |
| `Models/my_paper_ae_6.h5` | optional archive/release model asset, not committed by default |

## Issues fixed in the cleaned structure

- Replaced local Windows paths with CLI arguments.
- Replaced internal dataset filenames with public Zenodo filenames.
- Split long scripts into reusable modules.
- Added stable layer names for the cleaned CAE architecture.
- Added automatic support for legacy Keras layer names.
- Avoided storing large intermediate `.npy` feature tensors.
- Added unit tests using synthetic arrays.
- Added docs for dataset, method, reproducibility and confidentiality.

## Items intentionally not included

The cleaned repository does not include raw old autosave scripts by default. They should be kept outside the public repo or added only under a clearly marked `archive/` folder after manual review.
