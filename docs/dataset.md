# Dataset

The dataset is hosted on Zenodo and is not stored in this repository.

**Dataset DOI:** `10.5281/zenodo.14298279`

## Public files expected by this repository

| Key | Public filename | Expected shape | Role |
|---|---|---:|---|
| `normal_train` | `Normal_Data_Training.npy` | `(1347, 221, 375)` | normal WST data for CAE training and reference-map averaging |
| `normal_validation` | `Normal_Data_Validation.npy` | `(1347, 221, 375)` | normal validation data for classifier training/tuning |
| `anomalous_validation` | `Anomalous_Data_Validation.npy` | `(80, 221, 375)` | anomalous validation data for classifier training/tuning |
| `normal_test` | `Normal_Data_Test.npy` | `(1347, 221, 375)` | held-out normal test data |
| `anomalous_test` | `Anomalous_Data_Test.npy` | `(79, 221, 375)` | held-out anomalous test data |

## Legacy filename mapping

The original research scripts used internal filenames. They are mapped to public Zenodo filenames in `src/audio_anomaly/data.py`.

| Legacy/internal filename | Public filename |
|---|---|
| `REVISIONS_combined_training_DS_scat.npy` | `Normal_Data_Training.npy` |
| `REVISIONS_combined_val_DS.npy` | `Normal_Data_Validation.npy` |
| `REVISIONS_anomalies_validation.npy` | `Anomalous_Data_Validation.npy` |
| `REVISIONS_combined_test_DS.npy` | `Normal_Data_Test.npy` |
| `REVISIONS_anomalies_test.npy` | `Anomalous_Data_Test.npy` |

## Split logic

The original dataset explanation records the split as:

- Training: normal data from the first three minutes of `normal_op`, `n1`, and `n12`.
- Validation normal: first three minutes of `normal_new`, `n2`, and `n10`.
- Validation anomalous: even-ID samples from the 159 anomalous samples.
- Test normal: between six and nine minutes of `normal_new`, plus the first three minutes of `n5` and `n8`.
- Test anomalous: odd-ID samples from the 159 anomalous samples.

## Why data are excluded from Git

The processed WST arrays are large. Keeping them on Zenodo provides a stable DOI, clearer citation path and avoids bloating the Git repository.
