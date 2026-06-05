# Method notes

## Why WST data are used directly

The public dataset contains WST-processed arrays. This keeps the repository focused on the anomaly-detection method rather than raw audio processing. It also avoids the need to distribute raw facility audio.

## Why reconstruction-error baselines are included

The original workflow attempted standard reconstruction-error anomaly detection using CAE and PCA. This did not separate normal and anomalous samples sufficiently, largely because normal samples varied strongly within the same class.

The script `scripts/evaluate_reconstruction_baselines.py` is kept for reproducibility of that negative result.

## Why feature maps 18 and 19 are defaults

Feature-map IDs 18 and 19 produced the smallest normal/anomalous histogram overlap when scored with NCC. They are therefore the default `map_ids` in `configs/default.yaml` and in the evaluation scripts.

## Layer naming

The cleaned CAE architecture names the second convolutional layer `encoder_conv2`.

The historical `.h5` model used Keras-generated names. For the latent-24 model provided in the original ZIP, the equivalent layer is `conv2d_31`. The helper `find_feature_layer()` can automatically select the first 64-filter Conv2D layer when a preferred cleaned name is not present.

## Memory handling

The selected hidden layer can be large. The cleaned implementation uses batching for:

- reference-map averaging;
- validation/test NCC scoring;
- benchmark inference.

This avoids saving large intermediate feature tensors such as `Rescue_features_*.npy`.
