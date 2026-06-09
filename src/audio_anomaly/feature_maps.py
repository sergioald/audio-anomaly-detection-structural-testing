"""Feature-map extraction and NCC scoring utilities."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from .data import ensure_channel_axis
from .metrics import normalized_cross_correlation_batch
from .model import _import_keras, find_feature_layer


def make_feature_extractor(model, layer_name: str | None = None):
    """Create a Keras model returning the selected feature layer output."""
    keras = _import_keras()
    selected = layer_name or find_feature_layer(model)
    return keras.Model(inputs=model.input, outputs=model.get_layer(selected).output)


def _batched(array: np.ndarray, batch_size: int):
    for start in range(0, len(array), batch_size):
        yield array[start : start + batch_size]


def average_reference_maps(
    model,
    normal_data: np.ndarray,
    layer_name: str | None = None,
    map_ids: Sequence[int] = (18, 19),
    batch_size: int = 16,
) -> dict[int, np.ndarray]:
    """Compute averaged normal feature maps for selected feature-map IDs.

    This is batched to avoid materialising the full hidden-layer tensor for the
    complete training dataset.
    """
    extractor = make_feature_extractor(model, layer_name)
    map_ids = tuple(int(i) for i in map_ids)
    sums: dict[int, np.ndarray] = {}
    count = 0

    for batch in _batched(ensure_channel_axis(normal_data), batch_size):
        features = extractor.predict(batch, verbose=0)
        if features.ndim != 4:
            raise ValueError(f"Expected feature tensor with shape (n, h, w, c), got {features.shape}")
        for map_id in map_ids:
            if map_id < 0 or map_id >= features.shape[-1]:
                raise ValueError(f"Map ID {map_id} out of range for feature tensor with {features.shape[-1]} maps")
            current = np.sum(features[..., map_id], axis=0)
            sums[map_id] = current if map_id not in sums else sums[map_id] + current
        count += features.shape[0]

    if count == 0:
        raise ValueError("Cannot compute reference maps from an empty dataset")
    return {map_id: sums[map_id] / count for map_id in map_ids}


def score_dataset_against_references(
    model,
    data: np.ndarray,
    reference_maps: dict[int, np.ndarray],
    layer_name: str | None = None,
    batch_size: int = 16,
) -> np.ndarray:
    """Compute NCC score matrix for data against selected reference maps.

    Returns an array with shape `(n_samples, n_reference_maps)`.
    """
    extractor = make_feature_extractor(model, layer_name)
    map_ids = tuple(reference_maps.keys())
    score_batches: list[np.ndarray] = []

    for batch in _batched(ensure_channel_axis(data), batch_size):
        features = extractor.predict(batch, verbose=0)
        scores_for_maps = []
        for map_id in map_ids:
            scores = normalized_cross_correlation_batch(features[..., map_id], reference_maps[map_id])
            scores_for_maps.append(scores)
        score_batches.append(np.column_stack(scores_for_maps))

    if not score_batches:
        return np.empty((0, len(map_ids)))
    return np.vstack(score_batches)


def score_precomputed_feature_maps(
    features: np.ndarray,
    reference_maps: dict[int, np.ndarray],
) -> np.ndarray:
    """Compute NCC score matrix from an already extracted feature tensor."""
    if features.ndim != 4:
        raise ValueError(f"Expected feature tensor with shape (n, h, w, c), got {features.shape}")
    scores_for_maps = []
    for map_id, reference in reference_maps.items():
        scores_for_maps.append(normalized_cross_correlation_batch(features[..., map_id], reference))
    return np.column_stack(scores_for_maps)


def rank_feature_maps_by_overlap(
    normal_features: np.ndarray,
    anomalous_features: np.ndarray,
    reference_features: np.ndarray,
    metric: str = "ncc",
) -> list[tuple[int, float]]:
    """Rank feature maps by normal/anomalous histogram overlap.

    This helper is for exploratory work and mirrors the original paper scripts.
    Currently only NCC is implemented because it was the successful metric in the
    final workflow.
    """
    from .metrics import histogram_overlap

    if metric != "ncc":
        raise NotImplementedError("Only NCC ranking is implemented in the cleaned workflow")
    if normal_features.shape[-1] != anomalous_features.shape[-1]:
        raise ValueError("Normal and anomalous feature tensors must have the same number of maps")

    ranked: list[tuple[int, float]] = []
    for map_id in range(normal_features.shape[-1]):
        reference = reference_features[..., map_id]
        normal_scores = normalized_cross_correlation_batch(normal_features[..., map_id], reference)
        anomalous_scores = normalized_cross_correlation_batch(anomalous_features[..., map_id], reference)
        ranked.append((map_id, histogram_overlap(normal_scores, anomalous_scores)))
    return sorted(ranked, key=lambda item: item[1])


def save_reference_maps(reference_maps: dict[int, np.ndarray], path: str | Path) -> Path:
    """Save reference maps as a compressed `.npz` file."""
    from pathlib import Path

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **{f"map_{int(map_id)}": value for map_id, value in reference_maps.items()})
    return output


def load_reference_maps(path: str | Path) -> dict[int, np.ndarray]:
    """Load reference maps saved by `save_reference_maps`."""
    from pathlib import Path

    loaded = np.load(Path(path))
    reference_maps: dict[int, np.ndarray] = {}
    for key in loaded.files:
        if not key.startswith("map_"):
            continue
        reference_maps[int(key.split("_", 1)[1])] = loaded[key]
    if not reference_maps:
        raise ValueError(f"No reference maps found in {path}")
    return dict(sorted(reference_maps.items()))
