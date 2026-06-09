"""Raw-audio preparation utilities for advanced workflows.

The primary public workflow uses pre-computed WST `.npy` arrays from Zenodo.
This module supports the optional advanced path where users start from new raw
`.wav` recordings and compute Wavelet Scattering Transform (WST) feature arrays
with the same high-level processing assumptions used in the paper workflow.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


SUPPORTED_AUDIO_SUFFIXES = {".wav"}


@dataclass(frozen=True)
class AudioWindowRecord:
    """Metadata for one audio window converted into a WST feature map."""

    source_file: str
    window_index: int
    start_seconds: float
    end_seconds: float


@dataclass(frozen=True)
class WSTBatch:
    """Feature maps and metadata computed from a folder of audio files."""

    features: np.ndarray
    records: tuple[AudioWindowRecord, ...]


def read_wav_mono(path: str | Path) -> tuple[int, np.ndarray]:
    """Read a `.wav` file and return `(sample_rate, mono_float_signal)`.

    Integer PCM data are scaled to approximately `[-1, 1]`. Multi-channel audio
    is averaged to mono. The function imports SciPy lazily so the base package can
    be installed without audio/WST dependencies.
    """
    try:
        from scipy.io import wavfile
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError("Install audio/WST dependencies with `pip install -e .[wst]`.") from exc

    sample_rate, signal = wavfile.read(Path(path))
    signal = np.asarray(signal)
    if signal.ndim == 2:
        signal = signal.mean(axis=1)
    return int(sample_rate), normalise_audio_signal(signal)


def normalise_audio_signal(signal: np.ndarray) -> np.ndarray:
    """Convert common PCM/float audio arrays to `float32` in a stable range."""
    arr = np.asarray(signal)
    if np.issubdtype(arr.dtype, np.integer):
        info = np.iinfo(arr.dtype)
        scale = max(abs(info.min), abs(info.max))
        return (arr.astype(np.float32) / float(scale)).astype(np.float32)
    return arr.astype(np.float32)


def resample_if_needed(signal: np.ndarray, sample_rate: int, target_sample_rate: int) -> np.ndarray:
    """Resample audio if the input rate differs from the target rate."""
    if sample_rate == target_sample_rate:
        return np.asarray(signal, dtype=np.float32)
    try:
        from scipy.signal import resample_poly
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError("Install audio/WST dependencies with `pip install -e .[wst]`.") from exc

    from math import gcd

    divisor = gcd(sample_rate, target_sample_rate)
    up = target_sample_rate // divisor
    down = sample_rate // divisor
    return resample_poly(signal, up=up, down=down).astype(np.float32)


def segment_audio(
    signal: np.ndarray,
    sample_rate: int = 48_000,
    sample_seconds: float = 0.5,
    overlap_seconds: float = 0.1,
) -> np.ndarray:
    """Split an audio signal into overlapping fixed-length windows.

    The paper workflow used 0.5-second windows with 0.1-second overlap. Only full
    windows are returned; partial trailing windows are ignored.
    """
    arr = np.asarray(signal, dtype=np.float32).ravel()
    window_length = int(round(sample_seconds * sample_rate))
    overlap = int(round(overlap_seconds * sample_rate))
    hop = window_length - overlap
    if window_length <= 0:
        raise ValueError("sample_seconds must produce a positive window length")
    if hop <= 0:
        raise ValueError("overlap_seconds must be smaller than sample_seconds")
    if len(arr) < window_length:
        return np.empty((0, window_length), dtype=np.float32)

    starts = range(0, len(arr) - window_length + 1, hop)
    return np.stack([arr[start : start + window_length] for start in starts]).astype(np.float32)


def _minmax_normalise(array: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    arr = np.asarray(array, dtype=np.float32)
    lower = float(np.min(arr))
    upper = float(np.max(arr))
    if upper - lower < eps:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - lower) / (upper - lower)).astype(np.float32)


def compute_wst_feature(
    window: np.ndarray,
    j: int = 6,
    q: int = 16,
    shape_check: tuple[int, int] | None = (221, 375),
) -> np.ndarray:
    """Compute the concatenated 1st/2nd-order WST feature map for one window.

    This follows the legacy paper scripts: compute `Scattering1D(J=6, T=24000,
    Q=16)`, split outputs by scattering order, min-max normalise order 1 and
    order 2 separately, then concatenate along the coefficient axis.
    """
    try:
        from kymatio.numpy import Scattering1D
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError("Install WST dependencies with `pip install -e .[wst]`.") from exc

    arr = np.asarray(window, dtype=np.float32).ravel()
    scattering = Scattering1D(J=j, shape=len(arr), Q=q)
    meta = scattering.meta()
    sx = scattering(arr)
    order1 = sx[np.where(meta["order"] == 1)]
    order2 = sx[np.where(meta["order"] == 2)]
    feature = np.concatenate((_minmax_normalise(order1), _minmax_normalise(order2)), axis=0)
    if shape_check is not None and feature.shape != shape_check:
        raise ValueError(
            f"Unexpected WST feature shape {feature.shape}; expected {shape_check}. "
            "Check the sample rate, sample duration, J and Q settings."
        )
    return feature.astype(np.float32)


def compute_wst_features(
    windows: np.ndarray,
    j: int = 6,
    q: int = 16,
    shape_check: tuple[int, int] | None = (221, 375),
    progress: bool = True,
) -> np.ndarray:
    """Compute WST feature maps for many windows."""
    window_arr = np.asarray(windows, dtype=np.float32)
    features = []
    for index, window in enumerate(window_arr):
        if progress:
            print(f"Computing WST window {index + 1}/{len(window_arr)}", end="\r")
        features.append(compute_wst_feature(window, j=j, q=q, shape_check=shape_check))
    if progress and len(window_arr):
        print()
    if not features:
        h, w = shape_check or (0, 0)
        return np.empty((0, h, w), dtype=np.float32)
    return np.stack(features).astype(np.float32)


def iter_audio_files(audio_dir: str | Path, suffixes: Sequence[str] = tuple(SUPPORTED_AUDIO_SUFFIXES)) -> list[Path]:
    """Return sorted supported audio files from a directory."""
    directory = Path(audio_dir)
    suffix_set = {suffix.lower() for suffix in suffixes}
    files = sorted(path for path in directory.rglob("*") if path.is_file() and path.suffix.lower() in suffix_set)
    if not files:
        raise FileNotFoundError(f"No supported audio files found in {directory}. Supported suffixes: {sorted(suffix_set)}")
    return files


def compute_wst_batch_from_audio_dir(
    audio_dir: str | Path,
    target_sample_rate: int = 48_000,
    sample_seconds: float = 0.5,
    overlap_seconds: float = 0.1,
    j: int = 6,
    q: int = 16,
    shape_check: tuple[int, int] | None = (221, 375),
    progress: bool = True,
) -> WSTBatch:
    """Read all `.wav` files in a folder and compute WST feature maps."""
    all_features: list[np.ndarray] = []
    records: list[AudioWindowRecord] = []
    for audio_path in iter_audio_files(audio_dir):
        sample_rate, signal = read_wav_mono(audio_path)
        signal = resample_if_needed(signal, sample_rate, target_sample_rate)
        windows = segment_audio(
            signal,
            sample_rate=target_sample_rate,
            sample_seconds=sample_seconds,
            overlap_seconds=overlap_seconds,
        )
        if progress:
            print(f"{audio_path}: {len(windows)} windows")
        features = compute_wst_features(windows, j=j, q=q, shape_check=shape_check, progress=progress)
        all_features.append(features)
        hop_seconds = sample_seconds - overlap_seconds
        for index in range(len(windows)):
            start = index * hop_seconds
            records.append(
                AudioWindowRecord(
                    source_file=str(audio_path),
                    window_index=index,
                    start_seconds=start,
                    end_seconds=start + sample_seconds,
                )
            )

    if not all_features:
        h, w = shape_check or (0, 0)
        feature_array = np.empty((0, h, w), dtype=np.float32)
    else:
        feature_array = np.concatenate(all_features, axis=0)
    return WSTBatch(features=feature_array, records=tuple(records))


def write_window_records_csv(records: Iterable[AudioWindowRecord], path: str | Path) -> Path:
    """Write audio-window metadata to CSV."""
    import csv

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["source_file", "window_index", "start_seconds", "end_seconds"])
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "source_file": record.source_file,
                    "window_index": record.window_index,
                    "start_seconds": f"{record.start_seconds:.6f}",
                    "end_seconds": f"{record.end_seconds:.6f}",
                }
            )
    return output
