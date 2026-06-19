import numpy as np
import pytest

from audio_anomaly.audio import (
    AudioWindowRecord,
    _minmax_normalise,
    iter_audio_files,
    normalise_audio_signal,
    segment_audio,
    write_window_records_csv,
)


def test_normalise_audio_signal_scales_integer_pcm_to_float32():
    signal = np.array([np.iinfo(np.int16).min, 0, np.iinfo(np.int16).max], dtype=np.int16)

    normalised = normalise_audio_signal(signal)

    assert normalised.dtype == np.float32
    assert np.isclose(normalised[0], -1.0)
    assert normalised[1] == 0.0
    assert normalised[2] <= 1.0


def test_normalise_audio_signal_preserves_float_values_as_float32():
    signal = np.array([-0.25, 0.0, 0.25], dtype=np.float64)

    normalised = normalise_audio_signal(signal)

    assert normalised.dtype == np.float32
    assert np.allclose(normalised, signal.astype(np.float32))


def test_segment_audio_returns_overlapping_full_windows_only():
    signal = np.arange(10, dtype=np.float32)

    windows = segment_audio(signal, sample_rate=10, sample_seconds=0.4, overlap_seconds=0.1)

    assert windows.shape == (3, 4)
    assert np.array_equal(windows[0], np.array([0, 1, 2, 3], dtype=np.float32))
    assert np.array_equal(windows[1], np.array([3, 4, 5, 6], dtype=np.float32))
    assert np.array_equal(windows[2], np.array([6, 7, 8, 9], dtype=np.float32))


def test_segment_audio_returns_empty_when_signal_shorter_than_window():
    windows = segment_audio(np.arange(3), sample_rate=10, sample_seconds=0.5, overlap_seconds=0.1)

    assert windows.shape == (0, 5)
    assert windows.dtype == np.float32


def test_segment_audio_rejects_invalid_window_and_overlap():
    with pytest.raises(ValueError, match="positive window length"):
        segment_audio(np.arange(10), sample_rate=10, sample_seconds=0.0)

    with pytest.raises(ValueError, match="overlap_seconds"):
        segment_audio(np.arange(10), sample_rate=10, sample_seconds=0.5, overlap_seconds=0.5)


def test_minmax_normalise_scales_to_unit_interval_and_handles_constant_arrays():
    array = np.array([2.0, 4.0, 6.0])

    normalised = _minmax_normalise(array)

    assert normalised.dtype == np.float32
    assert np.allclose(normalised, np.array([0.0, 0.5, 1.0], dtype=np.float32))
    assert np.array_equal(_minmax_normalise(np.ones(3)), np.zeros(3, dtype=np.float32))


def test_iter_audio_files_returns_sorted_wav_files_recursively(tmp_path):
    nested = tmp_path / "nested"
    nested.mkdir()
    wav_b = nested / "b.wav"
    wav_a = tmp_path / "a.WAV"
    txt = tmp_path / "ignore.txt"

    wav_b.write_bytes(b"not really wav")
    wav_a.write_bytes(b"not really wav")
    txt.write_text("ignore", encoding="utf-8")

    files = iter_audio_files(tmp_path)

    assert files == sorted([wav_a, wav_b])


def test_iter_audio_files_raises_when_no_supported_files(tmp_path):
    (tmp_path / "ignore.txt").write_text("ignore", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="No supported audio files"):
        iter_audio_files(tmp_path)


def test_write_window_records_csv(tmp_path):
    records = [
        AudioWindowRecord("a.wav", 0, 0.0, 0.5),
        AudioWindowRecord("a.wav", 1, 0.4, 0.9),
    ]

    path = write_window_records_csv(records, tmp_path / "metadata" / "windows.csv")

    text = path.read_text(encoding="utf-8")
    assert "source_file,window_index,start_seconds,end_seconds" in text
    assert "a.wav,0,0.000000,0.500000" in text
    assert "a.wav,1,0.400000,0.900000" in text
