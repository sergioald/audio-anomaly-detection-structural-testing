import numpy as np

from audio_anomaly.audio import normalise_audio_signal, segment_audio


def test_segment_audio_uses_expected_overlap():
    signal = np.arange(10, dtype=np.float32)
    windows = segment_audio(signal, sample_rate=10, sample_seconds=0.5, overlap_seconds=0.2)
    assert windows.shape == (2, 5)
    assert np.array_equal(windows[0], np.array([0, 1, 2, 3, 4], dtype=np.float32))
    assert np.array_equal(windows[1], np.array([3, 4, 5, 6, 7], dtype=np.float32))


def test_segment_audio_returns_empty_for_short_signal():
    windows = segment_audio(np.arange(3), sample_rate=10, sample_seconds=0.5, overlap_seconds=0.1)
    assert windows.shape == (0, 5)


def test_normalise_audio_signal_handles_int16():
    audio = np.array([-32768, 0, 32767], dtype=np.int16)
    normalised = normalise_audio_signal(audio)
    assert normalised.dtype == np.float32
    assert np.isclose(normalised[0], -1.0)
    assert normalised[2] <= 1.0
