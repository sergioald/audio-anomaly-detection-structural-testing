import audio_anomaly


def test_public_package_exports():
    assert audio_anomaly.__version__ == "0.1.0"
    assert "normal_train" in audio_anomaly.DATASET_FILES
    assert callable(audio_anomaly.load_dataset_bundle)
    assert callable(audio_anomaly.validate_dataset_files)
    assert callable(audio_anomaly.normalized_cross_correlation)
