from scripts.predict_new_audio import _records_for_features


def test_records_for_features_creates_placeholder_metadata():
    records = _records_for_features(3)

    assert len(records) == 3
    assert [record.window_index for record in records] == [0, 1, 2]
    assert all(record.source_file == "features.npy" for record in records)
    assert all(record.start_seconds != record.start_seconds for record in records)  # NaN
    assert all(record.end_seconds != record.end_seconds for record in records)  # NaN
