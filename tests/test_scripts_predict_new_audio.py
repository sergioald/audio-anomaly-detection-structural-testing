import importlib.util
from pathlib import Path


def _load_predict_new_audio_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "predict_new_audio.py"
    spec = importlib.util.spec_from_file_location("predict_new_audio_script", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_records_for_features_creates_placeholder_metadata():
    module = _load_predict_new_audio_module()

    records = module._records_for_features(3)

    assert len(records) == 3
    assert [record.window_index for record in records] == [0, 1, 2]
    assert all(record.source_file == "features.npy" for record in records)
    assert all(record.start_seconds != record.start_seconds for record in records)  # NaN
    assert all(record.end_seconds != record.end_seconds for record in records)  # NaN
