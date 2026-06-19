import builtins

import pytest

from audio_anomaly.model import _import_keras, feature_layer_names, find_feature_layer


class FakeShape:
    def __init__(self, dims):
        self._dims = dims

    def __len__(self):
        return len(self._dims)


class FakeOutput:
    def __init__(self, dims):
        self.shape = FakeShape(dims)


class Conv2D:
    def __init__(self, name, filters=None, output_dims=(None, 2, 2, 1)):
        self.name = name
        self.filters = filters
        self.output = FakeOutput(output_dims)


class Dense:
    def __init__(self, name, output_dims=(None, 10)):
        self.name = name
        self.output = FakeOutput(output_dims)


class FakeModel:
    def __init__(self, layers):
        self.layers = layers

    def get_layer(self, name):
        for layer in self.layers:
            if layer.name == name:
                return layer
        raise ValueError(name)


def test_feature_layer_names_returns_layers_with_4d_outputs():
    model = FakeModel([
        Dense("flat", output_dims=(None, 10)),
        Conv2D("conv", output_dims=(None, 4, 4, 8)),
    ])

    assert feature_layer_names(model) == ["conv"]


def test_find_feature_layer_prefers_named_layer():
    model = FakeModel([
        Conv2D("encoder_conv2", filters=64),
        Conv2D("other_conv", filters=64),
    ])

    assert find_feature_layer(model, preferred="encoder_conv2") == "encoder_conv2"


def test_find_feature_layer_falls_back_to_first_64_filter_conv2d():
    model = FakeModel([
        Conv2D("conv_a", filters=32),
        Conv2D("conv_b", filters=64),
    ])

    assert find_feature_layer(model, preferred="missing") == "conv_b"


def test_find_feature_layer_reports_available_layers_when_missing():
    model = FakeModel([Conv2D("conv_a", filters=32)])

    with pytest.raises(ValueError, match="Could not find preferred layer"):
        find_feature_layer(model, preferred="missing", filters=64)


def test_import_keras_reports_helpful_message_when_tensorflow_missing(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "tensorflow":
            raise ImportError("No module named tensorflow")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ImportError, match="TensorFlow is required"):
        _import_keras()
