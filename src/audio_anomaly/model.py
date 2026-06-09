"""Convolutional autoencoder architecture and model-loading helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _import_keras() -> Any:
    try:
        from tensorflow import keras
    except ImportError as exc:
        raise ImportError(
            "TensorFlow is required for model operations. Install with `pip install -e .[deep-learning]`."
        ) from exc
    return keras


def build_cae_latent24(input_shape: tuple[int, int, int] = (221, 375, 1)):
    """Build the CAE architecture used for the latent-24 paper model.

    The layer dimensions follow the appendix architecture. Layers are given stable
    names so public scripts do not depend on fragile legacy names such as
    `conv2d_31`.
    """
    keras = _import_keras()
    layers = keras.layers

    input_img = keras.Input(shape=input_shape, name="wst_input")
    x = layers.ZeroPadding2D(padding=((17, 18), (4, 5)), name="input_padding")(input_img)

    x = layers.Conv2D(128, (7, 7), activation="leaky_relu", padding="same", name="encoder_conv1")(x)
    x = layers.MaxPooling2D((2, 2), padding="same", name="encoder_pool1")(x)

    x = layers.Conv2D(64, (5, 5), activation="leaky_relu", padding="same", name="encoder_conv2")(x)
    x = layers.MaxPooling2D((2, 2), padding="same", name="encoder_pool2")(x)

    x = layers.Conv2D(64, (3, 3), strides=2, activation="leaky_relu", padding="same", name="encoder_conv3")(x)
    x = layers.MaxPooling2D((2, 2), padding="same", name="encoder_pool3")(x)

    x = layers.Conv2D(32, (2, 2), activation="leaky_relu", padding="same", name="encoder_conv4")(x)
    x = layers.MaxPooling2D((2, 2), padding="same", name="encoder_pool4")(x)

    x = layers.Conv2D(16, (2, 2), activation="leaky_relu", padding="same", name="encoder_conv5")(x)
    x = layers.MaxPooling2D((2, 2), padding="same", name="encoder_pool5")(x)

    encoded = layers.Conv2D(1, (2, 2), activation="leaky_relu", padding="same", name="latent_code")(x)

    x = layers.Conv2DTranspose(16, (2, 2), activation="leaky_relu", padding="same", name="decoder_deconv1")(encoded)
    x = layers.UpSampling2D((2, 2), name="decoder_upsample1")(x)
    x = layers.Conv2DTranspose(32, (2, 2), activation="leaky_relu", padding="same", name="decoder_deconv2")(x)
    x = layers.UpSampling2D((2, 2), name="decoder_upsample2")(x)
    x = layers.Conv2DTranspose(64, (2, 2), activation="leaky_relu", padding="same", name="decoder_deconv3")(x)
    x = layers.UpSampling2D((2, 2), name="decoder_upsample3")(x)
    x = layers.Conv2DTranspose(64, (3, 3), strides=2, activation="leaky_relu", padding="same", name="decoder_deconv4")(x)
    x = layers.UpSampling2D((2, 2), name="decoder_upsample4")(x)
    x = layers.Conv2DTranspose(128, (5, 5), activation="leaky_relu", padding="same", name="decoder_deconv5")(x)
    x = layers.UpSampling2D((2, 2), name="decoder_upsample5")(x)
    x = layers.Conv2DTranspose(1, (7, 7), activation="sigmoid", padding="same", name="decoder_output_conv")(x)
    output = layers.Cropping2D(((17, 18), (4, 5)), name="output_cropping")(x)

    return keras.Model(input_img, output, name="cae_latent24")


def compile_autoencoder(model, optimizer: str = "adam", loss: str = "binary_crossentropy"):
    """Compile a CAE model."""
    model.compile(optimizer=optimizer, loss=loss)
    return model


def load_autoencoder(model_path: str | Path, compile_model: bool = False):
    """Load a TensorFlow/Keras model from `.keras` or legacy `.h5` format."""
    keras = _import_keras()
    return keras.models.load_model(model_path, compile=compile_model)


def feature_layer_names(model) -> list[str]:
    """Return layers with 4D outputs, useful for selecting feature-map layers."""
    names: list[str] = []
    for layer in model.layers:
        try:
            shape = layer.output.shape
        except Exception:
            continue
        if len(shape) == 4:
            names.append(layer.name)
    return names


def find_feature_layer(model, preferred: str | None = "encoder_conv2", filters: int = 64) -> str:
    """Find a feature layer in a model.

    If `preferred` exists, it is returned. Otherwise the first Conv2D layer with
    the requested number of filters is returned. This supports both cleaned models
    and legacy `.h5` models whose layer names look like `conv2d_31`.
    """
    if preferred:
        try:
            model.get_layer(preferred)
            return preferred
        except Exception:
            pass

    for layer in model.layers:
        if layer.__class__.__name__ == "Conv2D" and getattr(layer, "filters", None) == filters:
            return layer.name

    available = ", ".join(feature_layer_names(model))
    raise ValueError(
        f"Could not find preferred layer {preferred!r} or Conv2D layer with {filters} filters. "
        f"Available feature-like layers: {available}"
    )
