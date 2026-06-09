"""Audio anomaly detection research-software utilities."""

from .data import DATASET_FILES, DatasetBundle, load_dataset_bundle, validate_dataset_files
from .metrics import normalized_cross_correlation

__all__ = [
    "DATASET_FILES",
    "DatasetBundle",
    "load_dataset_bundle",
    "validate_dataset_files",
    "normalized_cross_correlation",
]

__version__ = "0.1.0"
