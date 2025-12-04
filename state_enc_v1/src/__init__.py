"""
STATE-ENC v1 — Market State Encoder

Module encode chuỗi N bar M1 thành embedding z_t đại diện market state.
"""

from .config import StateEncDatasetConfig, StateEncModelConfig, StateEncTrainConfig
from .features_spec import FEATURE_SPEC, NUMERIC_FEATURES, CATEGORICAL_FEATURES
from .normalization import FeatureNormalizer
from .dataset_encoder import StateEncDataset, StateEncSample

__version__ = "1.0.0"
__all__ = [
    "StateEncDatasetConfig",
    "StateEncModelConfig", 
    "StateEncTrainConfig",
    "FEATURE_SPEC",
    "NUMERIC_FEATURES",
    "CATEGORICAL_FEATURES",
    "FeatureNormalizer",
    "StateEncDataset",
    "StateEncSample",
]
