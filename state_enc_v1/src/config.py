"""
Configuration classes for STATE-ENC v1
"""

import json
from dataclasses import dataclass, field
from typing import Dict, Optional, Any
from pathlib import Path


@dataclass
class StateEncDatasetConfig:
    """Configuration for dataset building"""
    raw_bars_path: str
    output_path: str
    feature_config_path: str
    sequence_length: int = 128
    stride: int = 4
    future_bars: int = 5
    future_dir_threshold_up: float = 0.0005
    future_dir_threshold_down: float = -0.0005
    tick_size: float = 0.25
    min_bars_per_session: int = 64
    max_samples: Optional[int] = None  # Limit total samples (for debug)
    sessions: Dict[str, Dict[str, int]] = field(default_factory=lambda: {
        "ASIA": {"start_hour": 18, "end_hour": 2},
        "LDN": {"start_hour": 2, "end_hour": 8},
        "NY": {"start_hour": 8, "end_hour": 17}
    })
    symbol: str = "NQ"
    timeframe: str = "M1"
    
    @classmethod
    def from_json(cls, path: str) -> "StateEncDatasetConfig":
        with open(path, "r") as f:
            data = json.load(f)
        return cls(**data)
    
    def to_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.__dict__, f, indent=2)


@dataclass
class HeadConfig:
    """Configuration for model heads"""
    enabled: bool = True
    num_dir_classes: int = 3
    predict_return: bool = True
    num_classes: int = 6
    output_dim: int = 4


@dataclass
class StateEncModelConfig:
    """Configuration for model architecture v1.1"""
    input_dim: int = 95
    d_model: int = 96
    num_heads: int = 4
    num_layers: int = 4
    dim_feedforward: int = 512
    dropout: float = 0.05
    sequence_length: int = 128
    pooling: str = "last"
    conv_kernel: int = 5
    use_rms_norm: bool = False
    heads: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "mfm": {"enabled": True, "output_dim": 95},
        "contrastive": {"enabled": True, "proj_dim": 96},
        "self_supervised": {"enabled": True, "num_dir_classes": 3, "predict_return": True},
        "regime": {"enabled": True, "num_classes": 6},
        "meta_s4": {"enabled": False, "output_dim": 4}
    })
    
    @classmethod
    def from_json(cls, path: str) -> "StateEncModelConfig":
        with open(path, "r") as f:
            data = json.load(f)
        return cls(**data)
    
    def to_json(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.__dict__, f, indent=2)


@dataclass
class StateEncTrainConfig:
    """Configuration for training v1.1"""
    dataset_path: str
    feature_config_path: str
    model_config_path: str
    output_dir: str
    batch_size: int = 32
    max_epochs: int = 100
    learning_rate: float = 0.0001
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    patience: int = 15
    val_split: float = 0.15
    test_split: float = 0.1
    loss_weights: Dict[str, float] = field(default_factory=lambda: {
        "mfm": 1.0,
        "contrastive": 0.7,
        "future_dir": 0.5,
        "future_return": 0.1,
        "regime": 0.3,
        "temperature": 0.1
    })
    device: str = "cuda"
    seed: int = 42
    num_workers: int = 4
    save_every_n_epochs: int = 5
    log_every_n_steps: int = 100
    
    @classmethod
    def from_json(cls, path: str) -> "StateEncTrainConfig":
        with open(path, "r") as f:
            data = json.load(f)
        return cls(**data)
    
    def to_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.__dict__, f, indent=2)
