"""Configuration classes for ASM v2"""

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from pathlib import Path


@dataclass
class AsmDatasetConfig:
    """Configuration for ASM dataset building"""
    encoder_dataset_path: str
    state_enc_model_path: str
    state_enc_model_config_path: str
    state_enc_feature_config_path: str
    output_path: str
    splits_output_path: str
    feature_config_output_path: str
    symbol: str = "GC"
    timeframe: str = "M1"
    train_ratio: float = 0.8
    meta_features: List[str] = field(default_factory=lambda: [
        "session_id", "pos_in_session_range", "inside_value",
        "above_value", "below_value", "minute_of_day_norm"
    ])
    regime_mapping: Dict[str, str] = field(default_factory=lambda: {
        "1": "trend_up", "2": "trend_down", "3": "balance",
        "4": "opening_drive_up", "5": "opening_drive_down"
    })
    
    @classmethod
    def from_json(cls, path: str) -> "AsmDatasetConfig":
        with open(path, "r") as f:
            data = json.load(f)
        return cls(**data)


@dataclass
class AsmModelConfig:
    """Configuration for ASM model"""
    z_dim: int = 64
    meta_dim: int = 6
    hidden_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.1
    num_classes: int = 5
    use_grn: bool = True
    
    @classmethod
    def from_json(cls, path: str) -> "AsmModelConfig":
        with open(path, "r") as f:
            data = json.load(f)
        return cls(**data)
    
    def to_json(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.__dict__, f, indent=2)


@dataclass
class AsmTrainConfig:
    """Configuration for ASM training"""
    dataset_path: str
    splits_path: str
    model_config_path: str
    feature_config_path: str
    output_dir: str
    batch_size: int = 64
    epochs: int = 10
    learning_rate: float = 0.001
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    label_smoothing: float = 0.05
    device: str = "cuda"
    seed: int = 42
    save_best: bool = True
    
    @classmethod
    def from_json(cls, path: str) -> "AsmTrainConfig":
        with open(path, "r") as f:
            data = json.load(f)
        return cls(**data)


@dataclass
class AsmEvalConfig:
    """Configuration for ASM evaluation"""
    model_path: str
    model_config_path: str
    feature_config_path: str
    dataset_path: str
    splits_path: str
    output_dir: str
    device: str = "cuda"
    
    @classmethod
    def from_json(cls, path: str) -> "AsmEvalConfig":
        with open(path, "r") as f:
            data = json.load(f)
        return cls(**data)
