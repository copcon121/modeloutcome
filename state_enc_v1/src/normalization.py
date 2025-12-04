"""
Normalization utilities for STATE-ENC v1

Handles feature normalization (z-score, min-max) for numeric features.
"""

import json
import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path

from .features_spec import (
    FEATURE_SPEC, 
    NUMERIC_FEATURES, 
    CATEGORICAL_FEATURES,
    BINARY_FEATURES,
    get_feature_names,
    get_feature_defaults,
    FeatureType
)


@dataclass
class FeatureStats:
    """Statistics for a single feature"""
    name: str
    mean: float = 0.0
    std: float = 1.0
    min_val: float = 0.0
    max_val: float = 1.0
    count: int = 0


@dataclass
class NormalizerConfig:
    """Configuration for normalizer"""
    method: str = "zscore"  # "zscore" or "minmax"
    clip_zscore: float = 5.0  # Clip z-scores to [-clip, clip]
    eps: float = 1e-8


class FeatureNormalizer:
    """
    Normalizer for bar features.
    
    Supports:
    - Z-score normalization for numeric features
    - Min-max normalization (optional)
    - Categorical features passed through as-is
    - Binary features passed through as-is
    """
    
    def __init__(self, config: Optional[NormalizerConfig] = None):
        self.config = config or NormalizerConfig()
        self.stats: Dict[str, FeatureStats] = {}
        self.feature_names: List[str] = get_feature_names()
        self.defaults: Dict[str, float] = get_feature_defaults()
        self.is_fitted: bool = False
        
    def fit(self, bars: List[Dict[str, Any]]) -> "FeatureNormalizer":
        """
        Compute statistics from list of bar dicts.
        
        Args:
            bars: List of bar dictionaries with feature values
            
        Returns:
            self for chaining
        """
        if not bars:
            raise ValueError("Cannot fit on empty data")
            
        # Initialize accumulators
        accum: Dict[str, List[float]] = {name: [] for name in NUMERIC_FEATURES}
        
        # Collect values
        for bar in bars:
            for name in NUMERIC_FEATURES:
                val = bar.get(name, self.defaults.get(name, 0.0))
                if val is not None and not np.isnan(val):
                    accum[name].append(float(val))
        
        # Compute stats
        for name in NUMERIC_FEATURES:
            values = accum[name]
            if values:
                arr = np.array(values)
                self.stats[name] = FeatureStats(
                    name=name,
                    mean=float(np.mean(arr)),
                    std=float(np.std(arr)) + self.config.eps,
                    min_val=float(np.min(arr)),
                    max_val=float(np.max(arr)),
                    count=len(values)
                )
            else:
                # No valid values, use defaults
                self.stats[name] = FeatureStats(name=name)
                
        self.is_fitted = True
        return self
    
    def transform_bar(self, bar: Dict[str, Any]) -> np.ndarray:
        """
        Transform a single bar dict to normalized feature vector.
        
        Args:
            bar: Dictionary with feature values
            
        Returns:
            numpy array of shape [D] with normalized features
        """
        if not self.is_fitted:
            raise RuntimeError("Normalizer not fitted. Call fit() first.")
            
        result = np.zeros(len(self.feature_names), dtype=np.float32)
        
        for i, name in enumerate(self.feature_names):
            raw_val = bar.get(name, self.defaults.get(name, 0.0))
            
            if raw_val is None or (isinstance(raw_val, float) and np.isnan(raw_val)):
                raw_val = self.defaults.get(name, 0.0)
            
            # Normalize based on feature type
            if name in NUMERIC_FEATURES:
                stats = self.stats.get(name)
                if stats and self.config.method == "zscore":
                    normalized = (raw_val - stats.mean) / stats.std
                    # Clip extreme values
                    normalized = np.clip(normalized, -self.config.clip_zscore, self.config.clip_zscore)
                elif stats and self.config.method == "minmax":
                    range_val = stats.max_val - stats.min_val + self.config.eps
                    normalized = (raw_val - stats.min_val) / range_val
                    normalized = np.clip(normalized, 0.0, 1.0)
                else:
                    normalized = raw_val
                result[i] = normalized
            else:
                # Categorical and binary: pass through
                result[i] = float(raw_val)
                
        return result
    
    def transform_sequence(self, bars: List[Dict[str, Any]]) -> np.ndarray:
        """
        Transform sequence of bars to normalized tensor.
        
        Args:
            bars: List of bar dictionaries
            
        Returns:
            numpy array of shape [N, D]
        """
        return np.stack([self.transform_bar(bar) for bar in bars], axis=0)
    
    def inverse_transform_bar(self, vector: np.ndarray) -> Dict[str, float]:
        """
        Inverse transform normalized vector back to original scale.
        Only works for numeric features.
        """
        if not self.is_fitted:
            raise RuntimeError("Normalizer not fitted.")
            
        result = {}
        for i, name in enumerate(self.feature_names):
            val = vector[i]
            
            if name in NUMERIC_FEATURES:
                stats = self.stats.get(name)
                if stats and self.config.method == "zscore":
                    original = val * stats.std + stats.mean
                elif stats and self.config.method == "minmax":
                    range_val = stats.max_val - stats.min_val
                    original = val * range_val + stats.min_val
                else:
                    original = val
                result[name] = float(original)
            else:
                result[name] = float(val)
                
        return result
    
    def save(self, path: str) -> None:
        """Save normalizer state to JSON"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "config": {
                "method": self.config.method,
                "clip_zscore": self.config.clip_zscore,
                "eps": self.config.eps
            },
            "feature_names": self.feature_names,
            "stats": {
                name: {
                    "mean": s.mean,
                    "std": s.std,
                    "min_val": s.min_val,
                    "max_val": s.max_val,
                    "count": s.count
                }
                for name, s in self.stats.items()
            },
            "is_fitted": self.is_fitted
        }
        
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
    
    def load(self, data: Dict[str, Any]) -> "FeatureNormalizer":
        """Load normalizer state from dict (or path)"""
        if isinstance(data, str):
            with open(data, "r") as f:
                data = json.load(f)
        
        # Load config
        cfg = data.get("config", {})
        self.config = NormalizerConfig(
            method=cfg.get("method", "zscore"),
            clip_zscore=cfg.get("clip_zscore", 5.0),
            eps=cfg.get("eps", 1e-8)
        )
        
        # Load stats
        self.stats = {}
        for name, s in data.get("stats", {}).items():
            self.stats[name] = FeatureStats(
                name=name,
                mean=s.get("mean", 0.0),
                std=s.get("std", 1.0),
                min_val=s.get("min_val", 0.0),
                max_val=s.get("max_val", 1.0),
                count=s.get("count", 0)
            )
        
        self.is_fitted = data.get("is_fitted", True)
        return self
    
    @classmethod
    def from_file(cls, path: str) -> "FeatureNormalizer":
        """Load normalizer from file"""
        normalizer = cls()
        with open(path, "r") as f:
            data = json.load(f)
        return normalizer.load(data)
    
    def get_stats_summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary of feature statistics"""
        return {
            name: {
                "mean": s.mean,
                "std": s.std,
                "min": s.min_val,
                "max": s.max_val
            }
            for name, s in self.stats.items()
        }


def compute_session_stats(bars: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Compute session-level statistics for normalization.
    
    Returns dict with:
    - session_avg_volume
    - session_avg_range
    - session_high
    - session_low
    """
    if not bars:
        return {
            "session_avg_volume": 1.0,
            "session_avg_range": 1.0,
            "session_high": 0.0,
            "session_low": 0.0
        }
    
    volumes = [b.get("volume", 0) for b in bars]
    ranges = [b.get("h", 0) - b.get("l", 0) for b in bars]
    highs = [b.get("h", 0) for b in bars]
    lows = [b.get("l", float("inf")) for b in bars]
    
    return {
        "session_avg_volume": np.mean(volumes) if volumes else 1.0,
        "session_avg_range": np.mean(ranges) if ranges else 1.0,
        "session_high": max(highs) if highs else 0.0,
        "session_low": min(lows) if lows else 0.0
    }
