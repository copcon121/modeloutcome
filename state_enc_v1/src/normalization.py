"""
Normalization utilities for STATE-ENC v1.1

Normalization strategies:
- RobustScaler (median/IQR): volume, delta, tick_count, buy/sell ratio
- log1p scaling: atr, range, volatility features
- z-score: price (o, h, l, c)
- Min-max: distance features
"""

import json
import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from pathlib import Path

from .features_spec import (
    FEATURE_SPEC, NUMERIC_FEATURES, get_feature_names, get_feature_defaults
)


# Feature groups for different normalization strategies
ROBUST_FEATURES = [
    "volume", "delta", "delta_abs", "tick_count", "tick_speed",
    "buy_volume", "sell_volume", "buy_ratio", "sell_ratio",
    "imbalance_buy_sell", "cum_delta_session"
]

LOG1P_FEATURES = [
    "atr_m1_14", "true_range", "hl_range", "volume_zscore",
    "range_vs_session_avg", "volume_vs_session_avg"
]

ZSCORE_FEATURES = ["o", "h", "l", "c", "swing_high", "swing_low", "vah", "val", "poc", "session_high", "session_low"]

MINMAX_FEATURES = [
    "distance_to_swing_high", "distance_to_swing_low",
    "distance_to_swing_high_norm", "distance_to_swing_low_norm",
    "dist_to_vah", "dist_to_val", "dist_to_poc",
    "distance_to_nearest_ob", "distance_to_nearest_fvg",
    "dist_to_session_high_norm", "dist_to_session_low_norm",
    "pos_in_session_range", "price_vs_swing_mid"
]


@dataclass
class FeatureStats:
    """Statistics for a single feature"""
    name: str
    mean: float = 0.0
    std: float = 1.0
    median: float = 0.0
    iqr: float = 1.0
    min_val: float = 0.0
    max_val: float = 1.0
    count: int = 0


class FeatureNormalizer:
    """
    Feature normalizer v1.1 with multiple strategies
    """
    
    def __init__(self, clip_value: float = 5.0, eps: float = 1e-8):
        self.clip_value = clip_value
        self.eps = eps
        self.stats: Dict[str, FeatureStats] = {}
        self.feature_names: List[str] = get_feature_names()
        self.defaults: Dict[str, float] = get_feature_defaults()
        self.is_fitted: bool = False
    
    def fit(self, bars: List[Dict[str, Any]]) -> "FeatureNormalizer":
        """Compute statistics from bars"""
        if not bars:
            raise ValueError("Cannot fit on empty data")
        
        # Collect values per feature
        accum: Dict[str, List[float]] = {name: [] for name in NUMERIC_FEATURES}
        
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
                q1, q3 = np.percentile(arr, [25, 75])
                self.stats[name] = FeatureStats(
                    name=name,
                    mean=float(np.mean(arr)),
                    std=float(np.std(arr)) + self.eps,
                    median=float(np.median(arr)),
                    iqr=float(q3 - q1) + self.eps,
                    min_val=float(np.min(arr)),
                    max_val=float(np.max(arr)),
                    count=len(values)
                )
            else:
                self.stats[name] = FeatureStats(name=name)
        
        self.is_fitted = True
        return self
    
    def _get_norm_strategy(self, name: str) -> str:
        """Determine normalization strategy for feature"""
        if name in ROBUST_FEATURES:
            return "robust"
        elif name in LOG1P_FEATURES:
            return "log1p"
        elif name in ZSCORE_FEATURES:
            return "zscore"
        elif name in MINMAX_FEATURES:
            return "minmax"
        else:
            return "zscore"  # default
    
    def transform_bar(self, bar: Dict[str, Any]) -> np.ndarray:
        """Transform single bar to normalized vector"""
        if not self.is_fitted:
            raise RuntimeError("Normalizer not fitted")
        
        result = np.zeros(len(self.feature_names), dtype=np.float32)
        
        for i, name in enumerate(self.feature_names):
            raw_val = bar.get(name, self.defaults.get(name, 0.0))
            
            if raw_val is None or (isinstance(raw_val, float) and np.isnan(raw_val)):
                raw_val = self.defaults.get(name, 0.0)
            
            if name in NUMERIC_FEATURES:
                stats = self.stats.get(name)
                strategy = self._get_norm_strategy(name)
                
                if stats:
                    if strategy == "robust":
                        # RobustScaler: (x - median) / IQR
                        normalized = (raw_val - stats.median) / stats.iqr
                    elif strategy == "log1p":
                        # log1p scaling
                        sign = 1 if raw_val >= 0 else -1
                        normalized = sign * np.log1p(abs(raw_val))
                        # Then z-score
                        normalized = (normalized - stats.mean) / stats.std
                    elif strategy == "minmax":
                        # Min-max to [0, 1]
                        range_val = stats.max_val - stats.min_val + self.eps
                        normalized = (raw_val - stats.min_val) / range_val
                        normalized = np.clip(normalized, 0.0, 1.0)
                    else:  # zscore
                        normalized = (raw_val - stats.mean) / stats.std
                    
                    # Clip extreme values
                    if strategy != "minmax":
                        normalized = np.clip(normalized, -self.clip_value, self.clip_value)
                else:
                    normalized = raw_val
                
                result[i] = normalized
            else:
                # Categorical/binary: pass through
                result[i] = float(raw_val)
        
        return result
    
    def transform_sequence(self, bars: List[Dict[str, Any]]) -> np.ndarray:
        """Transform sequence of bars"""
        return np.stack([self.transform_bar(bar) for bar in bars], axis=0)
    
    def save(self, path: str) -> None:
        """Save normalizer state"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "version": "1.1",
            "clip_value": self.clip_value,
            "eps": self.eps,
            "feature_names": self.feature_names,
            "stats": {
                name: {
                    "mean": s.mean, "std": s.std,
                    "median": s.median, "iqr": s.iqr,
                    "min_val": s.min_val, "max_val": s.max_val,
                    "count": s.count
                }
                for name, s in self.stats.items()
            },
            "normalization_groups": {
                "robust": ROBUST_FEATURES,
                "log1p": LOG1P_FEATURES,
                "zscore": ZSCORE_FEATURES,
                "minmax": MINMAX_FEATURES
            },
            "is_fitted": self.is_fitted
        }
        
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
    
    def load(self, data: Dict[str, Any]) -> "FeatureNormalizer":
        """Load normalizer state"""
        if isinstance(data, str):
            with open(data, "r") as f:
                data = json.load(f)
        
        self.clip_value = data.get("clip_value", 5.0)
        self.eps = data.get("eps", 1e-8)
        
        self.stats = {}
        for name, s in data.get("stats", {}).items():
            self.stats[name] = FeatureStats(
                name=name,
                mean=s.get("mean", 0.0),
                std=s.get("std", 1.0),
                median=s.get("median", 0.0),
                iqr=s.get("iqr", 1.0),
                min_val=s.get("min_val", 0.0),
                max_val=s.get("max_val", 1.0),
                count=s.get("count", 0)
            )
        
        self.is_fitted = data.get("is_fitted", True)
        return self
    
    @classmethod
    def from_file(cls, path: str) -> "FeatureNormalizer":
        normalizer = cls()
        with open(path, "r") as f:
            data = json.load(f)
        return normalizer.load(data)


def compute_session_stats(bars: List[Dict[str, Any]]) -> Dict[str, float]:
    """Compute session-level statistics"""
    if not bars:
        return {"session_avg_volume": 1.0, "session_avg_range": 1.0, "session_high": 0.0, "session_low": 0.0}
    
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
