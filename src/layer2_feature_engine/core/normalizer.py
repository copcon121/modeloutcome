"""
Feature normalization utilities
Supports Z-score and Min-Max normalization with fit/transform pattern
"""

import json
from typing import Dict, List, Optional
import numpy as np
from pathlib import Path


class Normalizer:
    """
    Feature normalizer with fit/transform interface
    Supports saving and loading normalization statistics
    """

    def __init__(self, method: str = 'zscore'):
        """
        Args:
            method: 'zscore' or 'minmax'
        """
        self.method = method
        self.stats: Dict[str, Dict[str, float]] = {}
        self.fitted = False

    def fit(self, features: List[Dict[str, float]]) -> 'Normalizer':
        """
        Compute normalization statistics from training data

        Args:
            features: List of feature dictionaries

        Returns:
            self (for chaining)
        """
        if not features:
            raise ValueError("Cannot fit on empty feature list")

        # Collect all feature values
        feature_names = features[0].keys()
        feature_arrays = {name: [] for name in feature_names}

        for feat_dict in features:
            for name in feature_names:
                feature_arrays[name].append(feat_dict.get(name, 0.0))

        # Compute statistics for each feature
        for name, values in feature_arrays.items():
            arr = np.array(values)

            if self.method == 'zscore':
                mean = float(np.mean(arr))
                std = float(np.std(arr))
                # Avoid division by zero
                if std < 1e-8:
                    std = 1.0
                self.stats[name] = {'mean': mean, 'std': std}

            elif self.method == 'minmax':
                min_val = float(np.min(arr))
                max_val = float(np.max(arr))
                # Avoid division by zero
                if max_val - min_val < 1e-8:
                    max_val = min_val + 1.0
                self.stats[name] = {'min': min_val, 'max': max_val}

            else:
                raise ValueError(f"Unknown normalization method: {self.method}")

        self.fitted = True
        return self

    def transform(self, features: Dict[str, float]) -> Dict[str, float]:
        """
        Apply normalization to a single feature dictionary

        Args:
            features: Feature dictionary to normalize

        Returns:
            Normalized feature dictionary
        """
        if not self.fitted:
            raise RuntimeError("Normalizer must be fitted before transform")

        normalized = {}

        for name, value in features.items():
            if name not in self.stats:
                # Unknown feature, pass through as-is
                normalized[name] = value
                continue

            if self.method == 'zscore':
                mean = self.stats[name]['mean']
                std = self.stats[name]['std']
                normalized[name] = (value - mean) / std

            elif self.method == 'minmax':
                min_val = self.stats[name]['min']
                max_val = self.stats[name]['max']
                normalized[name] = (value - min_val) / (max_val - min_val)

        return normalized

    def fit_transform(self, features: List[Dict[str, float]]) -> List[Dict[str, float]]:
        """
        Fit and transform in one step

        Args:
            features: List of feature dictionaries

        Returns:
            List of normalized feature dictionaries
        """
        self.fit(features)
        return [self.transform(f) for f in features]

    def save(self, filepath: Path) -> None:
        """
        Save normalization statistics to JSON file

        Args:
            filepath: Path to save JSON file
        """
        if not self.fitted:
            raise RuntimeError("Cannot save unfitted normalizer")

        data = {
            'method': self.method,
            'stats': self.stats
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def load(self, filepath: Path) -> 'Normalizer':
        """
        Load normalization statistics from JSON file

        Args:
            filepath: Path to JSON file

        Returns:
            self (for chaining)
        """
        with open(filepath, 'r') as f:
            data = json.load(f)

        self.method = data['method']
        self.stats = data['stats']
        self.fitted = True

        return self

    def __repr__(self) -> str:
        return f"Normalizer(method='{self.method}', fitted={self.fitted}, n_features={len(self.stats)})"
