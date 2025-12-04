"""
S4 Policy Meta-Model - ML-based trade filter using z_t + regime + meta

This module provides:
- Simple logistic regression model (numpy-based, no sklearn dependency)
- MLP model (PyTorch-based)
- Training and evaluation utilities
"""

from typing import List, Tuple, Optional, Dict, Any
import numpy as np
from pathlib import Path
import json

from .policy_dataset import S4TradeEnriched, S4PolicyDataset


class LogisticRegressionSimple:
    """Simple logistic regression implemented with numpy.
    
    Binary classification: predict p(win) from features.
    """
    
    def __init__(self, input_dim: int, lr: float = 0.01, max_iter: int = 1000):
        self.input_dim = input_dim
        self.lr = lr
        self.max_iter = max_iter
        self.weights = np.zeros(input_dim + 1)  # +1 for bias
        self.trained = False
    
    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        # Clip to avoid overflow
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def _add_bias(self, X: np.ndarray) -> np.ndarray:
        """Add bias column."""
        return np.hstack([np.ones((X.shape[0], 1)), X])
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LogisticRegressionSimple':
        """Train the model using gradient descent."""
        X_b = self._add_bias(X)
        n_samples = X_b.shape[0]
        
        self.weights = np.zeros(X_b.shape[1])
        
        for _ in range(self.max_iter):
            # Forward pass
            z = X_b @ self.weights
            predictions = self._sigmoid(z)
            
            # Gradient
            error = predictions - y
            gradient = (X_b.T @ error) / n_samples
            
            # Update
            self.weights -= self.lr * gradient
        
        self.trained = True
        return self
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probability of win."""
        X_b = self._add_bias(X)
        return self._sigmoid(X_b @ self.weights)
    
    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predict binary labels."""
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute accuracy."""
        predictions = self.predict(X)
        return np.mean(predictions == y)


class S4MetaModel:
    """Meta-model for S4 trade filtering.
    
    Uses z_t embedding + regime + meta features to predict p(win).
    """
    
    def __init__(self, z_dim: int = 64, meta_dim: int = 7):
        self.z_dim = z_dim
        self.meta_dim = meta_dim
        self.input_dim = z_dim + meta_dim
        self.model = LogisticRegressionSimple(self.input_dim, lr=0.1, max_iter=500)
        self.trained = False
        
        # Feature normalization stats
        self.mean = None
        self.std = None
    
    def _extract_features(self, trade: S4TradeEnriched) -> np.ndarray:
        """Extract feature vector from trade."""
        z_t = np.array(trade.z_t, dtype=np.float32)
        meta = np.array([
            trade.regime,
            trade.session_id,
            trade.pos_in_session_range,
            trade.inside_value,
            trade.above_value,
            trade.below_value,
            1 if trade.direction == 'long' else 0,
        ], dtype=np.float32)
        return np.concatenate([z_t, meta])
    
    def _normalize(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        """Normalize features."""
        if fit:
            self.mean = np.mean(X, axis=0)
            self.std = np.std(X, axis=0) + 1e-8
        
        if self.mean is not None:
            return (X - self.mean) / self.std
        return X
    
    def fit(self, dataset: S4PolicyDataset) -> 'S4MetaModel':
        """Train on dataset."""
        # Filter to win/loss only
        trades = [t for t in dataset.trades if t.label in ['win', 'loss']]
        
        if len(trades) < 10:
            print("Warning: Too few trades for training")
            return self
        
        # Extract features
        X = np.array([self._extract_features(t) for t in trades])
        y = np.array([1 if t.label == 'win' else 0 for t in trades])
        
        # Normalize
        X = self._normalize(X, fit=True)
        
        # Train
        self.model.fit(X, y)
        self.trained = True
        
        return self
    
    def predict_proba_single(self, trade: S4TradeEnriched) -> float:
        """Predict p(win) for single trade."""
        if not self.trained:
            return 0.5
        
        X = self._extract_features(trade).reshape(1, -1)
        X = self._normalize(X)
        return float(self.model.predict_proba(X)[0])
    
    def predict_proba_batch(self, trades: List[S4TradeEnriched]) -> np.ndarray:
        """Predict p(win) for batch of trades."""
        if not self.trained:
            return np.full(len(trades), 0.5)
        
        X = np.array([self._extract_features(t) for t in trades])
        X = self._normalize(X)
        return self.model.predict_proba(X)
    
    def evaluate(self, dataset: S4PolicyDataset) -> Dict[str, float]:
        """Evaluate on dataset."""
        trades = [t for t in dataset.trades if t.label in ['win', 'loss']]
        
        if not trades:
            return {'accuracy': 0.0, 'n_samples': 0}
        
        X = np.array([self._extract_features(t) for t in trades])
        y = np.array([1 if t.label == 'win' else 0 for t in trades])
        
        X = self._normalize(X)
        
        accuracy = self.model.score(X, y)
        predictions = self.model.predict(X)
        
        # Compute metrics
        tp = np.sum((predictions == 1) & (y == 1))
        fp = np.sum((predictions == 1) & (y == 0))
        fn = np.sum((predictions == 0) & (y == 1))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'n_samples': len(trades),
        }
    
    def save(self, path: str):
        """Save model to file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        data = {
            'z_dim': self.z_dim,
            'meta_dim': self.meta_dim,
            'weights': self.model.weights.tolist(),
            'mean': self.mean.tolist() if self.mean is not None else None,
            'std': self.std.tolist() if self.std is not None else None,
            'trained': self.trained,
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'S4MetaModel':
        """Load model from file."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        model = cls(z_dim=data['z_dim'], meta_dim=data['meta_dim'])
        model.model.weights = np.array(data['weights'])
        model.mean = np.array(data['mean']) if data['mean'] else None
        model.std = np.array(data['std']) if data['std'] else None
        model.trained = data['trained']
        
        return model


def train_meta_model(
    train_dataset: S4PolicyDataset,
    val_dataset: Optional[S4PolicyDataset] = None,
) -> Tuple[S4MetaModel, Dict[str, Any]]:
    """Train meta-model and return with evaluation results.
    
    Returns:
        model: Trained S4MetaModel
        results: Dict with train/val metrics
    """
    model = S4MetaModel()
    model.fit(train_dataset)
    
    results = {
        'train': model.evaluate(train_dataset),
    }
    
    if val_dataset:
        results['val'] = model.evaluate(val_dataset)
    
    return model, results
