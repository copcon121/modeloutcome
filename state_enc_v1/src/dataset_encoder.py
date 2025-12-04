"""
PyTorch Dataset for STATE-ENC v1

Loads sequence samples and provides tensors for training.
"""

import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path

from .features_spec import get_feature_names, get_feature_defaults, TOTAL_FEATURE_DIM
from .normalization import FeatureNormalizer


@dataclass
class StateEncSample:
    """Single sample for state encoder"""
    symbol: str
    tf: str
    date: str
    session: str
    start_time: str
    end_time: str
    seq: List[Dict[str, Any]]  # List of N bar dicts
    aux: Dict[str, Any]  # future_return_5, future_dir_5, future_range_5
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "StateEncSample":
        return cls(
            symbol=d.get("symbol", ""),
            tf=d.get("tf", "M1"),
            date=d.get("date", ""),
            session=d.get("session", ""),
            start_time=d.get("start_time", ""),
            end_time=d.get("end_time", ""),
            seq=d.get("seq", []),
            aux=d.get("aux", {})
        )


class StateEncDataset(Dataset):
    """
    PyTorch Dataset for state encoder training.
    
    Loads samples from JSONL file and returns normalized tensors.
    """
    
    def __init__(self,
                 dataset_path: str,
                 feature_config_path: Optional[str] = None,
                 normalizer: Optional[FeatureNormalizer] = None,
                 transform: Optional[callable] = None,
                 max_samples: Optional[int] = None):
        """
        Args:
            dataset_path: Path to encoder_dataset.jsonl
            feature_config_path: Path to feature_config.json (for loading normalizer)
            normalizer: Pre-fitted normalizer (if None, loads from feature_config)
            transform: Optional transform function
            max_samples: Limit number of samples (for debugging)
        """
        self.dataset_path = dataset_path
        self.transform = transform
        self.feature_names = get_feature_names()
        self.feature_dim = len(self.feature_names)
        self.defaults = get_feature_defaults()
        
        # Load normalizer
        if normalizer is not None:
            self.normalizer = normalizer
        elif feature_config_path is not None:
            self.normalizer = self._load_normalizer(feature_config_path)
        else:
            raise ValueError("Must provide either normalizer or feature_config_path")
        
        # Load samples
        self.samples = self._load_samples(dataset_path, max_samples)
        
    def _load_normalizer(self, path: str) -> FeatureNormalizer:
        """Load normalizer from feature config"""
        with open(path, "r") as f:
            config = json.load(f)
        
        normalizer = FeatureNormalizer()
        normalizer.load(config.get("normalization", {}))
        return normalizer
    
    def _load_samples(self, path: str, max_samples: Optional[int] = None) -> List[StateEncSample]:
        """Load samples from JSONL file"""
        samples = []
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    samples.append(StateEncSample.from_dict(d))
                except json.JSONDecodeError:
                    continue
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get single sample.
        
        Returns:
            Dict with:
                - X: tensor [N, D] normalized features
                - future_dir_5: long tensor (label 0, 1, 2 for -1, 0, +1)
                - future_return_5: float tensor
                - regime_hint: long tensor
                - session_id: long tensor
        """
        sample = self.samples[idx]
        
        # Convert sequence to tensor
        X = self._encode_sequence(sample.seq)
        
        # Get targets
        aux = sample.aux
        
        # Map future_dir from {-1, 0, 1} to {0, 1, 2} for CrossEntropy
        future_dir = aux.get("future_dir_5", 0)
        future_dir_label = future_dir + 1  # -1 -> 0, 0 -> 1, 1 -> 2
        
        future_return = aux.get("future_return_5", 0.0)
        
        # Get regime hint from last bar
        last_bar = sample.seq[-1] if sample.seq else {}
        regime_hint = int(last_bar.get("asm_regime_hint", 0))
        
        # Session ID
        session_map = {"ASIA": 0, "LDN": 1, "NY": 2}
        session_id = session_map.get(sample.session, 0)
        
        result = {
            "X": torch.tensor(X, dtype=torch.float32),
            "future_dir_5": torch.tensor(future_dir_label, dtype=torch.long),
            "future_return_5": torch.tensor(future_return, dtype=torch.float32),
            "regime_hint": torch.tensor(regime_hint, dtype=torch.long),
            "session_id": torch.tensor(session_id, dtype=torch.long),
        }
        
        if self.transform:
            result = self.transform(result)
        
        return result
    
    def _encode_sequence(self, seq: List[Dict[str, Any]]) -> np.ndarray:
        """
        Encode sequence of bars to normalized tensor.
        
        Args:
            seq: List of bar dicts
            
        Returns:
            numpy array [N, D]
        """
        return self.normalizer.transform_sequence(seq)
    
    def get_sample_metadata(self, idx: int) -> Dict[str, Any]:
        """Get metadata for sample (for debugging/analysis)"""
        sample = self.samples[idx]
        return {
            "symbol": sample.symbol,
            "date": sample.date,
            "session": sample.session,
            "start_time": sample.start_time,
            "end_time": sample.end_time,
            "aux": sample.aux
        }


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Collate function for DataLoader.
    
    Args:
        batch: List of sample dicts from __getitem__
        
    Returns:
        Dict with batched tensors:
            - X: [B, N, D]
            - future_dir_5: [B]
            - future_return_5: [B]
            - regime_hint: [B]
            - session_id: [B]
    """
    return {
        "X": torch.stack([s["X"] for s in batch], dim=0),
        "future_dir_5": torch.stack([s["future_dir_5"] for s in batch], dim=0),
        "future_return_5": torch.stack([s["future_return_5"] for s in batch], dim=0),
        "regime_hint": torch.stack([s["regime_hint"] for s in batch], dim=0),
        "session_id": torch.stack([s["session_id"] for s in batch], dim=0),
    }


def create_dataloaders(dataset_path: str,
                       feature_config_path: str,
                       batch_size: int = 32,
                       val_split: float = 0.15,
                       test_split: float = 0.1,
                       num_workers: int = 0,
                       seed: int = 42) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/val/test dataloaders.
    
    Args:
        dataset_path: Path to encoder_dataset.jsonl
        feature_config_path: Path to feature_config.json
        batch_size: Batch size
        val_split: Validation split ratio
        test_split: Test split ratio
        num_workers: Number of workers for DataLoader
        seed: Random seed for splitting
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Load full dataset
    full_dataset = StateEncDataset(dataset_path, feature_config_path)
    
    # Split indices
    n = len(full_dataset)
    indices = list(range(n))
    
    np.random.seed(seed)
    np.random.shuffle(indices)
    
    test_size = int(n * test_split)
    val_size = int(n * val_split)
    train_size = n - test_size - val_size
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    # Create subset datasets
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
    test_dataset = torch.utils.data.Subset(full_dataset, test_indices)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test dataset loading
    dataset = StateEncDataset(
        "state_enc_v1/artifacts/encoder_dataset.jsonl",
        "state_enc_v1/artifacts/feature_config.json",
        max_samples=100
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"Sample X shape: {sample['X'].shape}")
        print(f"Future dir: {sample['future_dir_5']}")
        print(f"Future return: {sample['future_return_5']}")
        print(f"Regime hint: {sample['regime_hint']}")
