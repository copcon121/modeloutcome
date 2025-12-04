"""
PyTorch Dataset for ASM v2

Loads ASM dataset (z_t + meta + label) for training.
"""

import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple
from pathlib import Path


class AsmDataset(Dataset):
    """PyTorch Dataset for ASM v2 training"""
    
    def __init__(self,
                 dataset_path: str,
                 indices: Optional[List[int]] = None,
                 z_dim: int = 64,
                 meta_features: List[str] = None):
        self.dataset_path = dataset_path
        self.z_dim = z_dim
        self.meta_features = meta_features or [
            "session_id", "pos_in_session_range", "inside_value",
            "above_value", "below_value", "minute_of_day_norm"
        ]
        
        # Load all samples
        self.samples = self._load_samples(dataset_path)
        
        # Filter by indices if provided
        if indices is not None:
            self.samples = [self.samples[i] for i in indices]
    
    def _load_samples(self, path: str) -> List[Dict]:
        samples = []
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        
        # z_t embedding
        z_t = torch.tensor(sample["z_t"], dtype=torch.float32)
        
        # Meta features
        meta = sample["meta"]
        meta_vec = torch.tensor([
            meta.get("session_id", 0),
            meta.get("pos_in_session_range", 0.5),
            meta.get("inside_value", 0),
            meta.get("above_value", 0),
            meta.get("below_value", 0),
            meta.get("minute_of_day_norm", 0.0)
        ], dtype=torch.float32)
        
        # Concatenate z_t + meta
        x = torch.cat([z_t, meta_vec], dim=0)
        
        # Label
        label = torch.tensor(sample["label_regime"], dtype=torch.long)
        
        return {
            "x": x,
            "label": label,
            "z_t": z_t,
            "meta": meta_vec
        }


def asm_collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate function for ASM DataLoader"""
    return {
        "x": torch.stack([s["x"] for s in batch]),
        "label": torch.stack([s["label"] for s in batch]),
        "z_t": torch.stack([s["z_t"] for s in batch]),
        "meta": torch.stack([s["meta"] for s in batch])
    }


def create_asm_dataloaders(
    dataset_path: str,
    splits_path: str,
    batch_size: int = 64,
    num_workers: int = 0,
    z_dim: int = 64
) -> Tuple[DataLoader, DataLoader]:
    """Create train/val dataloaders for ASM"""
    
    # Load splits
    with open(splits_path, "r") as f:
        splits = json.load(f)
    
    train_indices = splits["train_indices"]
    val_indices = splits["val_indices"]
    
    # Create datasets
    train_dataset = AsmDataset(dataset_path, indices=train_indices, z_dim=z_dim)
    val_dataset = AsmDataset(dataset_path, indices=val_indices, z_dim=z_dim)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=asm_collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=asm_collate_fn,
        pin_memory=True
    )
    
    return train_loader, val_loader
