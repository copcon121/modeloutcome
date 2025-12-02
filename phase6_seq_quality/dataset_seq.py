"""
Sequence Dataset Wrapper

Loads Phase 4/5 datasets and returns full [60, D] sequences.
"""

from pathlib import Path
import torch
from torch.utils.data import Dataset
import numpy as np


class QualitySequenceDataset(Dataset):
    """
    Dataset for sequence-based quality model
    
    Returns:
        - X_seq: [60, D] time-series
        - side: [1] float (+1 or -1)
        - y: [1] long (0 or 1)
    """
    
    def __init__(self, dataset_path, normalizer_stats=None):
        """
        Args:
            dataset_path: Path to Phase 4/5 .pt dataset file
            normalizer_stats: dict with 'mean' and 'std' tensors [D]
        """
        # Load dataset
        data = torch.load(dataset_path)
        
        self.X = data['X']  # [N, 60, D]
        self.side = data['side']  # [N]
        self.y = data['y_quality']  # [N]
        
        # Store normalizer
        self.normalizer_stats = normalizer_stats
        
        print(f"[QualitySequenceDataset] Loaded {len(self)} samples from {Path(dataset_path).name}")
        print(f"  X shape: {self.X.shape}")
        print(f"  Labels: KEEP={((self.y==1).sum())} DROP={((self.y==0).sum())}")
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        X_seq = self.X[idx]  # [60, D]
        side_val = self.side[idx]  # scalar
        y_val = self.y[idx]  # scalar
        
        # Normalize if stats provided
        if self.normalizer_stats is not None:
            mean = self.normalizer_stats['mean']  # [D]
            std = self.normalizer_stats['std']  # [D]
            
            # Normalize across time dimension
            # X_seq: [60, D], mean/std: [D]
            X_seq = (X_seq - mean) / (std + 1e-8)
        
        # Convert side to float
        side_tensor = torch.FloatTensor([float(side_val)])
        
        # Return dict
        return {
            'X_seq': X_seq.float(),  # [60, D]
            'side': side_tensor,  # [1]
            'y': y_val.long()  # scalar
        }


def compute_normalizer_stats(train_dataset_path):
    """
    Compute per-feature mean and std from training set
    
    Args:
        train_dataset_path: Path to train .pt file
    
    Returns:
        dict with 'mean' [D] and 'std' [D]
    """
    print(f"\n[Normalizer] Computing stats from train set...")
    
    # Load train data
    data = torch.load(train_dataset_path)
    X_train = data['X']  # [N, 60, D]
    
    N, T, D = X_train.shape
    print(f"  Train shape: {X_train.shape}")
    
    # Compute mean and std over samples and time
    # Reshape to [N*T, D]
    X_flat = X_train.reshape(N * T, D)
    
    mean = X_flat.mean(dim=0)  # [D]
    std = X_flat.std(dim=0)  # [D]
    
    # Avoid division by zero
    std = torch.clamp(std, min=1e-8)
    
    print(f"  Mean shape: {mean.shape}, Std shape: {std.shape}")
    print(f"  Mean range: [{mean.min():.4f}, {mean.max():.4f}]")
    print(f"  Std range: [{std.min():.4f}, {std.max():.4f}]")
    
    return {
        'mean': mean,
        'std': std
    }


def create_dataloaders(train_path, val_path, normalizer_stats, batch_size=64):
    """
    Create train and val dataloaders
    
    Args:
        train_path: Path to train .pt
        val_path: Path to val .pt
        normalizer_stats: Normalizer dict
        batch_size: Batch size
    
    Returns:
        train_loader, val_loader
    """
    from torch.utils.data import DataLoader
    
    train_dataset = QualitySequenceDataset(train_path, normalizer_stats)
    val_dataset = QualitySequenceDataset(val_path, normalizer_stats)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    return train_loader, val_loader


if __name__ == "__main__":
    # Test dataset loading
    from pathlib import Path
    
    root = Path(__file__).parent.parent
    train_path = root / "output/phase4_quality/dataset_p2_quality_v1_train.pt"
    
    print("="*60)
    print("TESTING SEQUENCE DATASET")
    print("="*60)
    
    # Compute normalizer
    normalizer = compute_normalizer_stats(train_path)
    
    # Create dataset
    dataset = QualitySequenceDataset(train_path, normalizer)
    
    # Test sample
    sample = dataset[0]
    print(f"\nSample 0:")
    print(f"  X_seq: {sample['X_seq'].shape}")
    print(f"  side: {sample['side']}")
    print(f"  y: {sample['y']}")
    
    print(f"\nDataset test complete!")
