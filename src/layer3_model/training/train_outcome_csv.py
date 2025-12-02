import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import glob
import logging
import sys
import os

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training_csv_v4.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class CSVOutcomeDataset(Dataset):
    """PyTorch Dataset from CSV files"""

    def __init__(self, data_dir: str, context_len: int = 64, feature_cols: list = None):
        """
        Args:
            data_dir: Directory containing labeled CSV files
            context_len: Sequence length
            feature_cols: List of feature column names (if None, use all numeric except targets)
        """
        self.context_len = context_len
        
        # Load all CSVs
        files = sorted(glob.glob(os.path.join(data_dir, "*_labeled.csv")))
        logger.info(f"Loading {len(files)} CSV files from {data_dir}")
        
        dfs = []
        for f in files:
            try:
                df = pd.read_csv(f)
                # Sort by timestamp if needed, but usually they are time-ordered
                if 'timestamp' in df.columns:
                    df = df.sort_values('timestamp')
                dfs.append(df)
            except Exception as e:
                logger.error(f"Error loading {f}: {e}")
                
        if not dfs:
            raise ValueError("No data found!")
            
        # Concatenate (careful with boundaries between files - we should probably not cross file boundaries)
        # For simplicity, we'll process each file into sequences and then concat the sequences.
        
        self.sequences = []
        self.labels = []
        
        # Define feature columns
        if feature_cols is None:
            # Exclude non-features
            exclude = ['timestamp', 'bar_index', 'label', 'max_up_R', 'max_down_R', 'signal_type', 'open', 'high', 'low', 'close'] 
            # Note: 'close' etc might be features if normalized, but usually we use derived features.
            # Let's check what we have.
            # We'll use all float columns that are not excluded.
            sample_df = dfs[0]
            feature_cols = [c for c in sample_df.columns if c not in exclude and sample_df[c].dtype in [np.float64, np.float32, np.int64]]
            logger.info(f"Auto-detected {len(feature_cols)} features: {feature_cols}")
            
        self.feature_cols = feature_cols
        
    # Split into Train/Val indices first to avoid leakage
        # We need to know which indices belong to train/val
        # But random_split is done later.
        # To fix this properly, we should move split logic inside Dataset or pass a 'split' argument.
        # OR: We just compute mean/std on the first 80% of data (assuming time-ordered).
        
        # Since we concatenated DFS, they are likely time-ordered if file names are time-ordered.
        # smc_export_...20250901, 20250908... yes they are.
        
        full_df = pd.concat(dfs, ignore_index=True)
        
        # STRICT TIME SPLIT
        split_idx = int(len(full_df) * 0.8)
        
        # Compute stats on TRAIN ONLY
        train_df_stats = full_df.iloc[:split_idx]
        self.mean = train_df_stats[feature_cols].mean().values
        self.std = train_df_stats[feature_cols].std().values
        self.std[self.std < 1e-8] = 1.0
        
        logger.info(f"Normalization stats computed on first {split_idx} samples (Train Split)")
        
        # Load Masks
        mask_dir = r"c:\Users\Administrator\Desktop\modeloutcome\data\p2_masks"
        masks = []
        for f in files:
            basename = os.path.basename(f)
            mask_name = basename.replace("_labeled.csv", "_p2_mask.csv")
            mask_path = os.path.join(mask_dir, mask_name)
            
            logger.info(f"Looking for mask: {mask_path}")
            
            if os.path.exists(mask_path):
                mask_df = pd.read_csv(mask_path)
                mask_vals = mask_df['p2_mask'].astype(str).str.lower()
                mask_bools = mask_vals.isin(['true', '1', '1.0'])
                masks.append(mask_bools.values)
            else:
                # FAIL FAST
                raise FileNotFoundError(f"CRITICAL: Mask not found for {f}. Cannot proceed without P2 mask.")

        logger.info("Generating sequences (Filtered by P2)...")
        
        self.sequences = []
        self.labels = []
        self.timestamps = [] # Track timestamps for splitting
        
        # We need to reconstruct the full timeline to split correctly
        # But we process file by file.
        # Let's collect all sequences first, then split.
        # OR: Split the DFs first?
        # Splitting DFs is safer.
        
        # Actually, we have `full_df`.
        # Let's process `full_df` and `full_mask`.
        
        full_mask = np.concatenate(masks)
        
        if len(full_mask) != len(full_df):
            raise ValueError(f"Mask length {len(full_mask)} != Data length {len(full_df)}")
            
        # Normalize FULL dataset (using Train stats)
        # X_full = full_df[feature_cols].values
        # X_full = (X_full - self.mean) / self.std
        # y_full = full_df['label'].values
        
        # Generate Sequences
        # We need to respect file boundaries? 
        # If we concatenated, we might cross boundaries.
        # Ideally we process per file.
        # But for simplicity and since we want a global split, let's assume continuity or accept minor boundary jump.
        # Better: Process per file, collect all, then split.
        
        current_idx = 0
        for df, mask in zip(dfs, masks):
            # Normalize this chunk
            X = df[feature_cols].values
            X = (X - self.mean) / self.std
            y = df['label'].values
            
            num_samples = len(df) - context_len + 1
            if num_samples <= 0: continue
            
            for i in range(num_samples):
                end_idx = i + context_len
                
                # Check P2 mask at target
                if not mask[end_idx - 1]:
                    continue
                    
                seq = X[i : end_idx]
                label = y[end_idx - 1]
                
                self.sequences.append(seq)
                self.labels.append(label)
                # We track the global index of the target bar to split by time
                # Global index = current_idx + end_idx - 1
                self.timestamps.append(current_idx + end_idx - 1)
                
            current_idx += len(df)
            
        self.sequences = np.array(self.sequences, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.int64) + 1 
        self.timestamps = np.array(self.timestamps)
        
        # Time Split Logic
        # We want to split based on the original time (index).
        # Split point is `split_idx` (calculated from full_df).
        # Train = sequences where target_index < split_idx
        # Val = sequences where target_index >= split_idx
        
        train_mask = self.timestamps < split_idx
        val_mask = self.timestamps >= split_idx
        
        self.train_sequences = self.sequences[train_mask]
        self.train_labels = self.labels[train_mask]
        
        self.val_sequences = self.sequences[val_mask]
        self.val_labels = self.labels[val_mask]
        
        logger.info(f"Total Sequences: {len(self.sequences)}")
        logger.info(f"Train Split: {len(self.train_sequences)} (Target < {split_idx})")
        logger.info(f"Val Split: {len(self.val_sequences)} (Target >= {split_idx})")
        
        # Class weights (from Train only)
        counts = np.bincount(self.train_labels)
        logger.info(f"Train Class counts: Short={counts[0]}, Neutral={counts[1]}, Long={counts[2]}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Default to full dataset (not used directly if we use custom loaders)
        return torch.from_numpy(self.sequences[idx]), self.labels[idx]
        
    def get_train_val_datasets(self):
        """Return separate TensorDatasets for Train and Val"""
        from torch.utils.data import TensorDataset
        
        train_ds = TensorDataset(
            torch.from_numpy(self.train_sequences),
            torch.from_numpy(self.train_labels)
        )
        
        val_ds = TensorDataset(
            torch.from_numpy(self.val_sequences),
            torch.from_numpy(self.val_labels)
        )
        
        return train_ds, val_ds

class SimpleGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        # x: [batch, seq, feature]
        out, _ = self.gru(x)
        # Take last step
        out = out[:, -1, :]
        out = self.fc(out)
        return out

def train():
    # Config
    DATA_DIR = r"c:\Users\Administrator\Desktop\modeloutcome\data\labeled_outcome"
    CONTEXT_LEN = 64
    BATCH_SIZE = 128
    EPOCHS = 40
    LR = 0.001
    
    device = torch.device('cpu')
    logger.info(f"Using device: {device}")
    
    # Selected Features (Top 30 from XGBoost v2, excluding absolute prices)
    SELECTED_FEATURES = [
        'bars_since_ext_swing_high', 'ext_swing_high_distance', 'h1_trend_up', 'dist_to_h1_swing_high',
        'vp_dist_to_val', 'bars_since_ext_swing_low', 'ext_swing_low_distance', 'vp_dist_to_poc',
        'ext_in_bear_ob', 'dist_to_vwap', 'ext_trend_dir', 'int_trend_dir',
        'vp_dist_to_vah', 'atr', 'dist_to_h1_swing_low', 'int_swing_low_distance',
        'nearest_fvg_size', 'in_bear_fvg', 'cum_delta_20', 'tick_speed',
        'h1_premium', 'h1_trend_down', 'dist_to_nearest_ob', 'near_bear_fvg',
        'cum_delta_10', 'bars_since_int_swing_high', 'near_h1_fvg', 'impulse_strength',
        'int_swing_high_distance', 'int_in_bull_ob'
    ]
    
    # Dataset
    dataset = CSVOutcomeDataset(DATA_DIR, context_len=CONTEXT_LEN, feature_cols=SELECTED_FEATURES)
    
    # Get Time-Split Datasets
    train_dataset, val_dataset = dataset.get_train_val_datasets()
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Model
    input_dim = dataset.sequences.shape[2]
    # v2 Config: Dropout 0.1
    model = SimpleGRU(input_dim=input_dim, hidden_dim=128, num_layers=2, num_classes=3, dropout=0.1).to(device)
    
    # Weights for imbalance
    # Short (~25%), Neutral (~50%), Long (~25%)
    # We want to encourage trading, so we penalize missing Short/Long more.
    # Weights: Short=2.0, Neutral=1.0, Long=2.0
    weights = torch.tensor([2.0, 1.0, 2.0]).to(device)
    
    criterion = nn.CrossEntropyLoss(weight=weights)
    # v2 Config: AdamW, Weight Decay 1e-4
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    
    logger.info(f"Starting training loop for {EPOCHS} epochs...")
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            
            optimizer.zero_grad()
            out = model(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
            
        val_acc = validate(model, val_loader, device)
        logger.info(f"Epoch {epoch+1}: Train Loss={total_loss/len(train_loader):.4f}, Train Acc={correct/total:.4f}, Val Acc={val_acc:.4f}")
        
    # Save
    torch.save(model.state_dict(), "models/outcome_gru_v4.pt")
    logger.info("Model saved to models/outcome_gru_v4.pt")

def validate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            out = model(X)
            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total

if __name__ == "__main__":
    train()
