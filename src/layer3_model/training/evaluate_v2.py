import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import logging
import sys
import os

# Import the dataset and model class from the training script
# We need to make sure we can import them. 
# Since train_outcome_csv.py is a script, importing might be tricky if not structured as a module.
# I will copy the minimal necessary classes here to be safe and avoid side effects of importing a script.

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes, dropout=0.1):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

def load_data_and_split(data_dir, context_len=64):
    import glob
    files = sorted(glob.glob(os.path.join(data_dir, "*_labeled.csv")))
    
    dfs = []
    masks = []
    mask_dir = r"c:\Users\Administrator\Desktop\modeloutcome\data\p2_masks"
    
    for f in files:
        df = pd.read_csv(f)
        if 'timestamp' in df.columns:
            df = df.sort_values('timestamp')
        dfs.append(df)
        
        basename = os.path.basename(f)
        mask_name = basename.replace("_labeled.csv", "_p2_mask.csv")
        mask_path = os.path.join(mask_dir, mask_name)
        
        if os.path.exists(mask_path):
            mask_df = pd.read_csv(mask_path)
            mask_vals = mask_df['p2_mask'].astype(str).str.lower()
            mask_bools = mask_vals.isin(['true', '1', '1.0'])
            masks.append(mask_bools.values)
        else:
            raise FileNotFoundError(f"Mask not found: {mask_path}")

    full_df = pd.concat(dfs, ignore_index=True)
    
    # Feature Cols (Auto-detect logic from train script)
    exclude = ['timestamp', 'bar_index', 'label', 'max_up_R', 'max_down_R', 'signal_type', 'open', 'high', 'low', 'close'] 
    sample_df = dfs[0]
    feature_cols = [c for c in sample_df.columns if c not in exclude and sample_df[c].dtype in [np.float64, np.float32, np.int64]]
    
    # Split Index
    split_idx = int(len(full_df) * 0.8)
    
    # Normalization Stats (Train Only)
    train_df_stats = full_df.iloc[:split_idx]
    mean = train_df_stats[feature_cols].mean().values
    std = train_df_stats[feature_cols].std().values
    std[std < 1e-8] = 1.0
    
    sequences = []
    labels = []
    timestamps = []
    
    current_idx = 0
    for df, mask in zip(dfs, masks):
        X = df[feature_cols].values
        X = (X - mean) / std
        y = df['label'].values
        
        num_samples = len(df) - context_len + 1
        if num_samples <= 0: continue
        
        for i in range(num_samples):
            end_idx = i + context_len
            if not mask[end_idx - 1]: continue
            
            seq = X[i : end_idx]
            label = y[end_idx - 1]
            sequences.append(seq)
            labels.append(label)
            timestamps.append(current_idx + end_idx - 1)
            
        current_idx += len(df)
        
    sequences = np.array(sequences, dtype=np.float32)
    labels = np.array(labels, dtype=np.int64) + 1 # -1,0,1 -> 0,1,2
    timestamps = np.array(timestamps)
    
    # Val Split
    val_mask = timestamps >= split_idx
    X_val = sequences[val_mask]
    y_val = labels[val_mask]
    
    return X_val, y_val, feature_cols

def evaluate():
    DATA_DIR = r"c:\Users\Administrator\Desktop\modeloutcome\data\labeled_outcome"
    MODEL_PATH = "models/outcome_gru_v3.pt"
    
    logger.info("Loading Validation Data...")
    X_val, y_val, feature_cols = load_data_and_split(DATA_DIR)
    logger.info(f"Validation Samples: {len(y_val)}")
    
    device = torch.device('cpu')
    input_dim = X_val.shape[2]
    
    model = SimpleGRU(input_dim=input_dim, hidden_dim=128, num_layers=2, num_classes=3, dropout=0.1)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    
    logger.info("Running Inference...")
    with torch.no_grad():
        X_tensor = torch.from_numpy(X_val).to(device)
        out = model(X_tensor)
        probs = torch.softmax(out, dim=1)
        preds = out.argmax(dim=1).cpu().numpy()
        
    # Metrics
    target_names = ['Short', 'Neutral', 'Long']
    print("\nClassification Report:")
    print(classification_report(y_val, preds, target_names=target_names))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_val, preds)
    print(cm)
    
    # Class Distribution
    unique, counts = np.unique(y_val, return_counts=True)
    dist = dict(zip(unique, counts))
    total = len(y_val)
    print("\nValidation Class Distribution:")
    for k, v in dist.items():
        print(f"{target_names[k]}: {v} ({v/total:.2%})")

    # --- Expectancy & Drawdown Analysis ---
    print("\n--- Trading Simulation (2R Target / 1R Stop) ---")
    
    # Logic:
    # Pred Short (0): If Label=0 (Win, +2R), else (Loss, -1R)
    # Pred Neutral (1): 0R
    # Pred Long (2): If Label=2 (Win, +2R), else (Loss, -1R)
    
    r_results = []
    wins = 0
    losses = 0
    trades = 0
    
    for p, y in zip(preds, y_val):
        if p == 1: # Neutral / Skip
            continue
            
        trades += 1
        if p == y:
            r = 2.0
            wins += 1
        else:
            r = -1.0
            losses += 1
        r_results.append(r)
        
    if trades > 0:
        total_r = sum(r_results)
        expectancy = total_r / trades
        win_rate = wins / trades
        
        print(f"Total Trades: {trades}")
        print(f"Win Rate: {win_rate:.2%} ({wins}/{trades})")
        print(f"Total Return: {total_r:.2f}R")
        print(f"Expectancy: {expectancy:.4f}R per trade")
        
        # Drawdown
        equity_curve = np.cumsum(r_results)
        peak = np.maximum.accumulate(equity_curve)
        drawdown = peak - equity_curve
        max_dd = np.max(drawdown)
        print(f"Max Drawdown: {max_dd:.2f}R")
        
        # Quality Check
        if expectancy > 0.5:
            print("Rating: PREMIUM (>0.5R)")
        elif expectancy > 0.4:
            print("Rating: VERY GOOD (>0.4R)")
        elif expectancy > 0.2:
            print("Rating: TRADEABLE (>0.2R)")
        else:
            print("Rating: NOT TRADEABLE (<0.2R)")
            
    else:
        print("No trades taken (Model predicted Neutral for all samples).")


if __name__ == "__main__":
    evaluate()
