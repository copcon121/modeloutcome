#!/usr/bin/env python3
"""
ASM Dataset Builder v1
======================
Build Auction State Model dataset with VA-shift labels.

Reference: PLAN_AuctionStateModel_v1.md

Label Logic:
- UP: VA shifts up significantly AND price holds above old VAH
- DOWN: VA shifts down significantly AND price holds below old VAL  
- NEUTRAL: VA stable or breakout failed

Usage:
    python scripts/build_asm_dataset_v1.py
"""

import os
import json
import glob
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Label parameters (from PLAN_AuctionStateModel_v1.md Section 2.2)
K = 30                      # Lookahead bars for VA shift
VA_SHIFT_THRESHOLD = 10     # Ticks to consider VA shift significant
BREAKOUT_HOLD_RATIO = 0.6   # Min ratio of bars holding breakout

# Context parameters
CONTEXT_LENGTH = 60         # Bars of history for context

# Train/Val split
TRAIN_RATIO = 0.7           # 70% train, 30% val (time-based)

# Data paths
DATA_PATHS = [
    "data/processed_v2/*.csv",           # 10 weeks training data
    "output/new_data_features/smc_export_*.csv",  # 6 weeks validation data
]

# Output paths
OUTPUT_DIR = Path("output/asm_dataset_v1")

# VA column mapping
VA_COLS = {
    "vah": "vp_vah_price",
    "val": "vp_val_price", 
    "poc": "vp_poc_price",
}

# Label encoding
LABEL_MAP = {"UP": 0, "DOWN": 1, "NEUTRAL": 2}
LABEL_NAMES = {0: "UP", 1: "DOWN", 2: "NEUTRAL"}

# ==============================================================================
# FEATURE COLUMNS
# ==============================================================================

# Features to use for context (exclude timestamp, bar_index, open, high, low, close for raw)
# We'll dynamically detect numeric columns
EXCLUDE_COLS = ["timestamp", "bar_index"]

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def load_all_data(data_paths: List[str]) -> pd.DataFrame:
    """Load and concatenate all CSV files, sorted by timestamp."""
    all_files = []
    for pattern in data_paths:
        files = glob.glob(pattern)
        all_files.extend(files)
    
    if not all_files:
        raise FileNotFoundError(f"No CSV files found in: {data_paths}")
    
    print(f"Found {len(all_files)} CSV files")
    
    dfs = []
    for f in sorted(all_files):
        df = pd.read_csv(f)
        df["_source_file"] = os.path.basename(f)
        dfs.append(df)
        print(f"  Loaded {f}: {len(df)} rows")
    
    combined = pd.concat(dfs, ignore_index=True)
    
    # Parse timestamp and sort
    combined["timestamp"] = pd.to_datetime(combined["timestamp"])
    combined = combined.sort_values("timestamp").reset_index(drop=True)
    
    print(f"Total rows after combining: {len(combined)}")
    return combined


def check_va_columns(df: pd.DataFrame) -> bool:
    """Check if VA columns exist in dataframe."""
    missing = []
    for name, col in VA_COLS.items():
        if col not in df.columns:
            missing.append(col)
    
    if missing:
        print(f"WARNING: Missing VA columns: {missing}")
        print("TODO: VA features need to be added to feature engine")
        return False
    return True


def compute_va_shift_label(
    df: pd.DataFrame,
    t: int,
    k: int = K,
    va_shift_threshold: float = VA_SHIFT_THRESHOLD,
    breakout_hold_ratio: float = BREAKOUT_HOLD_RATIO,
) -> Optional[str]:
    """
    Compute VA shift label for bar t.
    
    Returns:
        "UP", "DOWN", "NEUTRAL", or None if cannot compute (boundary)
    """
    # Check boundaries
    if t + k >= len(df):
        return None  # Not enough future data
    
    # Get VA at time t
    vah_t = df.iloc[t][VA_COLS["vah"]]
    val_t = df.iloc[t][VA_COLS["val"]]
    va_center_t = (vah_t + val_t) / 2
    
    # Get VA at time t+K
    vah_t_k = df.iloc[t + k][VA_COLS["vah"]]
    val_t_k = df.iloc[t + k][VA_COLS["val"]]
    va_center_t_k = (vah_t_k + val_t_k) / 2
    
    # Calculate VA shift
    va_shift = va_center_t_k - va_center_t
    
    # Count bars outside VA in direction (t+1 to t+K inclusive)
    future_closes = df.iloc[t + 1 : t + k + 1]["close"].values
    bars_above_vah = np.sum(future_closes > vah_t)
    bars_below_val = np.sum(future_closes < val_t)
    
    # Label assignment (following PLAN exactly)
    if va_shift >= va_shift_threshold:
        if bars_above_vah / k >= breakout_hold_ratio:
            return "UP"
        else:
            return "NEUTRAL"  # Breakout failed
    elif va_shift <= -va_shift_threshold:
        if bars_below_val / k >= breakout_hold_ratio:
            return "DOWN"
        else:
            return "NEUTRAL"  # Breakout failed
    else:
        return "NEUTRAL"  # No significant VA shift


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Get list of feature columns (numeric, excluding metadata)."""
    feature_cols = []
    for col in df.columns:
        if col in EXCLUDE_COLS:
            continue
        if col.startswith("_"):
            continue
        if df[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
            feature_cols.append(col)
    return feature_cols


def detect_session(timestamp: pd.Timestamp) -> str:
    """Detect trading session from timestamp (UTC assumed)."""
    hour = timestamp.hour
    if 0 <= hour < 8:
        return "Asia"
    elif 8 <= hour < 14:
        return "London"
    else:
        return "NY"


def build_dataset(
    df: pd.DataFrame,
    feature_cols: List[str],
    context_length: int = CONTEXT_LENGTH,
    k: int = K,
) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
    """
    Build ASM dataset with context sequences and labels.
    
    Returns:
        X: (N, context_length, num_features) - context sequences
        y: (N,) - labels (0=UP, 1=DOWN, 2=NEUTRAL)
        meta: list of dicts with metadata per sample
    """
    # Valid range: need context_length bars before and k bars after
    start_idx = context_length
    end_idx = len(df) - k
    total_bars = end_idx - start_idx
    
    print(f"Building samples from index {start_idx} to {end_idx} ({total_bars} bars)...")
    print(f"  Context length: {context_length}, Lookahead K: {k}")
    
    # Pre-extract numpy arrays for speed
    feature_data = df[feature_cols].values.astype(np.float32)
    close_data = df["close"].values
    vah_data = df[VA_COLS["vah"]].values
    val_data = df[VA_COLS["val"]].values
    timestamps = df["timestamp"].values
    bar_indices = df["bar_index"].values if "bar_index" in df.columns else np.arange(len(df))
    source_files = df["_source_file"].values if "_source_file" in df.columns else np.array(["unknown"] * len(df))
    
    # Pre-allocate arrays (estimate max size)
    X_list = []
    y_list = []
    meta_list = []
    
    skipped_nan = 0
    label_counts = {"UP": 0, "DOWN": 0, "NEUTRAL": 0}
    
    # Progress tracking
    progress_interval = max(1, total_bars // 20)  # 5% intervals
    
    for i, t in enumerate(range(start_idx, end_idx)):
        # Progress
        if i % progress_interval == 0:
            pct = 100 * i / total_bars
            print(f"  Progress: {pct:.0f}% ({i}/{total_bars})")
        
        # Get VA at time t
        vah_t = vah_data[t]
        val_t = val_data[t]
        va_center_t = (vah_t + val_t) / 2
        
        # Get VA at time t+K
        vah_t_k = vah_data[t + k]
        val_t_k = val_data[t + k]
        va_center_t_k = (vah_t_k + val_t_k) / 2
        
        # Calculate VA shift
        va_shift = va_center_t_k - va_center_t
        
        # Count bars outside VA in direction (t+1 to t+K inclusive)
        future_closes = close_data[t + 1 : t + k + 1]
        bars_above_vah = np.sum(future_closes > vah_t)
        bars_below_val = np.sum(future_closes < val_t)
        
        # Label assignment (following PLAN exactly)
        if va_shift >= VA_SHIFT_THRESHOLD:
            if bars_above_vah / k >= BREAKOUT_HOLD_RATIO:
                label = "UP"
            else:
                label = "NEUTRAL"
        elif va_shift <= -VA_SHIFT_THRESHOLD:
            if bars_below_val / k >= BREAKOUT_HOLD_RATIO:
                label = "DOWN"
            else:
                label = "NEUTRAL"
        else:
            label = "NEUTRAL"
        
        # Extract context (past data only: t-context_length to t-1)
        context = feature_data[t - context_length : t]
        
        # Check for NaN in context
        if np.isnan(context).any():
            skipped_nan += 1
            continue
        
        # Build metadata
        ts = pd.Timestamp(timestamps[t])
        meta = {
            "timestamp": str(ts),
            "bar_index": int(bar_indices[t]),
            "session": detect_session(ts),
            "source_file": str(source_files[t]),
            "va_center": float(va_center_t),
        }
        
        X_list.append(context)
        y_list.append(LABEL_MAP[label])
        meta_list.append(meta)
        label_counts[label] += 1
    
    print(f"  Progress: 100% ({total_bars}/{total_bars})")
    print(f"  Skipped (NaN): {skipped_nan}")
    print(f"  Label counts: {label_counts}")
    print(f"  Total samples: {len(X_list)}")
    
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)
    
    return X, y, meta_list


def time_based_split(
    X: np.ndarray,
    y: np.ndarray,
    meta: List[Dict],
    train_ratio: float = TRAIN_RATIO,
) -> Tuple[np.ndarray, np.ndarray, List[Dict], np.ndarray, np.ndarray, List[Dict]]:
    """Split data by time (first train_ratio for train, rest for val)."""
    n = len(X)
    split_idx = int(n * train_ratio)
    
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    meta_train, meta_val = meta[:split_idx], meta[split_idx:]
    
    return X_train, y_train, meta_train, X_val, y_val, meta_val


def compute_stats(
    y_train: np.ndarray,
    y_val: np.ndarray,
    meta_train: List[Dict],
    meta_val: List[Dict],
    feature_cols: List[str],
) -> Dict:
    """Compute dataset statistics."""
    
    def label_dist(y):
        unique, counts = np.unique(y, return_counts=True)
        dist = {LABEL_NAMES[int(u)]: int(c) for u, c in zip(unique, counts)}
        return dist
    
    def label_pct(y):
        unique, counts = np.unique(y, return_counts=True)
        total = len(y)
        pct = {LABEL_NAMES[int(u)]: round(100 * c / total, 2) for u, c in zip(unique, counts)}
        return pct
    
    def session_dist(meta):
        sessions = [m["session"] for m in meta]
        unique, counts = np.unique(sessions, return_counts=True)
        return {str(u): int(c) for u, c in zip(unique, counts)}
    
    def time_range(meta):
        timestamps = [m["timestamp"] for m in meta]
        return {"start": min(timestamps), "end": max(timestamps)}
    
    stats = {
        "created_at": datetime.now().isoformat(),
        "parameters": {
            "K": K,
            "VA_SHIFT_THRESHOLD": VA_SHIFT_THRESHOLD,
            "BREAKOUT_HOLD_RATIO": BREAKOUT_HOLD_RATIO,
            "CONTEXT_LENGTH": CONTEXT_LENGTH,
            "TRAIN_RATIO": TRAIN_RATIO,
        },
        "train": {
            "n_samples": len(y_train),
            "label_distribution": label_dist(y_train),
            "label_percentage": label_pct(y_train),
            "session_distribution": session_dist(meta_train),
            "time_range": time_range(meta_train),
        },
        "val": {
            "n_samples": len(y_val),
            "label_distribution": label_dist(y_val),
            "label_percentage": label_pct(y_val),
            "session_distribution": session_dist(meta_val),
            "time_range": time_range(meta_val),
        },
        "features": {
            "n_features": len(feature_cols),
            "feature_names": feature_cols,
        },
        "notes": {
            "imbalance": "NEUTRAL dominates (~95%). Use class weights during training.",
            "nan_handling": "Samples with NaN in context are skipped. Consider fillna for more samples.",
        },
    }
    
    return stats


def save_dataset(
    X_train: np.ndarray,
    y_train: np.ndarray,
    meta_train: List[Dict],
    X_val: np.ndarray,
    y_val: np.ndarray,
    meta_val: List[Dict],
    stats: Dict,
    output_dir: Path,
):
    """Save dataset to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save PyTorch tensors
    train_path = output_dir / "asm_dataset_v1_train.pt"
    val_path = output_dir / "asm_dataset_v1_val.pt"
    stats_path = output_dir / "asm_dataset_v1_stats.json"
    
    torch.save({
        "X": torch.from_numpy(X_train),
        "y": torch.from_numpy(y_train),
        "meta": meta_train,
    }, train_path)
    print(f"Saved train dataset: {train_path}")
    
    torch.save({
        "X": torch.from_numpy(X_val),
        "y": torch.from_numpy(y_val),
        "meta": meta_val,
    }, val_path)
    print(f"Saved val dataset: {val_path}")
    
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved stats: {stats_path}")


def print_sample_examples(
    X: np.ndarray,
    y: np.ndarray,
    meta: List[Dict],
    n_examples: int = 5,
    split_name: str = "train",
):
    """Print example samples for sanity check."""
    print(f"\n{'='*60}")
    print(f"Sample Examples ({split_name}, first {n_examples}):")
    print(f"{'='*60}")
    
    for i in range(min(n_examples, len(X))):
        label_name = LABEL_NAMES[y[i]]
        m = meta[i]
        print(f"  [{i}] Label: {label_name:8s} | ts: {m['timestamp']} | session: {m['session']:7s} | file: {m['source_file']}")


def sanity_check(
    X_train: np.ndarray,
    y_train: np.ndarray,
    meta_train: List[Dict],
    X_val: np.ndarray,
    y_val: np.ndarray,
    meta_val: List[Dict],
):
    """Run sanity checks on the dataset."""
    print(f"\n{'='*60}")
    print("Sanity Checks:")
    print(f"{'='*60}")
    
    # Check shapes
    print(f"  Train X shape: {X_train.shape}")
    print(f"  Train y shape: {y_train.shape}")
    print(f"  Val X shape:   {X_val.shape}")
    print(f"  Val y shape:   {y_val.shape}")
    
    # Check no NaN
    train_nan = np.isnan(X_train).sum()
    val_nan = np.isnan(X_val).sum()
    print(f"  Train NaN count: {train_nan}")
    print(f"  Val NaN count:   {val_nan}")
    
    # Check time ordering (val should be after train)
    train_end = meta_train[-1]["timestamp"]
    val_start = meta_val[0]["timestamp"]
    print(f"  Train end time: {train_end}")
    print(f"  Val start time: {val_start}")
    print(f"  Time order OK:  {val_start >= train_end}")
    
    # Check label distribution
    for name, y in [("Train", y_train), ("Val", y_val)]:
        unique, counts = np.unique(y, return_counts=True)
        total = len(y)
        dist_str = ", ".join([f"{LABEL_NAMES[u]}: {c} ({100*c/total:.1f}%)" for u, c in zip(unique, counts)])
        print(f"  {name} labels: {dist_str}")


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    print("="*60)
    print("ASM Dataset Builder v1")
    print("="*60)
    print(f"Parameters:")
    print(f"  K (lookahead):          {K}")
    print(f"  VA_SHIFT_THRESHOLD:     {VA_SHIFT_THRESHOLD}")
    print(f"  BREAKOUT_HOLD_RATIO:    {BREAKOUT_HOLD_RATIO}")
    print(f"  CONTEXT_LENGTH:         {CONTEXT_LENGTH}")
    print(f"  TRAIN_RATIO:            {TRAIN_RATIO}")
    print()
    
    # Load data
    print("Loading data...")
    df = load_all_data(DATA_PATHS)
    
    # Check VA columns
    has_va = check_va_columns(df)
    if not has_va:
        print("\nERROR: Cannot build ASM dataset without VA columns.")
        print("Please ensure feature engine exports: vp_vah_price, vp_val_price, vp_poc_price")
        return
    
    # Get feature columns
    feature_cols = get_feature_columns(df)
    print(f"\nUsing {len(feature_cols)} feature columns")
    
    # Build dataset
    print("\nBuilding dataset...")
    X, y, meta = build_dataset(df, feature_cols)
    
    if len(X) == 0:
        print("ERROR: No samples generated!")
        return
    
    # Time-based split
    print("\nSplitting train/val (time-based)...")
    X_train, y_train, meta_train, X_val, y_val, meta_val = time_based_split(X, y, meta)
    
    # Compute stats
    stats = compute_stats(y_train, y_val, meta_train, meta_val, feature_cols)
    
    # Sanity checks
    sanity_check(X_train, y_train, meta_train, X_val, y_val, meta_val)
    
    # Print examples
    print_sample_examples(X_train, y_train, meta_train, n_examples=5, split_name="train")
    print_sample_examples(X_val, y_val, meta_val, n_examples=5, split_name="val")
    
    # Save
    print("\nSaving dataset...")
    save_dataset(X_train, y_train, meta_train, X_val, y_val, meta_val, stats, OUTPUT_DIR)
    
    # Final summary
    print(f"\n{'='*60}")
    print("DONE!")
    print(f"{'='*60}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Train samples:    {len(X_train)}")
    print(f"Val samples:      {len(X_val)}")
    print(f"Feature dim:      {X_train.shape[2]}")
    print(f"Context length:   {X_train.shape[1]}")


if __name__ == "__main__":
    main()
