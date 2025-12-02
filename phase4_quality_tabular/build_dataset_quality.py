"""
Build Quality Dataset for Phase 4

Loads Phase 3 labeled events + sequences, creates quality labels (KEEP/DROP),
and builds PyTorch datasets for training.

Output:
- events_p2_labeled_quality_v1.jsonl (enriched with quality labels)
- dataset_p2_quality_v1.pt (full dataset)
- dataset_p2_quality_v1_train.pt (80% train)
- dataset_p2_quality_v1_val.pt (20% val)
"""

import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from collections import Counter

print("="*80)
print("PHASE 4: BUILD QUALITY DATASET v1")
print("="*80)

# Paths (env-overridable)
import os
ROOT = Path(__file__).parent.parent
PHASE3_DIR = Path(os.getenv("PHASE3_DIR_OVERRIDE", ROOT / "output/phase3_labeled"))
PHASE2_DIR = Path(os.getenv("PHASE2_DIR_OVERRIDE", ROOT / "output/production_10weeks_v2"))
OUTPUT_DIR = Path(os.getenv("PHASE4_OUTPUT_DIR_OVERRIDE", ROOT / "output/phase4_quality"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load Phase 3 labeled events
events_path = PHASE3_DIR / "events_p2_labeled_10weeks.jsonl"
print(f"\n[1/5] Loading Phase 3 events: {events_path}")

events = []
with open(events_path, 'r') as f:
    for line in f:
        events.append(json.loads(line))

print(f"  Loaded {len(events):,} events")

# Load ALL features for metadata (symbol, session, SMC context)
# We will index with the global bar indices stored alongside sequences
features_path = PHASE2_DIR / "features_all_10weeks.csv"
print(f"\n[2/5] Loading features: {features_path}")
features_df = pd.read_csv(features_path)
print(f"  Features shape: {features_df.shape}")

# Extract symbol_root from filename pattern if exists, else make dummy
# For GC, symbol_root = "GC"
symbol_root = "GC"  # Hardcoded for now, can extract from data later

# Session detection (simple heuristic based on hour)
def get_session(timestamp_str):
    """Determine session from timestamp"""
    from datetime import datetime
    dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
    hour = dt.hour
    
    # Simple session mapping (adjust for your timezone)
    if 0 <= hour < 7:
        return "Asia"
    elif 7 <= hour < 13:
        return "Europe"
    elif 13 <= hour < 20:
        return "NY"
    else:
        return "Asia"  # Late night

# Load sequences
sequences_path = PHASE2_DIR / "sequences_p2_10weeks_sequences.npy"
print(f"\n[3/5] Loading sequences: {sequences_path}")
sequences = np.load(sequences_path)
print(f"  Sequences shape: {sequences.shape}")
WINDOW_SIZE = sequences.shape[1]

# Load sequence -> GLOBAL bar indices (prevents misalignment)
indices_path = PHASE2_DIR / "sequences_p2_10weeks_indices.npy"
if not indices_path.exists():
    raise FileNotFoundError(f"Missing indices file: {indices_path}")
p2_indices = np.load(indices_path)
print(f"  Indices shape: {p2_indices.shape}")

if len(p2_indices) != len(sequences):
    raise ValueError(f"indices length {len(p2_indices)} does not match sequences length {len(sequences)}")

# Soft-label thresholds (env-overridable)
# Defaults skew conservative to reduce drawdown:
#   KEEP: strong push (>=1.8R) and limited adverse (<0.4R)
#   DROP: clear adverse (<=-0.6R) and no real push (<0.1R)
RUNUP_KEEP = float(os.getenv("RUNUP_KEEP", "1.8"))
DRAWUP_KEEP = float(os.getenv("DRAWUP_KEEP", "-0.4"))
DRAW_DROP = float(os.getenv("DRAW_DROP", "-0.6"))
RUNUP_DROP = float(os.getenv("RUNUP_DROP", "0.1"))

print(f"\n[4/5] Creating quality labels (soft criteria, tighter)")
print(f"  KEEP if runup >= {RUNUP_KEEP}R and drawdown > {DRAWUP_KEEP}R")
print(f"  DROP if drawdown <= {DRAW_DROP}R and runup < {RUNUP_DROP}R")

# Build enriched events + quality labels
enriched_events = []
candidates = []  # For PyTorch dataset (only long/short)

for i, event in enumerate(events):
    # Copy original event
    enriched = event.copy()
    
    # Add metadata
    enriched['symbol'] = f"{symbol_root} {enriched['timestamp'][:7]}"  # e.g. "GC 2025-09"
    enriched['symbol_root'] = symbol_root
    enriched['session'] = get_session(enriched['timestamp'])
    
    # SMC context: align by GLOBAL bar index from indices file (prevents misalignment)
    if i < len(p2_indices):
        idx = int(p2_indices[i])
        if idx >= len(features_df):
            raise IndexError(f"Sequence index {idx} out of bounds for features_df ({len(features_df)})")
        feat_row = features_df.iloc[idx]
        enriched['ext_trend_dir'] = int(feat_row.get('ext_trend_dir', 0))
        enriched['in_bull_fvg'] = bool(feat_row.get('in_bull_fvg', False))
        enriched['in_bear_fvg'] = bool(feat_row.get('in_bear_fvg', False))
        enriched['int_in_bull_ob'] = bool(feat_row.get('int_in_bull_ob', False))
        enriched['int_in_bear_ob'] = bool(feat_row.get('int_in_bear_ob', False))
        enriched['ext_in_bull_ob'] = bool(feat_row.get('ext_in_bull_ob', False))
        enriched['ext_in_bear_ob'] = bool(feat_row.get('ext_in_bear_ob', False))
    else:
        enriched.setdefault('ext_trend_dir', 0)
        enriched.setdefault('in_bull_fvg', False)
        enriched.setdefault('in_bear_fvg', False)
        enriched.setdefault('int_in_bull_ob', False)
        enriched.setdefault('int_in_bear_ob', False)
        enriched.setdefault('ext_in_bull_ob', False)
        enriched.setdefault('ext_in_bear_ob', False)
    
    # Determine if candidate
    signal_side = enriched['signal_side']
    is_candidate = signal_side in ['long', 'short']
    enriched['is_candidate'] = is_candidate
    
    # Quality label
    if is_candidate:
        max_runup_rr = enriched.get('max_runup_rr', 0.0)
        max_drawdown_rr = enriched.get('max_drawdown_rr', 0.0)

        keep = (max_runup_rr >= RUNUP_KEEP) and (max_drawdown_rr > DRAWUP_KEEP)
        drop = (max_drawdown_rr <= DRAW_DROP) and (max_runup_rr < RUNUP_DROP)

        if not (keep or drop):
            enriched['quality_label'] = None
            enriched_events.append(enriched)
            continue

        quality_label = 1 if keep else 0
        
        enriched['quality_label'] = quality_label
        
        # Prepare for PyTorch dataset
        side_encoding = +1 if signal_side == 'long' else -1
        
        candidates.append({
            'event_id': i,
            'X': sequences[i],  # [60, 66]
            'y_quality': quality_label,
            'side': side_encoding,
            'meta': {
                'event_id': i,
                'symbol_root': enriched['symbol_root'],
                'timestamp': enriched['timestamp'],
                'session': enriched['session'],
                'signal_side': signal_side,
                'hit': enriched.get('hit'),
                'outcome_rr': enriched.get('outcome_rr'),
                'max_runup_rr': max_runup_rr,
                'max_drawdown_rr': max_drawdown_rr,
                'entry_price': enriched.get('entry_price'),
                'sl_price': enriched.get('sl_price'),
                'tp_price': enriched.get('tp_price'),
                # For leakage-safe splitting
                'end_idx': int(p2_indices[i]),
                'start_idx_est': int(p2_indices[i]) - (WINDOW_SIZE - 1),
            }
        })
    else:
        enriched['quality_label'] = None  # Skip events have no quality label
    
    enriched_events.append(enriched)

print(f"  Total events: {len(enriched_events):,}")
print(f"  Candidates (long/short after soft filter): {len(candidates):,}")
print(f"  Skipped (incl. SKIP + ambiguous soft rules): {len(enriched_events) - len(candidates):,}")

# Class distribution
labels = [c['y_quality'] for c in candidates]
label_counts = Counter(labels)
print(f"\n  Quality label distribution:")
print(f"    KEEP (1): {label_counts[1]:,} ({label_counts[1]/len(labels)*100:.1f}%)")
print(f"    DROP (0): {label_counts[0]:,} ({label_counts[0]/len(labels)*100:.1f}%)")

# Export enriched JSONL
jsonl_path = OUTPUT_DIR / "events_p2_labeled_quality_v1.jsonl"
print(f"\n[5/5] Exporting enriched events: {jsonl_path}")
with open(jsonl_path, 'w') as f:
    for event in enriched_events:
        f.write(json.dumps(event) + '\n')
print(f"  Saved {len(enriched_events):,} events")

# Build PyTorch dataset
print(f"\nBuilding PyTorch dataset...")

X = torch.FloatTensor([c['X'] for c in candidates])  # [N, 60, 66]
y_quality = torch.LongTensor([c['y_quality'] for c in candidates])  # [N]
side = torch.LongTensor([c['side'] for c in candidates])  # [N]
meta = [c['meta'] for c in candidates]

dataset = {
    'X': X,
    'y_quality': y_quality,
    'side': side,
    'meta': meta
}

print(f"  X shape: {X.shape}")
print(f"  y_quality shape: {y_quality.shape}")
print(f"  side shape: {side.shape}")
print(f"  meta length: {len(meta)}")

# Save full dataset
full_dataset_path = OUTPUT_DIR / "dataset_p2_quality_v1.pt"
torch.save(dataset, full_dataset_path)
print(f"\nSaved full dataset: {full_dataset_path}")

# Train/Val split - TEMPORAL with window-overlap guard (CRITICAL for time-series)
print(f"\nSplitting train/val TEMPORALLY (80/20) with window overlap guard...")
timestamps = [m['timestamp'] for m in meta]
end_indices = np.array([m['end_idx'] for m in meta])

# Sort by end index (global order)
sorted_indices = np.argsort(end_indices)

# Split point at 80% in sorted order
split_point = int(len(sorted_indices) * 0.8)
split_global = end_indices[sorted_indices[split_point]]

# Guard: ensure windows do not cross the split
# Train: All sequences ending at or before the split point
train_idx = [idx for idx in sorted_indices if end_indices[idx] <= split_global]

# Val: All sequences starting strictly AFTER the split point
# Start = End - Window + 1
# We want Start > split_global
# End - Window + 1 > split_global  =>  End > split_global + Window - 1
val_idx = [idx for idx in sorted_indices if end_indices[idx] > split_global + (WINDOW_SIZE - 1)]

dropped = len(sorted_indices) - (len(train_idx) + len(val_idx))

print(f"\nTemporal split ranges:")
if len(train_idx) > 0:
    print(f"  Train: {timestamps[train_idx[0]]} to {timestamps[train_idx[-1]]}")
if len(val_idx) > 0:
    print(f"  Val:   {timestamps[val_idx[0]]} to {timestamps[val_idx[-1]]}")
print(f"  Train size: {len(train_idx):,}")
print(f"  Val size:   {len(val_idx):,}")
if dropped > 0:
    print(f"  Dropped near-boundary sequences to prevent overlap: {dropped}")

# Normalization (StandardScaler)
print(f"\nApplying Normalization (StandardScaler)...")
from sklearn.preprocessing import StandardScaler
import joblib

# Flatten X for scaling: [N, Window, Features] -> [N*Window, Features]
X_train = X[train_idx].numpy()
N_train, W, F = X_train.shape
X_train_flat = X_train.reshape(-1, F)

scaler = StandardScaler()
scaler.fit(X_train_flat)
print(f"  Scaler fit on {N_train*W:,} samples")

# Save scaler
scaler_path = OUTPUT_DIR / "scaler_p2_quality_v1.joblib"
joblib.dump(scaler, scaler_path)
print(f"  Saved scaler: {scaler_path}")

# Transform Train
X_train_scaled = scaler.transform(X_train_flat).reshape(N_train, W, F)
X_train_final = torch.FloatTensor(X_train_scaled)

# Transform Val
X_val = X[val_idx].numpy()
N_val, _, _ = X_val.shape
X_val_flat = X_val.reshape(-1, F)
X_val_scaled = scaler.transform(X_val_flat).reshape(N_val, W, F)
X_val_final = torch.FloatTensor(X_val_scaled)

# Train dataset
train_dataset = {
    'X': X_train_final,
    'y_quality': y_quality[train_idx],
    'side': side[train_idx],
    'meta': [meta[i] for i in train_idx]
}

# Val dataset
val_dataset = {
    'X': X_val_final,
    'y_quality': y_quality[val_idx],
    'side': side[val_idx],
    'meta': [meta[i] for i in val_idx]
}

# Save splits
train_path = OUTPUT_DIR / "dataset_p2_quality_v1_train.pt"
val_path = OUTPUT_DIR / "dataset_p2_quality_v1_val.pt"

torch.save(train_dataset, train_path)
torch.save(val_dataset, val_path)

print(f"\nSaved train: {train_path}")
print(f"Saved val:   {val_path}")

# Verify split distribution
train_labels = train_dataset['y_quality'].numpy()
val_labels = val_dataset['y_quality'].numpy()

print(f"\nTrain distribution:")
print(f"  KEEP: {(train_labels == 1).sum():,} ({(train_labels == 1).sum()/len(train_labels)*100:.1f}%)")
print(f"  DROP: {(train_labels == 0).sum():,} ({(train_labels == 0).sum()/len(train_labels)*100:.1f}%)")

print(f"\nVal distribution:")
print(f"  KEEP: {(val_labels == 1).sum():,} ({(val_labels == 1).sum()/len(val_labels)*100:.1f}%)")
print(f"  DROP: {(val_labels == 0).sum():,} ({(val_labels == 0).sum()/len(val_labels)*100:.1f}%)")

print("\n" + "="*80)
print("DATASET BUILD COMPLETE!")
print("="*80)
print(f"\nOutput files:")
print(f"  - {jsonl_path}")
print(f"  - {full_dataset_path}")
print(f"  - {train_path}")
print(f"  - {val_path}")
print(f"\nReady for training!")
