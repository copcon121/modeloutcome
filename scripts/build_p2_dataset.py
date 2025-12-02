"""
Build P2 Dataset from processed_v2 CSV files
1. Merge all CSV files
2. Apply P2 filter
3. Build sequences (60-bar windows)
4. Export for training
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

# Output directory
OUTPUT_DIR = Path("output/production_10weeks_v2")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("="*60)
print("BUILD P2 DATASET FROM PROCESSED_V2")
print("="*60)

# Step 1: Load and merge all CSV files
print("\n[1/4] Loading CSV files...")
csv_dir = Path("data/processed_v2")
csv_files = sorted(csv_dir.glob("*.csv"))
print(f"  Found {len(csv_files)} files")

dfs = []
for f in csv_files:
    df = pd.read_csv(f)
    df['source_file'] = f.name
    dfs.append(df)
    print(f"    {f.name}: {len(df)} rows")

df_all = pd.concat(dfs, ignore_index=True)
print(f"\n  Total: {len(df_all):,} bars, {len(df_all.columns)} columns")

# Save merged features
merged_path = OUTPUT_DIR / "features_all_10weeks.csv"
df_all.to_csv(merged_path, index=False)
print(f"  Saved: {merged_path}")

# Step 2: Apply P2 filter
print("\n[2/4] Applying P2 filter...")

# P2 filter logic (simplified from event_filter.py)
# Keep: structure OR high_volatility OR (zone+volatility) OR liquidity_sweep

def compute_p2_mask(df, rolling_window=100):
    """Compute P2 filter mask"""
    n = len(df)
    mask = np.zeros(n, dtype=bool)
    
    for i in tqdm(range(n), desc="Computing P2 mask"):
        # Structure events
        has_bos_choch = (
            df.iloc[i].get('int_bos_up', 0) or df.iloc[i].get('int_bos_down', 0) or
            df.iloc[i].get('int_choch_up', 0) or df.iloc[i].get('int_choch_down', 0) or
            df.iloc[i].get('ext_bos_up', 0) or df.iloc[i].get('ext_bos_down', 0) or
            df.iloc[i].get('ext_choch_up', 0) or df.iloc[i].get('ext_choch_down', 0)
        )
        
        # Zone events
        in_zone = (
            df.iloc[i].get('in_bull_fvg', 0) or df.iloc[i].get('in_bear_fvg', 0) or
            df.iloc[i].get('int_in_bull_ob', 0) or df.iloc[i].get('int_in_bear_ob', 0) or
            df.iloc[i].get('ext_in_bull_ob', 0) or df.iloc[i].get('ext_in_bear_ob', 0)
        )
        
        # Liquidity sweep
        ls_event = (
            df.iloc[i].get('swept_prev_int_high', 0) or 
            df.iloc[i].get('swept_prev_int_low', 0)
        )
        
        # Rolling averages for volatility
        start_idx = max(0, i - rolling_window)
        window = df.iloc[start_idx:i+1]
        
        if len(window) > 10:
            avg_range = window['high_low_range'].mean()
            avg_volume = window['volume'].mean()
            avg_delta = window['delta'].abs().mean()
        else:
            avg_range = 2.0
            avg_volume = 100
            avg_delta = 20
        
        # Volatility flags
        high_range = df.iloc[i]['high_low_range'] > avg_range * 1.3
        high_volume = df.iloc[i]['volume'] > avg_volume * 2.0
        
        delta_over_vol = df.iloc[i].get('delta_over_volume', 0)
        delta_abs = abs(df.iloc[i].get('delta', 0))
        vol = df.iloc[i].get('volume', 0)
        high_delta = (abs(delta_over_vol) > 0.60 and delta_abs > avg_delta * 1.5 and vol > 50)
        
        high_volatility = high_range or high_volume or high_delta
        
        # P2 filter
        mask[i] = (
            has_bos_choch or
            high_volatility or
            (in_zone and high_volatility) or
            ls_event
        )
    
    return mask

p2_mask = compute_p2_mask(df_all)
p2_count = p2_mask.sum()
print(f"\n  P2 events: {p2_count:,} ({p2_count/len(df_all)*100:.1f}%)")

# Get P2 indices
p2_indices = np.where(p2_mask)[0]

# Step 3: Build sequences
print("\n[3/4] Building sequences (window=60)...")

WINDOW_SIZE = 60
FEATURE_COLS = [c for c in df_all.columns if c not in ['timestamp', 'source_file']]
print(f"  Feature columns: {len(FEATURE_COLS)}")

# Convert to numpy for faster access
features_array = df_all[FEATURE_COLS].values.astype(np.float32)

sequences = []
valid_indices = []

for idx in tqdm(p2_indices, desc="Building sequences"):
    if idx >= WINDOW_SIZE - 1:
        # Get 60-bar window ending at this index
        seq = features_array[idx - WINDOW_SIZE + 1 : idx + 1]
        sequences.append(seq)
        valid_indices.append(idx)

sequences = np.array(sequences, dtype=np.float32)
valid_indices = np.array(valid_indices, dtype=np.int64)

print(f"\n  Sequences shape: {sequences.shape}")
print(f"  Valid indices: {len(valid_indices)}")

# Step 4: Export
print("\n[4/4] Exporting...")

# Save sequences
seq_path = OUTPUT_DIR / "sequences_p2_10weeks_sequences.npy"
np.save(seq_path, sequences)
print(f"  Saved: {seq_path}")

# Save indices
idx_path = OUTPUT_DIR / "sequences_p2_10weeks_indices.npy"
np.save(idx_path, valid_indices)
print(f"  Saved: {idx_path}")

# Save P2 filtered features
df_p2 = df_all.iloc[p2_indices].copy()
p2_csv_path = OUTPUT_DIR / "features_p2_filtered_10weeks.csv"
df_p2.to_csv(p2_csv_path, index=False)
print(f"  Saved: {p2_csv_path}")

# Summary
print("\n" + "="*60)
print("DATASET BUILD COMPLETE!")
print("="*60)
print(f"\nSummary:")
print(f"  Total bars: {len(df_all):,}")
print(f"  P2 events: {p2_count:,} ({p2_count/len(df_all)*100:.1f}%)")
print(f"  Sequences: {sequences.shape}")
print(f"  Features: {len(FEATURE_COLS)}")
print(f"\nOutput directory: {OUTPUT_DIR}")
