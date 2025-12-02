"""
Process NEW DATA (6 weeks Apr-Jun 2025) to generate features for Pattern A validation.

This script:
1. Processes raw JSONL files from data/raw/new_data/
2. Generates features using SMCContextManager (same as training data)
3. Saves to output/new_data_features/

Usage:
    python scripts/process_new_data_features.py
"""

import os
import sys
import glob
import json
import pandas as pd
from datetime import datetime
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.layer2_feature_engine_v2.context_manager import SMCContextManager
from src.layer2_feature_engine_v2.config import GC_M1_SMC_CONFIG
from src.layer2_feature_engine_v2.schema import RawBar

print("="*70)
print("PROCESS NEW DATA FEATURES FOR PATTERN A VALIDATION")
print("="*70)

# Directories
RAW_DIR = ROOT / "data/raw/new_data"
OUTPUT_DIR = ROOT / "output/new_data_features"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_raw_bar(data):
    """Convert JSON dict to RawBar"""
    ts_str = data['timestamp']
    ts = datetime.fromisoformat(ts_str)
    
    bar_data = data['bar']
    tick_data = data.get('tick_features', {})
    
    return RawBar(
        symbol=data.get('symbol', 'GC'),
        timeframe=data.get('timeframe', 'M1'),
        timestamp=ts,
        bar_index=data['bar_index'],
        o=bar_data['o'],
        h=bar_data['h'],
        l=bar_data['l'],
        c=bar_data['c'],
        volume=bar_data['volume'],
        delta=bar_data.get('delta', 0),
        buy_volume=bar_data.get('buy_volume', 0),
        sell_volume=bar_data.get('sell_volume', 0),
        best_bid=bar_data.get('best_bid', bar_data['c']),
        best_ask=bar_data.get('best_ask', bar_data['c']),
        tick_speed=tick_data.get('tick_speed', 0),
        aggr_buy_speed=tick_data.get('aggr_buy_speed', 0),
        aggr_sell_speed=tick_data.get('aggr_sell_speed', 0),
        price_speed=tick_data.get('price_speed', bar_data['h'] - bar_data['l']),
        vwap_daily=bar_data.get('vwap_daily', 0.0)
    )

def process_file(input_path, output_path):
    """Process single JSONL file"""
    print(f"\n  Processing {input_path.name}...")
    
    # Initialize Manager (fresh for each file to reset state)
    manager = SMCContextManager(GC_M1_SMC_CONFIG, tick_size=0.1)
    
    processed_bars = []
    
    with open(input_path, 'r') as f:
        for line_num, line in enumerate(f):
            try:
                data = json.loads(line)
                raw_bar = load_raw_bar(data)
                feature_bar = manager.update(raw_bar)
                
                # Convert to dict
                fb_dict = feature_bar.to_dict()
                fb_dict['timestamp'] = raw_bar.timestamp.isoformat()
                fb_dict['bar_index'] = raw_bar.bar_index
                fb_dict['global_bar_index'] = len(processed_bars)  # For tracking
                # Add raw OHLC
                fb_dict['open'] = raw_bar.o
                fb_dict['high'] = raw_bar.h
                fb_dict['low'] = raw_bar.l
                processed_bars.append(fb_dict)
                
            except Exception as e:
                print(f"    Error line {line_num}: {e}")
                continue
    
    # Save to CSV
    df = pd.DataFrame(processed_bars)
    df.to_csv(output_path, index=False)
    print(f"    Saved {len(df)} bars to {output_path.name}")
    return df

# Find all new data files
files = sorted(RAW_DIR.glob("smc_export_gc_m1_v3_*.jsonl"))
print(f"\nFound {len(files)} files to process:")
for f in files:
    print(f"  - {f.name}")

# Process each file
all_dfs = []
for input_path in files:
    output_filename = input_path.name.replace(".jsonl", "_features.csv")
    output_path = OUTPUT_DIR / output_filename
    
    df = process_file(input_path, output_path)
    all_dfs.append(df)

# Merge all files
print("\n" + "="*70)
print("MERGING ALL FILES...")
print("="*70)

df_all = pd.concat(all_dfs, ignore_index=True)

# Reassign global_bar_index
df_all['global_bar_index'] = range(len(df_all))

# Save merged file
merged_path = OUTPUT_DIR / "features_all_new6w.csv"
df_all.to_csv(merged_path, index=False)
print(f"\nMerged features: {len(df_all):,} bars")
print(f"Columns: {len(df_all.columns)}")
print(f"Saved to: {merged_path}")

# Check features
print("\n" + "="*70)
print("FEATURE CHECK")
print("="*70)

# Load model to check required features
import torch
checkpoint = torch.load(ROOT / "output/pattern_dataset_v1/pattern_a_gru_best.pt", weights_only=False)
model_features = checkpoint['features']

print(f"\nModel expects {len(model_features)} features")
print(f"New data has {len(df_all.columns)} columns")

missing = [f for f in model_features if f not in df_all.columns]
if missing:
    print(f"\nMISSING FEATURES ({len(missing)}):")
    for f in missing:
        print(f"  - {f}")
else:
    print("\n✓ All required features present!")

# Show sample columns
print(f"\nSample columns: {list(df_all.columns[:20])}")

print("\n" + "="*70)
print("DONE!")
print("="*70)
