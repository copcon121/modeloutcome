#!/usr/bin/env python3
"""
Test Feature Order Match between Training and Live Gateway
"""

import json
from pathlib import Path
import sys

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# Load training feature order
with open(ROOT / "output/asm_dataset_v1/asm_dataset_v1_stats.json") as f:
    stats = json.load(f)

train_features = stats["features"]["feature_names"][:100]

# Load live gateway feature order
from services.live_gateway.context_store import ASM_FEATURE_COLS

print("=" * 70)
print("FEATURE ORDER COMPARISON")
print("=" * 70)

print(f"\nTraining features: {len(train_features)}")
print(f"Live gateway features: {len(ASM_FEATURE_COLS)}")

# Compare
mismatches = []
for i, (train_f, live_f) in enumerate(zip(train_features, ASM_FEATURE_COLS)):
    if train_f != live_f:
        mismatches.append((i, train_f, live_f))

if mismatches:
    print(f"\n❌ MISMATCHES FOUND: {len(mismatches)}")
    for idx, train_f, live_f in mismatches[:20]:
        print(f"  [{idx:3d}] Train: {train_f:30s} | Live: {live_f}")
else:
    print("\n✅ ALL FEATURES MATCH!")

# Test with actual feature bar using context_store
print("\n" + "=" * 70)
print("TEST WITH ACTUAL FEATURE BAR (via context_store)")
print("=" * 70)

from src.layer2_feature_engine_v2.schema import RawBar
from services.live_gateway.context_store import ContextStore
from datetime import datetime

# Create fresh context store
store = ContextStore()

# Create dummy bar
raw_bar = RawBar(
    symbol='GC', timeframe='M1', timestamp=datetime.now(), bar_index=0,
    o=3300, h=3301, l=3299, c=3300.5, volume=100, delta=10,
    buy_volume=55, sell_volume=45, best_bid=3300.4, best_ask=3300.6,
    tick_speed=10, aggr_buy_speed=5, aggr_sell_speed=5, price_speed=2, vwap_daily=3300
)

# Update via context store (this adds open, high, low, global_bar_index)
feature_bar, fb_dict = store.update('GC', 'M1', raw_bar)

# Check all features exist
missing_features = []
for col in ASM_FEATURE_COLS:
    if col not in fb_dict:
        missing_features.append(col)

if missing_features:
    print(f"\n❌ MISSING FEATURES IN FEATURE BAR: {len(missing_features)}")
    for f in missing_features:
        print(f"  - {f}")
else:
    print("\n✅ ALL FEATURES EXIST IN FEATURE BAR!")

# Show sample values
print("\nSample feature values:")
for i, col in enumerate(ASM_FEATURE_COLS[:10]):
    val = fb_dict.get(col, "MISSING")
    print(f"  [{i:3d}] {col:30s} = {val}")

print("\n" + "=" * 70)
print("DONE!")
print("=" * 70)
