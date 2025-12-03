#!/usr/bin/env python3
"""
Test ASM Inference with Fixed Feature Order
Simulates what live gateway does after the fix
"""

import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.layer2_feature_engine_v2.schema import RawBar
from services.live_gateway.context_store import ContextStore, ASM_FEATURE_COLS, ASM_SEQ_LEN
from services.live_gateway.asm_inference import ASMModelLoader
from datetime import datetime, timedelta

print("=" * 70)
print("TEST ASM INFERENCE WITH FIXED FEATURE ORDER")
print("=" * 70)

# Load ASM model
print("\nLoading ASM model...")
asm_model = ASMModelLoader()
asm_model.load()

# Create context store
store = ContextStore()

# Generate 100 bars to warm up context
print(f"\nGenerating {ASM_SEQ_LEN + 40} bars to warm up context...")
base_price = 3300.0
base_time = datetime(2025, 5, 1, 8, 0, 0)

for i in range(ASM_SEQ_LEN + 40):
    # Simulate price movement
    price_change = np.random.randn() * 2
    close = base_price + price_change
    high = close + abs(np.random.randn()) * 1
    low = close - abs(np.random.randn()) * 1
    open_price = base_price
    base_price = close
    
    raw_bar = RawBar(
        symbol='GC', timeframe='M1',
        timestamp=base_time + timedelta(minutes=i),
        bar_index=i,
        o=open_price, h=high, l=low, c=close,
        volume=100 + np.random.randint(-50, 50),
        delta=np.random.randint(-20, 20),
        buy_volume=50 + np.random.randint(-20, 20),
        sell_volume=50 + np.random.randint(-20, 20),
        best_bid=close - 0.1, best_ask=close + 0.1,
        tick_speed=10, aggr_buy_speed=5, aggr_sell_speed=5,
        price_speed=high - low, vwap_daily=3300
    )
    
    store.update('GC', 'M1', raw_bar)

print(f"  Context buffer size: {len(store.contexts[('GC', 'M1')].feature_buffer)}")

# Get ASM context
print("\nGetting ASM context...")
context = store.get_asm_context('GC', 'M1')

if context is None:
    print("ERROR: Context is None!")
    sys.exit(1)

print(f"  Context shape: {context.shape}")
print(f"  Context dtype: {context.dtype}")
print(f"  NaN count: {np.isnan(context).sum()}")
print(f"  Min value: {context.min():.4f}")
print(f"  Max value: {context.max():.4f}")

# Run ASM inference
print("\nRunning ASM inference...")
probs = asm_model.predict_proba(context)

print(f"\n{'='*70}")
print("ASM PREDICTION RESULTS")
print(f"{'='*70}")
print(f"  p_up:      {probs['p_up']:.4f}")
print(f"  p_down:    {probs['p_down']:.4f}")
print(f"  p_neutral: {probs['p_neutral']:.4f}")
print(f"  p_shift:   {probs['p_shift']:.4f}")

# Check if p_shift is reasonable
if probs['p_shift'] > 0.9:
    print(f"\n⚠️  WARNING: p_shift is very high ({probs['p_shift']:.2%})")
    print("    This might indicate feature mismatch or model issue")
elif probs['p_shift'] < 0.5:
    print(f"\n✅ p_shift is reasonable ({probs['p_shift']:.2%})")
    print("    Model predicts stable/neutral market")
else:
    print(f"\n📊 p_shift is moderate ({probs['p_shift']:.2%})")

# Compare with random input
print("\n" + "=" * 70)
print("COMPARISON: Random Input vs Structured Input")
print("=" * 70)

random_context = np.random.randn(60, 100).astype(np.float32)
random_probs = asm_model.predict_proba(random_context)

print(f"\nRandom input:")
print(f"  p_up:      {random_probs['p_up']:.4f}")
print(f"  p_down:    {random_probs['p_down']:.4f}")
print(f"  p_neutral: {random_probs['p_neutral']:.4f}")
print(f"  p_shift:   {random_probs['p_shift']:.4f}")

print(f"\nStructured input (from context_store):")
print(f"  p_up:      {probs['p_up']:.4f}")
print(f"  p_down:    {probs['p_down']:.4f}")
print(f"  p_neutral: {probs['p_neutral']:.4f}")
print(f"  p_shift:   {probs['p_shift']:.4f}")

print("\n" + "=" * 70)
print("DONE!")
print("=" * 70)
