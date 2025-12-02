"""
Test Feature Engine v0.1 - Orderflow per-bar (without SMC for now)
Tests basic feature extraction: OHLCV + Tick/Orderflow features
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from layer2_feature_engine.core.data_loader import load_raw_bars
from layer2_feature_engine.core.context_manager import ContextManager
from layer2_feature_engine.core.normalizer import Normalizer


def test_basic_feature_extraction():
    """Test basic OHLCV + Tick feature extraction"""
    print("\n" + "="*80)
    print("TEST: Feature Engine v0.1 - Orderflow Features")
    print("="*80 + "\n")

    # Load raw data
    jsonl_path = "data/raw/smc_export_gc_m1_v3_20250901.jsonl"
    print(f"Loading data from: {jsonl_path}")

    bars = load_raw_bars(jsonl_path)
    print(f"✓ Loaded {len(bars)} bars")
    print(f"  First bar: {bars[0].ts}")
    print(f"  Last bar:  {bars[-1].ts}\n")

    # Initialize Context Manager
    context_len = 60
    print(f"Initializing ContextManager (context_len={context_len})...")

    context_mgr = ContextManager(
        context_len=context_len,
        max_history=200,
        normalizer=None  # Don't normalize yet
    )
    print("✓ ContextManager initialized\n")

    # Add bars in batch
    print(f"Adding {len(bars)} bars to context...")
    context_mgr.add_bars_batch(bars)
    print(f"✓ Added {len(context_mgr.bars)} bars to context\n")

    # Build features
    print("Building features for all bars...")
    try:
        feature_bars = context_mgr.build_features()
        print(f"✓ Built features for {len(feature_bars)} bars\n")
    except Exception as e:
        print(f"✗ Error building features: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Analyze features
    if not feature_bars:
        print("✗ No feature bars generated!")
        return False

    # Get feature dimensions
    first_features = feature_bars[0].features
    feature_names = sorted(first_features.keys())
    feature_dim = len(feature_names)

    print(f"Feature Analysis:")
    print(f"  Total features: {feature_dim}")
    print(f"  Feature bars:   {len(feature_bars)}\n")

    # Check for NaN/Inf
    print("Checking for NaN/Inf values...")
    nan_count = 0
    inf_count = 0

    for fb in feature_bars:
        for name, value in fb.features.items():
            if np.isnan(value):
                nan_count += 1
            elif np.isinf(value):
                inf_count += 1

    if nan_count > 0:
        print(f"✗ Found {nan_count} NaN values!")
    else:
        print(f"✓ No NaN values")

    if inf_count > 0:
        print(f"✗ Found {inf_count} Inf values!")
    else:
        print(f"✓ No Inf values")

    print()

    # Print feature categories
    print("Feature Categories:")

    ohlcv_features = [f for f in feature_names if any(x in f for x in ['open', 'high', 'low', 'close', 'range', 'body', 'wick', 'volume'])]
    tick_features = [f for f in feature_names if any(x in f for x in ['tick', 'aggr', 'buy', 'sell', 'delta', 'price_speed'])]
    smc_features = [f for f in feature_names if any(x in f for x in ['swing', 'bos', 'choch', 'ob', 'fvg'])]
    vp_features = [f for f in feature_names if any(x in f for x in ['vah', 'val', 'poc', 'value_area', 'hvn', 'lvn'])]
    l2_features = [f for f in feature_names if 'l2' in f]
    time_features = [f for f in feature_names if any(x in f for x in ['time', 'session', 'day', 'weekend'])]

    print(f"  OHLCV:       {len(ohlcv_features)} features")
    print(f"  Tick/Flow:   {len(tick_features)} features")
    print(f"  SMC:         {len(smc_features)} features")
    print(f"  Vol Profile: {len(vp_features)} features")
    print(f"  Level 2:     {len(l2_features)} features")
    print(f"  Time:        {len(time_features)} features\n")

    # Sample features from first 5 bars
    print("Sample features from first 5 feature bars:")
    print("-" * 80)

    for i in range(min(5, len(feature_bars))):
        fb = feature_bars[i]
        print(f"\nBar {i+1} at {fb.ts}:")

        # OHLCV
        print(f"  OHLCV: close_norm={fb.features.get('close_norm', 0):.4f}, "
              f"volume_log={fb.features.get('volume_log', 0):.2f}")

        # Tick features
        print(f"  Tick:  tick_speed_norm={fb.features.get('tick_speed_norm', 0):.2f}, "
              f"buy_sell_ratio={fb.features.get('buy_sell_ratio', 0):.2f}")
        print(f"         buying_pressure={fb.features.get('buying_pressure_index', 0):.3f}, "
              f"activity={fb.features.get('activity_intensity', 0):.3f}")

    print("\n" + "="*80)

    # Create a Record (for ML)
    print("\nCreating ML Record (context window)...")
    try:
        record = context_mgr.create_record(label=None)
        print(f"✓ Created record:")
        print(f"  Context length: {record.context_len}")
        print(f"  Feature dim:    {record.feature_dim}")
        print(f"  Entry price:    {record.entry_price}")
        print(f"  ATR:            {record.atr:.2f}")
    except Exception as e:
        print(f"✗ Error creating record: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n" + "="*80)
    print("✓ Feature Engine v0.1 Test PASSED!")
    print("="*80 + "\n")

    return True


if __name__ == "__main__":
    success = test_basic_feature_extraction()
    sys.exit(0 if success else 1)
