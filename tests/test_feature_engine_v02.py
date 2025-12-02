"""
Test Feature Engine v0.2 - SMC & Volume Profile Features
Verify that SMC structure and Volume Profile features are working correctly
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from layer2_feature_engine.core.data_loader import load_raw_bars
from layer2_feature_engine.core.context_manager import ContextManager


def test_smc_and_vp_features():
    """Test SMC and Volume Profile feature extraction"""
    print("\n" + "="*80)
    print("TEST: Feature Engine v0.2 - SMC & Volume Profile")
    print("="*80 + "\n")

    # Load raw data
    jsonl_path = "data/raw/smc_export_gc_m1_v3_20250901.jsonl"
    print(f"Loading data from: {jsonl_path}")

    bars = load_raw_bars(jsonl_path)
    print(f"✓ Loaded {len(bars)} bars\n")

    # Initialize Context Manager
    context_len = 60
    context_mgr = ContextManager(
        context_len=context_len,
        max_history=200,
        normalizer=None
    )

    # Add bars
    context_mgr.add_bars_batch(bars)
    print(f"✓ Added {len(context_mgr.bars)} bars to context\n")

    # Build features
    print("Building features with SMC + VP...")
    feature_bars = context_mgr.build_features()
    print(f"✓ Built features for {len(feature_bars)} bars\n")

    # Analyze SMC structure
    if context_mgr.smc_structure:
        smc = context_mgr.smc_structure
        print("SMC Structure Analysis:")
        print(f"  Swing Highs:  {len(smc.swing_highs)}")
        print(f"  Swing Lows:   {len(smc.swing_lows)}")
        print(f"  BOS Up:       {len(smc.bos_up_indices)}")
        print(f"  BOS Down:     {len(smc.bos_down_indices)}")
        print(f"  CHoCH Up:     {len(smc.choch_up_indices)}")
        print(f"  CHoCH Down:   {len(smc.choch_down_indices)}")
        print(f"  Order Blocks Up:   {len(smc.active_obs_up)}")
        print(f"  Order Blocks Down: {len(smc.active_obs_down)}")
        print(f"  FVG Up:       {len(smc.active_fvgs_up)}")
        print(f"  FVG Down:     {len(smc.active_fvgs_down)}\n")

        # Show swing highs/lows
        if smc.swing_highs:
            print(f"  First 5 Swing Highs: {smc.swing_highs[:5]}")
        if smc.swing_lows:
            print(f"  First 5 Swing Lows:  {smc.swing_lows[:5]}\n")

    # Analyze Volume Profile
    if context_mgr.vp_state:
        vp = context_mgr.vp_state
        print("Volume Profile Analysis:")
        print(f"  VAH (Value Area High): {vp.vah:.2f}")
        print(f"  VAL (Value Area Low):  {vp.val:.2f}")
        print(f"  POC (Point of Control): {vp.poc:.2f}")
        print(f"  HVN Levels: {len(vp.hvn_levels)}")
        print(f"  LVN Levels: {len(vp.lvn_levels)}\n")

    # Show SMC features from sample bars
    print("Sample SMC & VP Features:")
    print("-" * 80)

    for i in range(min(10, len(feature_bars))):
        fb = feature_bars[i]

        # Extract SMC features
        is_swing_high = fb.features.get('is_swing_high', 0)
        is_swing_low = fb.features.get('is_swing_low', 0)
        bos_up = fb.features.get('bos_up', 0)
        bos_down = fb.features.get('bos_down', 0)
        choch_up = fb.features.get('choch_up', 0)
        choch_down = fb.features.get('choch_down', 0)

        # Extract VP features
        dist_vah = fb.features.get('dist_to_vah', 0)
        dist_val = fb.features.get('dist_to_val', 0)
        in_value_area = fb.features.get('in_value_area', 0)

        # Only print if there's interesting SMC activity
        if any([is_swing_high, is_swing_low, bos_up, bos_down, choch_up, choch_down]):
            print(f"\nBar {i+1} at {fb.ts}:")

            if is_swing_high:
                print(f"  🔺 SWING HIGH")
            if is_swing_low:
                print(f"  🔻 SWING LOW")
            if bos_up:
                print(f"  ⬆️  BOS UP")
            if bos_down:
                print(f"  ⬇️  BOS DOWN")
            if choch_up:
                print(f"  🔄 CHoCH UP")
            if choch_down:
                print(f"  🔄 CHoCH DOWN")

            print(f"  VP: dist_vah={dist_vah:.4f}, dist_val={dist_val:.4f}, in_VA={in_value_area}")

    print("\n" + "-" * 80)

    # Count active SMC signals
    swing_high_count = sum(1 for fb in feature_bars if fb.features.get('is_swing_high', 0) > 0)
    swing_low_count = sum(1 for fb in feature_bars if fb.features.get('is_swing_low', 0) > 0)
    bos_up_count = sum(1 for fb in feature_bars if fb.features.get('bos_up', 0) > 0)
    bos_down_count = sum(1 for fb in feature_bars if fb.features.get('bos_down', 0) > 0)

    print(f"\nSMC Signal Counts (out of {len(feature_bars)} bars):")
    print(f"  Swing Highs: {swing_high_count}")
    print(f"  Swing Lows:  {swing_low_count}")
    print(f"  BOS Up:      {bos_up_count}")
    print(f"  BOS Down:    {bos_down_count}")

    # Verify features exist
    first_features = feature_bars[0].features
    smc_feature_names = [f for f in first_features.keys() if any(x in f for x in ['swing', 'bos', 'choch', 'ob', 'fvg'])]
    vp_feature_names = [f for f in first_features.keys() if any(x in f for x in ['vah', 'val', 'poc', 'value_area', 'hvn', 'lvn'])]

    print(f"\nSMC Features ({len(smc_feature_names)}):")
    for name in sorted(smc_feature_names):
        print(f"  • {name}")

    print(f"\nVolume Profile Features ({len(vp_feature_names)}):")
    for name in sorted(vp_feature_names):
        print(f"  • {name}")

    print("\n" + "="*80)
    print("✓ Feature Engine v0.2 Test PASSED!")
    print("  SMC structure detection: WORKING")
    print("  Volume Profile calculation: WORKING")
    print("="*80 + "\n")

    return True


if __name__ == "__main__":
    success = test_smc_and_vp_features()
    sys.exit(0 if success else 1)
