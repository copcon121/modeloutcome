"""
SMC Debug Tool - Self-check for 2-layer swing detection

Usage:
    python -m src.layer2_feature_engine.smc.debug_smc

This will load sample GC M1 data and print:
- Internal swing counts (wave 5)
- External swing counts (wave 50)
- BOS/CHoCH events
- Sample bars with SMC states
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

from layer2_feature_engine.core.data_loader import load_raw_bars
from layer2_feature_engine.smc.smc_engine import SMCEngine, SMCState
from layer2_feature_engine.smc.config import GC_M1_SMC_CONFIG


def debug_smc_on_sample(data_path: str, n: int = 300):
    """
    Load n bars from JSONL, run SMC engine, print summary

    Args:
        data_path: Path to raw JSONL file
        n: Number of bars to process (default 300)
    """
    print("\n" + "="*80)
    print("SMC 2-LAYER DEBUG TOOL - Wave 5 (int) + Wave 50 (ext)")
    print("="*80 + "\n")

    # Load bars
    print(f"Loading bars from: {data_path}")
    bars = load_raw_bars(data_path)

    if n is not None and n < len(bars):
        bars = bars[:n]

    print(f"✓ Loaded {len(bars)} bars")
    print(f"  First: {bars[0].ts}")
    print(f"  Last:  {bars[-1].ts}\n")

    # Initialize SMC engine
    print("Initializing SMC Engine (GC M1 config)...")
    print(f"  Internal swing: wave {GC_M1_SMC_CONFIG.swing_int_strength} ({GC_M1_SMC_CONFIG.min_int_move_ticks} ticks)")
    print(f"  External swing: wave {GC_M1_SMC_CONFIG.swing_ext_strength} ({GC_M1_SMC_CONFIG.min_ext_move_ticks} ticks)")
    print()

    engine = SMCEngine(GC_M1_SMC_CONFIG)

    # Process all bars
    print(f"Processing {len(bars)} bars...\n")

    smc_states = []
    int_swing_high_count = 0
    int_swing_low_count = 0
    ext_swing_high_count = 0
    ext_swing_low_count = 0

    int_bos_up_count = 0
    int_bos_down_count = 0
    int_choch_up_count = 0
    int_choch_down_count = 0

    ext_bos_up_count = 0
    ext_bos_down_count = 0
    ext_choch_up_count = 0
    ext_choch_down_count = 0

    for i in range(len(bars)):
        smc_state = engine.update(bars, i)
        smc_states.append(smc_state)

        # Count events
        if smc_state.int_swing_high_bars_ago == 0:
            int_swing_high_count += 1
        if smc_state.int_swing_low_bars_ago == 0:
            int_swing_low_count += 1
        if smc_state.ext_swing_high_bars_ago == 0:
            ext_swing_high_count += 1
        if smc_state.ext_swing_low_bars_ago == 0:
            ext_swing_low_count += 1

        if smc_state.int_bos_up:
            int_bos_up_count += 1
        if smc_state.int_bos_down:
            int_bos_down_count += 1
        if smc_state.int_choch_up:
            int_choch_up_count += 1
        if smc_state.int_choch_down:
            int_choch_down_count += 1

        if smc_state.ext_bos_up:
            ext_bos_up_count += 1
        if smc_state.ext_bos_down:
            ext_bos_down_count += 1
        if smc_state.ext_choch_up:
            ext_choch_up_count += 1
        if smc_state.ext_choch_down:
            ext_choch_down_count += 1

    # Print summary
    print("="*80)
    print("SUMMARY")
    print("="*80 + "\n")

    print(f"Total bars processed: {len(bars)}\n")

    print("Internal Structure (wave 5):")
    print(f"  Swing Highs:  {int_swing_high_count}")
    print(f"  Swing Lows:   {int_swing_low_count}")
    print(f"  BOS Up:       {int_bos_up_count}")
    print(f"  BOS Down:     {int_bos_down_count}")
    print(f"  CHoCH Up:     {int_choch_up_count}")
    print(f"  CHoCH Down:   {int_choch_down_count}")
    print()

    print("External Structure (wave 50):")
    print(f"  Swing Highs:  {ext_swing_high_count}")
    print(f"  Swing Lows:   {ext_swing_low_count}")
    print(f"  BOS Up:       {ext_bos_up_count}")
    print(f"  BOS Down:     {ext_bos_down_count}")
    print(f"  CHoCH Up:     {ext_choch_up_count}")
    print(f"  CHoCH Down:   {ext_choch_down_count}")
    print()

    # Invariant check
    print("Invariant Checks:")
    ext_to_int_ratio = (ext_swing_high_count + ext_swing_low_count) / max(1, int_swing_high_count + int_swing_low_count)
    print(f"  Ext/Int swing ratio: {ext_to_int_ratio:.2f}")
    if ext_to_int_ratio < 0.3:
        print(f"  ✓ External swings significantly fewer than internal (expected)")
    else:
        print(f"  ⚠ Warning: Ext/Int ratio seems high (should be < 0.3)")
    print()

    # Print last 10 bars
    print("="*80)
    print("LAST 10 BARS")
    print("="*80 + "\n")

    print(f"{'Time':<20} {'Close':<8} {'Int Dir':<8} {'Ext Dir':<8} {'Events':<30}")
    print("-"*80)

    for i in range(max(0, len(bars) - 10), len(bars)):
        bar = bars[i]
        state = smc_states[i]

        events = []
        if state.int_bos_up:
            events.append("int_BOS↑")
        if state.int_bos_down:
            events.append("int_BOS↓")
        if state.int_choch_up:
            events.append("int_CHoCH↑")
        if state.int_choch_down:
            events.append("int_CHoCH↓")
        if state.ext_bos_up:
            events.append("EXT_BOS↑")
        if state.ext_bos_down:
            events.append("EXT_BOS↓")
        if state.ext_choch_up:
            events.append("EXT_CHoCH↑")
        if state.ext_choch_down:
            events.append("EXT_CHoCH↓")

        int_dir_str = {1: "↑ Bull", -1: "↓ Bear", 0: "─ Neutral"}.get(state.int_trend_dir, "?")
        ext_dir_str = {1: "↑ Bull", -1: "↓ Bear", 0: "─ Neutral"}.get(state.ext_trend_dir, "?")

        events_str = ", ".join(events) if events else "-"

        print(f"{str(bar.ts):<20} {bar.close:<8.1f} {int_dir_str:<8} {ext_dir_str:<8} {events_str:<30}")

    print("\n" + "="*80)
    print("✓ SMC Debug Complete!")
    print("="*80 + "\n")

    print("Next steps:")
    print("1. Compare swing counts with NinjaTrader indicator")
    print("2. Verify ext swings are ~1/5 to 1/10 of int swings")
    print("3. Check BOS/CHoCH events make sense with price action")
    print()

    return smc_states


if __name__ == "__main__":
    # Default: GC M1 data, 300 bars
    data_path = "/home/user/modeloutcome/data/raw/smc_export_gc_m1_v3.jsonl"

    # Process 300 bars (about 5 hours of M1 data)
    smc_states = debug_smc_on_sample(data_path, n=300)

    print(f"\nProcessed {len(smc_states)} bars successfully!")
