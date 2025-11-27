"""Check if swings are detected but not counted"""
import sys
sys.path.insert(0, '/home/user/modeloutcome/src')

from layer2_feature_engine.core.data_loader import load_raw_bars
from layer2_feature_engine.smc.smc_engine import SMCEngine
from layer2_feature_engine.smc.config import GC_M1_SMC_CONFIG

bars = load_raw_bars("/home/user/modeloutcome/data/raw/smc_export_gc_m1_v3.jsonl")[:300]

engine = SMCEngine(GC_M1_SMC_CONFIG)

int_swing_high_count_method1 = 0  # bars_ago == 0
int_swing_low_count_method1 = 0

int_swing_high_count_method2 = 0  # track changes in swing_high_idx
int_swing_low_count_method2 = 0

prev_swing_high_idx = -1
prev_swing_low_idx = -1

for i in range(len(bars)):
    smc_state = engine.update(bars, i)

    # Method 1: Original counting (bars_ago == 0)
    if smc_state.int_swing_high_bars_ago == 0:
        int_swing_high_count_method1 += 1
        print(f"Bar {i}: NEW swing high at bar {i} (bars_ago=0)")

    if smc_state.int_swing_low_bars_ago == 0:
        int_swing_low_count_method1 += 1
        print(f"Bar {i}: NEW swing low at bar {i} (bars_ago=0)")

    # Method 2: Track changes in swing_high_idx
    current_swing_high_idx = engine.int_swing_detector.state.swing_high_idx
    current_swing_low_idx = engine.int_swing_detector.state.swing_low_idx

    if current_swing_high_idx != prev_swing_high_idx and current_swing_high_idx >= 0:
        int_swing_high_count_method2 += 1
        bars_ago = i - current_swing_high_idx
        print(f"Bar {i}: CONFIRMED swing high at bar {current_swing_high_idx} (bars_ago={bars_ago})")

    if current_swing_low_idx != prev_swing_low_idx and current_swing_low_idx >= 0:
        int_swing_low_count_method2 += 1
        bars_ago = i - current_swing_low_idx
        print(f"Bar {i}: CONFIRMED swing low at bar {current_swing_low_idx} (bars_ago={bars_ago})")

    prev_swing_high_idx = current_swing_high_idx
    prev_swing_low_idx = current_swing_low_idx

print("\n" + "="*80)
print("COUNTING RESULTS:")
print("="*80)
print(f"Method 1 (bars_ago == 0):")
print(f"  Swing Highs: {int_swing_high_count_method1}")
print(f"  Swing Lows:  {int_swing_low_count_method1}")
print()
print(f"Method 2 (track idx changes):")
print(f"  Swing Highs: {int_swing_high_count_method2}")
print(f"  Swing Lows:  {int_swing_low_count_method2}")
