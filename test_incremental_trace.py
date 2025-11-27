"""Trace incremental swing detection logic"""
import sys
sys.path.insert(0, '/home/user/modeloutcome/src')

from layer2_feature_engine.core.data_loader import load_raw_bars
from layer2_feature_engine.smc.swing import InternalSwingDetector
from layer2_feature_engine.smc.config import GC_M1_SMC_CONFIG

bars = load_raw_bars("/home/user/modeloutcome/data/raw/smc_export_gc_m1_v3.jsonl")[:50]

print(f"Total bars: {len(bars)}")
print(f"Config: fractal_left={GC_M1_SMC_CONFIG.fractal_left}, fractal_right={GC_M1_SMC_CONFIG.fractal_right}")
print(f"Config: min_bars_between_swings={GC_M1_SMC_CONFIG.min_bars_between_swings}\n")

detector = InternalSwingDetector(GC_M1_SMC_CONFIG)

# Manually trace first 20 bars
for i in range(20):
    fractal_check_idx = i - GC_M1_SMC_CONFIG.fractal_right

    print(f"Bar {i}: close={bars[i].close:.1f}, fractal_check_idx={fractal_check_idx}")

    if fractal_check_idx >= GC_M1_SMC_CONFIG.fractal_left:
        pivot_bar = bars[fractal_check_idx]

        # Check fractal high
        is_fractal_high = True
        for j in range(fractal_check_idx - GC_M1_SMC_CONFIG.fractal_left, fractal_check_idx):
            if bars[j].high >= pivot_bar.high:
                is_fractal_high = False
                break

        if is_fractal_high:
            for j in range(fractal_check_idx + 1, fractal_check_idx + GC_M1_SMC_CONFIG.fractal_right + 1):
                if bars[j].high >= pivot_bar.high:
                    is_fractal_high = False
                    break

        # Check fractal low
        is_fractal_low = True
        for j in range(fractal_check_idx - GC_M1_SMC_CONFIG.fractal_left, fractal_check_idx):
            if bars[j].low <= pivot_bar.low:
                is_fractal_low = False
                break

        if is_fractal_low:
            for j in range(fractal_check_idx + 1, fractal_check_idx + GC_M1_SMC_CONFIG.fractal_right + 1):
                if bars[j].low <= pivot_bar.low:
                    is_fractal_low = False
                    break

        if is_fractal_high or is_fractal_low:
            print(f"  → Fractal at idx {fractal_check_idx}: high={is_fractal_high}, low={is_fractal_low}, price H={pivot_bar.high:.1f} L={pivot_bar.low:.1f}")

    # Now update with actual detector
    state = detector.update(bars, i)

    if state.swing_high_idx >= 0 or state.swing_low_idx >= 0:
        print(f"  ✓ CONFIRMED: swing_high_idx={state.swing_high_idx}, swing_low_idx={state.swing_low_idx}")

print("\n" + "="*80)
print("SUMMARY:")
print(f"Final swing_high_idx: {detector.state.swing_high_idx}")
print(f"Final swing_low_idx: {detector.state.swing_low_idx}")
