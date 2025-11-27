"""
Detailed SMC test output for NinjaTrader comparison
Shows every detected swing with timestamp, price, and context
"""
import sys
sys.path.insert(0, '/home/user/modeloutcome/src')

from layer2_feature_engine.core.data_loader import load_raw_bars
from layer2_feature_engine.smc.smc_engine import SMCEngine
from layer2_feature_engine.smc.config import GC_M1_SMC_CONFIG

# Load data - FULL DATASET
bars = load_raw_bars("/home/user/modeloutcome/data/raw/smc_export_gc_m1_v3.jsonl")

print("="*100)
print("SMC 2-LAYER SWING DETECTION - DETAILED OUTPUT FOR NINJATRADER COMPARISON")
print("="*100)
print()
print("CONFIGURATION:")
print(f"  Internal Swing (wave 5):")
print(f"    - Fractal Left:  {GC_M1_SMC_CONFIG.fractal_left} bars")
print(f"    - Fractal Right: {GC_M1_SMC_CONFIG.fractal_right} bars")
print(f"    - Min Move:      {GC_M1_SMC_CONFIG.min_int_move_ticks} ticks")
print(f"    - Min Spacing:   {GC_M1_SMC_CONFIG.min_bars_between_swings} bars")
print()
print(f"  External Swing (wave 50):")
print(f"    - Min Move:      {GC_M1_SMC_CONFIG.min_ext_move_ticks} ticks")
print()
print(f"Total bars: {len(bars)}")
print(f"Period: {bars[0].ts} → {bars[-1].ts}")
print()

# Initialize engine
engine = SMCEngine(GC_M1_SMC_CONFIG)

# Track swing changes
prev_int_high_idx = -1
prev_int_low_idx = -1
prev_ext_high_idx = -1
prev_ext_low_idx = -1

int_swings = []
ext_swings = []

# Process bars
for i in range(len(bars)):
    smc_state = engine.update(bars, i)

    # Detect new internal swings
    current_int_high_idx = engine.int_swing_detector.state.swing_high_idx
    current_int_low_idx = engine.int_swing_detector.state.swing_low_idx

    if current_int_high_idx != prev_int_high_idx and current_int_high_idx >= 0:
        swing_bar = bars[current_int_high_idx]
        int_swings.append({
            'type': 'HIGH',
            'detected_at_bar': i,
            'swing_at_bar': current_int_high_idx,
            'timestamp': swing_bar.ts.strftime('%Y-%m-%d %H:%M:%S'),
            'price': swing_bar.high,
            'bars_ago': i - current_int_high_idx
        })

    if current_int_low_idx != prev_int_low_idx and current_int_low_idx >= 0:
        swing_bar = bars[current_int_low_idx]
        int_swings.append({
            'type': 'LOW',
            'detected_at_bar': i,
            'swing_at_bar': current_int_low_idx,
            'timestamp': swing_bar.ts.strftime('%Y-%m-%d %H:%M:%S'),
            'price': swing_bar.low,
            'bars_ago': i - current_int_low_idx
        })

    # Detect new external swings
    current_ext_high_idx = engine.ext_swing_detector.state.swing_high_idx
    current_ext_low_idx = engine.ext_swing_detector.state.swing_low_idx

    if current_ext_high_idx != prev_ext_high_idx and current_ext_high_idx >= 0:
        swing_bar = bars[current_ext_high_idx]
        ext_swings.append({
            'type': 'HIGH',
            'detected_at_bar': i,
            'swing_at_bar': current_ext_high_idx,
            'timestamp': swing_bar.ts.strftime('%Y-%m-%d %H:%M:%S'),
            'price': swing_bar.high,
            'bars_ago': i - current_ext_high_idx
        })

    if current_ext_low_idx != prev_ext_low_idx and current_ext_low_idx >= 0:
        swing_bar = bars[current_ext_low_idx]
        ext_swings.append({
            'type': 'LOW',
            'detected_at_bar': i,
            'swing_at_bar': current_ext_low_idx,
            'timestamp': swing_bar.ts.strftime('%Y-%m-%d %H:%M:%S'),
            'price': swing_bar.low,
            'bars_ago': i - current_ext_low_idx
        })

    prev_int_high_idx = current_int_high_idx
    prev_int_low_idx = current_int_low_idx
    prev_ext_high_idx = current_ext_high_idx
    prev_ext_low_idx = current_ext_low_idx

# Sort swings by detection order
int_swings.sort(key=lambda x: x['detected_at_bar'])
ext_swings.sort(key=lambda x: x['detected_at_bar'])

# Print internal swings
print("="*120)
print("INTERNAL SWINGS (wave 5) - ALL DETECTIONS")
print("="*120)
print(f"{'#':<4} {'Type':<6} {'Bar':<5} {'Date & Time':<22} {'Price':<10} {'Detected':<10} {'Lag':<5}")
print("-"*120)

for idx, swing in enumerate(int_swings, 1):
    print(f"{idx:<4} {swing['type']:<6} {swing['swing_at_bar']:<5} {swing['timestamp']:<22} "
          f"{swing['price']:<10.1f} Bar {swing['detected_at_bar']:<4} {swing['bars_ago']} bars")

print()
print(f"Total Internal Swings: {len(int_swings)}")
print(f"  - Highs: {len([s for s in int_swings if s['type'] == 'HIGH'])}")
print(f"  - Lows:  {len([s for s in int_swings if s['type'] == 'LOW'])}")
print()

# Print external swings
print("="*120)
print("EXTERNAL SWINGS (wave 50) - ALL DETECTIONS")
print("="*120)
print(f"{'#':<4} {'Type':<6} {'Bar':<5} {'Date & Time':<22} {'Price':<10} {'Detected':<10} {'Lag':<5}")
print("-"*120)

for idx, swing in enumerate(ext_swings, 1):
    print(f"{idx:<4} {swing['type']:<6} {swing['swing_at_bar']:<5} {swing['timestamp']:<22} "
          f"{swing['price']:<10.1f} Bar {swing['detected_at_bar']:<4} {swing['bars_ago']} bars")

print()
print(f"Total External Swings: {len(ext_swings)}")
print(f"  - Highs: {len([s for s in ext_swings if s['type'] == 'HIGH'])}")
print(f"  - Lows:  {len([s for s in ext_swings if s['type'] == 'LOW'])}")
print()

# Print first 100 bars with context
print("="*120)
print("FIRST 100 BARS - OHLC CONTEXT")
print("="*120)
print(f"{'Bar':<5} {'Date & Time':<22} {'Open':<9} {'High':<9} {'Low':<9} {'Close':<9} {'Swing':<25}")
print("-"*120)

for i in range(min(100, len(bars))):
    bar = bars[i]

    # Check if this bar has a swing
    swing_label = ""
    for swing in int_swings:
        if swing['swing_at_bar'] == i:
            swing_label = f"INT_{swing['type']}"
    for swing in ext_swings:
        if swing['swing_at_bar'] == i:
            swing_label += f" EXT_{swing['type']}"

    print(f"{i:<5} {bar.ts.strftime('%Y-%m-%d %H:%M:%S'):<22} {bar.open:<9.1f} {bar.high:<9.1f} "
          f"{bar.low:<9.1f} {bar.close:<9.1f} {swing_label:<25}")

print()
print("="*100)
print("SUMMARY")
print("="*100)
print(f"Internal Swings: {len([s for s in int_swings if s['type'] == 'HIGH'])} highs, "
      f"{len([s for s in int_swings if s['type'] == 'LOW'])} lows")
print(f"External Swings: {len([s for s in ext_swings if s['type'] == 'HIGH'])} highs, "
      f"{len([s for s in ext_swings if s['type'] == 'LOW'])} lows")
print(f"Ext/Int Ratio: {len(ext_swings) / max(len(int_swings), 1):.3f}")
print()
print("="*100)
print("NOTE FOR NINJATRADER COMPARISON:")
print("="*100)
print("1. Swing detection lag = fractal_right (2 bars)")
print("   Example: Swing at bar 10 is detected at bar 12")
print()
print("2. Internal swings use 2-2 fractal (left=2, right=2)")
print("   A swing high at bar i requires:")
print("   - bars[i].high > bars[i-2].high AND bars[i].high > bars[i-1].high")
print("   - bars[i].high > bars[i+1].high AND bars[i].high > bars[i+2].high")
print()
print("3. External swings require min 50 ticks movement from last ext swing")
print()
print("4. Compare the swing bar numbers and prices with your NinjaTrader indicator")
print("="*100)
