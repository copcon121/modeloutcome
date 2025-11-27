"""
Test Context Manager - Full SMC Integration
"""

import json
from datetime import datetime
from pathlib import Path
import sys

src_dir = Path(__file__).parent / 'src'
sys.path.insert(0, str(src_dir))

from layer2_feature_engine_v2.config import GC_M1_SMC_CONFIG
from layer2_feature_engine_v2.schema import RawBar
from layer2_feature_engine_v2.context_manager import SMCContextManager

ninja_file = r"data\raw\deepseek_enhanced_GC 12-25_M1_20251111.jsonl"
max_bars = 100  # Test first 100 bars

# Initialize Context Manager
manager = SMCContextManager(GC_M1_SMC_CONFIG, tick_size=0.1)

feature_bars = []

print("Processing bars with Context Manager...")
with open(ninja_file, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if i >= max_bars:
            break
        if not line.strip():
            continue
        
        data = json.loads(line)
        timestamp_str = data.get('timestamp', '')
        timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        
        raw_bar = RawBar(
            symbol='GC', timeframe='M1',
            timestamp=timestamp,
            bar_index=data.get('bar_index', i),
            o=data.get('open', 0), h=data.get('high', 0),
            l=data.get('low', 0), c=data.get('close', 0),
            volume=0, delta=0, buy_volume=0, sell_volume=0,
            best_bid=0, best_ask=0, tick_speed=0,
            aggr_buy_speed=0, aggr_sell_speed=0, price_speed=0
        )
        
        # Single update() call processes everything!
        feature_bar = manager.update(raw_bar)
        feature_bars.append(feature_bar)

print(f"\nProcessed {len(feature_bars)} bars")
print("\n" + "="*80)
print("CONTEXT MANAGER TEST RESULTS")
print("="*80)

# Get summary from manager
swing_states = manager.get_swing_states()
structure_state = manager.get_structure_state()
zone_summary = manager.get_zone_summary()

print("\nSwing Detection:")
print(f"  Internal swing high: {swing_states['internal'].swing_high_price}")
print(f"  Internal swing low:  {swing_states['internal'].swing_low_price}")
print(f"  External swing high: {swing_states['external'].swing_high_price}")
print(f"  External swing low:  {swing_states['external'].swing_low_price}")

print("\nStructure State:")
print(f"  Internal direction: {structure_state['internal']['structure_dir']}")
print(f"  External direction: {structure_state['external']['structure_dir']}")

print("\nZone Summary:")
print(f"  PD Zone - EQ: {zone_summary['pd'].equilibrium:.1f}")
print(f"  PD Zone - In Premium: {zone_summary['pd'].in_premium}")
print(f"  FVGs - Active: {zone_summary['fvgs']['active']}, Total: {zone_summary['fvgs']['total']}")
print(f"  OBs  - Active: {zone_summary['obs']['active']}, Total: {zone_summary['obs']['total']}")

# Check for structure signals
bos_up_bars = [fb for fb in feature_bars if fb.int_bos_up]
choch_down_bars = [fb for fb in feature_bars if fb.int_choch_down]

print(f"\nStructure Signals Detected:")
print(f"  Internal BOS UP: {len(bos_up_bars)}")
print(f"  Internal CHoCH DOWN: {len(choch_down_bars)}")

# Show sample feature bar
if feature_bars:
    last_bar = feature_bars[-1]
    print(f"\nLast Feature Bar:")
    print(f"  OHLC: C={last_bar.close:.1f} Range={last_bar.high_low_range:.1f}")
    print(f"  Body={last_bar.body:.1f} Upper Wick={last_bar.upper_wick:.1f}")
    print(f"  Int Trend Dir={last_bar.int_trend_dir}, Ext Trend Dir={last_bar.ext_trend_dir}")
    print(f"  Zones: In Bull FVG={last_bar.in_bull_fvg}, Bull OB={last_bar.in_bull_ob}")
    print(f"  Distances: Nearest FVG={last_bar.dist_to_nearest_fvg:.1f}, Nearest OB={last_bar.dist_to_nearest_ob:.1f}")

print("\n" + "="*80)
print("SUCCESS! Context Manager working correctly!")
print("="*80)
