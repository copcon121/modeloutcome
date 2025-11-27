"""
Test Event Filtering System
Validate P1/P2/P3 filtering with real data
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

import json
from datetime import datetime

from layer2_feature_engine_v2.schema import RawBar, FeatureBar
from layer2_feature_engine_v2.context_manager import SMCContextManager
from layer2_feature_engine_v2.config import GC_M1_SMC_CONFIG
from layer2_feature_engine_v2.event_filter import EventFilter

print("="*80)
print("EVENT FILTERING TEST")
print("="*80)

# Load bars and build features
jsonl_file = r"data\raw\smc_export_gc_m1_v3.jsonl"
max_bars = 500

print(f"\nLoading {max_bars} bars from {jsonl_file}...")

manager = SMCContextManager(GC_M1_SMC_CONFIG, tick_size=0.1)
feature_bars = []

with open(jsonl_file, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if i >= max_bars:
            break
        
        if not line.strip():
            continue
        
        data = json.loads(line)
        timestamp_str = data.get('timestamp', '')
        timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        
        # Extract from nested 'bar' object
        bar_data = data.get('bar', {})
        
        raw_bar = RawBar(
            symbol=data.get('symbol', 'GC'),
            timeframe=data.get('timeframe', 'M1'),
            timestamp=timestamp,
            bar_index=data.get('bar_index', i),
            o=bar_data.get('o', 0.0),
            h=bar_data.get('h', 0.0),
            l=bar_data.get('l', 0.0),
            c=bar_data.get('c', 0.0),
            volume=bar_data.get('volume', 0.0),
            delta=bar_data.get('delta', 0.0),
            buy_volume=bar_data.get('buy_volume', 0.0),
            sell_volume=bar_data.get('sell_volume', 0.0),
            best_bid=bar_data.get('best_bid', bar_data.get('c', 0.0)),
            best_ask=bar_data.get('best_ask', bar_data.get('c', 0.0)),
            tick_speed=data.get('tick_features', {}).get('tick_speed', 0.0),
            aggr_buy_speed=data.get('tick_features', {}).get('aggr_buy_speed', 0.0),
            aggr_sell_speed=data.get('tick_features', {}).get('aggr_sell_speed', 0.0),
            price_speed=data.get('tick_features', {}).get('price_speed', 0.0),
            vwap_daily=bar_data.get('vwap_daily', 0.0)
        )
        
        # Build features
        feature_bar = manager.update(raw_bar)
        feature_bars.append(feature_bar)

print(f"Built {len(feature_bars)} feature bars")

# Apply event filtering
print("\n" + "="*80)
print("APPLYING EVENT FILTERS")
print("="*80)

event_filter = EventFilter()

# Compute flags (single pass)
print("\nComputing event flags...")
flags_list = event_filter.compute_flags(feature_bars)

# Get filter statistics
stats = event_filter.get_filter_stats(flags_list)

print(f"\nFilter Statistics:")
print(f"  Total bars: {stats['total_bars']}")
print(f"  P1 (Strict):   {stats['p1_strict']['count']:4} bars ({stats['p1_strict']['pct']:.1f}%)")
print(f"  P2 (Moderate): {stats['p2_moderate']['count']:4} bars ({stats['p2_moderate']['pct']:.1f}%)")
print(f"  P3 (Loose):    {stats['p3_loose']['count']:4} bars ({stats['p3_loose']['pct']:.1f}%)")

# Apply Phase 2 filter (recommended)
mask_p2 = event_filter.apply_phase2_filter(flags_list)
filtered_bars_p2 = [fb for fb, keep in zip(feature_bars, mask_p2) if keep]

print(f"\nPhase 2 (Moderate) filter results:")
print(f"  Original: {len(feature_bars)} bars")
print(f"  Filtered: {len(filtered_bars_p2)} bars")
print(f"  Reduction: {(1 - len(filtered_bars_p2)/len(feature_bars))*100:.1f}%")

# Tag bars with phases
tags = event_filter.tag_bars_with_phase(flags_list)
phase_counts = {
    'P1': tags.count('P1'),
    'P2': tags.count('P2'),
    'P3': tags.count('P3'),
    'None': tags.count('None')
}

print(f"\nPhase tagging:")
for phase, count in phase_counts.items():
    print(f"  {phase}: {count} bars")

# Show some examples
print("\n" + "="*80)
print("SAMPLE FILTERED BARS (First 10 of P2)")
print("="*80)

p2_indices = [i for i, keep in enumerate(mask_p2) if keep][:10]
for idx in p2_indices:
    fb = feature_bars[idx]
    flags = flags_list[idx]
    print(f"\nBar {idx}:")
    print(f"  Close: {fb.close:.1f}")
    print(f"  Flags: BOS/CHoCH={flags.has_bos_choch}, "
          f"InZone={flags.in_zone}, HighVol={flags.high_volatility}")
    if fb.int_bos_up or fb.int_bos_down:
        print(f"  -> Internal BOS: UP={fb.int_bos_up} DOWN={fb.int_bos_down}")
    if fb.ext_bos_up or fb.ext_bos_down:
        print(f"  -> External BOS: UP={fb.ext_bos_up} DOWN={fb.ext_bos_down}")

print("\n" + "="*80)
print("EVENT FILTERING TEST COMPLETE")
print("="*80)
