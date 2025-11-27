"""
Test Volume Profile - Standalone
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

import json
from datetime import datetime
from collections import defaultdict

from layer2_feature_engine_v2.schema import RawBar
from layer2_feature_engine_v2.volume_profile import VolumeProfileBuilder, GC_M1_VP_CONFIG

print("="*80)
print("VOLUME PROFILE TEST")
print("="*80)
print(f"Mode: {GC_M1_VP_CONFIG.mode}")
print(f"Sessions: {[s.name for s in GC_M1_VP_CONFIG.sessions]}")
print("="*80)
print()

# Initialize VP builder
vp_builder = VolumeProfileBuilder(GC_M1_VP_CONFIG)

# Track profiles
profiles = defaultdict(lambda: {
    'poc': 0.0,
    'val': 0.0,
    'vah': 0.0,
    'total_vol': 0.0,
    'bars': 0
})

# Track last bars for output
last_bars = []

# Load and process bars
jsonl_file = r"data\raw\smc_export_gc_m1_v3.jsonl"
max_bars = 500

print(f"Processing {max_bars} bars from {jsonl_file}...")

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
            volume=bar_data.get('volume', 0.0),  # FROM BAR OBJECT!
            delta=bar_data.get('delta', 0.0),
            buy_volume=bar_data.get('buy_volume', 0.0),
            sell_volume=bar_data.get('sell_volume', 0.0),
            best_bid=bar_data.get('best_bid', bar_data.get('c', 0.0)),
            best_ask=bar_data.get('best_ask', bar_data.get('c', 0.0)),
            tick_speed=data.get('tick_features', {}).get('tick_speed', 0.0),
            aggr_buy_speed=data.get('tick_features', {}).get('aggr_buy_speed', 0.0),
            aggr_sell_speed=data.get('tick_features', {}).get('aggr_sell_speed', 0.0),
            price_speed=data.get('tick_features', {}).get('price_speed', 0.0)
        )
        
        # Update VP
        vp_state = vp_builder.update(raw_bar)
        
        # Track profile
        pid = vp_state.profile_id
        profiles[pid]['poc'] = vp_state.poc_price
        profiles[pid]['val'] = vp_state.val_price
        profiles[pid]['vah'] = vp_state.vah_price
        profiles[pid]['total_vol'] = vp_state.total_volume
        profiles[pid]['bars'] += 1
        
        # Save last bars
        last_bars.append({
            'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'close': raw_bar.c,
            'profile_id': pid,
            'poc': vp_state.poc_price,
            'val': vp_state.val_price,
            'vah': vp_state.vah_price,
            'in_va': vp_state.in_value_area,
            'dist_to_poc': vp_state.dist_to_poc
        })

# Print profiles summary
print()
print(f"Profiles detected: {len(profiles)}")
print()
print("Profile Summary:")
print("-" * 90)
print(f"{'Profile ID':30} {'POC':>8} {'VAL':>8} {'VAH':>8} {'Total Vol':>12} {'Bars':>6}")
print("-" * 90)
for pid, info in sorted(profiles.items()):
    print(f"{pid:30} {info['poc']:8.1f} {info['val']:8.1f} {info['vah']:8.1f} "
          f"{info['total_vol']:12.0f} {info['bars']:6}")

# Print last bars
print()
print("Last 10 bars:")
print("-" * 110)
print(f"{'Time':20} {'Close':>8} {'Profile':30} {'POC':>8} {'VAL':>8} {'VAH':>8} {'InVA':>6} {'Dist':>7}")
print("-" * 110)
for bar_info in last_bars[-10:]:
    print(f"{bar_info['timestamp']:20} "
          f"{bar_info['close']:8.1f} "
          f"{bar_info['profile_id']:30} "
          f"{bar_info['poc']:8.1f} "
          f"{bar_info['val']:8.1f} "
          f"{bar_info['vah']:8.1f} "
          f"{str(bar_info['in_va']):>6} "
          f"{bar_info['dist_to_poc']:+7.0f}")

print()
print("="*80)
print("VP TEST COMPLETE")
print("="*80)
