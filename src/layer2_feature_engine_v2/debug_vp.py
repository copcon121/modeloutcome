"""
Volume Profile Debug Tools
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict
from collections import defaultdict

from ..schema import RawBar
from ..volume_profile import VolumeProfileBuilder, VPConfig, GC_M1_VP_CONFIG


def debug_vp_summary(
    jsonl_path: str,
    n: int = 300,
    vp_config: VPConfig = GC_M1_VP_CONFIG
) -> None:
    """
    Debug Volume Profile - load bars and show VP summary
    
    Args:
        jsonl_path: Path to JSONL file
        n: Number of bars to process
        vp_config: VP configuration
    """
    print("="*80)
    print("VOLUME PROFILE DEBUG SUMMARY")
    print("="*80)
    print(f"File: {jsonl_path}")
    print(f"Bars to process: {n}")
    print(f"Mode: {vp_config.mode}")
    if vp_config.mode == "session":
        print(f"Sessions: {[s.name for s in vp_config.sessions]}")
    print("="*80)
    print()
    
    # Initialize VP builder
    vp_builder = VolumeProfileBuilder(vp_config)
    
    # Track profiles
    profiles: Dict[str, dict] = defaultdict(lambda: {
        'poc': 0.0,
        'val': 0.0,
        'vah': 0.0,
        'total_vol': 0.0,
        'bars': 0
    })
    
    # Track last bars for output
    last_bars = []
    
    # Process bars
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            
            if not line.strip():
                continue
            
            data = json.loads(line)
            timestamp_str = data.get('timestamp', '')
            timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            
            raw_bar = RawBar(
                symbol=data.get('symbol', 'GC'),
                timeframe=data.get('timeframe', 'M1'),
                timestamp=timestamp,
                bar_index=data.get('bar_index', i),
                o=data.get('open', 0.0),
                h=data.get('high', 0.0),
                l=data.get('low', 0.0),
                c=data.get('close', 0.0),
                volume=data.get('volume', 0.0),
                delta=data.get('delta', 0.0),
                buy_volume=data.get('buy_volume', 0.0),
                sell_volume=data.get('sell_volume', 0.0),
                best_bid=data.get('best_bid', data.get('close', 0.0)),
                best_ask=data.get('best_ask', data.get('close', 0.0)),
                tick_speed=data.get('tick_speed', 0.0),
                aggr_buy_speed=data.get('aggr_buy_speed', 0.0),
                aggr_sell_speed=data.get('aggr_sell_speed', 0.0),
                price_speed=data.get('price_speed', 0.0)
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
    print(f"\nProfiles detected: {len(profiles)}")
    print()
    print("Profile Summary:")
    print("-" * 80)
    for pid, info in sorted(profiles.items()):
        print(f"  {pid:25} POC={info['poc']:7.1f} VAL={info['val']:7.1f} VAH={info['vah']:7.1f} "
              f"Vol={info['total_vol']:8.0f} Bars={info['bars']:3}")
    
    # Print last bars
    print()
    print("Last 10 bars:")
    print("-" * 80)
    print(f"{'Time':20} {'Close':>7} {'Profile':25} {'POC':>7} {'VAL':>7} {'VAH':>7} {'InVA':>5} {'Dist':>6}")
    print("-" * 80)
    for bar_info in last_bars[-10:]:
        print(f"{bar_info['timestamp']:20} "
              f"{bar_info['close']:7.1f} "
              f"{bar_info['profile_id']:25} "
              f"{bar_info['poc']:7.1f} "
              f"{bar_info['val']:7.1f} "
              f"{bar_info['vah']:7.1f} "
              f"{str(bar_info['in_va']):>5} "
              f"{bar_info['dist_to_poc']:+6.0f}")
    
    print()
    print("="*80)
    print("VP Debug Summary Complete")
    print("="*80)


if __name__ == "__main__":
    # Test with default file
    test_file = r"data\raw\deepseek_enhanced_GC 12-25_M1_20251111.jsonl"
    debug_vp_summary(test_file, n=500, vp_config=GC_M1_VP_CONFIG)
