#!/usr/bin/env python3
"""
Build GC M1 Features for NEW DATA (OOS Phase 3)

This script processes raw JSONL files from data/raw/new_data/ and generates
enhanced bar features using the SAME SMC feature pipeline as training data.

Output: state_enc_v1/artifacts/gc_m1_new/bars_enhanced_gc_m1_newdata.jsonl

Usage:
    python scripts/build_gc_m1_features_newdata.py
"""

import os
import sys
import glob
import json
from datetime import datetime
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.layer2_feature_engine_v2.context_manager import SMCContextManager
from src.layer2_feature_engine_v2.config import GC_M1_SMC_CONFIG
from src.layer2_feature_engine_v2.schema import RawBar


def load_raw_bar(data: dict) -> RawBar:
    """Convert JSON dict to RawBar."""
    ts_str = data['timestamp']
    ts = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
    
    bar_data = data['bar']
    tick_data = data.get('tick_features', {})
    
    return RawBar(
        symbol=data.get('symbol', 'GC'),
        timeframe=data.get('timeframe', 'M1'),
        timestamp=ts,
        bar_index=data['bar_index'],
        o=bar_data['o'],
        h=bar_data['h'],
        l=bar_data['l'],
        c=bar_data['c'],
        volume=bar_data['volume'],
        delta=bar_data.get('delta', 0),
        buy_volume=bar_data.get('buy_volume', 0),
        sell_volume=bar_data.get('sell_volume', 0),
        best_bid=bar_data.get('best_bid', bar_data['c']),
        best_ask=bar_data.get('best_ask', bar_data['c']),
        tick_speed=tick_data.get('tick_speed', 0),
        aggr_buy_speed=tick_data.get('aggr_buy_speed', 0),
        aggr_sell_speed=tick_data.get('aggr_sell_speed', 0),
        price_speed=tick_data.get('price_speed', bar_data['h'] - bar_data['l']),
        vwap_daily=bar_data.get('vwap_daily', 0.0)
    )


def get_session(hour: int) -> str:
    """Determine trading session from hour (UTC)."""
    if 2 <= hour < 8:
        return 'ASIA'
    elif 8 <= hour < 14:
        return 'LDN'
    else:
        return 'NY'


def process_file(input_path: Path, manager: SMCContextManager) -> list:
    """Process single JSONL file and return enhanced bars."""
    print(f"  Processing: {input_path.name}")
    
    enhanced_bars = []
    
    with open(input_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            
            try:
                data = json.loads(line)
                raw_bar = load_raw_bar(data)
                feature_bar = manager.update(raw_bar)
                
                # Convert to dict
                fb_dict = feature_bar.to_dict()
                
                # Add metadata (same fields as original bars_enhanced)
                fb_dict['time'] = raw_bar.timestamp.isoformat()
                fb_dict['bar_time'] = raw_bar.timestamp.isoformat()
                fb_dict['bar_index'] = raw_bar.bar_index
                fb_dict['symbol'] = raw_bar.symbol
                fb_dict['timeframe'] = raw_bar.timeframe
                fb_dict['session'] = get_session(raw_bar.timestamp.hour)
                
                # Add raw OHLCV (for compatibility)
                fb_dict['o'] = raw_bar.o
                fb_dict['h'] = raw_bar.h
                fb_dict['l'] = raw_bar.l
                fb_dict['c'] = raw_bar.c
                fb_dict['open'] = raw_bar.o
                fb_dict['high'] = raw_bar.h
                fb_dict['low'] = raw_bar.l
                
                enhanced_bars.append(fb_dict)
                
            except Exception as e:
                print(f"    Error line {line_num}: {e}")
                continue
    
    print(f"    Processed {len(enhanced_bars)} bars")
    return enhanced_bars


def main():
    print("=" * 80)
    print("BUILD GC M1 FEATURES FOR NEW DATA (OOS Phase 3)")
    print("=" * 80)
    
    # Directories
    raw_dir = ROOT / "data/raw/new_data"
    output_dir = ROOT / "state_enc_v1/artifacts/gc_m1_new"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / "bars_enhanced_gc_m1_newdata.jsonl"
    
    # Find all raw data files
    files = sorted(raw_dir.glob("smc_export_gc_m1_v3_*.jsonl"))
    print(f"\nFound {len(files)} raw data files:")
    for f in files:
        print(f"  - {f.name}")
    
    if not files:
        print("ERROR: No raw data files found!")
        sys.exit(1)
    
    # Initialize SMC Context Manager (SAME config as training)
    print(f"\nInitializing SMCContextManager with GC_M1_SMC_CONFIG...")
    manager = SMCContextManager(GC_M1_SMC_CONFIG, tick_size=0.1)
    
    # Process all files
    print(f"\nProcessing files...")
    all_bars = []
    
    for input_path in files:
        # Reset manager for each file (fresh state per week)
        manager = SMCContextManager(GC_M1_SMC_CONFIG, tick_size=0.1)
        bars = process_file(input_path, manager)
        all_bars.extend(bars)
    
    # Sort by time
    all_bars.sort(key=lambda b: b.get('time', ''))
    
    # Save to JSONL
    print(f"\nSaving to: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        for bar in all_bars:
            f.write(json.dumps(bar) + '\n')
    
    # Statistics
    print(f"\n" + "=" * 80)
    print("FEATURE BUILD STATISTICS")
    print("=" * 80)
    
    dates = sorted(set(b['time'][:10] for b in all_bars))
    sessions = {}
    for b in all_bars:
        s = b.get('session', 'UNKNOWN')
        sessions[s] = sessions.get(s, 0) + 1
    
    print(f"  Total bars: {len(all_bars):,}")
    print(f"  Date range: {dates[0]} to {dates[-1]}")
    print(f"  Dates: {len(dates)}")
    print(f"  Sessions: {sessions}")
    
    # Feature count
    sample_bar = all_bars[0] if all_bars else {}
    feature_count = len([k for k in sample_bar.keys() if k not in ['time', 'bar_time', 'bar_index', 'symbol', 'timeframe', 'session', 'o', 'h', 'l', 'c', 'open', 'high', 'low']])
    print(f"  Features per bar: {feature_count}")
    
    print(f"\n" + "=" * 80)
    print(f"✅ Features built successfully!")
    print(f"   Output: {output_path}")
    print("=" * 80)


if __name__ == '__main__':
    main()
