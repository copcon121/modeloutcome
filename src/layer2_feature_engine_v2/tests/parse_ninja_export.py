"""
Parse NinjaTrader Enhanced Exporter JSONL and extract SMC signals
For validation against Python implementation
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
from collections import defaultdict

def parse_ninja_export(jsonl_path: str) -> List[Dict[str, Any]]:
    """Parse NinjaTrader enhanced export JSONL"""
    bars = []
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            
            try:
                data = json.loads(line)
                bars.append(data)
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping malformed JSON at line {line_num}: {e}")
                continue
    
    return bars


def extract_swing_signals(bars: List[Dict]) -> Dict[str, List]:
    """Extract swing-related signals from bars"""
    swings = {
        'internal_high': [],
        'internal_low': [],
        'external_high': [],
        'external_low': []
    }
    
    prev_int_swing_high = None
    prev_int_swing_low = None
    prev_ext_swing_high = None
    prev_ext_swing_low = None
    
    for bar in bars:
        bar_data = bar.get('bar', {})
        timestamp = bar.get('timestamp', '')
        bar_index = bar.get('bar_index', -1)
        
        # Internal swings
        int_swing_high = bar_data.get('last_swing_high')
        int_swing_low = bar_data.get('last_swing_low')
        
        # Detect NEW swing (value changed)
        if int_swing_high and int_swing_high != prev_int_swing_high and int_swing_high != 0:
            swings['internal_high'].append({
                'timestamp': timestamp,
                'bar_index': bar_index,
                'price': int_swing_high
            })
            prev_int_swing_high = int_swing_high
        
        if int_swing_low and int_swing_low != prev_int_swing_low and int_swing_low != 0:
            swings['internal_low'].append({
                'timestamp': timestamp,
                'bar_index': bar_index,
                'price': int_swing_low
            })
            prev_int_swing_low = int_swing_low
        
        # External swings (same logic)
        # Note: enhanced exporter might not have separate external swing tracking
        # We'll use swing_pattern or other indicators
    
    return swings


def extract_bos_choch_signals(bars: List[Dict]) -> Dict[str, List]:
    """Extract BOS/CHoCH signals"""
    signals = {
        'bos_up': [],
        'bos_down': [],
        'choch_up': [],
        'choch_down': []
    }
    
    for bar in bars:
        bar_data = bar.get('bar', {})
        timestamp = bar.get('timestamp', '')
        bar_index = bar.get('bar_index', -1)
        close_price = bar.get('close', 0)
        
        # External BOS/CHoCH
        if bar_data.get('ext_bos_up'):
            signals['bos_up'].append({
                'timestamp': timestamp,
                'bar_index': bar_index,
                'price': close_price,
                'layer': 'EXTERNAL'
            })
        
        if bar_data.get('ext_bos_down'):
            signals['bos_down'].append({
                'timestamp': timestamp,
                'bar_index': bar_index,
                'price': close_price,
                'layer': 'EXTERNAL'
            })
        
        if bar_data.get('ext_choch_up'):
            signals['choch_up'].append({
                'timestamp': timestamp,
                'bar_index': bar_index,
                'price': close_price,
                'layer': 'EXTERNAL'
            })
        
        if bar_data.get('ext_choch_down'):
            signals['choch_down'].append({
                'timestamp': timestamp,
                'bar_index': bar_index,
                'price': close_price,
                'layer': 'EXTERNAL'
            })
        
       # Internal BOS/CHoCH
        if bar_data.get('int_bos_up'):
            signals['bos_up'].append({
                'timestamp': timestamp,
                'bar_index': bar_index,
                'price': close_price,
                'layer': 'INTERNAL'
            })
        
        if bar_data.get('int_bos_down'):
            signals['bos_down'].append({
                'timestamp': timestamp,
                'bar_index': bar_index,
                'price': close_price,
                'layer': 'INTERNAL'
            })
        
        if bar_data.get('int_choch_up'):
            signals['choch_up'].append({
                'timestamp': timestamp,
                'bar_index': bar_index,
                'price': close_price,
                'layer': 'INTERNAL'
            })
        
        if bar_data.get('int_choch_down'):
            signals['choch_down'].append({
                'timestamp': timestamp,
                'bar_index': bar_index,
                'price': close_price,
                'layer': 'INTERNAL'
            })
    
    return signals


def print_signal_summary(bars: List[Dict], swings: Dict, bos_choch: Dict):
    """Print summary of extracted signals"""
    print("=" * 80)
    print("NINJATRADER EXPORT ANALYSIS")
    print("=" * 80)
    print(f"Total bars: {len(bars)}")
    
    if bars:
        first_bar = bars[0]
        last_bar = bars[-1]
        print(f"Time range: {first_bar.get('timestamp')} to {last_bar.get('timestamp')}")
        print(f"Bar index range: {first_bar.get('bar_index')} to {last_bar.get('bar_index')}")
    
    print()
    print("=" * 80)
    print("SWING SIGNALS")
    print("=" * 80)
    print(f"Internal Swing Highs: {len(swings['internal_high'])}")
    print(f"Internal Swing Lows:  {len(swings['internal_low'])}")
    print(f"External Swing Highs: {len(swings['external_high'])}")
    print(f"External Swing Lows:  {len(swings['external_low'])}")
    
    # Show first few swings
    if swings['internal_high']:
        print("\nFirst 5 Internal Highs:")
        for s in swings['internal_high'][:5]:
            print(f"  {s['timestamp']} | Bar {s['bar_index']} @ {s['price']}")
    
    if swings['internal_low']:
        print("\nFirst 5 Internal Lows:")
        for s in swings['internal_low'][:5]:
            print(f"  {s['timestamp']} | Bar {s['bar_index']} @ {s['price']}")
    
    print()
    print("=" * 80)
    print("BOS/CHoCH SIGNALS")
    print("=" * 80)
    print(f"BOS Up:    {len(bos_choch['bos_up'])}")
    print(f"BOS Down:  {len(bos_choch['bos_down'])}")
    print(f"CHoCH Up:  {len(bos_choch['choch_up'])}")
    print(f"CHoCH Down: {len(bos_choch['choch_down'])}")
    
    # Show first few
    for signal_type in ['bos_up', 'bos_down', 'choch_up', 'choch_down']:
        signals = bos_choch[signal_type]
        if signals:
            print(f"\nFirst 3 {signal_type.upper()}:")
            for s in signals[:3]:
                print(f"  {s['timestamp']} | Bar {s['bar_index']} @ {s['price']} [{s['layer']}]")
    
    print()
    print("=" * 80)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python parse_ninja_export.py <jsonl_file>")
        print("\nExample:")
        print('  python parse_ninja_export.py "data/raw/deepseek_enhanced_GC 12-25_M1_20251111.jsonl"')
        sys.exit(1)
    
    jsonl_path = sys.argv[1]
    
    if not Path(jsonl_path).exists():
        print(f"Error: File not found: {jsonl_path}")
        sys.exit(1)
    
    print(f"Parsing {jsonl_path}...")
    print()
    
    bars = parse_ninja_export(jsonl_path)
    swings = extract_swing_signals(bars)
    bos_choch = extract_bos_choch_signals(bars)
    
    print_signal_summary(bars, swings, bos_choch)
    
    print("\n✅ Parsing complete!")
    print(f"Extracted {len(bars)} bars with signals ready for validation")
