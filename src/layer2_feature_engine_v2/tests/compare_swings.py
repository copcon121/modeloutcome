"""
Compare Python swing detection with NinjaTrader export
Focus on is_swing_high/is_swing_low fields (local swings)
"""

import json
import sys
from pathlib import Path

# Add src to path
src_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(src_dir))

from layer2_feature_engine_v2.config import GC_M1_SMC_CONFIG
from layer2_feature_engine_v2.loaders import iter_raw_bars
from layer2_feature_engine_v2.smc_core.swing import InternalSwingDetector, ExternalSwingDetector


def load_ninja_swings(ninja_jsonl: str, max_bars=500):
    """Load STRUCTURAL swing signals from NinjaTrader export"""
    swings = {
        'swing_highs': [],
        'swing_lows': [],
        'all_bars': []
    }
    
    prev_swing_high = None
    prev_swing_low = None
    
    with open(ninja_jsonl, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= max_bars:
                break
            if not line.strip():
                continue
            
            data = json.loads(line)
            bar_index = data.get('bar_index', -1)
            timestamp = data.get('timestamp', '')
            bar_data = data.get('bar', {})
            
            # Get STRUCTURAL swing levels (last_swing_high/low)
            current_swing_high = bar_data.get('last_swing_high')
            current_swing_low = bar_data.get('last_swing_low')
            
            swings['all_bars'].append({
                'bar_index': bar_index,
                'timestamp': timestamp,
                'swing_high': current_swing_high,
                'swing_low': current_swing_low
            })
            
            # Detect NEW structural swing (value changed and != 0)
            if current_swing_high and current_swing_high != 0:
                if current_swing_high != prev_swing_high:
                    swings['swing_highs'].append({
                        'bar_index': bar_index,
                        'timestamp': timestamp,
                        'price': current_swing_high
                    })
                    prev_swing_high = current_swing_high
            
            if current_swing_low and current_swing_low != 0:
                if current_swing_low != prev_swing_low:
                    swings['swing_lows'].append({
                        'bar_index': bar_index,
                        'timestamp': timestamp,
                        'price': current_swing_low
                    })
                    prev_swing_low = current_swing_low
    
    return swings


def run_python_swings(raw_jsonl: str, max_bars=500):
    """Run Python swing detector"""
    detector = InternalSwingDetector(GC_M1_SMC_CONFIG, tick_size=0.1)
    
    swings = {
        'swing_highs': [],
        'swing_lows': []
    }
    
    prev_high_idx = -1
    prev_low_idx = -1
    
    bar_count = 0
    for bar in iter_raw_bars(raw_jsonl):
        bar_count += 1
        if bar_count > max_bars:
            break
        
        state = detector.update(bar)
        
        # Detect NEW swing
        if state.swing_high_bar_index != prev_high_idx and state.swing_high_bar_index != -1:
            swings['swing_highs'].append({
                'bar_index': state.swing_high_bar_index,
                'timestamp': bar.timestamp.isoformat(),
                'price': state.swing_high_price
            })
            prev_high_idx = state.swing_high_bar_index
        
        if state.swing_low_bar_index != prev_low_idx and state.swing_low_bar_index != -1:
            swings['swing_lows'].append({
                'bar_index': state.swing_low_bar_index,
                'timestamp': bar.timestamp.isoformat(),
                'price': state.swing_low_price
            })
            prev_low_idx = state.swing_low_bar_index
    
    return swings


def compare_swings(ninja_swings, python_swings):
    """Compare swing detection results"""
    print("=" * 80)
    print("SWING DETECTION COMPARISON")
    print("=" * 80)
    print()
    
    # Counts
    ninja_highs = ninja_swings['swing_highs']
    ninja_lows = ninja_swings['swing_lows']
    python_highs = python_swings['swing_highs']
    python_lows = python_swings['swing_lows']
    
    print(f"Swing Highs - NinjaTrader: {len(ninja_highs)}, Python: {len(python_highs)}")
    print(f"Swing Lows  - NinjaTrader: {len(ninja_lows)}, Python: {len(python_lows)}")
    print()
    
    # Compare first 10 highs
    print("=" * 80)
    print("FIRST 10 SWING HIGHS COMPARISON")
    print("=" * 80)
    print(f"{'#':<4} {'NinjaTrader':<40} {'Python':<40} {'Match'}")
    print("-" * 80)
    
    for i in range(min(10, len(ninja_highs), len(python_highs))):
        ninja = ninja_highs[i]
        python = python_highs[i] if i < len(python_highs) else None
        
        ninja_str = f"Bar {ninja['bar_index']} @ {ninja['price']}"
        python_str = f"Bar {python['bar_index']} @ {python['price']}" if python else "N/A"
        
        match = "OK" if python and ninja['bar_index'] == python['bar_index'] else "DIFF"
        
        print(f"{i+1:<4} {ninja_str:<40} {python_str:<40} {match}")
    
    print()
    print("=" * 80)
    print("FIRST 10 SWING LOWS COMPARISON")
    print("=" * 80)
    print(f"{'#':<4} {'NinjaTrader':<40} {'Python':<40} {'Match'}")
    print("-" * 80)
    
    for i in range(min(10, len(ninja_lows), len(python_lows))):
        ninja = ninja_lows[i]
        python = python_lows[i] if i < len(python_lows) else None
        
        ninja_str = f"Bar {ninja['bar_index']} @ {ninja['price']}"
        python_str = f"Bar {python['bar_index']} @ {python['price']}" if python else "N/A"
        
        match = "OK" if python and ninja['bar_index'] == python['bar_index'] else "DIFF"
        
        print(f"{i+1:<4} {ninja_str:<40} {python_str:<40} {match}")
    
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    # Calculate match rate
    high_matches = sum(1 for i in range(min(len(ninja_highs), len(python_highs))) 
                       if ninja_highs[i]['bar_index'] == python_highs[i]['bar_index'])
    low_matches = sum(1 for i in range(min(len(ninja_lows), len(python_lows)))
                      if ninja_lows[i]['bar_index'] == python_lows[i]['bar_index'])
    
    high_match_rate = high_matches / min(len(ninja_highs), len(python_highs)) * 100 if min(len(ninja_highs), len(python_highs)) > 0 else 0
    low_match_rate = low_matches / min(len(ninja_lows), len(python_lows)) * 100 if min(len(ninja_lows), len(python_lows)) > 0 else 0
    
    print(f"Swing High Match Rate: {high_match_rate:.0f}% ({high_matches}/{min(len(ninja_highs), len(python_highs))})")
    print(f"Swing Low Match Rate:  {low_match_rate:.0f}% ({low_matches}/{min(len(ninja_lows), len(python_lows))})")
    
    if high_match_rate < 80 or low_match_rate < 80:
        print("\n[WARNING] Match rate < 80% - Python logic needs adjustment!")
    else:
        print("\n[OK] Good match rate!")


if __name__ == "__main__":
    ninja_file = r"data\raw\deepseek_enhanced_GC 12-25_M1_20251111.jsonl"
    raw_file = r"data\raw\smc_export_gc_m1_v3.jsonl"
    
    print("Loading NinjaTrader swings...")
    ninja_swings = load_ninja_swings(ninja_file, max_bars=500)
    
    print("Running Python swing detector...")
    python_swings = run_python_swings(raw_file, max_bars=500)
    
    print()
    compare_swings(ninja_swings, python_swings)
