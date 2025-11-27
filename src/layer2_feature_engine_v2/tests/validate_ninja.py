"""
Python Validation Tool - Compare Python SMC output with NinjaTrader signals
"""

import json
import sys
from pathlib import Path
from typing import List, Dict
from datetime import datetime

# Add src to path
src_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(src_dir))

from layer2_feature_engine_v2.config import GC_M1_SMC_CONFIG
from layer2_feature_engine_v2.loaders import iter_raw_bars
from layer2_feature_engine_v2.smc_core.swing import InternalSwingDetector, ExternalSwingDetector


def load_ninja_signals(jsonl_path: str) -> Dict[str, List[Dict]]:
    """Load NinjaTrader signals from JSONL"""
    signals = {
        'INTERNAL_HIGH': [],
        'INTERNAL_LOW': [],
        'EXTERNAL_HIGH': [],
        'EXTERNAL_LOW': [],
        'BOS': [],
        'CHOCH': [],
        'FVG': [],
        'OB': []
    }
    
    with open(jsonl_path, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            
            sig = json.loads(line)
            
            # Categorize signal
            if sig['signal_type'] == 'SWING':
                key = f"{sig['layer']}_{sig['direction']}"
                signals[key].append(sig)
            elif sig['signal_type'] in ['BOS', 'CHOCH', 'FVG', 'OB']:
                signals[sig['signal_type']].append(sig)
    
    return signals


def run_python_detector(raw_data_jsonl: str, tick_size: float = 0.1):
    """Run Python SMC detector and collect signals"""
    int_detector = InternalSwingDetector(GC_M1_SMC_CONFIG, tick_size)
    ext_detector = ExternalSwingDetector(GC_M1_SMC_CONFIG, tick_size)
    
    python_signals = {
        'INTERNAL_HIGH': [],
        'INTERNAL_LOW': [],
        'EXTERNAL_HIGH': [],
        'EXTERNAL_LOW': []
    }
    
    prev_int_high_idx = -1
    prev_int_low_idx = -1
    prev_ext_high_idx = -1
    prev_ext_low_idx = -1
    
    for bar in iter_raw_bars(raw_data_jsonl):
        # Update detectors
        int_state = int_detector.update(bar)
        ext_state = ext_detector.update(bar)
        
        # Detect NEW swings
        if int_state.swing_high_bar_index != prev_int_high_idx and int_state.swing_high_bar_index != -1:
            python_signals['INTERNAL_HIGH'].append({
                'timestamp': bar.timestamp.isoformat(),
                'bar_index': int_state.swing_high_bar_index,
                'price': int_state.swing_high_price,
                'leg': 'BEARISH_LEG' if int_state.last_leg == 1 else 'BULLISH_LEG'
            })
            prev_int_high_idx = int_state.swing_high_bar_index
        
        if int_state.swing_low_bar_index != prev_int_low_idx and int_state.swing_low_bar_index != -1:
            python_signals['INTERNAL_LOW'].append({
                'timestamp': bar.timestamp.isoformat(),
                'bar_index': int_state.swing_low_bar_index,
                'price': int_state.swing_low_price,
                'leg': 'BEARISH_LEG' if int_state.last_leg == 1 else 'BULLISH_LEG'
            })
            prev_int_low_idx = int_state.swing_low_bar_index
        
        if ext_state.swing_high_bar_index != prev_ext_high_idx and ext_state.swing_high_bar_index != -1:
            python_signals['EXTERNAL_HIGH'].append({
                'timestamp': bar.timestamp.isoformat(),
                'bar_index': ext_state.swing_high_bar_index,
                'price': ext_state.swing_high_price,
                'leg': 'BEARISH_LEG' if ext_state.last_leg == 1 else 'BULLISH_LEG'
            })
            prev_ext_high_idx = ext_state.swing_high_bar_index
        
        if ext_state.swing_low_bar_index != prev_ext_low_idx and ext_state.swing_low_bar_index != -1:
            python_signals['EXTERNAL_LOW'].append({
                'timestamp': bar.timestamp.isoformat(),
                'bar_index': ext_state.swing_low_bar_index,
                'price': ext_state.swing_low_price,
                'leg': 'BEARISH_LEG' if ext_state.last_leg == 1 else 'BULLISH_LEG'
            })
            prev_ext_low_idx = ext_state.swing_low_bar_index
    
    return python_signals


def compare_swings(ninja_signals: Dict, python_signals: Dict):
    """Compare swing signals"""
    print("=" * 80)
    print("SWING COMPARISON: Python vs NinjaTrader")
    print("=" * 80)
    print()
    
    categories = ['INTERNAL_HIGH', 'INTERNAL_LOW', 'EXTERNAL_HIGH', 'EXTERNAL_LOW']
    
    for cat in categories:
        ninja_swings = ninja_signals.get(cat, [])
        python_swings = python_signals.get(cat, [])
        
        print(f"--- {cat} ---")
        print(f"NinjaTrader: {len(ninja_swings)} swings")
        print(f"Python:      {len(python_swings)} swings")
        
        if len(ninja_swings) != len(python_swings):
            print(f"  ❌ COUNT MISMATCH!")
        else:
            print(f"  ✅ Count matches")
        
        # Compare details
        matches = 0
        mismatches = []
        
        for i in range(min(len(ninja_swings), len(python_swings))):
            ninja = ninja_swings[i]
            python = python_swings[i]
            
            # Compare bar_index
            ninja_idx = ninja['bar_index']
            python_idx = python['bar_index']
            
            # Compare price (±0.1 tolerance)
            ninja_price = ninja['price']
            python_price = python['price']
            price_diff = abs(ninja_price - python_price)
            
            if ninja_idx == python_idx and price_diff < 0.2:
                matches += 1
            else:
                mismatches.append({
                    'index': i,
                    'ninja_bar': ninja_idx,
                    'python_bar': python_idx,
                    'ninja_price': ninja_price,
                    'python_price': python_price,
                    'diff': price_diff
                })
        
        print(f"  Matches: {matches}/{min(len(ninja_swings), len(python_swings))}")
        
        if mismatches:
            print(f"  ❌ {len(mismatches)} mismatches:")
            for mm in mismatches[:5]:  # Show first 5
                print(f"     Swing #{mm['index']}: Ninja bar {mm['ninja_bar']} @ {mm['ninja_price']}, "
                      f"Python bar {mm['python_bar']} @ {mm['python_price']}")
        
        print()


def validate(ninja_jsonl: str, raw_data_jsonl: str):
    """Main validation function"""
    print("Loading NinjaTrader signals...")
    ninja_signals = load_ninja_signals(ninja_jsonl)
    
    print("Running Python detector...")
    python_signals = run_python_detector(raw_data_jsonl)
    
    print()
    compare_swings(ninja_signals, python_signals)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python validate_ninja.py <ninja_signals.jsonl> <raw_data.jsonl>")
        print()
        print("Example:")
        print("  python validate_ninja.py smc_signals_ninja.jsonl smc_export_gc_m1_v3.jsonl")
        sys.exit(1)
    
    ninja_jsonl = sys.argv[1]
    raw_data_jsonl = sys.argv[2]
    
    if not Path(ninja_jsonl).exists():
        print(f"Error: NinjaTrader signals file not found: {ninja_jsonl}")
        sys.exit(1)
    
    if not Path(raw_data_jsonl).exists():
        print(f"Error: Raw data file not found: {raw_data_jsonl}")
        sys.exit(1)
    
    validate(ninja_jsonl, raw_data_jsonl)
