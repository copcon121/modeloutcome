"""
Test Premium/Discount zone detection
"""

import json
import csv
from datetime import datetime
from pathlib import Path
import sys

# Add src to path
src_dir = Path(__file__).parent / 'src'
sys.path.insert(0, str(src_dir))

from layer2_feature_engine_v2.config import GC_M1_SMC_CONFIG
from layer2_feature_engine_v2.schema import RawBar
from layer2_feature_engine_v2.smc_core.swing import ExternalSwingDetector
from layer2_feature_engine_v2.smc_core.zones import PDZoneTracker


def test_pd_zones(ninja_file, output_csv, max_bars=500):
    """Test PD zone detection"""
    
    # Load bars
    all_bars_data = []
    with open(ninja_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= max_bars:
                break
            if not line.strip():
                continue
            
            data = json.loads(line)
            timestamp_str = data.get('timestamp', '')
            timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            
            all_bars_data.append({
                'data': data,
                'timestamp': timestamp,
                'bar_index': data.get('bar_index', i)
            })
    
    # Initialize detectors
    swing_detector = ExternalSwingDetector(GC_M1_SMC_CONFIG, tick_size=0.1)
    pd_tracker = PDZoneTracker(GC_M1_SMC_CONFIG)
    
    # Track PD state changes
    pd_states = []
    
    # Process bars
    for bar_info in all_bars_data:
        data = bar_info['data']
        
        # Create RawBar
        raw_bar = RawBar(
            symbol='GC', timeframe='M1',
            timestamp=bar_info['timestamp'],
            bar_index=bar_info['bar_index'],
            o=data.get('open', 0), h=data.get('high', 0),
            l=data.get('low', 0), c=data.get('close', 0),
            volume=0, delta=0, buy_volume=0, sell_volume=0,
            best_bid=0, best_ask=0, tick_speed=0,
            aggr_buy_speed=0, aggr_sell_speed=0, price_speed=0
        )
        
        # Update swing detector
        swing_state = swing_detector.update(raw_bar)
        
        # Update PD tracker
        pd_state = pd_tracker.update(raw_bar, swing_state)
        
        # Get NinjaTrader PD state
        bar_data = data.get('bar', {})
        ninja_in_premium = bar_data.get('in_premium', False)
        ninja_in_discount = bar_data.get('in_discount', False)
        
        # Record state
        pd_states.append({
            'bar_index': bar_info['bar_index'],
            'timestamp': bar_info['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
            'close': raw_bar.c,
            'py_trail_up': pd_state.trail_up,
            'py_trail_down': pd_state.trail_down,
            'py_eq': pd_state.equilibrium,
            'py_in_premium': pd_state.in_premium,
            'py_in_discount': pd_state.in_discount,
            'py_position_pct': pd_state.price_position_pct,
            'ninja_in_premium': ninja_in_premium,
            'ninja_in_discount': ninja_in_discount,
            'match': (pd_state.in_premium == ninja_in_premium and 
                     pd_state.in_discount == ninja_in_discount)
        })
    
    # Calculate match rate
    total_bars = len(pd_states)
    matches = sum(1 for s in pd_states if s['match'])
    match_rate = (matches / total_bars * 100) if total_bars > 0 else 0
    
    print("=" * 80)
    print("PREMIUM/DISCOUNT ZONE DETECTION")
    print("=" * 80)
    print(f"Total bars: {total_bars}")
    print(f"Matches: {matches} ({match_rate:.1f}%)")
    print()
    
    # Count states
    py_premium_count = sum(1 for s in pd_states if s['py_in_premium'])
    py_discount_count = sum(1 for s in pd_states if s['py_in_discount'])
    ninja_premium_count = sum(1 for s in pd_states if s['ninja_in_premium'])
    ninja_discount_count = sum(1 for s in pd_states if s['ninja_in_discount'])
    
    print(f"Python Premium bars:  {py_premium_count}")
    print(f"Python Discount bars: {py_discount_count}")
    print(f"Ninja Premium bars:   {ninja_premium_count}")
    print(f"Ninja Discount bars:  {ninja_discount_count}")
    print()
    
    # Show first few states
    print("First 20 bars:")
    print(f"{'Bar':<5} {'Time':<20} {'Close':<8} {'Py P/D':<8} {'Ninja P/D':<10} {'Match'}")
    print("-" * 80)
    for s in pd_states[:20]:
        py_pd = 'PREM' if s['py_in_premium'] else ('DISC' if s['py_in_discount'] else 'EQ')
        ninja_pd = 'PREM' if s['ninja_in_premium'] else ('DISC' if s['ninja_in_discount'] else 'EQ')
        match_str = 'OK' if s['match'] else 'DIFF'
        
        print(f"{s['bar_index']:<5} {s['timestamp']:<20} {s['close']:<8.1f} {py_pd:<8} {ninja_pd:<10} {match_str}")
    
    # Export to CSV
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=pd_states[0].keys())
        writer.writeheader()
        writer.writerows(pd_states)
    
    print(f"\nExported to: {output_csv}")


if __name__ == "__main__":
    ninja_file = r"data\raw\deepseek_enhanced_GC 12-25_M1_20251111.jsonl"
    output_csv = "pd_zones_comparison.csv"
    
    test_pd_zones(ninja_file, output_csv, max_bars=500)
