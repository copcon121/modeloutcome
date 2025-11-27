"""
Export Internal Swings (window=5) for verification
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
from layer2_feature_engine_v2.smc_core.swing import InternalSwingDetector

def export_internal_swings(ninja_file, output_csv, max_bars=500):
    """Export Internal swings to CSV"""
    
    # Load all bars
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
                'bar_index': data.get('bar_index', i),
                'h': data.get('high', 0),
                'l': data.get('low', 0)
            })
    
    # Run detector
    swings = []
    detector = InternalSwingDetector(GC_M1_SMC_CONFIG, tick_size=0.1)
    
    prev_high_idx = -1
    prev_low_idx = -1
    
    for bar_data in all_bars_data:
        raw_bar = RawBar(
            symbol='GC', timeframe='M1', 
            timestamp=bar_data['timestamp'],
            bar_index=bar_data['bar_index'],
            o=bar_data['data'].get('open', 0), 
            h=bar_data['data'].get('high', 0),
            l=bar_data['data'].get('low', 0), 
            c=bar_data['data'].get('close', 0),
            volume=0, delta=0, buy_volume=0, sell_volume=0,
            best_bid=0, best_ask=0, tick_speed=0,
            aggr_buy_speed=0, aggr_sell_speed=0, price_speed=0
        )
        
        state = detector.update(raw_bar)
        
        # Track highs
        if state.swing_high_bar_index != prev_high_idx and state.swing_high_bar_index != -1:
            swing_bar = next((b for b in all_bars_data if b['bar_index'] == state.swing_high_bar_index), None)
            if swing_bar:
                swings.append({
                    'type': 'HIGH',
                    'bar_index': state.swing_high_bar_index,
                    'timestamp': swing_bar['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
                    'price': state.swing_high_price,
                    'leg': 'BEARISH' if state.last_leg == 1 else 'BULLISH'
                })
            prev_high_idx = state.swing_high_bar_index
        
        # Track lows
        if state.swing_low_bar_index != prev_low_idx and state.swing_low_bar_index != -1:
            swing_bar = next((b for b in all_bars_data if b['bar_index'] == state.swing_low_bar_index), None)
            if swing_bar:
                swings.append({
                    'type': 'LOW',
                    'bar_index': state.swing_low_bar_index,
                    'timestamp': swing_bar['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
                    'price': state.swing_low_price,
                    'leg': 'BEARISH' if state.last_leg == 1 else 'BULLISH'
                })
            prev_low_idx = state.swing_low_bar_index
    
    # Sort by bar index
    swings.sort(key=lambda x: x['bar_index'])
    
    # Write to CSV
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Type', 'Bar Index', 'Time', 'Price', 'Leg'])
        
        for s in swings:
            writer.writerow([
                s['type'],
                s['bar_index'],
                s['timestamp'],
                s['price'],
                s['leg']
            ])
    
    print(f"Exported {len(swings)} internal swings to: {output_csv}")
    print(f"\nFirst 20 swings:")
    for i, s in enumerate(swings[:20]):
        print(f"  {i+1}. {s['type']:4} | Bar {s['bar_index']:>3} | {s['timestamp']} | {s['price']:>7.1f} | {s['leg']}")


if __name__ == "__main__":
    ninja_file = r"data\raw\deepseek_enhanced_GC 12-25_M1_20251111.jsonl"
    output_csv = "internal_swings_window5.csv"
    
    export_internal_swings(ninja_file, output_csv, max_bars=500)
