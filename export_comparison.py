"""
Export swing comparison to CSV for manual visual verification
"""

import json
import csv
from datetime import datetime
from pathlib import Path
import sys

# Add src to path
src_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(src_dir))

from layer2_feature_engine_v2.config import GC_M1_SMC_CONFIG
from layer2_feature_engine_v2.schema import RawBar
from layer2_feature_engine_v2.smc_core.swing import ExternalSwingDetector

def export_comparison_csv(ninja_file, output_csv, max_bars=500):
    """Export side-by-side comparison to CSV"""
    
    # First pass: load all bar data
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
    
    # Second pass: run detector and record swings with CORRECT timestamps
    python_swings = {'highs': [], 'lows': []}
    detector = ExternalSwingDetector(GC_M1_SMC_CONFIG, tick_size=0.1)
    
    prev_high_idx = -1
    prev_low_idx = -1
    
    for bar_data in all_bars_data:
        # Create RawBar
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
        
        # Run detector
        state = detector.update(raw_bar)
        
        # Track swings - lookup timestamp from bars array
        if state.swing_high_bar_index != prev_high_idx and state.swing_high_bar_index != -1:
            # Find the bar data for this swing
            swing_bar = next((b for b in all_bars_data if b['bar_index'] == state.swing_high_bar_index), None)
            if swing_bar:
                python_swings['highs'].append({
                    'python_bar_index': state.swing_high_bar_index,
                    'python_timestamp': swing_bar['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
                    'python_price': state.swing_high_price
                })
            prev_high_idx = state.swing_high_bar_index
        
        if state.swing_low_bar_index != prev_low_idx and state.swing_low_bar_index != -1:
            swing_bar = next((b for b in all_bars_data if b['bar_index'] == state.swing_low_bar_index), None)
            if swing_bar:
                python_swings['lows'].append({
                    'python_bar_index': state.swing_low_bar_index,
                    'python_timestamp': swing_bar['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
                    'python_price': state.swing_low_price
                })
            prev_low_idx = state.swing_low_bar_index
    
    
    # Extract NinjaTrader swings
    ninja_swings = {'highs': [], 'lows': []}
    prev_ninja_high = None
    prev_ninja_low = None
    
    for bar_info in all_bars_data:
        data = bar_info['data']
        bar_data = data.get('bar', {})
        ninja_high = bar_data.get('last_swing_high')
        ninja_low = bar_data.get('last_swing_low')
        
        if ninja_high and ninja_high != 0 and ninja_high != prev_ninja_high:
            ninja_swings['highs'].append({
                'ninja_bar_index': data.get('bar_index'),
                'ninja_timestamp': data.get('timestamp', '').replace('.000Z', '').replace('T', ' '),
                'ninja_price': ninja_high
            })
            prev_ninja_high = ninja_high
        
        if ninja_low and ninja_low != 0 and ninja_low != prev_ninja_low:
            ninja_swings['lows'].append({
                'ninja_bar_index': data.get('bar_index'),
               'ninja_timestamp': data.get('timestamp', '').replace('.000Z', '').replace('T', ' '),
                'ninja_price': ninja_low
            })
            prev_ninja_low = ninja_low
    
    # Write to CSV
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Header
        writer.writerow(['Type', 'NinjaTrader Bar', 'NinjaTrader Time', 'NinjaTrader Price', 
                         'Python Bar', 'Python Time', 'Python Price', 'Price Match', 'Index Diff'])
        
        # Swing Highs
        for i in range(max(len(ninja_swings['highs']), len(python_swings['highs']))):
            n = ninja_swings['highs'][i] if i < len(ninja_swings['highs']) else {}
            p = python_swings['highs'][i] if i < len(python_swings['highs']) else {}
            
            price_match = 'YES' if (n.get('ninja_price') and p.get('python_price') and 
                                    abs(n['ninja_price'] - p['python_price']) < 0.2) else 'NO'
            
            index_diff = ''
            if n.get('ninja_bar_index') and p.get('python_bar_index'):
                index_diff = n['ninja_bar_index'] - p['python_bar_index']
            
            writer.writerow([
                'SWING HIGH',
                n.get('ninja_bar_index', ''),
                n.get('ninja_timestamp', ''),
                n.get('ninja_price', ''),
                p.get('python_bar_index', ''),
                p.get('python_timestamp', ''),
                p.get('python_price', ''),
                price_match,
                index_diff
            ])
        
        # Empty row
        writer.writerow([])
        
        # Swing Lows
        for i in range(max(len(ninja_swings['lows']), len(python_swings['lows']))):
            n = ninja_swings['lows'][i] if i < len(ninja_swings['lows']) else {}
            p = python_swings['lows'][i] if i < len(python_swings['lows']) else {}
            
            price_match = 'YES' if (n.get('ninja_price') and p.get('python_price') and 
                                    abs(n['ninja_price'] - p['python_price']) < 0.2) else 'NO'
            
            index_diff = ''
            if n.get('ninja_bar_index') and p.get('python_bar_index'):
                index_diff = n['ninja_bar_index'] - p['python_bar_index']
            
            writer.writerow([
                'SWING LOW',
                n.get('ninja_bar_index', ''),
                n.get('ninja_timestamp', ''),
                n.get('ninja_price', ''),
                p.get('python_bar_index', ''),
                p.get('python_timestamp', ''),
                p.get('python_price', ''),
                price_match,
                index_diff
            ])
    
    print(f"Exported comparison to: {output_csv}")
    print(f"\nNinjaTrader: {len(ninja_swings['highs'])} highs, {len(ninja_swings['lows'])} lows")
    print(f"Python:      {len(python_swings['highs'])} highs, {len(python_swings['lows'])} lows")
    print(f"\nOpen in Excel to verify manually!")


if __name__ == "__main__":
    ninja_file = r"data\raw\deepseek_enhanced_GC 12-25_M1_20251111.jsonl"
    output_csv = "swing_comparison_visual_check.csv"
    
    export_comparison_csv(ninja_file, output_csv, max_bars=500)
