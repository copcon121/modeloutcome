"""
Test BOS/CHoCH detection and compare with NinjaTrader export
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
from layer2_feature_engine_v2.smc_core.swing import InternalSwingDetector, ExternalSwingDetector
from layer2_feature_engine_v2.smc_core.structure import StructureDetector


def test_bos_choch(ninja_file, output_csv, max_bars=500):
    """Test BOS/CHoCH detection"""
    
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
                'bar_index': data.get('bar_index', i)
            })
    
    # Initialize detectors
    int_swing = InternalSwingDetector(GC_M1_SMC_CONFIG, tick_size=0.1)
    ext_swing = ExternalSwingDetector(GC_M1_SMC_CONFIG, tick_size=0.1)
    structure = StructureDetector(GC_M1_SMC_CONFIG, tick_size=0.1)
    
    # Track signals
    python_signals = []
    ninja_signals = []
    
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
        
        # Update swing detectors
        int_state = int_swing.update(raw_bar)
        ext_state = ext_swing.update(raw_bar)
        
        # Update structure detector
        structure.update_internal(raw_bar, int_state)
        structure.update_external(raw_bar, ext_state)
        
        # Get states
        int_struct = structure.get_internal_state()
        ext_struct = structure.get_external_state()
        
        # Record Python signals
        if int_struct['bos_up']:
            python_signals.append({
                'layer': 'INTERNAL',
                'type': 'BOS_UP',
                'bar_index': bar_info['bar_index'],
                'timestamp': bar_info['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
                'price': raw_bar.c
            })
        
        if int_struct['bos_down']:
            python_signals.append({
                'layer': 'INTERNAL',
                'type': 'BOS_DOWN',
                'bar_index': bar_info['bar_index'],
                'timestamp': bar_info['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
                'price': raw_bar.c
            })
        
        if int_struct['choch_up']:
            python_signals.append({
                'layer': 'INTERNAL',
                'type': 'CHOCH_UP',
                'bar_index': bar_info['bar_index'],
                'timestamp': bar_info['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
                'price': raw_bar.c
            })
        
        if int_struct['choch_down']:
            python_signals.append({
                'layer': 'INTERNAL',
                'type': 'CHOCH_DOWN',
                'bar_index': bar_info['bar_index'],
                'timestamp': bar_info['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
                'price': raw_bar.c
            })
        
        # External signals
        if ext_struct['bos_up']:
            python_signals.append({
                'layer': 'EXTERNAL',
                'type': 'BOS_UP',
                'bar_index': bar_info['bar_index'],
                'timestamp': bar_info['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
                'price': raw_bar.c
            })
        
        if ext_struct['bos_down']:
            python_signals.append({
                'layer': 'EXTERNAL',
                'type': 'BOS_DOWN',
                'bar_index': bar_info['bar_index'],
                'timestamp': bar_info['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
                'price': raw_bar.c
            })
        
        if ext_struct['choch_up']:
            python_signals.append({
                'layer': 'EXTERNAL',
                'type': 'CHOCH_UP',
                'bar_index': bar_info['bar_index'],
                'timestamp': bar_info['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
                'price': raw_bar.c
            })
        
        if ext_struct['choch_down']:
            python_signals.append({
                'layer': 'EXTERNAL',
                'type': 'CHOCH_DOWN',
                'bar_index': bar_info['bar_index'],
                'timestamp': bar_info['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
                'price': raw_bar.c
            })
        
        # Extract NinjaTrader signals
        bar_data = data.get('bar', {})
        
        if bar_data.get('int_bos_up'):
            ninja_signals.append({
                'layer': 'INTERNAL',
                'type': 'BOS_UP',
                'bar_index': bar_info['bar_index'],
                'timestamp': bar_info['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
            })
        
        if bar_data.get('int_bos_down'):
            ninja_signals.append({
                'layer': 'INTERNAL',
                'type': 'BOS_DOWN',
                'bar_index': bar_info['bar_index'],
                'timestamp': bar_info['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
            })
        
        if bar_data.get('int_choch_up'):
            ninja_signals.append({
                'layer': 'INTERNAL',
                'type': 'CHOCH_UP',
                'bar_index': bar_info['bar_index'],
                'timestamp': bar_info['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
            })
        
        if bar_data.get('int_choch_down'):
            ninja_signals.append({
                'layer': 'INTERNAL',
                'type': 'CHOCH_DOWN',
                'bar_index': bar_info['bar_index'],
                'timestamp': bar_info['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
            })
        
        if bar_data.get('ext_bos_up'):
            ninja_signals.append({
                'layer': 'EXTERNAL',
                'type': 'BOS_UP',
                'bar_index': bar_info['bar_index'],
                'timestamp': bar_info['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
            })
        
        if bar_data.get('ext_bos_down'):
            ninja_signals.append({
                'layer': 'EXTERNAL',
                'type': 'BOS_DOWN',
                'bar_index': bar_info['bar_index'],
                'timestamp': bar_info['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
            })
        
        if bar_data.get('ext_choch_up'):
            ninja_signals.append({
                'layer': 'EXTERNAL',
                'type': 'CHOCH_UP',
                'bar_index': bar_info['bar_index'],
                'timestamp': bar_info['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
            })
        
        if bar_data.get('ext_choch_down'):
            ninja_signals.append({
                'layer': 'EXTERNAL',
                'type': 'CHOCH_DOWN',
                'bar_index': bar_info['bar_index'],
                'timestamp': bar_info['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
            })
    
    # Print summary
    print("=" * 80)
    print("BOS/CHoCH DETECTION RESULTS")
    print("=" * 80)
    print(f"\nPython detected: {len(python_signals)} signals")
    print(f"NinjaTrader:     {len(ninja_signals)} signals")
    
    # Count by type
    for layer in ['INTERNAL', 'EXTERNAL']:
        print(f"\n{layer}:")
        for sig_type in ['BOS_UP', 'BOS_DOWN', 'CHOCH_UP', 'CHOCH_DOWN']:
            py_count = len([s for s in python_signals if s['layer'] == layer and s['type'] == sig_type])
            nj_count = len([s for s in ninja_signals if s['layer'] == layer and s['type'] == sig_type])
            print(f"  {sig_type:12} - Python: {py_count:3}, Ninja: {nj_count:3}")
    
    # Export to CSV
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Source', 'Layer', 'Type', 'Bar Index', 'Time', 'Python Price'])
        
        for s in python_signals:
            writer.writerow(['PYTHON', s['layer'], s['type'], s['bar_index'], s['timestamp'], s.get('price', '')])
        
        writer.writerow([])  # Separator
        
        for s in ninja_signals:
            writer.writerow(['NINJA', s['layer'], s['type'], s['bar_index'], s['timestamp'], ''])
    
    print(f"\nExported to: {output_csv}")
    print("\nFirst 20 Python signals:")
    for i, s in enumerate(python_signals[:20]):
        print(f"  {i+1:2}. {s['layer']:8} {s['type']:12} | Bar {s['bar_index']:>3} | {s['timestamp']}")


if __name__ == "__main__":
    ninja_file = r"data\raw\deepseek_enhanced_GC 12-25_M1_20251111.jsonl"
    output_csv = "bos_choch_comparison.csv"
    
    test_bos_choch(ninja_file, output_csv, max_bars=500)
