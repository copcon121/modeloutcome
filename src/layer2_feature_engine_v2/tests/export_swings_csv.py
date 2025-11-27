"""Export swing points to CSV for easy comparison with NinjaTrader"""

import sys
import csv
from pathlib import Path

# Add src to path
src_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(src_dir))

from layer2_feature_engine_v2.config import GC_M1_SMC_CONFIG
from layer2_feature_engine_v2.loaders import iter_raw_bars
from layer2_feature_engine_v2.smc_core.swing import InternalSwingDetector, ExternalSwingDetector


def export_swings_to_csv(jsonl_path, output_csv, max_bars=500, tick_size=0.1):
    """Export swing points to CSV file"""
    
    print(f"Processing {jsonl_path}...")
    
    # Initialize detectors
    int_detector = InternalSwingDetector(GC_M1_SMC_CONFIG, tick_size)
    ext_detector = ExternalSwingDetector(GC_M1_SMC_CONFIG, tick_size)
    
    # Collect all swings
    swings = []
    bar_count = 0
    
    # Track previous swing indices to detect NEW swings
    prev_int_high_idx = -1
    prev_int_low_idx = -1
    prev_ext_high_idx = -1
    prev_ext_low_idx = -1
    
    for bar in iter_raw_bars(jsonl_path):
        bar_count += 1
        
        # Update detectors
        int_state = int_detector.update(bar)
        ext_state = ext_detector.update(bar)
        
        # Check for NEW internal swings (state changed)
        if int_state.swing_high_bar_index != prev_int_high_idx and int_state.swing_high_bar_index != -1:
            swings.append({
                'layer': 'INTERNAL',
                'type': 'HIGH',
                'date': bar.timestamp.strftime('%Y-%m-%d'),
                'time': bar.timestamp.strftime('%H:%M:%S'),
                'timestamp': bar.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                'bar_index': int_state.swing_high_bar_index,
                'price': int_state.swing_high_price,
                'leg': 'BEAR_LEG' if int_state.last_leg == 1 else 'BULL_LEG'
            })
            prev_int_high_idx = int_state.swing_high_bar_index
        
        if int_state.swing_low_bar_index != prev_int_low_idx and int_state.swing_low_bar_index != -1:
            swings.append({
                'layer': 'INTERNAL',
                'type': 'LOW',
                'date': bar.timestamp.strftime('%Y-%m-%d'),
                'time': bar.timestamp.strftime('%H:%M:%S'),
                'timestamp': bar.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                'bar_index': int_state.swing_low_bar_index,
                'price': int_state.swing_low_price,
                'leg': 'BEAR_LEG' if int_state.last_leg == 1 else 'BULL_LEG'
            })
            prev_int_low_idx = int_state.swing_low_bar_index
        
        # Check for NEW external swings
        if ext_state.swing_high_bar_index != prev_ext_high_idx and ext_state.swing_high_bar_index != -1:
            swings.append({
                'layer': 'EXTERNAL',
                'type': 'HIGH',
                'date': bar.timestamp.strftime('%Y-%m-%d'),
                'time': bar.timestamp.strftime('%H:%M:%S'),
                'timestamp': bar.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                'bar_index': ext_state.swing_high_bar_index,
                'price': ext_state.swing_high_price,
                'leg': 'BEAR_LEG' if ext_state.last_leg == 1 else 'BULL_LEG'
            })
            prev_ext_high_idx = ext_state.swing_high_bar_index
        
        if ext_state.swing_low_bar_index != prev_ext_low_idx and ext_state.swing_low_bar_index != -1:
            swings.append({
                'layer': 'EXTERNAL',
                'type': 'LOW',
                'date': bar.timestamp.strftime('%Y-%m-%d'),
                'time': bar.timestamp.strftime('%H:%M:%S'),
                'timestamp': bar.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                'bar_index': ext_state.swing_low_bar_index,
                'price': ext_state.swing_low_price,
                'leg': 'BEAR_LEG' if ext_state.last_leg == 1 else 'BULL_LEG'
            })
            prev_ext_low_idx = ext_state.swing_low_bar_index
        
        if max_bars and bar_count >= max_bars:
            break
    
    # Write to CSV
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['layer', 'type', 'date', 'time', 'timestamp', 'bar_index', 'price', 'leg']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        writer.writeheader()
        for swing in swings:
            writer.writerow(swing)
    
    print(f"\nExported {len(swings)} swing points to {output_csv}")
    print(f"Processed {bar_count} bars")
    
    # Summary
    int_swings = [s for s in swings if s['layer'] == 'INTERNAL']
    ext_swings = [s for s in swings if s['layer'] == 'EXTERNAL']
    
    print(f"\nInternal: {len(int_swings)} swings")
    print(f"External: {len(ext_swings)} swings")
    
    return swings


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python export_swings_csv.py <jsonl_file> [output_csv]")
        sys.exit(1)
    
    jsonl_path = sys.argv[1]
    output_csv = sys.argv[2] if len(sys.argv) > 2 else "swing_points.csv"
    
    if not Path(jsonl_path).exists():
        print(f"Error: File not found: {jsonl_path}")
        sys.exit(1)
    
    export_swings_to_csv(jsonl_path, output_csv, max_bars=500, tick_size=0.1)
    print(f"\nDone! Open {output_csv} in Excel to compare with NinjaTrader")
