"""
Test FVG and OB detection against NinjaTrader export
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
from layer2_feature_engine_v2.smc_core.zones import FVGDetector, OBDetector


def test_fvg_ob(ninja_file, max_bars=500):
    """Test FVG and OB detection"""
    
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
    int_swing_detector = InternalSwingDetector(GC_M1_SMC_CONFIG, tick_size=0.1)
    ext_swing_detector = ExternalSwingDetector(GC_M1_SMC_CONFIG, tick_size=0.1)  # For OBs!
    structure_detector = StructureDetector(GC_M1_SMC_CONFIG, tick_size=0.1)
    fvg_detector = FVGDetector(GC_M1_SMC_CONFIG, tick_size=0.1)
    ob_detector = OBDetector(GC_M1_SMC_CONFIG, tick_size=0.1)
    
    # Track detections
    python_fvgs = []
    python_obs = []
    ninja_fvgs = []
    ninja_obs = []
    
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
        
        # Update swing & structure
        int_swing_state = int_swing_detector.update(raw_bar)
        ext_swing_state = ext_swing_detector.update(raw_bar)  # For OBs!
        
        structure_detector.update_internal(raw_bar, int_swing_state)  # For FVG
        structure_detector.update_external(raw_bar, ext_swing_state)  # For OBs!
        
        int_struct = structure_detector.get_internal_state()
        ext_struct = structure_detector.get_external_state()  # For OBs!
        
        # Update FVG detector (uses internal swings)
        new_fvgs = fvg_detector.update(raw_bar)
        for fvg in new_fvgs:
            python_fvgs.append({
                'bar_index': fvg.bar_index,
                'timestamp': bar_info['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
                'is_bullish': fvg.is_bullish,
                'top': fvg.top,
                'bottom': fvg.bottom,
                'gap_size': fvg.top - fvg.bottom
            })
        
        # Update OB detector (uses EXTERNAL structure!)
        new_obs = ob_detector.update(raw_bar, ext_struct)
        for ob in new_obs:
            python_obs.append({
                'bar_index': ob.bar_index,
                'timestamp': bar_info['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
                'is_bullish': ob.is_bullish,
                'source_type': ob.source_type,
                'top': ob.top,
                'bottom': ob.bottom
            })
        
        # Extract NinjaTrader OB detection
        bar_data = data.get('bar', {})
        
        # Check if OB detected at this bar
        if bar_data.get('ob_detected'):
            ob_type = bar_data.get('ob_type', '')
            ninja_obs.append({
                'bar_index': bar_info['bar_index'],
                'timestamp': bar_info['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
                'is_bullish': 'bull' in ob_type.lower() if ob_type else False,
                'ob_type': ob_type,
                'top': bar_data.get('ob_top', 0),
                'bottom': bar_data.get('ob_bottom', 0)
            })
    
    # Print results
    print("=" * 80)
    print("FVG/OB DETECTION RESULTS")
    print("=" * 80)
    print()
    
    print("FVG Detection:")
    print(f"  Python detected: {len(python_fvgs)} FVGs")
    print(f"  Ninja detected:  {len(ninja_fvgs)} FVGs")
    py_bull_fvg = sum(1 for f in python_fvgs if f['is_bullish'])
    py_bear_fvg = sum(1 for f in python_fvgs if not f['is_bullish'])
    print(f"  Python: {py_bull_fvg} bullish, {py_bear_fvg} bearish")
    print()
    
    print("OB Detection:")
    print(f"  Python detected: {len(python_obs)} OBs")
    print(f"  Ninja detected:  {len(ninja_obs)} OBs")
    py_bull_ob = sum(1 for o in python_obs if o['is_bullish'])
    py_bear_ob = sum(1 for o in python_obs if not o['is_bullish'])
    print(f"  Python: {py_bull_ob} bullish, {py_bear_ob} bearish")
    print()
    
    # Show first 10 FVGs
    if python_fvgs:
        print("First 10 Python FVGs:")
        print(f"{'Bar':<5} {'Time':<20} {'Type':<8} {'Top':<8} {'Bottom':<8} {'Gap':<6}")
        print("-" * 70)
        for fvg in python_fvgs[:10]:
            fvg_type = 'BULL' if fvg['is_bullish'] else 'BEAR'
            print(f"{fvg['bar_index']:<5} {fvg['timestamp']:<20} {fvg_type:<8} "
                  f"{fvg['top']:<8.1f} {fvg['bottom']:<8.1f} {fvg['gap_size']:<6.1f}")
        print()
    
    # Show first 10 OBs
    if python_obs:
        print("First 10 Python OBs:")
        print(f"{'Bar':<5} {'Time':<20} {'Type':<8} {'Source':<12} {'Top':<8} {'Bottom':<8}")
        print("-" * 75)
        for ob in python_obs[:10]:
            ob_type = 'BULL' if ob['is_bullish'] else 'BEAR'
            print(f"{ob['bar_index']:<5} {ob['timestamp']:<20} {ob_type:<8} "
                  f"{ob['source_type']:<12} {ob['top']:<8.1f} {ob['bottom']:<8.1f}")
        print()
    
    # Export to CSV
    with open('fvg_detection.csv', 'w', newline='', encoding='utf-8') as f:
        if python_fvgs:
            writer = csv.DictWriter(f, fieldnames=python_fvgs[0].keys())
            writer.writeheader()
            writer.writerows(python_fvgs)
    
    with open('ob_detection.csv', 'w', newline='', encoding='utf-8') as f:
        if python_obs:
            writer = csv.DictWriter(f, fieldnames=python_obs[0].keys())
            writer.writeheader()
            writer.writerows(python_obs)
    
    print("Exported to: fvg_detection.csv, ob_detection.csv")
    print()
    print(f"Active FVGs: {len(fvg_detector.get_active_fvgs())}")
    print(f"Active OBs: {len(ob_detector.get_active_obs())}")


if __name__ == "__main__":
    ninja_file = r"data\raw\deepseek_enhanced_GC 12-25_M1_20251111.jsonl"
    test_fvg_ob(ninja_file, max_bars=500)
