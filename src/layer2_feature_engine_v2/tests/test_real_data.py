"""
Test swing detection with real JSONL data
Print swing points with full timestamp for NinjaTrader comparison
"""

import sys
from pathlib import Path

# Add src to path
src_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(src_dir))

from layer2_feature_engine_v2.config import GC_M1_SMC_CONFIG
from layer2_feature_engine_v2.loaders import iter_raw_bars
from layer2_feature_engine_v2.smc_core.swing import InternalSwingDetector, ExternalSwingDetector


def test_swing_with_real_data(jsonl_path, max_bars=500, tick_size=0.1):
    """
    Test swing detection with real JSONL data
    Print swing points for comparison with NinjaTrader
    """
    
    print("=" * 80)
    print("SWING DETECTION TEST - Real Data")
    print("=" * 80)
    print(f"File: {jsonl_path}")
    print(f"Tick Size: {tick_size}")
    print(f"Config: Internal window={GC_M1_SMC_CONFIG.swing_int_window}, External window={GC_M1_SMC_CONFIG.swing_ext_window}")
    print("=" * 80)
    print()
    
    # Initialize detectors
    int_detector = InternalSwingDetector(GC_M1_SMC_CONFIG, tick_size)
    ext_detector = ExternalSwingDetector(GC_M1_SMC_CONFIG, tick_size)
    
    # Track swing points
    int_swings = []
    ext_swings = []
    
    # Process bars
    bar_count = 0
    for bar in iter_raw_bars(jsonl_path):
        bar_count += 1
        
        # Update detectors
        int_state = int_detector.update(bar)
        ext_state = ext_detector.update(bar)
        
        # Check for new internal swings
        if int_state.swing_high_bar_index == bar.bar_index:
            int_swings.append({
                'type': 'HIGH',
                'timestamp': bar.timestamp,
                'bar_index': bar.bar_index,
                'price': int_state.swing_high_price,
                'leg': int_state.last_leg
            })
        
        if int_state.swing_low_bar_index == bar.bar_index:
            int_swings.append({
                'type': 'LOW',
                'timestamp': bar.timestamp,
                'bar_index': bar.bar_index,
                'price': int_state.swing_low_price,
                'leg': int_state.last_leg
            })
        
        # Check for new external swings
        if ext_state.swing_high_bar_index == bar.bar_index:
            ext_swings.append({
                'type': 'HIGH',
                'timestamp': bar.timestamp,
                'bar_index': bar.bar_index,
                'price': ext_state.swing_high_price,
                'leg': ext_state.last_leg
            })
        
        if ext_state.swing_low_bar_index == bar.bar_index:
            ext_swings.append({
                'type': 'LOW',
                'timestamp': bar.timestamp,
                'bar_index': bar.bar_index,
                'price': ext_state.swing_low_price,
                'leg': ext_state.last_leg
            })
        
        # Limit bars
        if max_bars and bar_count >= max_bars:
            break
    
    # Print results
    print(f"Processed {bar_count} bars")
    print()
    
    # Print Internal Swings
    print("=" * 80)
    print(f"INTERNAL SWINGS (wave 5, window={GC_M1_SMC_CONFIG.swing_int_window})")
    print("=" * 80)
    print(f"{'#':<4} {'Date & Time':<20} {'Type':<6} {'Price':>10} {'Bar Index':>10} {'Leg':<10}")
    print("-" * 80)
    
    for i, swing in enumerate(int_swings, 1):
        timestamp_str = swing['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
        leg_str = "BULL_LEG" if swing['leg'] == 0 else "BEAR_LEG"
        print(f"{i:<4} {timestamp_str:<20} {swing['type']:<6} {swing['price']:>10.1f} {swing['bar_index']:>10} {leg_str:<10}")
    
    print()
    print(f"Total Internal Swings: {len(int_swings)}")
    print()
    
    # Print External Swings
    print("=" * 80)
    print(f"EXTERNAL SWINGS (wave 50, window={GC_M1_SMC_CONFIG.swing_ext_window})")
    print("=" * 80)
    print(f"{'#':<4} {'Date & Time':<20} {'Type':<6} {'Price':>10} {'Bar Index':>10} {'Leg':<10}")
    print("-" * 80)
    
    if ext_swings:
        for i, swing in enumerate(ext_swings, 1):
            timestamp_str = swing['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
            leg_str = "BULL_LEG" if swing['leg'] == 0 else "BEAR_LEG"
            print(f"{i:<4} {timestamp_str:<20} {swing['type']:<6} {swing['price']:>10.1f} {swing['bar_index']:>10} {leg_str:<10}")
        print()
        print(f"Total External Swings: {len(ext_swings)}")
    else:
        print(f"(Need {GC_M1_SMC_CONFIG.swing_ext_window + 1} bars minimum, processed {bar_count})")
    
    print()
    print("=" * 80)
    
    # Summary
    print("\nSUMMARY FOR NINJATRADER COMPARISON:")
    print("-" * 80)
    print(f"Internal Swings: {len([s for s in int_swings if s['type'] == 'HIGH'])} highs, {len([s for s in int_swings if s['type'] == 'LOW'])} lows")
    print(f"External Swings: {len([s for s in ext_swings if s['type'] == 'HIGH'])} highs, {len([s for s in ext_swings if s['type'] == 'LOW'])} lows")
    print()
    
    if int_swings:
        print("First Internal Swing:")
        first = int_swings[0]
        print(f"  {first['timestamp'].strftime('%Y-%m-%d %H:%M:%S')} | {first['type']} @ {first['price']}")
        print()
        print("Last Internal Swing:")
        last = int_swings[-1]
        print(f"  {last['timestamp'].strftime('%Y-%m-%d %H:%M:%S')} | {last['type']} @ {last['price']}")
    
    print("=" * 80)


if __name__ == "__main__":
    import sys
    
    # Check for JSONL file argument
    if len(sys.argv) > 1:
        jsonl_path = sys.argv[1]
    else:
        # Try to find JSONL file automatically
        possible_paths = [
            "gc_export.jsonl",
            "data/gc_export.jsonl",
            "../data/gc_export.jsonl",
            "../../data/gc_export.jsonl",
        ]
        
        jsonl_path = None
        for path in possible_paths:
            if Path(path).exists():
                jsonl_path = path
                break
        
        if not jsonl_path:
            print("ERROR: No JSONL file found!")
            print("Usage: python test_real_data.py <path_to_jsonl>")
            print()
            print("Looked in:")
            for p in possible_paths:
                print(f"  - {p}")
            sys.exit(1)
    
    # Check if file exists
    if not Path(jsonl_path).exists():
        print(f"ERROR: File not found: {jsonl_path}")
        sys.exit(1)
    
    # Run test
    test_swing_with_real_data(jsonl_path, max_bars=500, tick_size=0.1)
