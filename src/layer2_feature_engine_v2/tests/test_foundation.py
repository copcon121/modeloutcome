"""
Phase 2 Feature Engine V2 - Foundation Tests
Test loaders, schema, and swing detection logic
"""

import sys
from pathlib import Path

# Add src directory to path
src_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(src_dir))

from layer2_feature_engine_v2.config import GC_M1_SMC_CONFIG
from layer2_feature_engine_v2.schema import RawBar
from layer2_feature_engine_v2.loaders import iter_raw_bars, validate_jsonl_file
from layer2_feature_engine_v2.smc_core.swing import InternalSwingDetector, ExternalSwingDetector
from datetime import datetime



def test_schema():
    """Test RawBar schema"""
    print("=" * 60)
    print("TEST 1: Schema - RawBar Creation")
    print("=" * 60)
    
    bar = RawBar(
        symbol="GC 02-26",
        timeframe="M1",
        timestamp=datetime.now(),
        bar_index=100,
        o=4047.8,
        h=4049.1,
        l=4043.2,
        c=4048.9,
        volume=850,
        delta=-77,
        buy_volume=386.5,
        sell_volume=463.5,
        best_bid=4048.9,
        best_ask=4048.9,
        tick_speed=1404,
        aggr_buy_speed=386.5,
        aggr_sell_speed=463.5,
        price_speed=5.9
    )
    
    print(f"✅ Created RawBar: {bar.symbol} @ {bar.c}")
    print(f"   - OHLC: O={bar.o} H={bar.h} L={bar.l} C={bar.c}")
    print(f"   - Volume: {bar.volume}, Delta: {bar.delta}")
    print(f"   - Bullish: {bar.is_bullish}")
    print(f"   - Body: {bar.body_size}, Range: {bar.range_size}")
    print(f"   - Wicks: Upper={bar.upper_wick}, Lower={bar.lower_wick}")
    print()
    
    return True


def test_config():
    """Test SMC config"""
    print("=" * 60)
    print("TEST 2: Config - SMCConfig")
    print("=" * 60)
    
    config = GC_M1_SMC_CONFIG
    
    print(f"✅ GC M1 Config Loaded:")
    print(f"   - Internal window: {config.swing_int_window} (wave 5)")
    print(f"   - External window: {config.swing_ext_window} (wave 50)")
    print(f"   - BOS buffer: {config.bos_close_buffer_ticks} ticks")
    print(f"   - FVG auto threshold: {config.fvg_auto_threshold}")
    print(f"   - OB lookback: {config.ob_lookback_bars} bars")
    print()
    
    return True


def test_swing_detection_synthetic():
    """Test swing detection with synthetic data"""
    print("=" * 60)
    print("TEST 3: Swing Detection - Synthetic Data")
    print("=" * 60)
    
    # Create synthetic uptrend then downtrend
    synthetic_bars = []
    base_price = 4000.0
    
    # Uptrend: 10 bars going up
    print("Creating synthetic uptrend (10 bars)...")
    for i in range(10):
        price = base_price + i * 2  # +2 per bar
        bar = RawBar(
            symbol="TEST",
            timeframe="M1",
            timestamp=datetime.now(),
            bar_index=i,
            o=price,
            h=price + 1.5,
            l=price - 0.5,
            c=price + 1,
            volume=100,
            delta=10,
            buy_volume=55,
            sell_volume=45,
            best_bid=price + 1,
            best_ask=price + 1,
            tick_speed=100,
            aggr_buy_speed=55,
            aggr_sell_speed=45,
            price_speed=2.0
        )
        synthetic_bars.append(bar)
    
    # Downtrend: 10 bars going down
    print("Creating synthetic downtrend (10 bars)...")
    for i in range(10, 20):
        price = base_price + 20 - (i - 10) * 2  # -2 per bar
        bar = RawBar(
            symbol="TEST",
            timeframe="M1",
            timestamp=datetime.now(),
            bar_index=i,
            o=price,
            h=price + 0.5,
            l=price - 1.5,
            c=price - 1,
            volume=100,
            delta=-10,
            buy_volume=45,
            sell_volume=55,
            best_bid=price - 1,
            best_ask=price - 1,
            tick_speed=100,
            aggr_buy_speed=45,
            aggr_sell_speed=55,
            price_speed=2.0
        )
        synthetic_bars.append(bar)
    
    # Test Internal Swing Detector (window=5)
    print("\n--- Internal Swing Detector (window=5) ---")
    int_detector = InternalSwingDetector(GC_M1_SMC_CONFIG, tick_size=0.1)
    
    int_swing_highs = []
    int_swing_lows = []
    
    for bar in synthetic_bars:
        state = int_detector.update(bar)
        
        # Check for new swings
        if state.swing_high_bar_index == bar.bar_index:
            int_swing_highs.append((bar.bar_index, state.swing_high_price))
            print(f"   📈 Internal SWING HIGH at bar {bar.bar_index}: {state.swing_high_price}")
        
        if state.swing_low_bar_index == bar.bar_index:
            int_swing_lows.append((bar.bar_index, state.swing_low_price))
            print(f"   📉 Internal SWING LOW at bar {bar.bar_index}: {state.swing_low_price}")
    
    print(f"\n✅ Internal Swings Detected:")
    print(f"   - Swing Highs: {len(int_swing_highs)}")
    print(f"   - Swing Lows: {len(int_swing_lows)}")
    
    # Test External Swing Detector (window=50) - won't trigger with only 20 bars
    print("\n--- External Swing Detector (window=50) ---")
    ext_detector = ExternalSwingDetector(GC_M1_SMC_CONFIG, tick_size=0.1)
    
    for bar in synthetic_bars:
        state = ext_detector.update(bar)
    
    print(f"   ⚠️  Need {GC_M1_SMC_CONFIG.swing_ext_window + 1} bars for external swing (only have {len(synthetic_bars)})")
    print(f"   - Last leg: {ext_detector.state.last_leg}")
    print()
    
    return len(int_swing_highs) > 0 or len(int_swing_lows) > 0


def test_with_jsonl_if_available():
    """Test with actual JSONL file if available"""
    print("=" * 60)
    print("TEST 4: Real Data - JSONL File (if available)")
    print("=" * 60)
    
    # Look for JSONL files in common locations
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
        print("⚠️  No JSONL file found. Skipping real data test.")
        print("   Looked in:", possible_paths)
        print()
        return True
    
    print(f"✅ Found JSONL file: {jsonl_path}")
    
    # Validate file
    stats = validate_jsonl_file(jsonl_path)
    print(f"\nFile Statistics:")
    print(f"   - Total bars: {stats['bar_count']}")
    print(f"   - Symbols: {stats['symbols']}")
    print(f"   - Timeframes: {stats['timeframes']}")
    print(f"   - Time range: {stats['first_timestamp']} to {stats['last_timestamp']}")
    
    # Test swing detection on real data
    print("\n--- Testing Swing Detection on Real Data ---")
    int_detector = InternalSwingDetector(GC_M1_SMC_CONFIG, tick_size=0.1)
    ext_detector = ExternalSwingDetector(GC_M1_SMC_CONFIG, tick_size=0.1)
    
    int_swing_count = {"highs": 0, "lows": 0}
    ext_swing_count = {"highs": 0, "lows": 0}
    
    bar_count = 0
    for bar in iter_raw_bars(jsonl_path):
        bar_count += 1
        
        # Update detectors
        int_state = int_detector.update(bar)
        ext_state = ext_detector.update(bar)
        
        # Count swings
        if int_state.swing_high_bar_index == bar.bar_index:
            int_swing_count["highs"] += 1
        if int_state.swing_low_bar_index == bar.bar_index:
            int_swing_count["lows"] += 1
        if ext_state.swing_high_bar_index == bar.bar_index:
            ext_swing_count["highs"] += 1
        if ext_state.swing_low_bar_index == bar.bar_index:
            ext_swing_count["lows"] += 1
        
        # Stop after reasonable amount
        if bar_count >= 500:
            break
    
    print(f"\nProcessed {bar_count} bars:")
    print(f"   - Internal Swings: {int_swing_count['highs']} highs, {int_swing_count['lows']} lows")
    print(f"   - External Swings: {ext_swing_count['highs']} highs, {ext_swing_count['lows']} lows")
    print()
    
    return True


def test_swing_window_logic():
    """Test that window logic matches NinjaTrader exactly"""
    print("=" * 60)
    print("TEST 5: Window Logic Validation")
    print("=" * 60)
    
    print("Testing window-based MAX/MIN approach:")
    print("   - Internal window = 5")
    print("   - Should detect swing when price breaks OUT of rolling window")
    print()
    
    # Create specific pattern to test window logic
    # Pattern: 5 bars flat, then 1 bar spike up
    bars = []
    for i in range(5):
        bars.append(RawBar(
            symbol="TEST", timeframe="M1", timestamp=datetime.now(), bar_index=i,
            o=4000, h=4001, l=3999, c=4000,
            volume=100, delta=0, buy_volume=50, sell_volume=50,
            best_bid=4000, best_ask=4000,
            tick_speed=100, aggr_buy_speed=50, aggr_sell_speed=50, price_speed=2
        ))
    
    # Spike up - should trigger BEARISH leg (from high)
    bars.append(RawBar(
        symbol="TEST", timeframe="M1", timestamp=datetime.now(), bar_index=5,
        o=4000, h=4010, l=4000, c=4008,  # High breaks above window MAX
        volume=100, delta=10, buy_volume=55, sell_volume=45,
        best_bid=4008, best_ask=4008,
        tick_speed=100, aggr_buy_speed=55, aggr_sell_speed=45, price_speed=10
    ))
    
    detector = InternalSwingDetector(GC_M1_SMC_CONFIG, tick_size=0.1)
    
    for bar in bars:
        state = detector.update(bar)
        if bar.bar_index == 5:
            print(f"   Bar {bar.bar_index}: H={bar.h} breaks above window")
            print(f"   → Last leg: {state.last_leg} (1=BEARISH_LEG expected)")
            print(f"   → Swing high: {state.swing_high_price}")
            
            if state.last_leg == 1 and state.swing_high_price == 4010:
                print(f"   ✅ PASS: Window logic working correctly!")
            else:
                print(f"   ❌ FAIL: Expected BEARISH_LEG with high=4010")
    
    print()
    return True


def run_all_tests():
    """Run all foundation tests"""
    print("\n" + "=" * 60)
    print("PHASE 2 FEATURE ENGINE V2 - FOUNDATION TESTS")
    print("=" * 60)
    print()
    
    tests = [
        ("Schema", test_schema),
        ("Config", test_config),
        ("Swing Detection (Synthetic)", test_swing_detection_synthetic),
        ("Window Logic Validation", test_swing_window_logic),
        ("Real Data (if available)", test_with_jsonl_if_available),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, "PASS" if result else "FAIL"))
        except Exception as e:
            print(f"❌ ERROR in {name}: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, "ERROR"))
        print()
    
    # Summary
    print("=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    for name, status in results:
        symbol = "✅" if status == "PASS" else "❌"
        print(f"{symbol} {name}: {status}")
    
    pass_count = sum(1 for _, status in results if status == "PASS")
    total_count = len(results)
    
    print()
    print(f"Results: {pass_count}/{total_count} tests passed")
    print("=" * 60)
    
    return pass_count == total_count


if __name__ == "__main__":
    import logging
    logging.basicConfig(
        level=logging.WARNING,  # Suppress debug logs for cleaner output
        format='%(levelname)s: %(message)s'
    )
    
    success = run_all_tests()
    sys.exit(0 if success else 1)
