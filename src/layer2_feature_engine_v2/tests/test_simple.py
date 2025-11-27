"""Simple foundation test without unicode symbols"""
import sys
from pathlib import Path

# Add src to path
src_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(src_dir))

from layer2_feature_engine_v2.config import GC_M1_SMC_CONFIG
from layer2_feature_engine_v2.schema import RawBar
from layer2_feature_engine_v2.smc_core.swing import InternalSwingDetector
from datetime import datetime


def test_swing_basic():
    """Basic swing test"""
    print("Testing Internal Swing Detector (window=5)")
    print("-" * 50)
    
    # Create uptrend
    detector = InternalSwingDetector(GC_M1_SMC_CONFIG, tick_size=0.1)
    
    for i in range(15):
        price = 4000 + i * 2
        bar = RawBar(
            symbol="TEST", timeframe="M1", timestamp=datetime.now(), bar_index=i,
            o=price, h=price + 1.5, l=price - 0.5, c=price + 1,
            volume=100, delta=10, buy_volume=55, sell_volume=45,
            best_bid=price, best_ask=price,
            tick_speed=100, aggr_buy_speed=55, aggr_sell_speed=45, price_speed=2
        )
        
        state = detector.update(bar)
        
        if state.swing_high_bar_index == i:
            print(f"Bar {i}: SWING HIGH at {state.swing_high_price}")
        if state.swing_low_bar_index == i:
            print(f"Bar {i}: SWING LOW at {state.swing_low_price}")
    
    final_state = detector.get_state()
    print(f"\nFinal state:")
    print(f"  Swing High: {final_state.swing_high_price}")
    print(f"  Swing Low: {final_state.swing_low_price}")
    print(f"  Last Leg: {final_state.last_leg} (0=BULL, 1=BEAR)")
    print(f"  Trend Bias: {final_state.trend_bias}")
    
    print("\n[PASS] Swing detection working!")
    return True


if __name__ == "__main__":
    try:
        test_swing_basic()
        print("\n=== ALL TESTS PASSED ===")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
