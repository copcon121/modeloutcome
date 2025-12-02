import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.layer2_feature_engine_v2.context_manager import SMCContextManager
from src.layer2_feature_engine_v2.config import GC_M1_SMC_CONFIG
from src.layer2_feature_engine_v2.schema import RawBar

def create_synthetic_bar(index, price, timestamp):
    return RawBar(
        symbol="GC", timeframe="M1", timestamp=timestamp, bar_index=index,
        o=price, h=price+1, l=price-1, c=price, volume=100,
        delta=0, buy_volume=50, sell_volume=50, best_bid=price, best_ask=price,
        tick_speed=10, aggr_buy_speed=5, aggr_sell_speed=5, price_speed=2, vwap_daily=price
    )

def run_logic_test():
    print("Running Level 1: Expected Output Tests...")
    
    # Initialize Manager with Test Config
    test_config = GC_M1_SMC_CONFIG
    test_config.htf_ema_period = 10
    test_config.htf_swing_length = 5
    
    manager = SMCContextManager(test_config, tick_size=0.1)
    
    # Scenario:
    # 1. Create 30 bars of UPTREND (enough for EMA 10 and Swing 5 on H1?)
    # Wait, H1 resampler needs 60 M1 bars to close 1 H1 bar.
    # To get 10 H1 bars (for EMA), we need 600 M1 bars.
    # To get 5 H1 bars (for Swing), we need 300 M1 bars.
    # Let's generate 1000 M1 bars (approx 16 hours).
    
    start_time = datetime(2025, 1, 1, 0, 0)
    bars = []
    
    # 1. Establish Uptrend (Bars 0-800)
    # Price moves from 1900 to 2000
    for i in range(800):
        price = 1900 + (i * 0.125) # Ends at 2000
        t = start_time + timedelta(minutes=i)
        bars.append(create_synthetic_bar(i, price, t))
        
    # 2. Pullback (Bars 800-1200)
    # Price drops to 1950
    for i in range(400):
        price = 2000 - (i * 0.125) # Ends at 1950
        t = start_time + timedelta(minutes=800+i)
        bars.append(create_synthetic_bar(800+i, price, t))
        
    print(f"Generated {len(bars)} synthetic bars.")
    
    # Run Context Manager
    last_fb = None
    
    for i, bar in enumerate(bars):
        fb = manager.update(bar)
        last_fb = fb
        
    # Final State Verification (at bar 999, price=1950)
    print("\n--- Final State Verification (Bar 999, Price=1950) ---")
    fb_dict = last_fb.to_dict()
    
    # 1. H1 Trend
    # We have 1000 mins = 16.6 hours.
    # EMA period 10. We have >10 H1 bars.
    # Price dropped from 2000 to 1950 in last 3 hours.
    # EMA 10 should be somewhere below 2000 but maybe above 1950?
    # Let's see.
    
    print(f"H1 Trend Up: {fb_dict['h1_trend_up']}")
    print(f"H1 Trend Down: {fb_dict['h1_trend_down']}")
    
    # DEBUG: Inspect Internal State
    print("\n--- DEBUG STATE ---")
    print(f"H1 Resampler EMA: {manager.h1_resampler.ema_value}")
    print(f"H1 Resampler Closed Bars: {len(manager.h1_resampler.closed_closes)}")
    print(f"H1 SMC Swing High: {manager.h1_smc.state.swing_high}")
    print(f"H1 SMC Swing Low: {manager.h1_smc.state.swing_low}")
    print(f"H1 SMC Last Leg: {manager.h1_smc.last_swing_leg}")
    print(f"H1 SMC FVGs: {len(manager.h1_smc.fvgs)}")
    
    # 2. Premium/Discount
    
    # 2. Premium/Discount
    # Needs Swing High/Low.
    # LuxSMC needs 50 bars for swing. We have 250 bars.
    # We should have a Swing Low at start (1900) and Swing High at 2000.
    # Midpoint = 1950.
    # Current Price = 1950.
    # If Price <= Midpoint, it is Discount.
    
    print(f"H1 Premium: {fb_dict['h1_premium']}")
    print(f"H1 Discount: {fb_dict['h1_discount']}")
    
    # 3. Distances
    # Swing High at 2000. Current 1950. Dist = 50 / ATR.
    # ATR is calculated on H1 bars. We have ~4 H1 bars.
    # ATR period 14. So ATR is also not fully ready?
    # Default ATR value?
    # HTFResampler: if len >= atr_period ... else atr_value=0.
    # FeatureBar: if atr > 0 else 1.0.
    # So ATR=1.0. Dist should be 50.0.
    
    print(f"Dist to H1 Swing High: {fb_dict['dist_to_h1_swing_high']} (Expected ~50.0)")
    print(f"Dist to H1 Swing Low: {fb_dict['dist_to_h1_swing_low']}")
    
    # Assertions
    failures = []
    
    # Trend might be Up or Down depending on EMA lag.
    # But it should NOT be 0 (we have enough data now).
    if fb_dict['h1_trend_up'] == 0 and fb_dict['h1_trend_down'] == 0:
        failures.append("H1 Trend is 0 (Should be Up or Down, we have enough data)")
        
    # We expect Discount (price 1950 <= mid 1950)
    if fb_dict['h1_discount'] != 1.0: 
        failures.append(f"Should be H1 Discount (Price 1950 <= Mid 1950). Got {fb_dict['h1_discount']}")
    
    # Dist to Swing High
    # Should be > 0.
    if fb_dict['dist_to_h1_swing_high'] <= 0:
        failures.append(f"Dist to H1 Swing High should be > 0. Got {fb_dict['dist_to_h1_swing_high']}")
        
    if failures:
        print("\nFAILED:")
        for f in failures: print(f"- {f}")
        sys.exit(1)
    else:
        print("\nSUCCESS: Logic Verification Passed!")
        sys.exit(0)

if __name__ == "__main__":
    run_logic_test()
