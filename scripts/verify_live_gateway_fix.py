#!/usr/bin/env python3
"""
Verify Live Gateway Fix
Send a few test bars and check p_shift values
"""

import requests
import json
from datetime import datetime, timedelta
import numpy as np

GATEWAY_URL = "http://localhost:8000"

def send_bar(bar_index: int, timestamp: datetime, price: float, volume: int = 100):
    """Send a bar to live gateway"""
    payload = {
        "symbol": "GC 06-25",
        "timeframe": "M1",
        "timestamp": timestamp.isoformat(),
        "bar_index": bar_index,
        "bar": {
            "o": price - 0.5,
            "h": price + 1.0,
            "l": price - 1.0,
            "c": price,
            "volume": volume,
            "delta": np.random.randint(-20, 20),
            "buy_volume": volume // 2 + np.random.randint(-10, 10),
            "sell_volume": volume // 2 + np.random.randint(-10, 10),
            "best_bid": price - 0.1,
            "best_ask": price + 0.1,
            "vwap_daily": 3300.0
        },
        "tick_features": {
            "tick_speed": 10.0,
            "aggr_buy_speed": 5.0,
            "aggr_sell_speed": 5.0,
            "price_speed": 2.0
        }
    }
    
    response = requests.post(f"{GATEWAY_URL}/live_bar", json=payload)
    return response.json()

def main():
    print("=" * 70)
    print("VERIFY LIVE GATEWAY FIX")
    print("=" * 70)
    
    # Check health
    print("\nChecking gateway health...")
    try:
        health = requests.get(f"{GATEWAY_URL}/health").json()
        print(f"  Status: {health['status']}")
        print(f"  Model loaded: {health['model_loaded']}")
        print(f"  Contexts active: {health['contexts_active']}")
    except Exception as e:
        print(f"  ERROR: Cannot connect to gateway: {e}")
        print("  Please start the gateway first!")
        return
    
    # Send warmup bars (need 60+ for ASM context)
    print(f"\nSending 100 warmup bars...")
    base_time = datetime(2025, 5, 1, 8, 0, 0)
    base_price = 3300.0
    
    for i in range(100):
        price_change = np.random.randn() * 2
        base_price += price_change
        
        result = send_bar(
            bar_index=i,
            timestamp=base_time + timedelta(minutes=i),
            price=base_price,
            volume=100 + np.random.randint(-30, 30)
        )
        
        if i % 20 == 0:
            print(f"  Sent bar {i}...")
    
    # Send test bars and check p_shift
    print(f"\nSending 10 test bars and checking p_shift...")
    p_shifts = []
    
    for i in range(100, 110):
        price_change = np.random.randn() * 2
        base_price += price_change
        
        result = send_bar(
            bar_index=i,
            timestamp=base_time + timedelta(minutes=i),
            price=base_price,
            volume=100 + np.random.randint(-30, 30)
        )
        
        p_shift = result.get('p_shift')
        s4_setup = result.get('s4_setup', False)
        has_signal = result.get('has_signal', False)
        
        if p_shift is not None:
            p_shifts.append(p_shift)
            print(f"  Bar {i}: p_shift={p_shift:.4f}, s4_setup={s4_setup}, has_signal={has_signal}")
        else:
            print(f"  Bar {i}: No ASM prediction (s4_setup={s4_setup})")
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    
    if p_shifts:
        print(f"  p_shift values collected: {len(p_shifts)}")
        print(f"  Min p_shift: {min(p_shifts):.4f}")
        print(f"  Max p_shift: {max(p_shifts):.4f}")
        print(f"  Avg p_shift: {np.mean(p_shifts):.4f}")
        
        if all(p > 0.9 for p in p_shifts):
            print(f"\n  ❌ All p_shift > 0.9 - Feature mismatch likely still exists!")
        elif any(p <= 0.2 for p in p_shifts):
            print(f"\n  ✅ Some p_shift <= 0.2 - Fix is working!")
        else:
            print(f"\n  📊 p_shift values are moderate - Fix applied, but no signals pass threshold")
    else:
        print("  No p_shift values collected (no S4 setups detected)")
    
    print(f"\n{'='*70}")
    print("DONE!")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
