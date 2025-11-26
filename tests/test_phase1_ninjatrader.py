"""
Phase 1 Test: NinjaTrader Data Export Validation
Tests the data flow from NinjaTrader to Feature Engine (without model server)
"""

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
from datetime import datetime
import json

app = FastAPI(title="Phase 1 Test Server", version="1.0.0")

# Track received data
received_requests = []


class BarData(BaseModel):
    """Single bar data"""
    ts: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    delta: Optional[float] = 0.0
    buy_volume: Optional[float] = 0.0
    sell_volume: Optional[float] = 0.0


class RawDataRequest(BaseModel):
    """Request from NinjaTrader"""
    symbol: str
    timeframe: str
    timestamp: str
    bars: List[BarData]


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Phase 1 Test Server - NinjaTrader Data Validation",
        "version": "1.0.0",
        "status": "running",
        "received_requests": len(received_requests)
    }


@app.get("/health")
async def health_check():
    """Health check"""
    return {
        "status": "ok",
        "received_requests": len(received_requests)
    }


@app.post("/raw")
async def receive_raw_data(request: RawDataRequest):
    """
    Receive and validate raw bars from NinjaTrader

    This is a simplified endpoint that just validates the data structure
    without calling the model server.
    """
    try:
        print("\n" + "="*80)
        print(f"✅ RECEIVED DATA FROM NINJATRADER")
        print("="*80)
        print(f"Symbol: {request.symbol}")
        print(f"Timeframe: {request.timeframe}")
        print(f"Timestamp: {request.timestamp}")
        print(f"Number of bars: {len(request.bars)}")

        # Validate bars
        if len(request.bars) == 0:
            print("⚠️  WARNING: No bars received!")
            return {"status": "error", "message": "No bars in request"}

        # Show first and last bar
        first_bar = request.bars[0]
        last_bar = request.bars[-1]

        print("\nFirst Bar:")
        print(f"  Timestamp: {first_bar.ts}")
        print(f"  OHLCV: O={first_bar.open:.2f}, H={first_bar.high:.2f}, L={first_bar.low:.2f}, C={first_bar.close:.2f}, V={first_bar.volume:.0f}")
        if first_bar.delta != 0.0:
            print(f"  Delta: {first_bar.delta:.0f} (Buy={first_bar.buy_volume:.0f}, Sell={first_bar.sell_volume:.0f})")

        print("\nLast Bar:")
        print(f"  Timestamp: {last_bar.ts}")
        print(f"  OHLCV: O={last_bar.open:.2f}, H={last_bar.high:.2f}, L={last_bar.low:.2f}, C={last_bar.close:.2f}, V={last_bar.volume:.0f}")
        if last_bar.delta != 0.0:
            print(f"  Delta: {last_bar.delta:.0f} (Buy={last_bar.buy_volume:.0f}, Sell={last_bar.sell_volume:.0f})")

        # Validation checks
        print("\n📊 VALIDATION CHECKS:")

        checks_passed = 0
        checks_total = 0

        # Check 1: All bars have valid timestamps
        checks_total += 1
        try:
            for bar in request.bars:
                datetime.fromisoformat(bar.ts)
            print("  ✅ All timestamps are valid ISO format")
            checks_passed += 1
        except Exception as e:
            print(f"  ❌ Invalid timestamp format: {e}")

        # Check 2: All bars have OHLCV data
        checks_total += 1
        valid_ohlcv = all(
            bar.open > 0 and bar.high > 0 and bar.low > 0 and bar.close > 0 and bar.volume >= 0
            for bar in request.bars
        )
        if valid_ohlcv:
            print("  ✅ All bars have valid OHLCV data (all positive)")
            checks_passed += 1
        else:
            print("  ❌ Some bars have invalid OHLCV data (zero or negative)")

        # Check 3: High >= Low
        checks_total += 1
        valid_hl = all(bar.high >= bar.low for bar in request.bars)
        if valid_hl:
            print("  ✅ All bars have High >= Low")
            checks_passed += 1
        else:
            print("  ❌ Some bars have High < Low (invalid)")

        # Check 4: Close within [Low, High]
        checks_total += 1
        valid_close = all(bar.low <= bar.close <= bar.high for bar in request.bars)
        if valid_close:
            print("  ✅ All bars have Close within [Low, High]")
            checks_passed += 1
        else:
            print("  ❌ Some bars have Close outside [Low, High] (invalid)")

        # Check 5: Bars are in chronological order
        checks_total += 1
        try:
            timestamps = [datetime.fromisoformat(bar.ts) for bar in request.bars]
            is_sorted = all(timestamps[i] <= timestamps[i+1] for i in range(len(timestamps)-1))
            if is_sorted:
                print("  ✅ Bars are in chronological order")
                checks_passed += 1
            else:
                print("  ⚠️  Bars are NOT in chronological order (may be intentional)")
        except Exception as e:
            print(f"  ❌ Error checking chronological order: {e}")

        # Summary
        print(f"\n📈 VALIDATION SUMMARY: {checks_passed}/{checks_total} checks passed")

        if checks_passed == checks_total:
            print("🎉 ALL CHECKS PASSED! Data format is correct.")
        elif checks_passed >= checks_total - 1:
            print("✅ MOSTLY CORRECT! Minor issues detected.")
        else:
            print("⚠️  ISSUES DETECTED! Please review the data.")

        print("="*80 + "\n")

        # Store request
        received_requests.append({
            "timestamp": request.timestamp,
            "symbol": request.symbol,
            "num_bars": len(request.bars),
            "checks_passed": checks_passed,
            "checks_total": checks_total
        })

        # Return success response (mock prediction for NinjaTrader)
        return {
            "status": "success",
            "symbol": request.symbol,
            "timestamp": request.timestamp,
            "bars_received": len(request.bars),
            "validation": {
                "checks_passed": checks_passed,
                "checks_total": checks_total,
                "result": "PASS" if checks_passed == checks_total else "PARTIAL"
            },
            # Mock prediction (not used in Phase 1, but NinjaTrader might expect these fields)
            "prob_long": 0.33,
            "prob_short": 0.33,
            "prob_skip": 0.34,
            "recommendation": "SKIP"
        }

    except Exception as e:
        print(f"\n❌ ERROR processing request: {e}\n")
        return {"status": "error", "message": str(e)}


@app.get("/received")
async def get_received_data():
    """Get summary of all received requests"""
    return {
        "total_requests": len(received_requests),
        "requests": received_requests
    }


if __name__ == "__main__":
    print("\n" + "="*80)
    print("🚀 PHASE 1 TEST SERVER - NinjaTrader Data Validation")
    print("="*80)
    print("\nThis server will:")
    print("  1. Listen on http://localhost:5001/raw")
    print("  2. Receive data from NinjaTrader ExportRawData strategy")
    print("  3. Validate the data structure and format")
    print("  4. Print detailed validation results")
    print("\n📋 SETUP INSTRUCTIONS:")
    print("  1. Start this server: python tests/test_phase1_ninjatrader.py")
    print("  2. Open NinjaTrader 8")
    print("  3. Attach ExportRawData strategy to a 1-minute chart")
    print("  4. Configure Endpoint URL: http://localhost:5001/raw")
    print("  5. Watch this console for incoming data!")
    print("\n🔍 To view received data summary:")
    print("  GET http://localhost:5001/received")
    print("="*80 + "\n")

    uvicorn.run(app, host="0.0.0.0", port=5001)
