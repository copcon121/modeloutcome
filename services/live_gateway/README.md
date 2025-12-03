# Live Gateway Service

**S4_LDN_ASM_LowShift_0.2_v1.1** - Real-time signal generation service

## Overview

FastAPI service that receives M1 bars from NinjaTrader SMC_Exporter_Pro_v3 and generates trade signals using:
- S4 HighVol FVG London rule engine
- ASM-GRU64-v1.0-C3 model for auction state prediction
- LowShift filter (p_shift ≤ 0.2)

## Architecture

```
NinjaTrader → SMC_Exporter_Pro_v3 → JSON → Live Gateway → Signal
                                            ↓
                                      [Context Store]
                                      [S4 Engine]
                                      [ASM Model]
```

## Files

- `app.py` - FastAPI application with endpoints
- `models.py` - Pydantic schemas for request/response
- `context_store.py` - In-memory context management
- `s4_engine.py` - S4 rule engine implementation
- `asm_inference.py` - ASM model loading and inference
- `run_server.py` - Server startup script

## API Endpoints

### `GET /health`
Health check with model status

### `POST /live_bar`
Process bar and return signal

**Request:**
```json
{
  "symbol": "GC 12-25",
  "timeframe": "M1",
  "timestamp": "2025-11-17T23:01:00",
  "bar_index": 12345,
  "bar": {
    "o": 4045.4, "h": 4046.1, "l": 4044.8, "c": 4045.9,
    "volume": 123, "delta": -45,
    "buy_volume": 39, "sell_volume": 84,
    "best_bid": 4045.8, "best_ask": 4046.0,
    "vwap_daily": 4044.9
  },
  "tick_features": {
    "tick_speed": 57, "aggr_buy_speed": 80.0,
    "aggr_sell_speed": 43.0, "price_speed": 1.3
  }
}
```

**Response (Signal):**
```json
{
  "has_signal": true,
  "module": "S4_LDN_ASM_LowShift_0.2_v1.1",
  "symbol": "GC 12-25",
  "side": "long",
  "entry": 4045.9,
  "sl": 4040.5,
  "tp": 4056.7,
  "rr": 2.0,
  "session": "London",
  "p_shift": 0.15
}
```

### `GET /stats`
Gateway statistics and configuration

## Usage

### Start Server
```bash
# Default (port 8000)
python services/live_gateway/run_server.py

# Custom port
python services/live_gateway/run_server.py --port 8001
```

### Test with Replay
```bash
python scripts/simulate_live_gateway_from_jsonl.py
```

## Configuration

### S4 Rules
- **Session**: London (08:00 - 14:00 UTC)
- **Regime**: HighVol (range > Q66 or volume > 2x avg)
- **Entry**: FVG retest in trend direction
- **RR**: 2.0 (1:2 risk/reward)

### ASM Filter
- **Model**: ASM-GRU64-v1.0-C3.pt
- **Context**: 60 bars M1 (100 features)
- **Filter**: p_shift ≤ 0.2

## Performance

**Expected Results** (NEW 6W backtest):
- Trades: ~258 per 6 weeks
- Winrate: ~61.6%
- Expectancy: ~+0.85R
- MaxDD: ~26R

## Dependencies

```bash
pip install -r requirements.txt
```
