# API Documentation

**Live Gateway API** - S4_LDN_ASM_LowShift_0.2_v1.1

## Overview

FastAPI service that receives M1 bars from NinjaTrader SMC_Exporter_Pro_v3 and generates trade signals using:
- S4 HighVol FVG London rule engine
- ASM-GRU64-v1.0-C3 model for auction state prediction
- LowShift filter (p_shift ≤ 0.2)

Base URL: `http://localhost:8000`

---

## Endpoints

### GET /health

Health check endpoint.

**Response:**
```json
{
  "status": "ok",
  "model_loaded": true,
  "contexts_active": 1
}
```

| Field | Type | Description |
|-------|------|-------------|
| status | string | "ok" if healthy |
| model_loaded | boolean | ASM model loaded status |
| contexts_active | integer | Number of active symbol contexts |

---

### POST /live_bar

Process a bar and return trade signal.

**Request Body:**
```json
{
  "symbol": "GC 12-25",
  "timeframe": "M1",
  "timestamp": "2025-11-17T23:01:00",
  "bar_index": 12345,
  "bar": {
    "o": 4045.4,
    "h": 4046.1,
    "l": 4044.8,
    "c": 4045.9,
    "volume": 123,
    "delta": -45,
    "buy_volume": 39,
    "sell_volume": 84,
    "best_bid": 4045.8,
    "best_ask": 4046.0,
    "vwap_daily": 4044.9
  },
  "tick_features": {
    "tick_speed": 57,
    "aggr_buy_speed": 80.0,
    "aggr_sell_speed": 43.0,
    "price_speed": 1.3
  }
}
```

**Request Schema:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| symbol | string | Yes | Instrument symbol (e.g., "GC 12-25") |
| timeframe | string | Yes | Bar timeframe ("M1") |
| timestamp | datetime | Yes | Bar close timestamp (ISO 8601) |
| bar_index | integer | Yes | Sequential bar index |
| bar | BarData | Yes | OHLCV + orderflow data |
| tick_features | TickFeatures | Yes | Aggregated tick features |

**BarData Schema:**

| Field | Type | Description |
|-------|------|-------------|
| o | float | Open price |
| h | float | High price |
| l | float | Low price |
| c | float | Close price |
| volume | float | Total volume |
| delta | float | Buy volume - Sell volume |
| buy_volume | float | Aggressive buy volume |
| sell_volume | float | Aggressive sell volume |
| best_bid | float | Best bid at bar close |
| best_ask | float | Best ask at bar close |
| vwap_daily | float | Daily VWAP |

**TickFeatures Schema:**

| Field | Type | Description |
|-------|------|-------------|
| tick_speed | float | Ticks per second |
| aggr_buy_speed | float | Aggressive buy speed |
| aggr_sell_speed | float | Aggressive sell speed |
| price_speed | float | Price movement speed |

**Response (Signal):**
```json
{
  "has_signal": true,
  "module": "S4_LDN_ASM_LowShift_0.2_v1.1",
  "symbol": "GC 12-25",
  "timeframe": "M1",
  "side": "long",
  "entry": 4045.9,
  "sl": 4040.5,
  "tp": 4056.7,
  "rr": 2.0,
  "session": "London",
  "p_shift": 0.15,
  "p_up": 0.08,
  "p_down": 0.07,
  "p_neutral": 0.85,
  "s4_setup": true,
  "high_vol": true,
  "in_fvg": true,
  "version": "S4_LDN_ASM_LowShift_0.2_v1.1"
}
```

**Response (No Signal):**
```json
{
  "has_signal": false,
  "module": "S4_LDN_ASM_LowShift_0.2_v1.1",
  "symbol": "GC 12-25",
  "timeframe": "M1",
  "session": "NY",
  "s4_setup": false,
  "high_vol": false,
  "in_fvg": false,
  "version": "S4_LDN_ASM_LowShift_0.2_v1.1"
}
```

**Response Schema:**

| Field | Type | Description |
|-------|------|-------------|
| has_signal | boolean | True if trade signal generated |
| module | string | Strategy module name |
| symbol | string | Instrument symbol |
| timeframe | string | Bar timeframe |
| side | string | "long" or "short" (null if no signal) |
| entry | float | Entry price (null if no signal) |
| sl | float | Stop loss price (null if no signal) |
| tp | float | Take profit price (null if no signal) |
| rr | float | Risk/Reward ratio (2.0) |
| session | string | Current session (London/NY/Asia/Off) |
| p_shift | float | ASM shift probability (p_up + p_down) |
| p_up | float | ASM up probability |
| p_down | float | ASM down probability |
| p_neutral | float | ASM neutral probability |
| s4_setup | boolean | S4 setup conditions met |
| high_vol | boolean | High volatility regime |
| in_fvg | boolean | Price in FVG zone |
| version | string | Strategy version |

---

### GET /stats

Get gateway statistics and configuration.

**Response:**
```json
{
  "version": "S4_LDN_ASM_LowShift_0.2_v1.1",
  "model_loaded": true,
  "contexts_active": 1,
  "p_shift_threshold": 0.2,
  "session_filter": "London",
  "rr_target": 2.0
}
```

---

## Error Handling

### HTTP Status Codes

| Code | Description |
|------|-------------|
| 200 | Success |
| 400 | Bad Request - Invalid input |
| 422 | Validation Error - Schema mismatch |
| 500 | Internal Server Error |

### Error Response Format
```json
{
  "detail": "Error message description"
}
```

### Common Errors

**422 Validation Error:**
```json
{
  "detail": [
    {
      "loc": ["body", "bar", "volume"],
      "msg": "field required",
      "type": "value_error.missing"
    }
  ]
}
```

**500 Internal Error:**
```json
{
  "detail": "ASM model not loaded"
}
```

---

## Rate Limiting & Performance

### Performance Targets
- Response time: <100ms per bar
- Throughput: 60+ bars/second
- Memory: <500MB

### Recommendations
- Send bars sequentially (one at a time)
- Wait for response before sending next bar
- Use connection pooling for HTTP clients
- Consider in-process calls for replay/backtest

### No Rate Limiting
Current implementation has no rate limiting. For production:
- Consider adding rate limiting middleware
- Monitor resource usage
- Scale horizontally if needed

---

## NinjaTrader Integration

### SMC_Exporter_Pro_v3 Configuration

1. **Enable HTTP Export:**
   ```csharp
   // In indicator settings
   ExportEndpoint = "http://localhost:8000/live_bar";
   ExportOnBarClose = true;
   ```

2. **JSON Format:**
   The exporter's `BuildJsonForClosedBar()` method produces the exact schema expected by `/live_bar`.

3. **Connection Flow:**
   ```
   NinjaTrader → SMC_Exporter_Pro_v3 → HTTP POST → Live Gateway → Response
   ```

### Shadow Mode
Gateway operates in shadow mode by default:
- Generates signals and logs them
- No actual trading execution
- Use for validation before live deployment

### Signal Handling in NinjaTrader
```csharp
// Parse response
var response = JsonConvert.DeserializeObject<LiveSignalResponse>(responseBody);

if (response.has_signal) {
    // Log signal for review
    Print($"Signal: {response.side} @ {response.entry}, SL={response.sl}, TP={response.tp}");
    
    // In shadow mode: just log
    // In live mode: execute trade (not implemented)
}
```

---

## Code Examples

### Python - Send Bar
```python
import requests

bar_event = {
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

response = requests.post("http://localhost:8000/live_bar", json=bar_event)
signal = response.json()

if signal["has_signal"]:
    print(f"SIGNAL: {signal['side']} @ {signal['entry']}")
```

### cURL - Health Check
```bash
curl http://localhost:8000/health
```

### cURL - Send Bar
```bash
curl -X POST http://localhost:8000/live_bar \
  -H "Content-Type: application/json" \
  -d '{"symbol":"GC 12-25","timeframe":"M1","timestamp":"2025-11-17T23:01:00","bar_index":12345,"bar":{"o":4045.4,"h":4046.1,"l":4044.8,"c":4045.9,"volume":123,"delta":-45,"buy_volume":39,"sell_volume":84,"best_bid":4045.8,"best_ask":4046.0,"vwap_daily":4044.9},"tick_features":{"tick_speed":57,"aggr_buy_speed":80.0,"aggr_sell_speed":43.0,"price_speed":1.3}}'
```

---

## Logging

### Signal Log Format
File: `logs/live_signals_s4_asm_v1.jsonl`

```json
{
  "timestamp": "2025-11-17T10:15:00",
  "symbol": "GC 12-25",
  "timeframe": "M1",
  "bar_index": 12345,
  "session": "London",
  "s4_setup": true,
  "side": "long",
  "entry": 4045.9,
  "sl": 4040.5,
  "tp": 4056.7,
  "high_vol": true,
  "in_fvg": true,
  "ext_trend": 1,
  "p_up": 0.08,
  "p_down": 0.07,
  "p_neutral": 0.85,
  "p_shift": 0.15,
  "filter_pass": true,
  "version": "S4_LDN_ASM_LowShift_0.2_v1.1"
}
```

---

**Version**: 1.0  
**Last Updated**: 2025-12-03
