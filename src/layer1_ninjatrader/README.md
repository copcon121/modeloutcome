# Layer 1: NinjaTrader Adapter - ExportRawData

## Overview
`ExportRawData.cs` is a NinjaTrader 8 strategy that exports raw OHLCV bar data to the Python Feature Engine (Layer 2) via HTTP POST requests.

## Features
- **Real-time data export**: Sends bar data to Feature Engine every N bars (configurable)
- **Historical context**: Includes last 100 bars (configurable) in each export for ML context
- **Non-blocking**: Uses async HTTP calls to avoid impacting chart performance
- **Extensible**: Placeholders for delta volume and Level 2 depth data (requires Rithmic API)

## Installation

### 1. Copy Strategy File
1. Open NinjaTrader 8
2. Go to `Tools` → `Edit NinjaScript` → `Strategy`
3. Click `+ New` or `Import`
4. Copy the contents of `ExportRawData.cs` into a new strategy file
5. Save as `ExportRawData`

### 2. Install Dependencies
The strategy uses `Newtonsoft.Json` for JSON serialization, which is included by default in NinjaTrader 8.

If you encounter compilation errors:
1. Right-click the strategy in NinjaScript Editor
2. Select `References`
3. Ensure `Newtonsoft.Json` is checked

### 3. Compile
1. Click `Compile` in the NinjaScript Editor
2. Verify no errors appear
3. Close the editor

## Configuration

### Attach Strategy to Chart
1. Open a chart (e.g., NQ 1 Minute)
2. Right-click chart → `Strategies`
3. Select `ExportRawData` from the list
4. Configure parameters (see below)
5. Click `OK`

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| **Endpoint URL** | `http://localhost:5001/raw` | URL of the Python Feature Engine API |
| **Bars To Export** | `100` | Number of bars to include in each export (context window) |
| **Export Interval Bars** | `1` | Export data every N bars (1 = every bar, 5 = every 5 bars) |
| **Include Delta** | `false` | Enable delta volume export (requires Rithmic integration) |
| **Include L2 Depth** | `false` | Enable Level 2 depth export (requires Rithmic integration) |

### Recommended Settings
- **For live trading**: Set `Export Interval Bars = 1` (export every bar)
- **For testing**: Set `Export Interval Bars = 5` to reduce HTTP traffic
- **Bars To Export**: 60-100 bars (provides sufficient ML context)

## Usage

### Start Exporting
1. Ensure the Python Feature Engine server is running on `http://localhost:5001`
2. Attach the strategy to a chart
3. Strategy will begin sending bar data on each bar close

### Verify Data Export
Check the NinjaTrader Output Window (`Tools` → `Output Window`) for logs:
- Success: No messages (silent by default to avoid spam)
- Errors: `ExportRawData ERROR: [error message]`

## JSON Payload Format

The strategy sends JSON in the following format:

```json
{
  "symbol": "NQ 03-25",
  "timeframe": "1Minute",
  "timestamp": "2025-01-26T14:30:00.123",
  "bars": [
    {
      "ts": "2025-01-26T12:50:00",
      "open": 17500.25,
      "high": 17505.50,
      "low": 17498.00,
      "close": 17502.75,
      "volume": 1250,
      "delta": 0.0,
      "buy_volume": 625.0,
      "sell_volume": 625.0,
      "l2_bid_depth": [],
      "l2_ask_depth": []
    },
    ...
  ]
}
```

**Notes**:
- `delta`, `buy_volume`, `sell_volume` are currently placeholder values (0.0 or 50/50 split)
- `l2_bid_depth` and `l2_ask_depth` are empty arrays (Rithmic integration required)

## Extending with Rithmic API

To enable real delta volume and Level 2 depth:

### Delta Volume
1. Subscribe to `OnMarketData()` or `OnMarketDepth()` events
2. Track bid vs ask volume per bar
3. Calculate delta: `delta = buy_volume - sell_volume`
4. Update the `CollectBarData()` method to use real values

Example:
```csharp
protected override void OnMarketData(MarketDataEventArgs marketDataUpdate)
{
    if (marketDataUpdate.MarketDataType == MarketDataType.Last)
    {
        // Track if trade was at bid or ask
        // Accumulate buy/sell volume per bar
    }
}
```

### Level 2 Depth
1. Subscribe to `OnMarketDepth()` events from Rithmic
2. Capture top 5-10 levels of bid/ask depth
3. Store in class variables
4. Include in JSON payload

Example:
```csharp
protected override void OnMarketDepth(MarketDepthEventArgs marketDepthUpdate)
{
    // Store bid/ask prices and volumes
    // Update l2_bid_depth and l2_ask_depth arrays
}
```

## Troubleshooting

### "HTTP POST failed with status 404"
- Ensure Python Feature Engine is running: `uvicorn src.layer2_feature_engine.api_server:app --port 5001`
- Check endpoint URL matches: `http://localhost:5001/raw`

### "Connection refused"
- Feature Engine server not started
- Firewall blocking localhost connections
- Wrong port number

### Strategy not exporting data
- Check `CurrentBar < BarsToExport` condition (need minimum bars on chart)
- Verify `Calculate = Calculate.OnBarClose` is set
- Check Output Window for errors

### High CPU usage
- Reduce `Export Interval Bars` (e.g., set to 5 instead of 1)
- Reduce `Bars To Export` (e.g., 60 instead of 100)
- Ensure Feature Engine server responds quickly

## Performance Notes
- **Latency**: Async HTTP calls typically add <10ms overhead
- **Network**: Uses localhost, no internet required
- **Threading**: Fire-and-forget pattern prevents UI blocking
- **Memory**: Minimal (only stores current bar data, no accumulation)

## Next Steps
After verifying Layer 1 works:
1. Proceed to Layer 2: Implement Feature Engine to receive and process bar data
2. Add Rithmic integration for delta and L2 features
3. Implement live inference pipeline (Layer 2 → Layer 3)

## Support
For issues or questions:
- Check NinjaTrader 8 documentation: https://ninjatrader.com/support/helpGuides/nt8/
- Review ARCHITECTURE.md for system design
- Refer to PROJECT_MASTER_PLAN.md Phase 1 checklist
