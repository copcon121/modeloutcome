# 📊 NINJATRADER DATA SPECIFICATION - PRO MODE

**Version**: 2.0 (Updated 2025-01-26)
**Architecture**: NinjaTrader exports RAW + TICK FEATURES → Python builds all derived features

---

## 🎯 OVERVIEW

NinjaTrader chỉ export **raw data + tick features** cho mỗi bar M1. Python sẽ xử lý tất cả:
- ✅ SMC structure (BOS/CHoCH/Sweep/OB/FVG)
- ✅ Volume Profile (VAH/VAL/POC)
- ✅ Multi-Timeframe (M5 tự build từ M1)
- ✅ Normalization & feature scaling

**Lý do**: Separation of concerns - NinjaTrader làm realtime data collection, Python làm feature engineering.

---

## 📦 EXPORT MODES

### ✅ PRO MODE (Recommended)
**Export**: RAW + TICK FEATURES

NinjaTrader exports:
1. **Basic bar data**: OHLCV
2. **Orderflow**: delta, buy_volume, sell_volume
3. **Market depth**: best_bid, best_ask, spread
4. **Tick features**: tick_speed, aggr_buy_speed, aggr_sell_speed, price_speed

Python handles:
- SMC structure detection
- Volume Profile calculation
- Multi-timeframe aggregation
- All feature engineering

---

## 📊 DATA FIELDS SPECIFICATION

### 1️⃣ BASIC BAR DATA

| Field | Type | Description | NinjaTrader Code |
|-------|------|-------------|------------------|
| `ts` | string | Timestamp (ISO 8601) | `Time[0].ToString("yyyy-MM-ddTHH:mm:ss")` |
| `open` | float | Open price | `Open[0]` |
| `high` | float | High price | `High[0]` |
| `low` | float | Low price | `Low[0]` |
| `close` | float | Close price | `Close[0]` |
| `volume` | float | Total volume | `Volume[0]` |

**Example**:
```json
{
  "ts": "2025-01-26T14:30:00",
  "open": 17502.75,
  "high": 17508.00,
  "low": 17501.50,
  "close": 17506.25,
  "volume": 1380
}
```

---

### 2️⃣ ORDERFLOW DATA

| Field | Type | Description | Calculation |
|-------|------|-------------|-------------|
| `delta` | float | Buy volume - Sell volume | Cumulative in bar |
| `buy_volume` | float | Aggressive buy (at ask) | Volume when price >= ask |
| `sell_volume` | float | Aggressive sell (at bid) | Volume when price <= bid |

**Implementation**:
```csharp
// In OnMarketData()
if (e.MarketDataType == MarketDataType.Last)
{
    double currentBid = GetCurrentBid();
    double currentAsk = GetCurrentAsk();

    if (e.Price >= currentAsk)
    {
        currentBarBuyVolume += e.Volume;
        currentBarDelta += e.Volume;
    }
    else if (e.Price <= currentBid)
    {
        currentBarSellVolume += e.Volume;
        currentBarDelta -= e.Volume;
    }
}
```

**Example**:
```json
{
  "delta": 125.0,
  "buy_volume": 752.5,
  "sell_volume": 627.5
}
```

**Validation**: `buy_volume - sell_volume ≈ delta`

---

### 3️⃣ MARKET DEPTH DATA

| Field | Type | Description | NinjaTrader Code |
|-------|------|-------------|------------------|
| `best_bid` | float | Best bid price | `GetCurrentBid()` |
| `best_ask` | float | Best ask price | `GetCurrentAsk()` |
| `spread` | float | Ask - Bid | `GetCurrentAsk() - GetCurrentBid()` |

**Example**:
```json
{
  "best_bid": 17506.00,
  "best_ask": 17506.25,
  "spread": 0.25
}
```

---

### 4️⃣ TICK FEATURES (🆕 NEW!)

**Critical**: These are per-bar metrics computed by NinjaTrader during bar formation.

| Field | Type | Unit | Description |
|-------|------|------|-------------|
| `tick_speed` | float | ticks/sec | Tốc độ tick mỗi giây (trading intensity) |
| `aggr_buy_speed` | float | contracts/sec | Tốc độ aggressive buy market orders/giây |
| `aggr_sell_speed` | float | contracts/sec | Tốc độ aggressive sell market orders/giây |
| `price_speed` | float | points/sec | Tốc độ di chuyển giá/giây |

---

#### 4.1 Tick Speed
**Definition**: Số lượng tick (price updates) mỗi giây trong bar.

**Formula**:
```
tick_speed = total_ticks_in_bar / bar_duration_seconds
```

**Example**:
- Bar duration: 60 seconds (M1)
- Total ticks: 1200 ticks
- tick_speed = 1200 / 60 = **20.0 ticks/sec**

**Interpretation**:
- High tick_speed (>30) → High trading activity, volatility
- Low tick_speed (<10) → Low activity, consolidation

**NinjaTrader Implementation**:
```csharp
private int tickCount = 0;
private DateTime barStartTime;

protected override void OnMarketData(MarketDataEventArgs e)
{
    if (e.MarketDataType == MarketDataType.Last)
    {
        tickCount++;
    }
}

protected override void OnBarUpdate()
{
    if (CurrentBar > 0)
    {
        double barDurationSeconds = (Time[0] - barStartTime).TotalSeconds;
        double tickSpeed = tickCount / barDurationSeconds;

        // Export tickSpeed
        // Reset for next bar
        tickCount = 0;
        barStartTime = Time[0];
    }
}
```

---

#### 4.2 Aggressive Buy Speed
**Definition**: Tốc độ aggressive buy market orders (at ask) mỗi giây.

**Formula**:
```
aggr_buy_speed = buy_volume / bar_duration_seconds
```

**Example**:
- Bar duration: 60 seconds
- Buy volume: 750 contracts
- aggr_buy_speed = 750 / 60 = **12.5 contracts/sec**

**Interpretation**:
- High aggr_buy_speed → Strong buying pressure
- aggr_buy_speed > aggr_sell_speed → Bullish momentum

**NinjaTrader Implementation**:
```csharp
protected override void OnBarUpdate()
{
    double barDurationSeconds = (Time[0] - barStartTime).TotalSeconds;
    double aggrBuySpeed = currentBarBuyVolume / barDurationSeconds;
    // Export aggrBuySpeed
}
```

---

#### 4.3 Aggressive Sell Speed
**Definition**: Tốc độ aggressive sell market orders (at bid) mỗi giây.

**Formula**:
```
aggr_sell_speed = sell_volume / bar_duration_seconds
```

**Example**:
- Bar duration: 60 seconds
- Sell volume: 625 contracts
- aggr_sell_speed = 625 / 60 = **10.42 contracts/sec**

**Interpretation**:
- High aggr_sell_speed → Strong selling pressure
- aggr_sell_speed > aggr_buy_speed → Bearish momentum

---

#### 4.4 Price Speed
**Definition**: Tốc độ di chuyển giá (price change) mỗi giây.

**Formula**:
```
price_speed = |close - open| / bar_duration_seconds
```

**Example**:
- Open: 17502.75
- Close: 17506.25
- Bar duration: 60 seconds
- price_speed = |17506.25 - 17502.75| / 60 = **0.058 points/sec**

**Interpretation**:
- High price_speed → Fast price movement, momentum
- Low price_speed → Slow movement, consolidation
- Combine with tick_speed to detect breakouts

**NinjaTrader Implementation**:
```csharp
protected override void OnBarUpdate()
{
    if (CurrentBar > 0)
    {
        double barDurationSeconds = 60.0; // M1 bars
        double priceChange = Math.Abs(Close[0] - Open[0]);
        double priceSpeed = priceChange / barDurationSeconds;
        // Export priceSpeed
    }
}
```

---

### 📊 Tick Features Summary

| Feature | High Value Meaning | Low Value Meaning |
|---------|-------------------|-------------------|
| `tick_speed` | High activity, volatility | Low activity, quiet |
| `aggr_buy_speed` | Strong buying pressure | Weak buying |
| `aggr_sell_speed` | Strong selling pressure | Weak selling |
| `price_speed` | Fast price movement | Slow/consolidation |

**Use Cases**:
- **Breakout detection**: High tick_speed + high price_speed
- **Momentum direction**: aggr_buy_speed vs aggr_sell_speed
- **Consolidation**: Low tick_speed + low price_speed
- **Absorption**: High aggr_sell_speed but price rises (bullish)

---

## 📋 COMPLETE JSON SCHEMA

### Per-Bar Export Format

```json
{
  "symbol": "ES 03-25",
  "timeframe": "1Minute",
  "timestamp": "2025-01-26T14:30:15.123",
  "bar_index": 1234,

  "bar": {
    "ts": "2025-01-26T14:30:00",
    "open": 17502.75,
    "high": 17508.00,
    "low": 17501.50,
    "close": 17506.25,
    "volume": 1380
  },

  "orderflow": {
    "delta": 125.0,
    "buy_volume": 752.5,
    "sell_volume": 627.5
  },

  "market_depth": {
    "best_bid": 17506.00,
    "best_ask": 17506.25,
    "spread": 0.25
  },

  "tick_features": {
    "tick_speed": 20.5,
    "aggr_buy_speed": 12.54,
    "aggr_sell_speed": 10.46,
    "price_speed": 0.058
  }
}
```

---

## 🎨 VISUAL INDICATORS (NinjaTrader Panel)

**Purpose**: Visualize tick features on chart để kiểm tra realtime.

### Panel Layout
```
┌─────────────────────────────────────┐
│  TICK FEATURES (M1)                 │
├─────────────────────────────────────┤
│  Tick Speed:      20.5 ticks/s  ▓▓▓▓▓▓░░░░  │
│  Aggr Buy Speed:  12.5 c/s      ▓▓▓▓▓░░░░░  │
│  Aggr Sell Speed: 10.4 c/s      ▓▓▓▓░░░░░░  │
│  Price Speed:     0.058 pts/s   ▓▓░░░░░░░░  │
├─────────────────────────────────────┤
│  Delta: +125  🟢 BULLISH            │
│  Spread: 0.25 pts                   │
└─────────────────────────────────────┘
```

### Implementation (NinjaTrader)
```csharp
protected override void OnRender(ChartControl chartControl, ChartScale chartScale)
{
    base.OnRender(chartControl, chartScale);

    // Draw panel
    int panelX = ChartPanel.X + 10;
    int panelY = ChartPanel.Y + 10;
    int panelWidth = 300;
    int panelHeight = 150;

    // Background
    SharpDX.Direct2D1.Brush bgBrush = chartControl.Properties.ChartBackground.ToDxBrush(RenderTarget);
    RenderTarget.FillRectangle(new RectangleF(panelX, panelY, panelWidth, panelHeight), bgBrush);

    // Draw tick features
    DrawText("Tick Speed: " + tickSpeed.ToString("F1") + " ticks/s", panelX + 10, panelY + 20);
    DrawText("Aggr Buy Speed: " + aggrBuySpeed.ToString("F1") + " c/s", panelX + 10, panelY + 40);
    DrawText("Aggr Sell Speed: " + aggrSellSpeed.ToString("F1") + " c/s", panelX + 10, panelY + 60);
    DrawText("Price Speed: " + priceSpeed.ToString("F3") + " pts/s", panelX + 10, panelY + 80);

    // Draw bar for tick_speed (normalized to 0-50 range)
    DrawBar(tickSpeed, 50, panelX + 200, panelY + 20);
}
```

**Color Coding**:
- 🟢 Green: Bullish (aggr_buy_speed > aggr_sell_speed)
- 🔴 Red: Bearish (aggr_sell_speed > aggr_buy_speed)
- 🟡 Yellow: Neutral/consolidation

---

## 🔧 NINJATRADER IMPLEMENTATION

### Complete Strategy Template

```csharp
#region Using declarations
using System;
using System.Collections.Generic;
using System.Net.Http;
using System.Text;
using System.Threading.Tasks;
using NinjaTrader.Cbi;
using NinjaTrader.Data;
using NinjaTrader.NinjaScript.Strategies;
using Newtonsoft.Json;
#endregion

namespace NinjaTrader.NinjaScript.Strategies
{
    public class ExportRawDataPro : Strategy
    {
        #region Variables

        private HttpClient httpClient;
        private string endpointUrl = "http://localhost:5001/raw";

        // Bar tracking
        private DateTime barStartTime;
        private int barIndex = 0;

        // Orderflow tracking
        private double currentBarBuyVolume = 0;
        private double currentBarSellVolume = 0;
        private double currentBarDelta = 0;

        // Tick features tracking
        private int tickCount = 0;
        private double tickSpeed = 0;
        private double aggrBuySpeed = 0;
        private double aggrSellSpeed = 0;
        private double priceSpeed = 0;

        #endregion

        protected override void OnStateChange()
        {
            if (State == State.SetDefaults)
            {
                Description = "Export RAW + TICK FEATURES to Python Feature Engine";
                Name = "ExportRawDataPro";
                Calculate = Calculate.OnBarClose;
            }
            else if (State == State.Configure)
            {
                httpClient = new HttpClient();
                httpClient.Timeout = TimeSpan.FromSeconds(5);
            }
            else if (State == State.Terminated)
            {
                if (httpClient != null)
                {
                    httpClient.Dispose();
                    httpClient = null;
                }
            }
        }

        protected override void OnBarUpdate()
        {
            if (CurrentBar == 0)
            {
                barStartTime = Time[0];
                return;
            }

            // Calculate bar duration
            double barDurationSeconds = (Time[0] - barStartTime).TotalSeconds;
            if (barDurationSeconds == 0) barDurationSeconds = 60.0; // Default M1

            // Calculate tick features
            tickSpeed = tickCount / barDurationSeconds;
            aggrBuySpeed = currentBarBuyVolume / barDurationSeconds;
            aggrSellSpeed = currentBarSellVolume / barDurationSeconds;
            priceSpeed = Math.Abs(Close[0] - Open[0]) / barDurationSeconds;

            // Build JSON payload
            var payload = new Dictionary<string, object>
            {
                { "symbol", Instrument.FullName },
                { "timeframe", BarsPeriod.Value.ToString() + BarsPeriod.BarsPeriodType.ToString() },
                { "timestamp", DateTime.Now.ToString("yyyy-MM-ddTHH:mm:ss.fff") },
                { "bar_index", barIndex },

                { "bar", new Dictionary<string, object>
                    {
                        { "ts", Time[0].ToString("yyyy-MM-ddTHH:mm:ss") },
                        { "open", Open[0] },
                        { "high", High[0] },
                        { "low", Low[0] },
                        { "close", Close[0] },
                        { "volume", Volume[0] }
                    }
                },

                { "orderflow", new Dictionary<string, object>
                    {
                        { "delta", currentBarDelta },
                        { "buy_volume", currentBarBuyVolume },
                        { "sell_volume", currentBarSellVolume }
                    }
                },

                { "market_depth", new Dictionary<string, object>
                    {
                        { "best_bid", GetCurrentBid() },
                        { "best_ask", GetCurrentAsk() },
                        { "spread", GetCurrentAsk() - GetCurrentBid() }
                    }
                },

                { "tick_features", new Dictionary<string, object>
                    {
                        { "tick_speed", tickSpeed },
                        { "aggr_buy_speed", aggrBuySpeed },
                        { "aggr_sell_speed", aggrSellSpeed },
                        { "price_speed", priceSpeed }
                    }
                }
            };

            // Send to Python
            string jsonPayload = JsonConvert.SerializeObject(payload, Formatting.None);
            _ = SendDataAsync(jsonPayload);

            // Reset for next bar
            barIndex++;
            barStartTime = Time[0];
            tickCount = 0;
            currentBarBuyVolume = 0;
            currentBarSellVolume = 0;
            currentBarDelta = 0;
        }

        protected override void OnMarketData(MarketDataEventArgs e)
        {
            if (e.MarketDataType == MarketDataType.Last)
            {
                tickCount++;

                double currentBid = GetCurrentBid();
                double currentAsk = GetCurrentAsk();

                if (e.Price >= currentAsk)
                {
                    currentBarBuyVolume += e.Volume;
                    currentBarDelta += e.Volume;
                }
                else if (e.Price <= currentBid)
                {
                    currentBarSellVolume += e.Volume;
                    currentBarDelta -= e.Volume;
                }
            }
        }

        private async Task SendDataAsync(string jsonPayload)
        {
            try
            {
                var content = new StringContent(jsonPayload, Encoding.UTF8, "application/json");
                var response = await httpClient.PostAsync(endpointUrl, content);

                if (!response.IsSuccessStatusCode)
                {
                    Print($"ExportRawDataPro: HTTP POST failed with status {response.StatusCode}");
                }
            }
            catch (Exception ex)
            {
                Print($"ExportRawDataPro ERROR: {ex.Message}");
            }
        }
    }
}
```

---

## 📊 PYTHON RESPONSIBILITIES

**Python Feature Engine handles**:

### 1. SMC Structure Detection
```python
# From raw M1 bars → detect structure
from layer2_feature_engine.smc import detect_swings, compute_structure_flags, detect_order_blocks

swings = detect_swings(bars, lookback=5)
structure = compute_structure_flags(bars, swings)  # BOS, CHoCH, Sweep
order_blocks = detect_order_blocks(bars, swings)
fvgs = detect_fair_value_gaps(bars)
```

### 2. Volume Profile
```python
# Build volume profile from M1 bars
from layer2_feature_engine.volume_profile import VolumeProfileBuilder

vp = VolumeProfileBuilder(bars, num_bins=50)
vp_state = vp.build()  # VAH, VAL, POC, HVN, LVN
```

### 3. Multi-Timeframe Aggregation
```python
# Build M5 from M1 bars
m5_bars = aggregate_bars(m1_bars, target_timeframe='5min')

# Build M15 from M5 bars
m15_bars = aggregate_bars(m5_bars, target_timeframe='15min')
```

### 4. Feature Engineering
```python
# Normalize all features
from layer2_feature_engine.core.normalizer import Normalizer

normalizer = Normalizer(method='zscore')
normalizer.fit(features)
normalized_features = normalizer.transform(features)
```

---

## ⚠️ IMPORTANT NOTES

### 1. Dataset Size: NO TOKEN LIMIT
**This is NOT an LLM**. This is a **numeric tabular model** (Transformer/MLP on numeric features).

- ❌ NO 1024 token limit
- ❌ NO text tokenization
- ✅ Context window = number of bars (e.g., 60 M1 bars)
- ✅ Each bar = N numeric features (e.g., 50-100 features)
- ✅ Input shape: `[batch, sequence_len, feature_dim]` e.g., `[32, 60, 80]`

**Example**:
- Context: 60 M1 bars
- Features per bar: 80 features
- Input tensor: `(60, 80)` → 4800 numeric values
- This is TINY compared to LLM (no token limit issue)

### 2. Bar Duration Handling
```csharp
// For M1 bars
double barDurationSeconds = (Time[0] - barStartTime).TotalSeconds;

// For irregular bars (tick, volume, range)
// Still track start time and calculate duration
```

### 3. Tick Features Edge Cases
- **First bar**: tick_speed may be 0 (no ticks yet) → handle in Python
- **Low activity**: tick_speed < 1 → normal for overnight/premarket
- **Spikes**: tick_speed > 100 → normal for high volatility events

---

## 🧪 TESTING & VALIDATION

### Phase 1: NinjaTrader Export Test
```bash
# Start test server
python tests/test_phase1_ninjatrader.py

# Expected output with new fields:
✅ Received bar with tick_features:
  tick_speed: 20.5 ticks/sec
  aggr_buy_speed: 12.5 contracts/sec
  aggr_sell_speed: 10.4 contracts/sec
  price_speed: 0.058 points/sec
```

### Phase 2: Python Feature Engine Test
```python
# Test raw bar ingestion
from layer2_feature_engine.core.schema import RawBar

bar = RawBar(
    ts=datetime.now(),
    open=17502.75,
    high=17508.00,
    low=17501.50,
    close=17506.25,
    volume=1380,
    delta=125.0,
    buy_volume=752.5,
    sell_volume=627.5,
    tick_speed=20.5,
    aggr_buy_speed=12.5,
    aggr_sell_speed=10.4,
    price_speed=0.058
)

# Build features
context_manager.add_bar(bar)
features = context_manager.build_features()
```

---

## 📚 SUMMARY

### NinjaTrader Exports:
1. ✅ OHLCV (basic bar data)
2. ✅ Delta, buy_volume, sell_volume (orderflow)
3. ✅ best_bid, best_ask, spread (market depth)
4. ✅ tick_speed, aggr_buy_speed, aggr_sell_speed, price_speed (tick features) 🆕

### Python Handles:
1. ✅ SMC structure (BOS/CHoCH/Sweep/OB/FVG)
2. ✅ Volume Profile (VAH/VAL/POC/HVN/LVN)
3. ✅ Multi-timeframe (M5, M15 from M1)
4. ✅ Feature engineering & normalization

### Dataset:
- ✅ NO token limit (numeric tabular model, not LLM)
- ✅ Context window = number of bars (e.g., 60 M1)
- ✅ Input shape: `[batch, seq_len, features]` e.g., `[32, 60, 80]`

---

**Version**: 2.0 (Pro Mode)
**Last Updated**: 2025-01-26
**Status**: ✅ Ready for Implementation
