# ARCHITECTURE - ML Outcome Model for Trading

## 1. Tổng quan Outcome Model

### Mục tiêu
Dự án này xây dựng một hệ thống ML trading hoàn chỉnh với mô hình **outcome-based**, tức là dự đoán kết quả trading dựa trên R-multiple (risk-reward ratio) thay vì dự đoán giá đơn thuần.

### Định nghĩa Outcome
- **Input Context**: 40-100 bar M1 (1 phút) với features phong phú:
  - OHLCV cơ bản
  - Delta và orderflow (buy_volume, sell_volume)
  - **Tick Features** (tick_speed, aggr_buy_speed, aggr_sell_speed, price_speed)
  - SMC structure (Swing, BOS, CHoCH, Sweep, OB, FVG) - *computed in Python*
  - Volume Profile (VAH, VAL, POC, HVN, LVN) - *computed in Python*
  - Market Depth (bid/ask, spread)

- **Output**: Xác suất 3 hành động cho mỗi bar candidate
  - `prob_long`: Xác suất nên vào lệnh LONG
  - `prob_short`: Xác suất nên vào lệnh SHORT
  - `prob_skip`: Xác suất nên bỏ qua (không trade)

- **Label Generation**: Dựa trên R-based outcome
  - Định nghĩa stop_R (ví dụ: 1R) và target_R (ví dụ: 2R)
  - Quét future window (30-50 bar) để tính:
    - `max_up_R`: Mức tăng tối đa (tính theo R)
    - `max_down_R`: Mức giảm tối đa (tính theo R)
  - **Label "long"**: nếu `max_up_R >= target_R` TRƯỚC KHI `max_down_R <= -1R`
  - **Label "short"**: nếu `max_down_R <= -target_R` TRƯỚC KHI `max_up_R >= +1R`
  - **Label "skip"**: nếu không thỏa mãn điều kiện trên

### Lợi ích của phương pháp này
1. **Phù hợp thực tế trading**: Tập trung vào risk-reward thay vì chỉ đúng/sai về hướng
2. **Giảm false signals**: Label "skip" giúp model học cách tránh setup kém
3. **Alignment với strategy**: Dễ dàng tích hợp với risk management (stop loss, take profit)

---

## 2. Kiến trúc 3 Layer

### Layer 1: NinjaTrader Adapter (C#) - SMC_Exporter_Pro_v3
**Nhiệm vụ**: Thu thập và xuất **raw OHLCV + tick features** từ NinjaTrader 8.0.28

**Indicator**: `SMC_Exporter_Pro_v3`
- Chạy trên khung M1 (có thể mở rộng M5, M15 sau)
- Export file `.jsonl` (1 bar = 1 dòng JSON) vào: `Documents/NinjaTrader 8/SMC_Exports/<FileName>.jsonl`

**Input**:
- Live market data stream từ NinjaTrader (OHLCV, Volume, Tick data)
- **Delta chuẩn** từ indicator `Volumdelta` (field `DeltasClose[1]`)
- OnMarketData events để đếm tick

**Output - Pro Mode Recommended**:
- File JSONL với mỗi bar M1 export:
  1. **OHLCV**: o, h, l, c, volume
  2. **Orderflow**: delta (từ Volumdelta), buy_volume, sell_volume (suy ra từ volume + delta)
  3. **Market Depth stub**: best_bid, best_ask (đặt = close của bar, không dùng spread thật)
  4. **Tick Features**: tick_speed, aggr_buy_speed, aggr_sell_speed, price_speed

**Lưu ý quan trọng**:
- ❌ **KHÔNG** dùng uptick/downtick thủ công để tính delta
- ✅ Delta trong JSON = delta từ indicator `Volumdelta` (gần footprint nhất, sai khác vài lot là chấp nhận được)
- ✅ `buy_volume = (volume + delta) / 2`, `sell_volume = volume - buy_volume`
- ✅ `best_bid` / `best_ask` chỉ là stub (= close), không export spread thật

**JSON Schema per bar** (Actual Format từ SMC_Exporter_Pro_v3):
```json
{
  "symbol": "GC 02-26",
  "timeframe": "M1",
  "timestamp": "2025-11-17T20:01:00.0000000",
  "bar_index": 1260,

  "bar": {
    "o": 4047.8,
    "h": 4049.1,
    "l": 4043.2,
    "c": 4048.9,
    "volume": 850,

    // Delta chuẩn lấy từ Volumdelta.DeltasClose[1]
    "delta": -77,

    // Suy ra từ volume + delta:
    // buy_volume = (volume + delta) / 2 = (850 + (-77)) / 2 = 386.5
    // sell_volume = volume - buy_volume = 850 - 386.5 = 463.5
    "buy_volume": 386.5,
    "sell_volume": 463.5,

    // Stub: dùng close của bar (không phải bid/ask thật)
    "best_bid": 4048.9,
    "best_ask": 4048.9
  },

  "tick_features": {
    // Tổng số Last tick mà NinjaTrader nhận được trong bar
    "tick_speed": 1404,

    // Dùng luôn buy/sell volume của bar như tốc độ giao dịch chủ động
    "aggr_buy_speed": 386.5,
    "aggr_sell_speed": 463.5,

    // Intrabar range = High[1] - Low[1]
    "price_speed": 5.9
  }
}
```

**Giải thích**:
- `delta`: Delta bar từ Volumdelta indicator (gần footprint nhất trong môi trường Ninja)
- `buy_volume` / `sell_volume`: Phân rã volume theo delta, dùng cho feature ML (không nhất thiết trùng 100% footprint)
- `tick_speed`: Tổng số tick (price updates) trong bar
- `aggr_buy_speed` / `aggr_sell_speed`: Dùng trực tiếp buy/sell volume (không chia cho thời gian)
- `price_speed`: Intrabar range (High - Low)
- `best_bid` / `best_ask`: Stub value = close (không export spread thật)

**Visual Verification Panel** (trong SMC_Exporter_Pro_v3):
- NinjaTrader hiển thị panel 4 dòng trên chart:
  - `tick_speed`: Số tick trong bar
  - `aggr_buy_speed`: Buy volume
  - `aggr_sell_speed`: Sell volume
  - `price_speed`: Intrabar range (H - L)
- Purpose: Visual validation realtime khi backtest/live

**Đặc điểm**:
- Non-blocking (fire-and-forget) để không ảnh hưởng chart rendering
- Có thể export historical data (backtest) hoặc live stream
- Calculate tick features realtime during bar formation

---

### Layer 2: Feature Engine (Python)
**Nhiệm vụ**: Nhận raw + tick features từ Layer 1, xây dựng **TẤT CẢ** derived features phức tạp

**Kiến trúc submodule**:

#### 2.1. `core/`
- **schema.py**: Định nghĩa data structures (RawBar với tick features, FeatureBar, Record)
- **normalizer.py**: Chuẩn hóa features (Z-score, Min-Max)
- **context_manager.py**: Quản lý sliding window context, orchestrate feature building
- **mtf_builder.py**: Xây dựng Multi-Timeframe M5 từ M1 bars

#### 2.2. `smc/` (Smart Money Concepts) - **Python xử lý 100%**
- **swing.py**: Phát hiện swing high/low từ raw M1 bars
- **structure.py**: Detect BOS (Break of Structure), CHoCH (Change of Character), sweep
- **zones.py**: Xác định và tracking Order Block (OB), Fair Value Gap (FVG)
- **Không có SMC logic trong NinjaTrader** - tất cả trong Python

#### 2.3. `volume_profile/` - **Python xử lý 100%**
- **vp_builder.py**: Xây dựng Volume Profile từ raw M1 bars
- Tính VAH (Value Area High), VAL (Value Area Low), POC (Point of Control)
- Identify HVN (High Volume Node), LVN (Low Volume Node)

#### 2.4. `tick_features/`
- **tick_analyzer.py**: Analyze tick features từ NinjaTrader
- Derive additional features: tick_speed_ma, tick_acceleration, etc.

#### 2.5. `utils/`
- **time_features.py**: Session detection (Asia/Europe/US), time-of-day encoding
- **logging_utils.py**: Centralized logging
- **config_loader.py**: Load YAML configs

**Input**: Raw JSON từ Layer 1 (OHLCV + tick features)

**Processing in Python**:
- ✅ SMC structure detection (BOS/CHoCH/Sweep/OB/FVG)
- ✅ Volume Profile calculation (VAH/VAL/POC)
- ✅ Multi-Timeframe M5 building từ M1
- ✅ Normalization và feature scaling

**Output**: Feature matrix `[context_len, feature_dim]` sẵn sàng cho model

---

### Layer 3: Model Training + Inference Server (Python)
**Nhiệm vụ**: Train model và serve predictions

#### 3.1. `training/`
- **labeler.py**:
  - Load historical OHLC data
  - Compute R-based outcomes (max_up_R, max_down_R)
  - Generate labels (long/short/skip)
  - Build dataset files

- **train_outcome_model.py**:
  - Define model architecture (Transformer hoặc MLP)
  - Training loop với CrossEntropyLoss
  - Validation và metrics tracking
  - Save trained model

#### 3.2. `inference/`
- **server.py**:
  - FastAPI server
  - Endpoint POST `/infer` nhận feature context
  - Return `{prob_long, prob_short, prob_skip}`
  - Load model vào RAM, inference nhanh (<50ms)

#### 3.3. `evaluation/`
- **metrics.py**: Custom metrics (precision@top-k, expected R, confusion matrix)

**Input Training**: Labeled dataset từ labeler

**Output Inference**: Real-time predictions qua REST API

---

## 3. Workflow Diagram - Full Pipeline Live

```
┌─────────────────────────────────────────────────────────────────────┐
│                         LIVE TRADING PIPELINE                        │
└─────────────────────────────────────────────────────────────────────┘

┌──────────────┐         ┌──────────────┐         ┌──────────────┐
│              │  JSON   │              │ Feature │              │
│ NinjaTrader  │ ──────> │   Feature    │ Vector  │    Model     │
│   (Layer 1)  │  POST   │   Engine     │ ──────> │   Server     │
│              │         │  (Layer 2)   │         │  (Layer 3)   │
└──────────────┘         └──────────────┘         └──────────────┘
      ▲                         │                         │
      │                         │                         │
      │                         ▼                         ▼
      │                  ┌─────────────┐         ┌──────────────┐
      │                  │  Context    │         │ prob_long    │
      │                  │  Manager    │         │ prob_short   │
      │                  │  (deque)    │         │ prob_skip    │
      │                  └─────────────┘         └──────────────┘
      │                         │                         │
      │                         │                         │
      │                         ▼                         ▼
      │                  ┌─────────────┐         ┌──────────────┐
      │                  │   SMC +     │         │   Decision   │
      │                  │   VP + L2   │         │    Logic     │
      │                  │  Features   │         │              │
      │                  └─────────────┘         └──────────────┘
      │                                                   │
      │                                                   │
      │                                                   ▼
      │                                          ┌──────────────┐
      │                                          │  Execution   │
      └──────────────────────────────────────────│  (Manual/    │
                                                 │   Auto)      │
                                                 └──────────────┘

FLOW CHI TIẾT:
1. NinjaTrader thu raw bars → POST JSON đến http://localhost:5001/raw
2. Feature Engine:
   - Nhận bars mới
   - Update context window (deque 100 bars)
   - Tính SMC structure, Volume Profile, L2 features
   - Chuẩn hóa và build feature vector
   - POST đến Model Server: http://localhost:5002/infer
3. Model Server:
   - Inference với trained model
   - Trả về probabilities {long, short, skip}
4. Decision Logic:
   - Nếu prob_long > threshold → signal BUY
   - Nếu prob_short > threshold → signal SELL
   - Nếu prob_skip cao nhất → wait
5. Execution:
   - Manual: hiển thị alert cho trader
   - Auto: gửi lệnh về NinjaTrader (future phase)
```

---

## 4. Dataset Specification

### Dataset Structure
Mỗi training sample (record) gồm:

```python
Record {
    context: List[FeatureBar]  # 40-100 bars, mỗi bar có ~50-100 features
    label: int                  # 0=long, 1=short, 2=skip
    max_up_R: float            # Metadata cho analysis
    max_down_R: float          # Metadata cho analysis
    entry_price: float         # Reference price
    atr: float                 # ATR tại thời điểm đó
}
```

### Feature Dimensions
Mỗi **FeatureBar** có khoảng 70-100 features:

**OHLCV Features** (10-15):
- open, high, low, close, volume (normalized)
- range, body_size, wick_upper, wick_lower
- volume_ma, volume_ratio

**Tick Features** (12-15) - **From NinjaTrader + Derived**:
- tick_speed (ticks/sec) - raw from NinjaTrader
- aggr_buy_speed (contracts/sec) - raw from NinjaTrader  
- aggr_sell_speed (contracts/sec) - raw from NinjaTrader
- price_speed (points/sec) - raw from NinjaTrader
- delta, buy_volume, sell_volume - raw from NinjaTrader
- best_bid, best_ask, spread - raw from NinjaTrader
- tick_speed_ma, tick_acceleration - derived in Python
- buy_sell_ratio, delta_normalized - derived in Python

**SMC Features** (15-20) - **Computed in Python**:
- is_swing_high, is_swing_low
- bos_up, bos_down, choch_up, choch_down
- sweep_high, sweep_low
- dist_to_nearest_ob_up, dist_to_nearest_ob_down
- dist_to_nearest_fvg_up, dist_to_nearest_fvg_down
- ob_strength, fvg_size

**Volume Profile Features** (8-10) - **Computed in Python**:
- dist_to_vah, dist_to_val, dist_to_poc
- at_hvn, at_lvn
- value_area_position (0-1)

**Time Features** (3-5):
- session_flag (one-hot: Asia/Europe/US)
- time_sin, time_cos

**Multi-Timeframe Features** (10-15) - **M5 built in Python**:
- m5_trend, m5_swing_high, m5_swing_low
- m5_ob_nearby, m5_fvg_nearby

**Context Size**:
- **NO TOKEN LIMIT** (this is tabular numeric data, not LLM)
- Training: Fixed 60 bars M1 (1 hour context)
- Inference: Sliding window 60-100 bars
- Input shape: `[batch, seq_len, features]` e.g., `[32, 60, 85]`
- Total numeric values: 60 × 85 = 5,100 (tiny compared to LLM)

---

## 4.1. Tick Features Explained

### Tick Speed (tick_speed)
**Định nghĩa**: Tổng số Last tick (price updates) mà NinjaTrader nhận được trong bar  
**Công thức**: `tick_speed = total_ticks_in_bar` (KHÔNG chia cho thời gian)  
**Ý nghĩa**:
- High tick_speed (>1000 cho M1) → High trading activity, volatility
- Low tick_speed (<500 cho M1) → Low activity, consolidation

### Aggressive Buy Speed (aggr_buy_speed)
**Định nghĩa**: Buy volume của bar (giao dịch chủ động mua)  
**Công thức**: `aggr_buy_speed = buy_volume = (volume + delta) / 2`  
**Ý nghĩa**:
- High aggr_buy_speed → Strong buying pressure
- aggr_buy_speed > aggr_sell_speed → Bullish momentum
- **Lưu ý**: Không chia cho thời gian, dùng trực tiếp volume

### Aggressive Sell Speed (aggr_sell_speed)
**Định nghĩa**: Sell volume của bar (giao dịch chủ động bán)  
**Công thức**: `aggr_sell_speed = sell_volume = volume - buy_volume`  
**Ý nghĩa**:
- High aggr_sell_speed → Strong selling pressure
- aggr_sell_speed > aggr_buy_speed → Bearish momentum
- **Lưu ý**: Không chia cho thời gian, dùng trực tiếp volume

### Price Speed (price_speed)
**Định nghĩa**: Intrabar range (biên độ giá trong bar)  
**Công thức**: `price_speed = High[1] - Low[1]` (KHÔNG chia cho thời gian)  
**Ý nghĩa**:
- High price_speed → Wide range, volatility
- Low price_speed → Narrow range, consolidation
- Combine với tick_speed để detect breakouts

**Use Cases**:
- **Breakout detection**: High tick_speed + high price_speed
- **Momentum direction**: aggr_buy_speed vs aggr_sell_speed
- **Consolidation**: Low tick_speed + low price_speed
- **Absorption**: High aggr_sell_speed but price rises (bullish)

---

## 5. Retraining Strategy

### Offline Training (Initial)
- Sử dụng 6-12 tháng historical data
- Rolling window validation (walk-forward)
- Train một model baseline

### Incremental Retraining
**KHÔNG daily retrain** (tránh overfitting short-term noise)

**Khi nào retrain**:
1. **Monthly review**: Đánh giá performance metrics
2. **Regime change detection**: Nếu market structure thay đổi rõ rệt (volatility spike, correlation shift)
3. **Performance degradation**: Nếu win-rate hoặc expected R giảm >20% so với backtest

**Rolling window approach**:
- Giữ train window = 6 tháng gần nhất
- Validation = 1 tháng tiếp theo
- Test = 2 tuần tiếp (out-of-sample)
- Slide window forward 1 tháng mỗi lần retrain

### Model Versioning
- Lưu model với timestamp: `outcome_v20250315.pt`
- A/B testing: Chạy song song model cũ vs mới trước khi switch hoàn toàn
- Rollback nhanh nếu model mới underperform

---

## 6. Technology Stack

### Layer 1 (NinjaTrader)
- **Language**: C# (.NET Framework 4.8)
- **Platform**: NinjaTrader 8
- **Data Source**: Rithmic API (for L2 depth)

### Layer 2 (Feature Engine)
- **Language**: Python 3.10+
- **Core libs**: NumPy, Pandas
- **API**: FastAPI (nhận data từ Layer 1)

### Layer 3 (Model)
- **ML Framework**: PyTorch 2.0+
- **Model types**: Transformer, MLP, potential ensemble
- **Serving**: FastAPI + Uvicorn
- **Containerization**: Docker

### Deployment
- **Local Dev**: Python venv, direct run
- **Production**: Docker containers, orchestrated với Docker Compose
- **Monitoring**: Custom logging + metrics export (future: Prometheus)

---

## 7. Next Steps & Extensibility

### Phase 1 Extensions
- Add more SMC features (liquidity sweeps multi-timeframe)
- Incorporate orderflow footprint patterns
- Multi-timeframe context (M1 + M5 aggregated)

### Phase 2 Enhancements
- Ensemble models (Transformer + LightGBM)
- Reinforcement Learning for position sizing
- Auto-calibration của stop_R và target_R based on volatility

### Phase 3 Production
- Real-time monitoring dashboard
- Auto-execution integration
- Cloud deployment (AWS/GCP) với low-latency requirements

---

**Document Version**: 1.0
**Last Updated**: 2025-01-26
**Author**: Senior ML Engineer + Quant Developer Team
