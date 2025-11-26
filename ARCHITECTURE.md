# ARCHITECTURE - ML Outcome Model for Trading

## 1. Tổng quan Outcome Model

### Mục tiêu
Dự án này xây dựng một hệ thống ML trading hoàn chỉnh với mô hình **outcome-based**, tức là dự đoán kết quả trading dựa trên R-multiple (risk-reward ratio) thay vì dự đoán giá đơn thuần.

### Định nghĩa Outcome
- **Input Context**: 40-100 bar M1 (1 phút) với features phong phú:
  - OHLCV cơ bản
  - Delta và orderflow
  - SMC structure (Swing, BOS, CHoCH, Sweep, OB, FVG)
  - Volume Profile (VAH, VAL, POC, HVN, LVN)
  - Level 2 Market Depth (bid/ask pressure, imbalance)

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

### Layer 1: NinjaTrader Adapter (C#)
**Nhiệm vụ**: Thu thập và xuất raw market data từ NinjaTrader 8

**Input**:
- Live market data stream từ NinjaTrader (OHLCV, Volume, Tick data)
- Rithmic API data (Delta, Level 2 depth)

**Output**:
- JSON payload qua HTTP POST đến Feature Engine
- Format: `{"symbol": "NQ", "timeframe": "1m", "bars": [...]}`

**Đặc điểm**:
- Non-blocking (fire-and-forget) để không ảnh hưởng chart rendering
- Có thể export historical data (backtest) hoặc live stream
- Extensible để thêm orderflow metrics từ Rithmic

---

### Layer 2: Feature Engine (Python)
**Nhiệm vụ**: Chuyển đổi raw bars thành feature vectors phong phú

**Kiến trúc submodule**:

#### 2.1. `core/`
- **schema.py**: Định nghĩa data structures (RawBar, FeatureBar, Record)
- **normalizer.py**: Chuẩn hóa features (Z-score, Min-Max)
- **context_manager.py**: Quản lý sliding window context, orchestrate feature building

#### 2.2. `smc/` (Smart Money Concepts)
- **swing.py**: Phát hiện swing high/low
- **structure.py**: Detect BOS (Break of Structure), CHoCH (Change of Character), sweep
- **zones.py**: Xác định và tracking Order Block (OB), Fair Value Gap (FVG)

#### 2.3. `volume_profile/`
- **vp_builder.py**: Xây dựng Volume Profile theo session/window
- Tính VAH (Value Area High), VAL (Value Area Low), POC (Point of Control)
- Identify HVN (High Volume Node), LVN (Low Volume Node)

#### 2.4. `orderflow_l2/`
- **l2_features.py**: Xử lý Level 2 market depth từ Rithmic
- Tính bid/ask pressure, depth imbalance, aggression indicators

#### 2.5. `utils/`
- **time_features.py**: Session detection (Asia/Europe/US), time-of-day encoding
- **logging_utils.py**: Centralized logging
- **config_loader.py**: Load YAML configs

**Input**: Raw JSON từ Layer 1

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
Mỗi **FeatureBar** có khoảng 60-80 features:

**OHLCV Features** (10-15):
- open, high, low, close, volume (normalized)
- range, body_size, wick_upper, wick_lower
- volume_delta, buy_volume, sell_volume

**SMC Features** (15-20):
- is_swing_high, is_swing_low
- bos_up, bos_down, choch_up, choch_down
- sweep_high, sweep_low
- dist_to_nearest_ob_up, dist_to_nearest_ob_down
- dist_to_nearest_fvg_up, dist_to_nearest_fvg_down
- ob_strength, fvg_size

**Volume Profile Features** (8-10):
- dist_to_vah, dist_to_val, dist_to_poc
- at_hvn, at_lvn
- value_area_position (0-1)

**Level 2 Features** (5-8):
- l2_bid_pressure, l2_ask_pressure
- l2_depth_imbalance
- l2_aggression_buy, l2_aggression_sell

**Time Features** (3-5):
- session_flag (one-hot: Asia/Europe/US)
- time_sin, time_cos

**Context Size**:
- Training: Fixed 60 bars (1 hour M1 data)
- Inference: Sliding window 60-100 bars

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
