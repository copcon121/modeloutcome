# ML Outcome Model for Trading

A complete 3-layer ML trading system with outcome-based predictions (long/short/skip) using Smart Money Concepts, Volume Profile, and Level 2 orderflow features.

## 🎯 Project Overview

This system predicts trading outcomes based on **R-multiple** (risk-reward ratio) rather than simple price direction. It uses a context window of 60-100 bars with rich features including:

- **OHLCV** + Delta/Orderflow
- **Tick Features** (tick_speed, aggr_buy_speed, aggr_sell_speed, price_speed) - from NinjaTrader
- **SMC** (Smart Money Concepts): Swing, BOS, CHoCH, Order Blocks, Fair Value Gaps - **built in Python**
- **Volume Profile**: VAH, VAL, POC, HVN, LVN - **built in Python**
- **Multi-Timeframe**: M5 built from M1 in Python
- **Market Depth**: Bid/Ask, spread

**Model Output**: Probabilities for 3 actions
- `prob_long`: Enter LONG position
- `prob_short`: Enter SHORT position
- `prob_skip`: Skip (don't trade)

## 📁 Project Structure

```
modeloutcome/
├── ARCHITECTURE.md           # Full system architecture documentation
├── PROJECT_MASTER_PLAN.md    # Phase-by-phase execution plan
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── .gitignore
│
├── config/                   # Configuration files
│   ├── model_config.yaml
│   ├── data_paths.yaml
│   └── feature_config.yaml
│
├── data/
│   ├── raw/                  # Raw OHLCV data from NinjaTrader
│   ├── processed/            # Processed features
│   └── datasets/             # Labeled training datasets
│
├── src/
│   ├── layer1_ninjatrader/   # NinjaTrader C# adapter
│   │   ├── ExportRawData.cs
│   │   └── README.md
│   │
│   ├── layer2_feature_engine/  # Python feature engineering
│   │   ├── core/             # Core data structures
│   │   ├── smc/              # Smart Money Concepts features
│   │   ├── volume_profile/   # Volume Profile features
│   │   ├── orderflow_l2/     # Level 2 depth features
│   │   └── utils/            # Utilities
│   │
│   └── layer3_model/         # ML model training & inference
│       ├── training/         # Labeler and training scripts
│       ├── inference/        # FastAPI inference server
│       └── evaluation/       # Metrics and evaluation
│
├── deployment/               # Docker and deployment scripts
│   ├── Dockerfile
│   └── run_model_server.sh
│
├── models/                   # Saved model weights
├── logs/                     # Application logs
└── notebooks/                # Jupyter notebooks for exploration
```

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Clone or navigate to project
cd modeloutcome

# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Verify Installation

```bash
python -c "import torch; import pandas; import fastapi; print('All imports successful!')"
```

### 3. Run Phase-by-Phase

Follow the detailed instructions in [PROJECT_MASTER_PLAN.md](PROJECT_MASTER_PLAN.md):

- **Phase 0**: Environment setup ✅
- **Phase 1**: Deploy NinjaTrader adapter
- **Phase 2**: Build Feature Engine
- **Phase 3**: Generate labeled dataset
- **Phase 4**: Train model
- **Phase 5**: Deploy inference server
- **Phase 6**: Live integration test

## 📊 System Architecture

### 🎯 Pro Mode (Recommended)

NinjaTrader exports **Raw + Tick Features**:
- OHLCV: open, high, low, close, volume
- Delta: delta, buy_volume, sell_volume
- Bid/Ask: best_bid, best_ask, spread
- Tick Features: tick_speed, aggr_buy_speed, aggr_sell_speed, price_speed

Python handles **ALL advanced processing**:
- SMC structure (BOS/CHOCH/Sweep/OB/FVG)
- Volume Profile (VAH/VAL/POC)
- Multi-Timeframe M5 from M1
- Normalization and scaling

### Layer 1: NinjaTrader Adapter (C#) - SMC_Exporter_Pro_v3
- Indicator: `SMC_Exporter_Pro_v3` trên NinjaTrader 8.0.28
- Export **raw OHLCV + tick features** per M1 bar vào file `.jsonl`
- Delta chuẩn từ indicator `Volumdelta` (DeltasClose[1])
- Tick features computed realtime:
  - `tick_speed`: Tổng số tick trong bar
  - `aggr_buy_speed`: Buy volume của bar
  - `aggr_sell_speed`: Sell volume của bar
  - `price_speed`: Intrabar range (H - L)
- Visual panel hiển thị 4 tick features để validation
- File xuất: `Documents/NinjaTrader 8/SMC_Exports/<FileName>.jsonl`
- See [src/layer1_ninjatrader/README.md](src/layer1_ninjatrader/README.md)

### Layer 2: Feature Engine (Python)
- Receives raw + tick features from Layer 1
- **Builds ALL derived features**:
  - SMC structure detection (BOS/CHOCH/Sweep/OB/FVG) from raw M1 bars
  - Volume Profile calculation (VAH/VAL/POC) from raw M1 bars
  - Multi-Timeframe M5 aggregation from M1 bars
  - Tick feature derivatives (tick_speed_ma, tick_acceleration, etc.)
  - Normalization and feature scaling
- Runs on port **5001**

### Layer 3: Model Server (Python)
- FastAPI inference server
- Loads trained PyTorch model
- Returns predictions via REST API
- Runs on port **5002**

**Full Pipeline Flow:**
```
NinjaTrader (Raw + Tick Features) → Feature Engine (5001) → Model Server (5002) → Predictions
```

---

## 📋 JSON Schema (Per Bar) - SMC_Exporter_Pro_v3 Format

NinjaTrader exports mỗi bar M1 theo schema này:

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

    // Delta chuẩn từ Volumdelta.DeltasClose[1]
    "delta": -77,

    // Suy ra từ volume + delta
    "buy_volume": 386.5,   // (volume + delta) / 2
    "sell_volume": 463.5,  // volume - buy_volume

    // Stub: dùng close của bar
    "best_bid": 4048.9,
    "best_ask": 4048.9
  },

  "tick_features": {
    // Tổng số Last tick trong bar
    "tick_speed": 1404,

    // Dùng trực tiếp buy/sell volume
    "aggr_buy_speed": 386.5,
    "aggr_sell_speed": 463.5,

    // Intrabar range
    "price_speed": 5.9  // High - Low
  }
}
```

### Tick Features Explained

- **delta**: Delta bar từ Volumdelta indicator (gần footprint nhất, sai khác vài lot chấp nhận được)
  - ❌ KHÔNG tính bằng uptick/downtick thủ công
  - ✅ Lấy trực tiếp từ `Volumdelta.DeltasClose[1]`

- **buy_volume / sell_volume**: Phân rã volume theo delta
  - `buy_volume = (volume + delta) / 2`
  - `sell_volume = volume - buy_volume`
  - Dùng cho feature ML (không nhất thiết trùng 100% footprint)

- **tick_speed**: Tổng số tick (price updates) trong bar
  - KHÔNG chia cho thời gian
  - High (>1000 cho M1) = high volatility/activity
  - Low (<500 cho M1) = consolidation

- **aggr_buy_speed**: Buy volume của bar (giao dịch chủ động mua)
  - Dùng trực tiếp buy_volume (KHÔNG chia cho thời gian)
  - So sánh với aggr_sell_speed cho momentum direction

- **aggr_sell_speed**: Sell volume của bar (giao dịch chủ động bán)
  - Dùng trực tiếp sell_volume (KHÔNG chia cho thời gian)
  - aggr_sell_speed > aggr_buy_speed = bearish momentum

- **price_speed**: Intrabar range (biên độ giá)
  - `price_speed = High - Low` (KHÔNG chia cho thời gian)
  - High value = wide range, volatility
  - Low value = narrow range, consolidation

## 🔧 Configuration

### model_config.yaml
```yaml
labeling:
  context_len: 60
  future_window: 50
  stop_R: 1.0
  target_R: 2.0

model:
  architecture: transformer  # or 'mlp'
  hidden_dim: 128
  num_layers: 4
  dropout: 0.1

training:
  batch_size: 32
  learning_rate: 0.0001
  epochs: 20
```

## 📈 Training a Model

```bash
# 1. Prepare historical data
# Export from NinjaTrader or place CSV in data/raw/

# 2. Generate labeled dataset
python src/layer3_model/training/labeler.py --config config/model_config.yaml

# 3. Train model
python src/layer3_model/training/train_outcome_model.py

# Model saved to: models/outcome_transformer_v1.pt
```

## 🌐 Running the Inference Server

```bash
# Start model server
uvicorn src.layer3_model.inference.server:app --port 5002

# Or use deployment script
./deployment/run_model_server.sh
```

**Test endpoint:**
```bash
curl -X POST http://localhost:5002/infer \
  -H "Content-Type: application/json" \
  -d '{"features": [[0.1, 0.2, ...], ...]}'
```

**Response:**
```json
{
  "prob_long": 0.35,
  "prob_short": 0.20,
  "prob_skip": 0.45
}
```

## 🔗 Live Trading Integration

1. **Start Model Server** (Layer 3): `./deployment/run_model_server.sh`
2. **Start Feature Engine** (Layer 2): `uvicorn src.layer2_feature_engine.api_server:app --port 5001`
3. **Attach NinjaTrader Strategy** (Layer 1): Configure endpoint to `http://localhost:5001/raw`

Monitor logs for real-time predictions!

## 📚 Documentation

- **[ARCHITECTURE.md](ARCHITECTURE.md)**: Detailed system design (Vietnamese)
- **[PROJECT_MASTER_PLAN.md](PROJECT_MASTER_PLAN.md)**: Phase-by-phase execution plan
- **[src/layer1_ninjatrader/README.md](src/layer1_ninjatrader/README.md)**: NinjaTrader setup guide

## 🧪 Testing

```bash
# Run feature extraction test
python tests/test_feature_pipeline.py

# Run inference test
pytest tests/test_inference.py
```

## 🐳 Docker Deployment

```bash
# Build image
docker build -t outcome-model-server -f deployment/Dockerfile .

# Run container
docker run -p 5002:5002 outcome-model-server
```

## 📊 Performance Targets

- **Inference latency**: <50ms per prediction
- **Feature extraction**: <100ms per bar update
- **End-to-end latency**: <200ms (NinjaTrader → Decision)
- **Model accuracy**: >50% (baseline: 33% for 3-class)
- **Dataset**: NO token limit (tabular numeric model, not LLM)

## 🛠️ Development

### Code Style
- Python: PEP8, type hints, English comments
- C#: Standard .NET conventions

### Adding New Features
1. Add feature extraction in `src/layer2_feature_engine/`
2. Update `ContextManager._extract_bar_features()`
3. Retrain normalizer with new features
4. Retrain model

### Retraining
- **Frequency**: Monthly or on regime change (not daily)
- **Method**: Rolling window (6 months train, 1 month val)
- **Versioning**: Save as `outcome_vYYYYMMDD.pt`

## 🚨 Troubleshooting

### "HTTP POST failed" from NinjaTrader
- Ensure Feature Engine server is running on port 5001
- Check firewall settings

### "Model not found" error
- Train a model first: `python src/layer3_model/training/train_outcome_model.py`
- Verify model exists in `models/` directory

### Import errors
- Activate virtual environment: `venv\Scripts\activate`
- Reinstall dependencies: `pip install -r requirements.txt`

## 📝 License

This project is for educational and research purposes.

## 👥 Contributors

Senior ML Engineer + Quant Developer + DevOps Team

---

**Version**: 1.0
**Last Updated**: 2025-01-26
**Status**: Ready for Phase 1 Execution
