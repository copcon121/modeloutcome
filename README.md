# ML Outcome Model for Trading

A complete 3-layer ML trading system with outcome-based predictions (long/short/skip) using Smart Money Concepts, Volume Profile, and Level 2 orderflow features.

## 🎯 Project Overview

This system predicts trading outcomes based on **R-multiple** (risk-reward ratio) rather than simple price direction. It uses a context window of 60-100 bars with rich features including:

- **OHLCV** + Delta/Orderflow
- **SMC** (Smart Money Concepts): Swing, BOS, CHoCH, Order Blocks, Fair Value Gaps
- **Volume Profile**: VAH, VAL, POC, HVN, LVN
- **Level 2 Market Depth**: Bid/Ask pressure, imbalance

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

### Layer 1: NinjaTrader Adapter (C#)
- Exports raw OHLCV bars via HTTP POST
- See [src/layer1_ninjatrader/README.md](src/layer1_ninjatrader/README.md)

### Layer 2: Feature Engine (Python)
- Receives bars from Layer 1
- Computes SMC, Volume Profile, L2 features
- Runs on port **5001**

### Layer 3: Model Server (Python)
- FastAPI inference server
- Loads trained PyTorch model
- Returns predictions via REST API
- Runs on port **5002**

**Full Pipeline Flow:**
```
NinjaTrader → Feature Engine (5001) → Model Server (5002) → Predictions
```

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
