# Setup Guide - ML Outcome Model for Trading

## 🚀 Quick Start (5 Minutes)

### Step 1: Navigate to Project
```bash
cd c:\Users\Administrator\Desktop\modeloutcome
```

### Step 2: Create Virtual Environment
```bash
python -m venv venv
```

### Step 3: Activate Virtual Environment
**Windows:**
```bash
venv\Scripts\activate
```

**Linux/Mac:**
```bash
source venv/bin/activate
```

### Step 4: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 5: Verify Installation
```bash
python -c "import torch; import pandas; import fastapi; print('✅ All dependencies installed successfully!')"
```

---

## 📂 Project Structure Created

Your project now has the following structure:

```
modeloutcome/
├── ARCHITECTURE.md              ⭐ Read this first - Full system design
├── PROJECT_MASTER_PLAN.md       ⭐ Phase-by-phase execution plan
├── README.md                    ⭐ Project overview
├── SETUP_GUIDE.md              ⭐ This file
├── requirements.txt
├── .gitignore
│
├── config/
│   ├── model_config.yaml       ⭐ Main configuration
│   └── data_paths.yaml
│
├── docs/
│   └── notes.md
│
├── data/
│   ├── raw/                    (Place your CSV data here)
│   ├── processed/
│   └── datasets/
│
├── notebooks/
│   └── exploration.ipynb
│
├── src/
│   ├── layer1_ninjatrader/
│   │   ├── ExportRawData.cs   ⭐ NinjaTrader strategy
│   │   └── README.md
│   │
│   ├── layer2_feature_engine/
│   │   ├── api_server.py      ⭐ Feature Engine API (port 5001)
│   │   ├── core/
│   │   │   ├── schema.py
│   │   │   ├── normalizer.py
│   │   │   └── context_manager.py
│   │   ├── smc/               (Smart Money Concepts)
│   │   ├── volume_profile/
│   │   ├── orderflow_l2/
│   │   └── utils/
│   │
│   └── layer3_model/
│       ├── training/
│       │   ├── labeler.py     ⭐ Generate labeled dataset
│       │   └── train_outcome_model.py  ⭐ Train model
│       ├── inference/
│       │   └── server.py      ⭐ Inference API (port 5002)
│       └── evaluation/
│           └── metrics.py
│
├── deployment/
│   ├── Dockerfile
│   ├── run_model_server.bat   ⭐ Windows: Start inference server
│   └── run_model_server.sh    ⭐ Linux/Mac: Start inference server
│
├── models/                     (Trained models saved here)
└── logs/                       (Application logs)
```

---

## 🎯 Next Steps

### Option A: Full Pipeline (Recommended for First-Time Setup)

Follow **PROJECT_MASTER_PLAN.md** from Phase 0 to Phase 6:

1. **Phase 0**: ✅ Already completed (environment setup)
2. **Phase 1**: Deploy NinjaTrader adapter
3. **Phase 2**: Test Feature Engine
4. **Phase 3**: Generate labeled dataset
5. **Phase 4**: Train model
6. **Phase 5**: Deploy inference server
7. **Phase 6**: Live integration test

### Option B: Quick Test (Without NinjaTrader)

If you just want to test the Python components:

```bash
# 1. Prepare sample data (you'll need to create this)
# Place a CSV file with columns: timestamp, open, high, low, close, volume
# Example: data/raw/NQ_M1_sample.csv

# 2. Generate labeled dataset
python src/layer3_model/training/labeler.py --input data/raw/NQ_M1_sample.csv

# 3. Train model
python src/layer3_model/training/train_outcome_model.py

# 4. Start inference server
deployment\run_model_server.bat   # Windows
# or
./deployment/run_model_server.sh  # Linux/Mac
```

---

## 🔍 Testing Endpoints

### Test Model Server (Layer 3)

```bash
# Health check
curl http://localhost:5002/health

# Inference (example)
curl -X POST http://localhost:5002/infer \
  -H "Content-Type: application/json" \
  -d '{"features": [[0.1, 0.2, 0.3, ...], ...]}'
```

### Test Feature Engine (Layer 2)

```bash
# Health check
curl http://localhost:5001/health

# Send raw bars (simulating NinjaTrader)
curl -X POST http://localhost:5001/raw \
  -H "Content-Type: application/json" \
  -d @sample_request.json
```

---

## 📚 Documentation References

1. **ARCHITECTURE.md** - Hiểu hệ thống (Tiếng Việt)
   - Giải thích Outcome Model
   - Kiến trúc 3 Layer
   - Dataset specification
   - Workflow diagram

2. **PROJECT_MASTER_PLAN.md** - Kế hoạch thực thi
   - Phase-by-phase checklist
   - Validation criteria
   - Deliverables per phase

3. **src/layer1_ninjatrader/README.md** - NinjaTrader setup
   - Installation guide
   - Configuration parameters
   - Troubleshooting

4. **README.md** - Quick reference
   - Project overview
   - Quick start commands
   - API documentation

---

## 🛠️ Troubleshooting

### Import Errors
```bash
# Make sure PYTHONPATH includes project root
export PYTHONPATH="${PYTHONPATH}:c:\Users\Administrator\Desktop\modeloutcome"
```

### Port Already in Use
```bash
# Check if ports 5001, 5002 are free
netstat -ano | findstr :5001
netstat -ano | findstr :5002

# Kill process if needed (Windows)
taskkill /PID <process_id> /F
```

### Model Not Found
```bash
# Train model first
python src/layer3_model/training/train_outcome_model.py
```

---

## 📊 Expected File Sizes (After Running)

- **Dataset** (outcome_train.pt): 100-500 MB (depends on historical data)
- **Trained Model** (outcome_transformer_v1.pt): 5-20 MB
- **Logs**: Growing over time, rotate regularly

---

## 🎓 Learning Resources

- **Smart Money Concepts**: https://www.tradingview.com/scripts/smartmoneyconcepts/
- **PyTorch Transformers**: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
- **FastAPI**: https://fastapi.tiangolo.com/tutorial/
- **NinjaTrader 8 API**: https://ninjatrader.com/support/helpGuides/nt8/

---

## ✅ Checklist Before Going Live

- [ ] Trained model exists in `models/` directory
- [ ] Inference server starts without errors
- [ ] Feature Engine API responds to health check
- [ ] NinjaTrader strategy compiles successfully
- [ ] Tested end-to-end with historical data
- [ ] Reviewed `config/model_config.yaml` settings
- [ ] Set up logging and monitoring
- [ ] Defined risk management rules (max position size, daily loss limit)

---

## 🤝 Support

For issues or questions:
- Check **docs/notes.md** for known issues and solutions
- Review **PROJECT_MASTER_PLAN.md** Phase checklists
- Refer to module-specific README files

---

**Version**: 1.0
**Last Updated**: 2025-01-26
**Status**: ✅ Complete and Ready for Execution
