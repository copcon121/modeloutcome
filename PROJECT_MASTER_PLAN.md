# PROJECT MASTER PLAN - ML Outcome Model for Trading

## Overview
This master plan provides a phase-by-phase execution roadmap for building the complete ML trading system. Each phase must be completed and validated before moving to the next.

**Execution Principle**:
- ✅ Complete one phase fully before starting the next
- ✅ Validate outputs at each phase
- ✅ Document any deviations or blockers
- ✅ Maintain clean code and documentation throughout

---

## Phase 0: Environment Setup & Project Initialization

### Objectives
- Set up development environment
- Create project structure
- Install dependencies
- Verify basic toolchain

### Checklist

#### 0.1 Create Directory Structure
```bash
- [ ] Create root directory: `modeloutcome/`
- [ ] Create subdirectories:
      - config/
      - docs/
      - data/raw/
      - data/processed/
      - data/datasets/
      - notebooks/
      - src/layer1_ninjatrader/
      - src/layer2_feature_engine/core/
      - src/layer2_feature_engine/smc/
      - src/layer2_feature_engine/volume_profile/
      - src/layer2_feature_engine/orderflow_l2/
      - src/layer2_feature_engine/utils/
      - src/layer3_model/training/
      - src/layer3_model/inference/
      - src/layer3_model/evaluation/
      - src/deployment/
      - models/ (for saved model weights)
      - logs/
```

#### 0.2 Create Core Documentation Files
```bash
- [ ] ARCHITECTURE.md (completed)
- [ ] PROJECT_MASTER_PLAN.md (this file)
- [ ] README.md (project overview)
- [ ] .gitignore
- [ ] docs/notes.md (for development notes)
```

#### 0.3 Python Environment Setup
```bash
- [ ] Install Python 3.10+ (verify: python --version)
- [ ] Create virtual environment: python -m venv venv
- [ ] Activate venv:
      Windows: venv\Scripts\activate
      Linux/Mac: source venv/bin/activate
- [ ] Create requirements.txt with dependencies
- [ ] Install dependencies: pip install -r requirements.txt
- [ ] Verify imports: python -c "import torch; import pandas; import fastapi"
```

#### 0.4 Configuration Files
```bash
- [ ] Create config/model_config.yaml
- [ ] Create config/data_paths.yaml
- [ ] Create config/feature_config.yaml (for feature engineering params)
```

#### 0.5 NinjaTrader Environment (if applicable)
```bash
- [ ] Verify NinjaTrader 8 installation
- [ ] Locate NinjaTrader custom strategy directory
- [ ] Prepare C# development environment (Visual Studio or Rider)
```

### Validation Criteria
- ✅ All directories exist
- ✅ Python environment activated and dependencies installed
- ✅ Configuration files created
- ✅ Git repository initialized (optional)

---

## Phase 1: Layer 1 - NinjaTrader Adapter (SMC_Exporter_Pro_v3)

### Module: Tầng 2 – EXPORT-SMC-JSON — NinjaTrader Exporter

**Tên**: `SMC_Exporter_Pro_v3`  
**Trạng thái**: ĐÃ FIX STABLE  
**Platform**: NinjaTrader 8.0.28  
**Khung thời gian**: M1 (có thể mở rộng M5, M15 sau)

###Objectives
- Implement indicator `SMC_Exporter_Pro_v3` để export raw OHLCV + tick features
- Đọc delta chuẩn từ indicator `Volumdelta` (field `DeltasClose[1]`)
- Export file `.jsonl` (1 bar = 1 dòng JSON)
- Visual panel 4 dòng để verify realtime

### Checklist

#### 1.1 Develop SMC_Exporter_Pro_v3 (Pro Mode) - ĐÃ FIX STABLE
```bash
- [x] Create SMC_Exporter_Pro_v3.cs trong src/layer1_ninjatrader/
- [x] OnBarUpdate() export mỗi bar M1 theo format:
      **OHLCV Data** (shorthand):
      - o, h, l, c, volume
      **Delta Features** (từ Volumdelta indicator):
      - delta = Volumdelta.DeltasClose[1] (CHÚ Ý: không tính uptick/downtick thủ công)
      - buy_volume = (volume + delta) / 2
      - sell_volume = volume - buy_volume
      - best_bid, best_ask = close của bar (stub)
- [x] Realtime tick features trong OnMarketData():
      - tick_speed = tổng số Last tick trong bar (KHÔNG chia cho thời gian)
      - aggr_buy_speed = buy_volume (dùng trực tiếp)
      - aggr_sell_speed = sell_volume (dùng trực tiếp)
      - price_speed = High[1] - Low[1] (intrabar range)
- [x] JSON schema per bar:
      {
        "symbol": "GC 02-26",
        "timeframe": "M1",
        "timestamp": "2025-11-17T20:01:00.0000000",
        "bar_index": 1260,
        "bar": {"o": ..., "h": ..., "l": ..., "c": ..., "volume": ..., 
                "delta": ..., "buy_volume": ..., "sell_volume": ...,
                "best_bid": ..., "best_ask": ...},
        "tick_features": {"tick_speed": ..., "aggr_buy_speed": ..., 
                         "aggr_sell_speed": ..., "price_speed": ...}
      }
- [x] Export vào file: Documents/NinjaTrader 8/SMC_Exports/<FileName>.jsonl
      - Mỗi bar = 1 dòng JSON
      - Append mode cho backtest/live stream
```

**Lưu ý kỹ thuật quan trọng**:
- ❌ **KHÔNG** dùng uptick/downtick thủ công để tính delta
- ✅ Delta lấy từ `Volumdelta.DeltasClose[1]` (gần footprint nhất)
- ✅ File xuất chuẩn `.jsonl`, 1 bar/line
- ✅ Đã có panel visual 4 line (tick_speed, buy_volume, sell_volume, price_speed)
- ✅ Khi backtest/live, chỉ cần bật 1 indicator này trên chart là có full orderflow features

#### 1.2 Visual Tick Features Panel (Verification)
```bash
- [ ] Implement OnRender() to display tick features panel:
      - Panel trên chart hiển thị realtime:
        * Tick Speed: X.X ticks/sec
        * Aggr Buy Speed: X.X contracts/sec
        * Aggr Sell Speed: X.X contracts/sec
        * Price Speed: X.XXX points/sec
      - Include progress bars for visualization
      - Color coding: Green (bullish), Red (bearish), Yellow (neutral)
- [ ] Purpose: Visual validation that calculations work correctly
```

#### 1.3 Create Layer 1 Documentation
```bash
- [ ] Write src/layer1_ninjatrader/README.md with:
      - Installation instructions
      - How to attach strategy to chart
      - Configuration parameters (endpoint URL, symbol)
      - **Tick features explanation** (definitions and formulas)
      - Troubleshooting common issues
      - Note: All SMC/VP/MTF processing done in Python, NOT NinjaTrader
```

#### 1.4 Testing
```bash
- [ ] Compile ExportRawDataPro.cs in NinjaTrader
- [ ] Attach to a chart (historical data)
- [ ] Set up a simple Python HTTP server to receive POSTs:
      python -m http.server 5001 (or simple FastAPI app)
- [ ] Verify JSON payloads with new tick_features fields:
      - Check tick_speed values are reasonable (5-50 range for M1)
      - Verify aggr_buy_speed + aggr_sell_speed ≈ volume/60
      - Confirm best_bid < best_ask, spread > 0
- [ ] Visual panel displays tick features correctly
- [ ] Test with live data (if market open)
- [ ] Log any errors and fix
```

### Deliverables
- ✅ ExportRawDataPro.cs (fully functional with tick features)
- ✅ Visual panel showing tick metrics
- ✅ README.md (clear instructions)
- ✅ Test results documented

### Validation Criteria
- ✅ Strategy compiles without errors
- ✅ HTTP POST successfully sends data with tick_features object
- ✅ JSON format matches schema exactly
- ✅ Visual panel displays tick features correctly
- ✅ Tick feature values are reasonable (no NaN, no extreme outliers)

---

## Phase 2: Layer 2 - Feature Engine Core

### Objectives
- Implement core data structures and pipeline
- Build foundational feature engineering modules
- Create SMC, Volume Profile, and L2 feature extractors

### Checklist

#### 2.1 Core Module (`core/`)
```bash
- [ ] Implement schema.py:
      - RawBar dataclass: ts, OHLCV + tick features (delta, buy_volume, sell_volume, tick_speed, etc.)
      - FeatureBar dataclass (contains all computed features as dict)
      - Record dataclass (context + label + metadata)
- [ ] Implement normalizer.py:
      - Normalizer class with fit() and transform()
      - Support for Z-score and Min-Max normalization
      - Save/load normalization stats to JSON
- [ ] Implement context_manager.py:
      - ContextManager class
      - Maintain deque of RawBar (maxlen=100)
      - add_bar(bar) method
      - build_features() method that orchestrates all feature modules
      - Return List[FeatureBar]
- [ ] Implement mtf_builder.py: 🆕
      - build_m5_from_m1(m1_bars) → aggregated M5 bars
      - Aggregate 5x M1 bars into 1x M5 bar
      - Preserve tick features (sum volumes, average speeds)
```

#### 2.2 SMC Module (`smc/`) - **Python xử lý 100%**
```bash
- [ ] Implement swing.py:
      - detect_swings(bars, lookback=2) → List of swing points (from raw M1 bars)
      - Return indices and types (high/low)
- [ ] Implement structure.py:
      - compute_structure_flags(bars, swings) → dict with BOS/CHoCH flags
      - extract_bar_structure_features(i, struct_flags) → dict
      - Features: bos_up, bos_down, choch_up, choch_down, sweep_high, sweep_low
      - **NO NinjaTrader SMC detection** - all done here in Python
- [ ] Implement zones.py:
      - detect_fvg(bars) → Fair Value Gaps (computed from raw M1 bars)
      - detect_ob(bars) → Order Blocks (computed from raw M1 bars)
      - extract_zone_features(i, bars) → dict
      - Features: dist_to_fvg_up, dist_to_fvg_down, dist_to_ob_up, dist_to_ob_down
```

#### 2.3 Volume Profile Module (`volume_profile/`) - **Python xử lý 100%**
```bash
- [ ] Implement vp_builder.py:
      - build_volume_profile(bars, price_bins=50) → VolumeProfile object (from raw M1 bars)
      - Compute VAH, VAL, POC
      - Identify HVN and LVN zones
      - extract_bar_vp_features(i, vp_state, bar) → dict
      - Features: dist_to_vah, dist_to_val, dist_to_poc, at_hvn, at_lvn
```

#### 2.4 Tick Features Module (`tick_features/`) 🆕
```bash
- [ ] Implement tick_analyzer.py:
      - Receive tick features from NinjaTrader (already in RawBar)
      - Derive additional features:
            * tick_speed_ma (moving average of tick_speed)
            * tick_acceleration (change in tick_speed)
            * buy_sell_ratio = aggr_buy_speed / aggr_sell_speed
            * delta_normalized = delta / volume
      - extract_bar_tick_features(i, bars) → dict
      - Features: tick_speed_ma, tick_accel, buy_sell_ratio, etc.
```

#### 2.5 Utils Module (`utils/`)
```bash
- [ ] Implement time_features.py:
      - extract_time_features(ts) → dict
      - Session detection (Asia: 18-02, Europe: 02-08, US: 08-17 EST)
      - Sinusoidal time encoding: time_sin, time_cos
- [ ] Implement logging_utils.py:
      - setup_logger(name, log_file, level=INFO)
- [ ] Implement config_loader.py:
      - load_config(yaml_path) → dict
      - Uses PyYAML
```

#### 2.6 Integration Testing
```bash
- [ ] Create test script: tests/test_feature_pipeline.py
- [ ] Load sample raw bars from JSON (with tick_features)
- [ ] Run through ContextManager.build_features()
- [ ] Verify output shape: [num_bars, ~70-100 features]
- [ ] Check for NaN values (should be minimal)
- [ ] Verify tick features are included and normalized
- [ ] Print sample features for manual inspection
```

### Deliverables
- ✅ All Layer 2 modules implemented (including mtf_builder and tick_analyzer)
- ✅ Integration test passing
- ✅ Sample feature output saved to `data/processed/sample_features.json`

### Validation Criteria
- ✅ No import errors
- ✅ Feature extraction runs without crashes
- ✅ Tick features correctly parsed and derived features computed
- ✅ SMC structure detected from raw M1 bars in Python
- ✅ Volume Profile built from raw M1 bars in Python
- ✅ Feature values are reasonable (no extreme outliers unless expected)
- ✅ Code is documented with comments

---

## Phase 3: Layer 3 - Outcome Labeler

### Objectives
- Implement R-based outcome calculation
- Generate labeled dataset from historical data
- Validate label distribution (long/short/skip balance)

### Checklist

#### 3.1 Implement Labeler (`training/labeler.py`)
```bash
- [ ] Function: load_historical_ohlc(symbol, start_date, end_date)
      - Load from CSV or database
      - Return DataFrame with OHLC
- [ ] Function: compute_atr(bars, period=14) → ATR values
- [ ] Function: compute_outcome_for_bar(index, prices, atr, config):
      - Extract future window (e.g., next 50 bars)
      - Calculate max_up_R and max_down_R
      - Apply labeling logic:
            if max_up_R >= target_R before max_down_R <= -stop_R: label = "long"
            elif max_down_R <= -target_R before max_up_R >= stop_R: label = "short"
            else: label = "skip"
      - Return (label, max_up_R, max_down_R)
- [ ] Function: build_dataset(raw_bars, config):
      - Loop through candidate bars (skip first context_len bars)
      - For each bar:
            * Extract context (previous 60 bars)
            * Build features using ContextManager
            * Compute outcome label
            * Create Record object
      - Save to data/datasets/outcome_train.pt (PyTorch) or .pkl (pickle)
```

#### 3.2 Configuration
```bash
- [ ] Add to config/model_config.yaml:
      labeling:
        context_len: 60
        future_window: 50
        stop_R: 1.0
        target_R: 2.0
        min_bars_after: 50  # Don't label bars too close to end
```

#### 3.3 Generate Dataset
```bash
- [ ] Prepare historical data:
      - Download or export 6 months of M1 OHLC for target symbol (e.g., NQ)
      - Save to data/raw/NQ_M1_6months.csv
- [ ] Run labeler script:
      python src/layer3_model/training/labeler.py --config config/model_config.yaml
- [ ] Output: data/datasets/outcome_train.pt
- [ ] Log statistics:
      - Total samples
      - Label distribution (% long, % short, % skip)
      - Average max_up_R and max_down_R per label
```

#### 3.4 Validation
```bash
- [ ] Load generated dataset
- [ ] Check label balance:
      - Ideally: skip ~50-60%, long ~20-25%, short ~20-25%
      - If skip > 80%: Adjust target_R or future_window
- [ ] Visualize samples:
      - Plot a few "long" labeled bars with context
      - Verify that futures actually hit target before stop
- [ ] Check for data leakage (future info in features)
```

### Deliverables
- ✅ labeler.py fully implemented
- ✅ Dataset file: data/datasets/outcome_train.pt
- ✅ Label statistics report

### Validation Criteria
- ✅ Dataset contains >10,000 samples
- ✅ No NaN or Inf values in features
- ✅ Label distribution is reasonable
- ✅ Spot-check confirms labels are correct

---

## Phase 4: Layer 3 - Model Training

### Objectives
- Define model architecture (Transformer or MLP)
- Implement training loop
- Train baseline model and save weights

### Checklist

#### 4.1 Implement Dataset Loader (`training/train_outcome_model.py`)
```bash
- [ ] Class OutcomeDataset(torch.utils.data.Dataset):
      - Load data/datasets/outcome_train.pt
      - __len__() and __getitem__()
      - Return (features_tensor, label_tensor)
      - Features shape: [context_len, feature_dim]
      - Label shape: [1] (class index 0/1/2)
```

#### 4.2 Define Model Architecture
```bash
- [ ] Option A: SimpleTransformer
      - Input: [batch, seq_len, feature_dim]
      - 2-4 Transformer encoder layers
      - Global pooling (mean or CLS token)
      - FC head → 3 classes (long/short/skip)
- [ ] Option B: SimpleMLP (fallback if Transformer too slow)
      - Flatten context: [batch, seq_len * feature_dim]
      - 2-3 hidden layers with ReLU
      - Output: [batch, 3]
- [ ] Implement in train_outcome_model.py
```

#### 4.3 Training Loop
```bash
- [ ] Split dataset: 80% train, 20% validation
- [ ] DataLoader with batch_size=32
- [ ] Optimizer: AdamW, lr=1e-4
- [ ] Loss: CrossEntropyLoss
- [ ] Training loop (10-20 epochs):
      - Forward pass
      - Compute loss
      - Backward + optimizer step
      - Log train loss every 100 batches
      - Validate every epoch
      - Save best model (based on val accuracy or F1)
- [ ] Save final model: models/outcome_transformer_v1.pt
```

#### 4.4 Logging & Metrics
```bash
- [ ] Track metrics:
      - Train/Val loss
      - Train/Val accuracy
      - Per-class precision/recall (especially for long/short)
- [ ] Save training log: logs/training_run_20250126.log
- [ ] Plot loss curves (optional: use matplotlib, save to docs/)
```

#### 4.5 Run Training
```bash
- [ ] Execute: python src/layer3_model/training/train_outcome_model.py
- [ ] Monitor output for convergence
- [ ] Expected val accuracy: >40% (random is 33% for 3-class)
      - Good target: 50-60% with skip class performing well
```

### Deliverables
- ✅ train_outcome_model.py implemented
- ✅ Trained model: models/outcome_transformer_v1.pt
- ✅ Training log with metrics

### Validation Criteria
- ✅ Training completes without errors
- ✅ Validation accuracy > 45%
- ✅ Model file is saveable and loadable
- ✅ No severe overfitting (train/val gap <15%)

---

## Phase 5: Layer 3 - Inference Server

### Objectives
- Build FastAPI server for model inference
- Load trained model and serve predictions
- Test inference latency and accuracy

### Checklist

#### 5.1 Implement Inference Server (`inference/server.py`)
```bash
- [ ] FastAPI app setup
- [ ] Load trained model at startup:
      - Path: models/outcome_transformer_v1.pt
      - Set model.eval()
      - Move to device (CPU or GPU)
- [ ] Endpoint: POST /infer
      - Input: JSON {"features": [[...], [...], ...]}  # shape [context_len, feature_dim]
      - Process:
            * Convert to tensor
            * Run inference (no_grad)
            * Get probabilities (softmax)
      - Output: {"prob_long": 0.3, "prob_short": 0.2, "prob_skip": 0.5}
- [ ] Health check endpoint: GET /health
      - Return {"status": "ok", "model_loaded": true}
```

#### 5.2 Pydantic Models
```bash
- [ ] Create InferRequest model:
      features: List[List[float]]
- [ ] Create InferResponse model:
      prob_long: float
      prob_short: float
      prob_skip: float
```

#### 5.3 Testing
```bash
- [ ] Start server: uvicorn src.layer3_model.inference.server:app --port 5002
- [ ] Test with curl or Postman:
      POST http://localhost:5002/infer
      Body: sample feature context (use one from dataset)
- [ ] Verify:
      - Response is valid JSON
      - Probabilities sum to ~1.0
      - Inference time <50ms
```

#### 5.4 Dockerization
```bash
- [ ] Create deployment/Dockerfile:
      - Base image: python:3.10-slim
      - COPY src/, config/, models/
      - Install requirements
      - CMD: uvicorn src.layer3_model.inference.server:app --host 0.0.0.0 --port 5002
- [ ] Build image: docker build -t outcome-model-server -f deployment/Dockerfile .
- [ ] Test container: docker run -p 5002:5002 outcome-model-server
- [ ] Verify /health and /infer endpoints work
```

#### 5.5 Deployment Script
```bash
- [ ] Create deployment/run_model_server.sh:
      #!/bin/bash
      source venv/bin/activate
      uvicorn src.layer3_model.inference.server:app --host 0.0.0.0 --port 5002 --reload
- [ ] Make executable: chmod +x deployment/run_model_server.sh
- [ ] Test run: ./deployment/run_model_server.sh
```

### Deliverables
- ✅ server.py fully implemented
- ✅ Dockerfile working
- ✅ run_model_server.sh script
- ✅ Inference latency <50ms confirmed

### Validation Criteria
- ✅ Server starts without errors
- ✅ /health returns 200 OK
- ✅ /infer returns valid predictions
- ✅ Docker container runs successfully

---

## Phase 6: Live Integration & End-to-End Testing

### Objectives
- Integrate all 3 layers
- Test full pipeline with live or simulated data
- Deploy Feature Engine server to receive NinjaTrader data

### Checklist

#### 6.1 Feature Engine Server
```bash
- [ ] Create src/layer2_feature_engine/api_server.py:
      - FastAPI app
      - POST /raw endpoint:
            * Receive JSON from NinjaTrader
            * Parse bars
            * Update ContextManager
            * Build features
            * Call Model Server /infer
            * Return prediction to caller (or log)
      - Run on port 5001
```

#### 6.2 Full Pipeline Test
```bash
- [ ] Start Model Server (Layer 3):
      ./deployment/run_model_server.sh
      Verify running on localhost:5002
- [ ] Start Feature Engine Server (Layer 2):
      uvicorn src.layer2_feature_engine.api_server:app --port 5001
      Verify running on localhost:5001
- [ ] Start NinjaTrader (Layer 1):
      - Attach ExportRawData strategy to chart
      - Configure endpoint: http://localhost:5001/raw
- [ ] Monitor logs:
      - NinjaTrader: should POST bars
      - Feature Engine: should receive, process, forward to Model Server
      - Model Server: should return predictions
- [ ] Verify end-to-end:
      - Predictions appear in Feature Engine logs
      - Latency: <200ms total (NinjaTrader → Feature → Model → Response)
```

#### 6.3 Decision Logic (Optional)
```bash
- [ ] Create src/decision_engine.py (simple script):
      - Subscribe to Feature Engine output (or add to api_server.py)
      - If prob_long > 0.6: Log "SIGNAL: BUY"
      - If prob_short > 0.6: Log "SIGNAL: SELL"
      - Else: Log "SKIP"
- [ ] Test with live market data
- [ ] Collect signals for manual review
```

#### 6.4 Documentation & Handoff
```bash
- [ ] Update README.md with:
      - Quick start guide
      - How to run each layer
      - Troubleshooting tips
- [ ] Create docs/deployment_guide.md:
      - Production deployment checklist
      - Docker Compose setup (future)
      - Monitoring and logging setup
- [ ] Record demo video or screenshots (optional)
```

### Deliverables
- ✅ api_server.py for Feature Engine
- ✅ Full pipeline tested and working
- ✅ Documentation updated

### Validation Criteria
- ✅ All 3 layers communicate successfully
- ✅ Predictions generated in real-time
- ✅ No crashes or timeouts under normal load
- ✅ Code is ready for production deployment

---

## Post-Phase 6: Iteration & Optimization

### Backlog (Future Enhancements)
- [ ] Add more sophisticated SMC features (multi-timeframe)
- [ ] Integrate real Rithmic L2 data
- [ ] Implement ensemble models (Transformer + LightGBM)
- [ ] Build monitoring dashboard (Streamlit or Grafana)
- [ ] Auto-execution module (with risk management)
- [ ] Cloud deployment (AWS EC2 or Lambda)
- [ ] A/B testing framework for model versions
- [ ] Reinforcement learning for position sizing

---

## PROMPT TEMPLATE FOR ANOTHER AGENT

Use this prompt to hand off the project to another ML engineer or agent:

```
You are a Senior ML Engineer tasked with implementing a trading ML system.

**Context:**
- Read `ARCHITECTURE.md` for system design
- Read `PROJECT_MASTER_PLAN.md` (this file) for execution phases

**Your Mission:**
Execute Phase 0 through Phase 6 sequentially:
1. Phase 0: Set up environment and create project structure
2. Phase 1: Implement NinjaTrader adapter (C#) for data export
3. Phase 2: Build Feature Engine (Python) with SMC, Volume Profile, L2 features
4. Phase 3: Implement outcome labeler (R-based labels)
5. Phase 4: Train baseline model (Transformer or MLP)
6. Phase 5: Build and deploy inference server (FastAPI + Docker)
7. Phase 6: Integrate all layers and test end-to-end

**Requirements:**
- Complete each phase fully before moving to next
- Write clean, documented code (English comments)
- Validate outputs at each phase
- Update this master plan if you encounter blockers or need to deviate
- Do NOT skip phases or create placeholder implementations

**Deliverable:**
A fully functional 3-layer ML trading system with:
- Live data ingestion from NinjaTrader
- Real-time feature engineering
- Model inference <50ms latency
- Outcome-based predictions (long/short/skip)

**Get started with Phase 0 now.**
```

---

**Document Version**: 1.0
**Last Updated**: 2025-01-26
**Status**: Ready for Execution
