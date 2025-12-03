# SMC Trading ML System

**ML-enhanced trading system using Smart Money Concepts (SMC) + Value Area (VA) for futures trading.**

---

## 🎯 Current Status (Dec 2025)

### Core System

| Component | Status | Description |
|-----------|--------|-------------|
| **S4_HighVol_FVG_London_v1** | ✅ LOCKED | Baseline strategy (rule-only, no ML) |
| **S4_LDN_ASM_LowShift_0.2_v1.1** | 🔒 LOCKED | S4 + ASM inverse filter (p_shift ≤ 0.2) – PASS extended validation |
| **Auction State Model (ASM) v1** | ✅ TRAINED | ML critic/filter for VA-shift evaluation |

### Key Philosophy

```
┌─────────────────────────────────────────────────────────────────────┐
│  CORE PRINCIPLE: ML là CRITIC, không phải DECISION MAKER           │
├─────────────────────────────────────────────────────────────────────┤
│  ❌ OLD: ML predict WIN/LOSS per trade → Noise, không bền vững     │
│  ✅ NEW: ML đo AUCTION STATE (VA-shift, cung-cầu)                  │
│  ✅ NEW: ML làm FILTER/CRITIC cho rule-based entries               │
└─────────────────────────────────────────────────────────────────────┘
```

### Performance Summary (6W backtest, Apr-Jun 2025)

| Strategy | Trades | WR% | Exp(R) | MaxDD |
|----------|--------|-----|--------|-------|
| **S4_London (Baseline)** | 479 | 50.9% | +0.54R | 37R |
| **S4_LDN_ASM_LowShift_0.2** | 109 | **82.6%** | **+1.48R** | **9R** |

---

## 📊 System Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    SMC + ML PIPELINE v2 (ASM-based)                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  TẦNG 0: DATA SOURCE                                                   │
│  ┌─────────────┐    ┌──────────────────┐                               │
│  │ NinjaTrader │───▶│ EXPORT-SMC-JSON  │───▶ smc_states_raw.jsonl      │
│  │ (SMC Core)  │    │ (raw events)     │                               │
│  └─────────────┘    └──────────────────┘                               │
│                                                                         │
│  TẦNG 1: FEATURE ENGINE                                                │
│  ┌──────────────────────────────────────┐                              │
│  │ SMCContextManager                    │                              │
│  │ ├── SMC features (BOS/CHoCH/FVG/OB) │───▶ features_*.csv            │
│  │ ├── VA/VP features                   │                              │
│  │ └── Wave/Delta features              │                              │
│  └──────────────────────────────────────┘                              │
│                                                                         │
│  TẦNG 2: STRATEGY LAYER (Rule-based)                                   │
│  ┌──────────────────────────────────────┐                              │
│  │ S4_London_v1 (BASELINE - LOCKED)     │                              │
│  │ ├── HighVol + FVG Trend Continuation │───▶ candidate entries        │
│  │ ├── London session (08:00-14:00 UTC) │                              │
│  │ └── Rule-only, no ML                 │                              │
│  └──────────────────────────────────────┘                              │
│                                                                         │
│  TẦNG 3: ML LAYER (Critic/Filter)                                      │
│  ┌──────────────────────────────────────┐                              │
│  │ AUCTION STATE MODEL (ASM) v1         │                              │
│  │ ├── Input: 60-bar context            │                              │
│  │ ├── Output: p_up, p_down, p_neutral  │───▶ filter decision          │
│  │ └── Role: Evaluate VA-shift prob     │                              │
│  └──────────────────────────────────────┘                              │
│                                                                         │
│  TẦNG 4: EXECUTION                                                     │
│  ┌─────────────────┐    ┌──────────────────┐                           │
│  │ DEPLOY-INFER    │───▶│ NT-AUTOBOT       │                           │
│  │ (API Service)   │    │ (Strategy/Exec)  │                           │
│  └─────────────────┘    └──────────────────┘                           │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Data Processing (Active)
```bash
# Process raw data → features
python -m src.layer2_feature_engine_v2.batch_process
```

### ASM Pipeline
```bash
# Build ASM dataset (112 features with Weekly VA)
python scripts/build_asm_dataset_v1.py

# Train ASM model
python scripts/train_asm_v1.py --config_name C3_focal_inv_os --loss_type focal --oversample_shift

# Shadow backtest ASM on S4_London
python scripts/shadow_backtest_s4_asm_v1.py
```

### ASM Models
- **ASM-GRU64-v1.0-C3**: Baseline model (AUC_SHIFT=0.712, AUC_UP=0.779, AUC_DOWN=0.857)
- Model path: `output/asm_models_v1/ASM-GRU64-v1.0-C3.pt`
- Shadow backtest results: `backtests/s4_highvol_fvg_london_asm_v1_shadow.json`

### Strategies / Modules
- **S4_HighVol_FVG_London_v1**: Baseline rule-only strategy (LOCKED)
- **S4_LDN_ASM_LowShift_0.2_v1.1**: 🔒 LOCKED – S4 HighVol FVG London + ASM v1.0 (p_shift ≤ 0.2). +59% expectancy vs baseline, -30% MaxDD trên NEW 6W data. See [PLAN_S4_HighVol_FVG_London_v1.md](PLAN_S4_HighVol_FVG_London_v1.md) and `backtests/s4_asm_lowshift_extval_new6w_v1.json`.

### Inference API
```bash
uvicorn phase8_deploy.infer_api:app --port 8000
```

> **NOTE**: Scripts trong `phase3_label_backtest/`, `phase4_quality_tabular/` (P2/P4 cũ) thuộc nhánh Archive. Xem mục Archive bên dưới.

---

## 📁 Project Structure

```
modeloutcome/
├── PLAN_PIPELINE_SMC_ML_v1.md         # Pipeline overview (ACTIVE)
├── PLAN_AuctionStateModel_v1.md       # ASM development (ACTIVE)
├── PLAN_S4_HighVol_FVG_London_v1.md   # Baseline strategy (LOCKED)
│
├── config/                            # Configuration files
├── data/
│   ├── raw/                           # NinjaTrader exports (.jsonl)
│   └── processed_v2/                  # Feature CSVs
├── src/
│   └── layer2_feature_engine_v2/      # Feature engine
│
├── scripts/
│   ├── build_asm_dataset_v1.py        # ASM dataset (TODO)
│   └── shadow_backtest_asm_on_S4.py   # ASM validation (TODO)
│
├── phase4_quality_tabular/
│   └── train_asm_v1.py                # ASM training (TODO)
│
├── phase8_deploy/
│   └── infer_api.py                   # Inference API
│
└── output/                            # Model artifacts
```

---

## 📖 Documentation

### Planning Documents
| Document | Description |
|----------|-------------|
| [PLAN_PIPELINE_SMC_ML_v1.md](PLAN_PIPELINE_SMC_ML_v1.md) | Overall pipeline architecture |
| [PLAN_AuctionStateModel_v1.md](PLAN_AuctionStateModel_v1.md) | ASM development plan |
| [PLAN_S4_HighVol_FVG_London_v1.md](PLAN_S4_HighVol_FVG_London_v1.md) | Baseline strategy spec |

### Deployment & Operations
| Document | Description |
|----------|-------------|
| [docs/DEPLOYMENT_GUIDE.md](docs/DEPLOYMENT_GUIDE.md) | Fresh install to production deployment |
| [docs/API_DOCUMENTATION.md](docs/API_DOCUMENTATION.md) | Live Gateway API reference |
| [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) | Common issues and solutions |
| [services/live_gateway/README.md](services/live_gateway/README.md) | Live Gateway service overview |

---

## 🗄️ Archive / History – Outcome & Quality Models

> **⚠️ ARCHIVED – NOT ACTIVE**
>
> Các model dưới đây đã được thử nghiệm nhưng KHÔNG còn là hướng phát triển chính.

### P2-Quality GRU Model

| Metric | Value | Status |
|--------|-------|--------|
| Best F1 | 0.6248 | ❌ NOT FOR PRODUCTION |
| AUC | ~0.53 | Near random |

**Lesson**: P2 quality labels quá noisy, không đủ signal để filter.

### Outcome Model v2.x (WIN/LOSS per trade)

| Metric | Value | Status |
|--------|-------|--------|
| Val AUC | ~0.51-0.53 | ❌ ABANDONED |

**Lesson**: Per-trade outcome prediction quá noisy, không bền vững.

### Pattern A Detector (FVG Continuation)

| Metric | Value | Status |
|--------|-------|--------|
| Val AUC | 0.7586 | ❌ ABANDONED |

**Lesson**: Pattern detection regime-dependent, overfit, không universal.

---

## 🎯 Roadmap

### Completed ✅
- [x] TẦNG 0: SMC Core + Export
- [x] TẦNG 1: Feature Engine (SMCContextManager)
- [x] TẦNG 2: S4_London_v1 baseline strategy (LOCKED)
- [x] TẦNG 3: ASM v1.x Dataset (112 features with Weekly VA)
- [x] TẦNG 3: ASM v1.0 Training (ASM-GRU64-v1.0-C3)
- [x] Shadow backtest ASM on S4_London
- [x] **Inverse filter discovery**: p_shift ≤ 0.2 → WR 82.6%, Exp +1.48R
- [x] Lock candidate strategy: S4_LDN_ASM_LowShift_0.2_v1.0
- [x] Archive: Outcome Model v2.x, Pattern A, P2-Quality

### In Progress 🔄
- [x] Extended validation: S4_LDN_ASM_LowShift_0.2_v1.0 on new 6W data → **PASS** (see `backtests/s4_asm_lowshift_extval_new6w_v1.json`)
- [ ] Integrate ASM filter vào NT-AUTOBOT execution pipeline

### Next Steps 📋
1. ~~Validate inverse filter on out-of-sample data~~ ✅ DONE
2. Implement ASM inference in production pipeline
3. Monitor live performance of S4_LDN_ASM_LowShift_0.2

---

## 🧪 Testing & Validation

### Health Check
```bash
python scripts/health_check.py
```

### Full Pipeline Test
```bash
# Quick test (1 week data)
python scripts/test_full_pipeline_v1.py --quick

# Full test (6 weeks data)
python scripts/test_full_pipeline_v1.py
```

### Live Gateway
```bash
# Start server
python services/live_gateway/run_server.py

# Validate with replay
python scripts/simulate_live_gateway_from_jsonl.py
```

---

## 📝 Experiment Log

| Date | Experiment | Result |
|------|------------|--------|
| 2025-12-03 | Extended validation: S4_LDN_ASM_LowShift_0.2_v1.0 on new 6W data | **PASS** - Exp +0.85R (+59% vs baseline), WR 61.6%, MaxDD 26R |
| 2025-12-03 | Lock S4_LDN_ASM_LowShift_0.2_v1.1 | 🔒 LOCKED – Ready for shadow trading |
| 2025-12-03 | Live Gateway implementation | ✅ **PASS** - 99.9%+ accuracy vs backtest |
| 2025-12-03 | Full Pipeline Test v1.0 | ✅ **PASS** - All 5 components working |

---

**Version**: 4.5 (Live Gateway + Documentation)  
**Last Updated**: 2025-12-03
