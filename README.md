# SMC Trading ML System

**ML-enhanced trading system using Smart Money Concepts (SMC) + Value Area (VA) for futures trading.**

---

## 🎯 Current Status (Dec 2025)

### Core System

| Component | Status | Description |
|-----------|--------|-------------|
| **S4_HighVol_FVG_London_v1** | ✅ LOCKED | Baseline strategy (rule-only, no ML) |
| **Auction State Model (ASM) v1** | 🔄 IN PROGRESS | ML critic/filter for VA-shift evaluation |

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

### Baseline Performance (S4_London, 6W backtest)

| Metric | Value |
|--------|-------|
| Trades | 297 |
| Winrate | 55.2% |
| Expectancy | +0.654R |
| MaxDD | 15.1R |

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

### ASM Pipeline (Upcoming)
```bash
# TODO: ASM dataset builder
python scripts/build_asm_dataset_v1.py

# TODO: ASM training
python phase4_quality_tabular/train_asm_v1.py

# TODO: Shadow backtest ASM on S4_London
python scripts/shadow_backtest_asm_on_S4.py
```

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

| Document | Description |
|----------|-------------|
| [PLAN_PIPELINE_SMC_ML_v1.md](PLAN_PIPELINE_SMC_ML_v1.md) | Overall pipeline architecture |
| [PLAN_AuctionStateModel_v1.md](PLAN_AuctionStateModel_v1.md) | ASM development plan |
| [PLAN_S4_HighVol_FVG_London_v1.md](PLAN_S4_HighVol_FVG_London_v1.md) | Baseline strategy spec |

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
- [x] Archive: Outcome Model v2.x, Pattern A, P2-Quality

### In Progress 🔄
- [ ] TẦNG 3: ASM v1 Dataset
- [ ] TẦNG 3: ASM v1 Training

### Next Steps 📋
1. Build ASM dataset (VA-shift labels)
2. Train ASM v1 (3-class classifier)
3. Shadow backtest ASM on S4_London
4. Integrate ASM filter vào execution pipeline

---

**Version**: 4.0 (ASM Focus)  
**Last Updated**: 2025-12-02
