# PLAN: SMC + ML Pipeline v1

**Updated**: 2025-12-02  
**Status**: Active Development  
**Current Focus**: Auction State Model (ASM)

---

## 1. Overview

### 1.1 Project Goal

Auto-trade futures (GC, ES, NQ) using:
- **SMC/VA (Smart Money Concepts + Value Area)**: FVG, OB, VA, CHoCH/BOS, Liquidity Sweeps
- **ML Enhancement**: Auction state evaluation + Context filtering

### 1.2 Key Philosophy

```
┌─────────────────────────────────────────────────────────────────────┐
│  CORE PRINCIPLE: ML là CRITIC, không phải DECISION MAKER           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ❌ OLD: ML predict WIN/LOSS per trade → Noise, không bền vững     │
│  ❌ OLD: ML detect pattern → Overfit, regime-dependent             │
│                                                                     │
│  ✅ NEW: ML đo AUCTION STATE (VA-shift, cung-cầu)                  │
│  ✅ NEW: ML làm FILTER/CRITIC cho rule-based entries               │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.3 Pipeline High-Level

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    SMC + ML PIPELINE v2 (ASM-based)                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  TẦNG 0: DATA SOURCE & EXPORT                                          │
│  ┌─────────────┐    ┌──────────────────┐                               │
│  │ NinjaTrader │───▶│ EXPORT-SMC-JSON  │───▶ smc_states_raw.jsonl      │
│  │ (SMC Core)  │    │ (raw events)     │                               │
│  └─────────────┘    └──────────────────┘                               │
│         │                                                               │
│         ▼                                                               │
│  TẦNG 1: FEATURE ENGINE                                                │
│  ┌──────────────────────────────────────┐                              │
│  │ SMCContextManager                    │                              │
│  │ ├── SMC features (BOS/CHoCH/FVG/OB) │───▶ features_*.csv            │
│  │ ├── VA/VP features                   │                              │
│  │ └── Wave/Delta features              │                              │
│  └──────────────────────────────────────┘                              │
│         │                                                               │
│         ▼                                                               │
│  TẦNG 2: STRATEGY LAYER (Rule-based)                                   │
│  ┌──────────────────────────────────────┐                              │
│  │ S4_London_v1 (BASELINE)              │                              │
│  │ ├── HighVol + FVG Trend Continuation │───▶ candidate entries        │
│  │ ├── London session focus             │                              │
│  │ └── Rule-only, no ML                 │                              │
│  └──────────────────────────────────────┘                              │
│         │                                                               │
│         ▼                                                               │
│  TẦNG 3: ML LAYER (Critic/Filter)                                      │
│  ┌──────────────────────────────────────┐                              │
│  │ AUCTION STATE MODEL (ASM) v1         │                              │
│  │ ├── Input: 60-bar context            │                              │
│  │ ├── Output: p_up, p_down, p_neutral  │───▶ filter decision          │
│  │ └── Role: Evaluate VA-shift prob     │                              │
│  └──────────────────────────────────────┘                              │
│         │                                                               │
│         ▼                                                               │
│  TẦNG 4: EXECUTION                                                     │
│  ┌─────────────────┐    ┌──────────────────┐                           │
│  │ DEPLOY-INFER    │───▶│ NT-AUTOBOT       │                           │
│  │ (API Service)   │    │ (Strategy/Exec)  │                           │
│  └─────────────────┘    └──────────────────┘                           │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Chi Tiết Từng Tầng

### 2.1 TẦNG 0 – DATA SOURCE & EXPORT

| Module | Description | Output |
|--------|-------------|--------|
| CORE-NT-SMC | SMC indicator trên Ninja | Real-time signals |
| EXPORT-SMC-JSON | Export raw bar+SMC state | `smc_states_raw.jsonl` |

**Data Fields**:
- OHLC, volume, delta, spread
- BOS/CHoCH (external/internal)
- FVG (bull/bear, in/near)
- OB (bull/bear, in/near)
- VA (VAH/VAL/POC, in_va)
- Macro M5/H1 (trend, premium/discount)

---

### 2.2 TẦNG 1 – FEATURE ENGINE

| Module | Description | Output |
|--------|-------------|--------|
| SMCContextManager | Process raw data → features | `features_*.csv` |

**Feature Groups**:
- Price/Candle: OHLC, spread, body, wicks
- Volume/Delta: vol, delta, cum_delta, delta_z
- SMC Structure: BOS/CHoCH, sweep, zone flags
- VA/VP: VAH, VAL, POC, in_va, dist_to_va
- Wave: impulse_strength, pullback_strength
- Regime: vol_regime, session

---

### 2.3 TẦNG 2 – STRATEGY LAYER (Rule-based)

**BASELINE**: S4_HighVol_FVG_London_v1

| Aspect | Specification |
|--------|---------------|
| Instrument | GC (Gold Futures) |
| Timeframe | M1 |
| Session | London (08:00-14:00 UTC) |
| Entry | FVG retest in trend |
| Regime | HighVol + Trend continuation |
| RR Target | 2.0R |
| ML Filter | NONE (rule-only baseline) |

**Performance** (NEW DATA 6W, Apr-Jun 2025):
- Trades: 297 (London only)
- Winrate: 55.2%
- Expectancy: +0.654R
- MaxDD: 15.1R

**Reference**: [PLAN_S4_HighVol_FVG_London_v1.md](PLAN_S4_HighVol_FVG_London_v1.md)

---

### 2.4 TẦNG 3 – ML LAYER (Critic/Filter)

**CURRENT FOCUS**: Auction State Model (ASM) v1

#### ASM Overview

| Aspect | Description |
|--------|-------------|
| Question | "VA sẽ shift theo hướng nào trong K bar tới?" |
| Output | p_up, p_down, p_neutral |
| Role | Filter candidate entries based on VA-shift probability |
| Philosophy | Auction/Wyckoff, không phải pattern matching |

#### ASM Label Design

```python
va_shift_label ∈ {UP, DOWN, NEUTRAL}

# At reference bar t, look ahead K bars:
# UP: VA shifts up significantly, price holds above old VAH
# DOWN: VA shifts down significantly, price holds below old VAL
# NEUTRAL: VA stable, no significant shift
```

#### ASM Integration Flow

```
1. Strategy generates candidate entry (e.g., S4_London long)
2. ASM evaluates context → p_up, p_down, p_neutral
3. Filter logic:
   - Long: trade if (p_up - p_down) >= threshold
   - Short: trade if (p_down - p_up) >= threshold
4. ASM does NOT modify entry/SL/TP, only filter
```

**Reference**: [PLAN_AuctionStateModel_v1.md](PLAN_AuctionStateModel_v1.md)

---

### 2.5 TẦNG 4 – EXECUTION

#### DEPLOY-INFER (API Service)

```
Ninja → API Request (context + candidate entry)
    → ASM inference → p_up, p_down, p_neutral
    → Filter decision
    → Response to Ninja
```

#### NT-AUTOBOT (Ninja Strategy)

- Receive filter decision from API
- Execute or skip based on ASM output
- Manage position, SL/TP, risk

---

## 3. Archive / Lessons Learned

> **Các approach dưới đây KHÔNG còn là hướng chính.**

### 3.1 Outcome Model v2.x (WIN/LOSS per trade) — ARCHIVED

- Val AUC: ~0.51-0.53 (near random)
- Lesson: Per-trade outcome prediction quá noisy

### 3.2 Pattern A Detector (FVG Continuation) — ARCHIVED

- Val AUC: 0.7586 (good)
- Extended validation: Không bền vững, overfit
- Lesson: Pattern detection regime-dependent, không universal

**Reference**: [PLAN_P4_outcome_v2.md](PLAN_P4_outcome_v2.md) (Archive section)

---

## 4. Current Status & Roadmap

### Completed ✅
- [x] TẦNG 0: SMC Core + Export
- [x] TẦNG 1: Feature Engine (SMCContextManager)
- [x] TẦNG 2: S4_London_v1 baseline strategy
- [x] Archive: Outcome Model v2.x, Pattern A Detector

### In Progress 🔄
- [ ] TẦNG 3: ASM v1 Dataset (build_asm_dataset_v1.py)
- [ ] TẦNG 3: ASM v1 Training (train_asm_v1.py)

### Next Steps 📋
1. **ASM Dataset v1** - Build VA-shift labels
2. **ASM Training v1** - 3-class classifier
3. **ASM Validation** - Shadow backtest on S4_London
4. **ASM Integration** - API + NT-AUTOBOT

### Future 🔮
- ASM v2: Multi-horizon prediction
- Regime detection model
- Catastrophic loss filter

---

## 5. Success Criteria

### ASM v1 Model
| Metric | Target |
|--------|--------|
| Macro F1 | > 0.45 |
| AUC (UP vs rest) | > 0.65 |
| AUC (DOWN vs rest) | > 0.65 |

### Trading Performance (with ASM filter)
| Metric | Target | Baseline (S4_London) |
|--------|--------|----------------------|
| Expectancy | > +0.70R | +0.654R |
| MaxDD | < 12R | 15.1R |
| Trade retention | > 50% | 100% |

---

## 6. File Structure Summary

```
modeloutcome/
├── PLAN_PIPELINE_SMC_ML_v1.md         # This file
├── PLAN_AuctionStateModel_v1.md       # ASM development (ACTIVE)
├── PLAN_S4_HighVol_FVG_London_v1.md   # Baseline strategy (LOCKED)
├── PLAN_P4_outcome_v2.md              # Archive + history
│
├── scripts/
│   ├── build_asm_dataset_v1.py        # ASM dataset builder (TODO)
│   ├── shadow_backtest_asm_on_S4.py   # ASM validation (TODO)
│   └── build_pattern_dataset_v1.py    # Archived
│
├── phase4_quality_tabular/
│   ├── train_asm_v1.py                # ASM training (TODO)
│   └── train_pattern_a.py             # Archived
│
├── output/
│   ├── asm_dataset_v1/                # ASM dataset (TODO)
│   └── pattern_dataset_v1/            # Archived
│
└── phase8_deploy/
    └── infer_api.py                   # Inference API
```

---

## 7. References

- [PLAN_AuctionStateModel_v1.md](PLAN_AuctionStateModel_v1.md) - ASM development
- [PLAN_S4_HighVol_FVG_London_v1.md](PLAN_S4_HighVol_FVG_London_v1.md) - Baseline strategy
- [PLAN_P4_outcome_v2.md](PLAN_P4_outcome_v2.md) - Archive & lessons learned

---

**Last Updated**: 2025-12-02
