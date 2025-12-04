# ASM v2 - Auction State Model (Regime Classifier)

## Overview
ASM v2 là regime classifier sử dụng embedding `z_t` từ STATE-ENC v1.2 để phân loại market regime.

## Architecture
- Input: `z_t` (64-dim) + meta features (6-dim) = 70-dim
- Model: 2-layer GRN (Gated Residual Network)
- Output: K regime classes (distilled từ asm_regime_hint rule-based)

## Regime Classes
- 1: trend_up
- 2: trend_down  
- 3: balance/range
- 4: opening_drive_up
- 5: opening_drive_down

## Usage

### 1. Build ASM Dataset
```bash
python asm_v2/scripts/build_asm_dataset_gc_m1.py --config asm_v2/configs/asm_dataset_gc_m1_v1.json
```

### 2. Train ASM v2
```bash
python asm_v2/scripts/train_asm_v2_gc_m1.py --config asm_v2/configs/asm_train_v1.json
```

### 3. Evaluate & Leak Tests
```bash
python asm_v2/scripts/eval_asm_v2_gc_m1.py --config asm_v2/configs/asm_eval_v1.json
python asm_v2/scripts/eval_asm_v2_leaks_gc_m1.py --config asm_v2/configs/asm_eval_v1.json
```

### 4. Enrich S4_LDN Trades
```bash
python asm_v2/scripts/enrich_s4_ldn_with_context.py \
    --bars-glob "data/raw/smc_export_gc_m1_v3_*.jsonl" \
    --s4-file "data/s4_ldn/gc_m1_trades_rule_only.jsonl"
```

### 5. End-to-End Test
```bash
python asm_v2/scripts/test_asm_v2_end2end.py
```

## Artifacts
- `artifacts/gc_m1/asm_dataset_gc_m1_v1.jsonl` - ASM dataset
- `artifacts/gc_m1/splits_gc_m1_v1.json` - Train/val splits
- `artifacts/final/asm_v2_gc_m1_v1.pt` - Trained model
- `artifacts/final/asm_model_config_v1.json` - Model config
- `artifacts/final/asm_feature_config_v1.json` - Feature config

---

# S4_LDN Policy & Backtest v1

## Overview
Module đánh giá và so sánh các policy trade/skip cho S4_LDN strategy dựa trên:
- Regime từ ASM v2 (trend_up, trend_down, balance)
- Embedding z_t từ STATE-ENC v1.2
- Meta features (session, pos_in_session, etc.)

## Policy Types

### Rule-based Policies
- **P0_baseline_all**: Giữ tất cả trades (baseline)
- **P1_regime_trend_down**: Chỉ trade khi regime = trend_down
- **P2_regime_no_balance**: Bỏ trades khi regime = balance
- **P3_session_ldn**: Chỉ trade trong phiên London
- **P4_direction_aligned**: Trade theo hướng regime (long+trend_up, short+trend_down)
- **P5_combo_***: Kết hợp nhiều điều kiện

### ML-based Policies
- **P_ML_thresh_X**: Dùng meta-model predict p(win), filter theo threshold

## Usage

### 1. Ensure enriched trades exist
```bash
python asm_v2/scripts/enrich_s4_ldn_with_context.py \
    --bars-glob "data/raw/smc_export_gc_m1_v3_*.jsonl" \
    --s4-file "data/s4_ldn/gc_m1_trades_rule_only.jsonl"
```

### 2. Run Policy Backtest
```bash
python asm_v2/scripts/run_s4_ldn_policy_backtest.py --config asm_v2/configs/s4_policy_config_v1.json
```

### 3. Run Sanity Tests
```bash
python asm_v2/scripts/eval_s4_ldn_policy_sanity.py --config asm_v2/configs/s4_policy_sanity_v1.json
```

## Output Artifacts
- `artifacts/s4_policy/gc_m1/s4_policy_results_v1.json` - Full results
- `artifacts/s4_policy/gc_m1/s4_policy_league_table_v1.csv` - Leaderboard
- `artifacts/s4_policy/gc_m1/s4_policy_selected_v1.json` - Best policy
- `artifacts/s4_policy/gc_m1/s4_policy_sanity_report_v1.json` - Leak tests

## Metrics
- **Expectancy**: Mean R per trade
- **Win Rate**: % winning trades
- **Max Drawdown**: Largest peak-to-trough in R
- **Profit Factor**: Gross profit / Gross loss
- **Sharpe-like**: Mean R / Std R


---

# PHASE 3 — GC M1 NEW DATA OOS EVALUATION

## Overview
Out-of-sample evaluation on 6 weeks of new GC M1 data (Apr-Jun 2025) using FROZEN models:
- STATE-ENC v1.2 (frozen)
- ASM v2 (frozen)
- P7_direction_aligned policy (frozen)

**NO RETRAINING** - chỉ inference và evaluation.

## Data Source
```
data/raw/new_data/
├── smc_export_gc_m1_v3_20250428.jsonl
├── smc_export_gc_m1_v3_20250505.jsonl
├── smc_export_gc_m1_v3_20250512.jsonl
├── smc_export_gc_m1_v3_20250519.jsonl
├── smc_export_gc_m1_v3_20250526.jsonl
└── smc_export_gc_m1_v3_20250602.jsonl
```

## Pipeline Commands

### Option 1: Run Full Pipeline (Recommended)
```bash
python asm_v2/scripts/run_phase3_newdata_pipeline.py
```

### Option 2: Run Individual Steps
```bash
# 0. Build SMC features from raw data
python scripts/build_gc_m1_features_newdata.py

# 1. Build encoder dataset from bars_enhanced
python state_enc_v1/scripts/build_encoder_dataset_gc_m1_newdata.py

# 2. Build ASM dataset with z_t embeddings
python asm_v2/scripts/build_asm_dataset_gc_m1_newdata.py

# 3. Run regime inference
python asm_v2/scripts/run_asm_infer_gc_m1_newdata.py

# 4. Build S4 trades from new data
python asm_v2/scripts/build_s4_ldn_trades_real_gc_m1_newdata.py

# 5. Enrich trades with z_t + regime
python asm_v2/scripts/run_s4_ldn_enrich_gc_m1_real_newdata.py

# 6. Run policy backtest
python asm_v2/scripts/run_s4_ldn_policy_backtest_v2_real_newdata.py

# 7. Run sanity/leak tests
python asm_v2/scripts/eval_s4_ldn_policy_sanity_v2_real_newdata.py

# 8. Generate final shadow report
python asm_v2/scripts/s4_ldn_shadow_report_gc_m1_real_newdata.py
```

## New Data Artifacts
```
state_enc_v1/artifacts/gc_m1_new/
├── bars_enhanced_gc_m1_newdata.jsonl          # SMC features
└── encoder_dataset_gc_m1_newdata_v1.2.jsonl   # Encoder sequences

asm_v2/artifacts/gc_m1_new/
├── asm_dataset_gc_m1_newdata_v1.jsonl         # ASM dataset
├── asm_regime_pred_gc_m1_newdata_v1.jsonl     # Regime predictions
├── s4_ldn_trades_gc_m1_newdata_v1.jsonl       # Raw S4 trades
├── s4_ldn_trades_enriched_gc_m1_real_newdata_v1.2.jsonl  # Enriched trades
├── s4_policy_league_gc_m1_real_newdata_v1.json/csv       # League table
├── s4_policy_best_gc_m1_real_newdata_v1.json             # Best policy
└── s4_policy_sanity_gc_m1_real_newdata_v1.json           # Sanity tests
```

## Evaluation Metrics
- **P7_direction_aligned** performance on TEST split
- Comparison with original REAL data results
- Sanity tests: time split, label shuffle, future field guard, regime leak guard
