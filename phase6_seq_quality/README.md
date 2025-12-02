# Phase 6 - Sequence Quality Model (Experimental)

## Overview

**Status**: 🧪 EXPERIMENTAL

GRU-based sequence model that preserves temporal structure (no flattening).  
Alternative to QUALITY_TABULAR_V1 baseline.

**Goal**: Test if sequential modeling improves quality prediction vs tabular MLP.

---

## Architecture

### QualitySeqGRU

```
Input: [B, 60, 66] sequence + [B, 1] side
  ↓
Concat side as extra feature channel → [B, 60, 67]
  ↓
GRU(hidden_dim=128, num_layers=1)
  ↓
Take last hidden state: [B, 128]
  ↓
MLP Head: Linear(128 → 128) + ReLU + Dropout + Linear(128 → 1)
  ↓
Output: [B, 1] logit
```

**Parameters**: ~92K (vs 1M in tabular)

**Key difference from tabular**:
- Tabular: Flattens [60×66] → 3,960 features
- Sequence: Processes [60, 66] with GRU (preserves time)

---

## Usage

### 1. Train Model

```bash
python -m phase6_seq_quality.train_seq_quality
```

- Batch size: 64
- Early stopping on val F1 (patience=5)
- Saves best model to `output/phase6_seq_quality/model_seq_quality_v1_best.pt`

### 2. Evaluate

```bash
# Val + Holdout evaluation
python -m phase6_seq_quality.eval_seq_backtest

# Threshold sweep (val only)
python -m phase6_seq_quality.threshold_sweep_seq
```

### 3. Compare to Tabular

Check reports:
- `output/phase6_seq_quality/val_backtest_seq_vs_tabular.txt`
- `output/phase6_seq_quality/holdout_backtest_seq_vs_tabular.txt`

---

## Files

```
phase6_seq_quality/
├── __init__.py
├── dataset_seq.py          # Sequence dataset wrapper
├── model_seq.py            # GRU architecture
├── train_seq_quality.py    # Training script
├── eval_seq_backtest.py    # Evaluation
├── threshold_sweep_seq.py  # Threshold analysis
└── README.md               # This file

output/phase6_seq_quality/
├── model_seq_quality_v1_best.pt          # Trained weights
├── normalizer_stats_seq.pt               # Per-feature mean/std
├── training_log_seq.csv                  # Epoch metrics
├── val_backtest_seq_vs_tabular.txt       # Comparison
├── holdout_backtest_seq_vs_tabular.txt   # Holdout results
└── threshold_sweep_seq_results.csv       # Threshold analysis
```

---

## Results (To be filled after training)

### Training
- Best val F1: [TBD]
- Epochs to convergence: [TBD]
- Training time: [TBD]

### Validation Set (t=0.5)
- Trades: [TBD]
- Winrate: [TBD]
- Expectancy: [TBD]
- Max DD: [TBD]

**vs Tabular**: 546 trades, 36.1% WR, +0.44R exp, 16R DD

### Holdout Set (t=0.7)
- Trades: [TBD]
- Winrate: [TBD]
- Expectancy: [TBD]
- Max DD: [TBD]

**vs Tabular**: 66 trades, 72.7% WR, +1.91R exp, 4R DD

---

## Key Questions

1. **Does temporal modeling help?**
   - Compare F1, AUC, expectancy vs tabular
   
2. **Is sequence model more efficient?**
   - 92K params vs 1M params
   - Training time, inference speed
   
3. **Does it generalize better?**
   - Holdout performance vs val performance

4. **What's the optimal threshold?**
   - Does sweet spot differ from tabular (0.5/0.7)?

---

## Next Steps Based on Results

### If sequence model BEATS tabular:
1. Consider promoting to production candidate
2. Test ensemble (tabular + sequence)
3. Explore deeper architectures (2-layer GRU, Transformer)

### If sequence model MATCHES tabular:
1. Use as validation/ensemble member
2. Investigate where each model excels
3. Consider hybrid approach

### If sequence model UNDERPERFORMS:
1. Document why (loss of information? model capacity?)
2. Keep as baseline experiment
3. Try alternative architectures (LSTM, Transformer)

---

## Status

**Phase**: Experimental R&D  
**Production Model**: QUALITY_TABULAR_V1 (unchanged)  
**Decision Criteria**: Must beat tabular on holdout to consider promotion

---

**Created**: 2025-11-28  
**Model**: QUALITY_SEQ_GRU_V1  
**Status**: In development
