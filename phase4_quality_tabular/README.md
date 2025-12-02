# Phase 4: Quality Model (Tabular MLP)

Binary classifier to filter SMC rule candidates by quality (KEEP vs DROP).

## Quick Start

```bash
# 1. Build quality dataset from Phase 3 outputs
python -m phase4_quality_tabular.build_dataset_quality

# 2. Train MLP classifier
python -m phase4_quality_tabular.train_tabular_quality

# 3. Evaluate with backtest
python -m phase4_quality_tabular.eval_quality_backtest
```

## Overview

**Phase 4 addresses the question**: *Which SMC setups are worth trading?*

- **Phase 3 (SMC Rule)**: Decides signal direction (long/short/skip) based on trend + zones
- **Phase 4 (Quality Model)**: Filters candidates by predicting KEEP (high quality) vs DROP (low quality)

## Dataset

### Quality Labels

For each P2 candidate where `signal_side ∈ {long, short}`:

- **KEEP (1)**: Event hit TP with `outcome_rr >=  target_rr`
- **DROP (0)**: Event hit SL, none, or `outcome_rr < target_rr`

### Stats (10-week dataset)

- **Total P2 events**: 20,542
- **Candidates** (long/short): 6,520
- **KEEP**: 1,713 (26.3%)
- **DROP**: 4,807 (73.7%)
- **Train/Val split**: 5,216 / 1,304 (80/20, stratified)

### Files

```
output/phase4_quality/
├── events_p2_labeled_quality_v1.jsonl  # Enriched events with quality labels
├── dataset_p2_quality_v1.pt            # Full PyTorch dataset
├── dataset_p2_quality_v1_train.pt      # Train split
└── dataset_p2_quality_v1_val.pt        # Val split
```

## Model Architecture

**Tabular MLP** (v1 baseline):

```
Input: Flattened [60×66] context + side encoding = 3,961 features

Layer 1: Linear(3961 → 256) + ReLU + Dropout(0.2)
Layer 2: Linear(256 → 128) + ReLU + Dropout(0.2)
Output:  Linear(128 → 1)  # Single logit (BCE loss)

Total parameters: 1,047,297
```

### Training Details

- **Loss**: BCEWithLogitsLoss with `pos_weight=2.81` (class balancing)
- **Optimizer**: Adam (lr=1e-3, weight_decay=1e-4)
- **Batch size**: 128
- **Epochs**: 50 (early stopping on val F1)
- **Best epoch**: 2 (Val F1=0.4432)

### Feature Normalization

- Z-score normalization on train set
- Mean/std applied to val/test
- Saved in `normalizer_stats.pt`

## Results

### Classification Metrics (Validation)

| Metric | Value |
|--------|-------|
| Accuracy | 0.6204 |
| Precision (KEEP) | 0.3608 |
| Recall (KEEP) | 0.5743 |
| F1 (KEEP) | 0.4432 |
| ROC AUC | 0.6365 |

### Trading Performance

**Baseline (Rule Only)**:
- Trades: 1,304
- Winrate: 26.3%
- Expectancy: **+0.0521R**
- Total PnL: +68R
- Max DD: 39R

**ML Filtered** (threshold=0.5):
- Trades: 546 (-58.1%)
- Winrate: **36.1%** (+9.8%)
- Expectancy: **+0.4432R** (+750%! 🎉)
- Total PnL: **+242R** (+256%)
- Max DD: **16R** (-59%)

### 🎯 **SUCCESS!**

ML quality filter dramatically improves trading performance:
- ✅ **8.5x better expectancy**
- ✅ **59% lower max drawdown**  
- ✅ **Maintains reasonable trade frequency** (546 trades on val set)

## Files & Artifacts

### Model Files

```
output/phase4_quality/
├── model_tabular_quality_v1_best.pt     # Trained model weights
├── normalizer_stats.pt                  # Feature normalization params
├── training_log.csv                     # Epoch-by-epoch metrics
├── report_tabular_v1.txt                # Full evaluation report
├── predictions.csv                      # Per-event predictions
└── backtest_results.txt                 # Trading metrics comparison
```

### Code Structure

```
phase4_quality_tabular/
├── __init__.py
├── model.py                   # MLP architecture + feature prep
├── build_dataset_quality.py   # Create quality labels & PyTorch datasets
├── train_tabular_quality.py   # Train MLP with early stopping
└── eval_quality_backtest.py   # Backtest evaluation
```

## Usage Examples

### Load Trained Model

```python
import torch
from phase4_quality_tabular.model import QualityMLP, prepare_features

# Load model
model = QualityMLP(input_dim=3961)
model.load_state_dict(torch.load('output/phase4_quality/model_tabular_quality_v1_best.pt'))
model.eval()

# Load normalizer
normalizer = torch.load('output/phase4_quality/normalizer_stats.pt')
mean, std = normalizer['mean'], normalizer['std']

# Predict on new event
# X: [1, 60, 66] context window
# side: [1] (+1 for long, -1 for short)
X_flat = prepare_features(X, side)
X_norm = (X_flat - mean) / std

with torch.no_grad():
    logit = model(X_norm)
    p_keep = torch.sigmoid(logit).item()

# Filter decision
if p_keep >= 0.5:
    print(f"KEEP this trade (p={p_keep:.3f})")
else:
    print(f"DROP this trade (p={p_keep:.3f})")
```

### Analyze Predictions

```python
import pandas as pd

# Load predictions
df = pd.read_csv('output/phase4_quality/predictions.csv')

# High-confidence KEEPs
high_quality = df[(df['predicted'] == 1) & (df['p_keep'] > 0.7)]
print(f"High-confidence KEEP trades: {len(high_quality)}")

# Analyze errors
false_positives = df[(df['predicted'] == 1) & (df['actual'] == 0)]
false_negatives = df[(df['predicted'] == 0) & (df['actual'] == 1)]
```

## Next Steps

### Potential Improvements (Phase 5)

1. **Advanced architectures**:
   - RNN/LSTM for temporal modeling
   - Transformer for attention mechanism
   - Skip flattening, use full [60, 66] sequences

2. **Multi-task learning**:
   - Predict both quality AND expected RR
   - Auxiliary tasks: predict hold_bars, exit_price

3. **Feature engineering**:
   - Add derived features (ratios, interactions)
   - Feature importance analysis
   - Dimensionality reduction (PCA/autoencoder)

4. **Threshold optimization**:
   - ROC curve analysis for optimal threshold
   - Risk-adjusted threshold (e.g., target Sharpe ratio)

5. **Ensemble methods**:
   - Multiple models with different architectures
   - Voting or stacking

## Notes

- **Training time**: ~30 seconds on CPU
- **Inference**: ~1ms per event
- **Memory**: ~300MB for full dataset
- **Overfitting observed**: Train loss drops to 0.01 but val stabilizes around epoch 2
  - Early stopping crucial
  - Could benefit from stronger regularization or simpler architecture

## License

Internal project - SMC ML Quality Filter v1.0
