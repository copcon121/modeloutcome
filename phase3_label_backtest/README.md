# Phase 3: SMC Rule-Based Labeling & Baseline Backtest

## Overview
This module implements SMC (Smart Money Concepts) rule-based signal generation for P2-filtered events, computes trading outcomes, builds ML datasets, and runs baseline backtests.

## Approach
- **Signal Logic**: Trend + Zone alignment (External trend + FVG/OB zones)
- **Labels**: long / short / skip
- **Outcomes**: Computed via forward simulation (TP/SL hits)
- **Baseline Backtest**: Rule-only performance before ML

## Module Structure

```
phase3_label_backtest/
├── __init__.py           # Package init
├── config.py             # Configuration (tick size, RR, paths)
├── loader.py             # Data loading (P2 sequences + full stream)
├── labeler.py            # SMC signal rules
├── outcome.py            # Entry/SL/TP + outcome computation
├── build_dataset.py      # Build ML dataset
├── backtest_baseline.py  # Rule-only backtest
└── README.md             # This file
```

## Input Data (from Phase 2)

- `output/production_10weeks/features_all_10weeks.csv` (67,709 bars)
- `output/production_10weeks/sequences_p2_10weeks_sequences.npy` (20,542 events, 60x62)
- `output/production_10weeks/sequences_p2_10weeks_indices.npy` (P2 indices)

## Workflow

### 1. Load Data
```bash
python -m phase3_label_backtest.loader --verify
```
Loads P2 sequences and full bar stream, verifies alignment.

### 2. Generate Labels
```bash
python -m phase3_label_backtest.labeler
```
Applies SMC rules to each P2 event:
- **LONG**: `ext_trend_dir > 0` AND (`in_bull_fvg` OR `in_bull_ob`)
- **SHORT**: `ext_trend_dir < 0` AND (`in_bear_fvg` OR `in_bear_ob`)
- **SKIP**: All other cases

Output: Signal side + reason for each event

### 3. Compute Outcomes
```bash
python -m phase3_label_backtest.outcome
```
For each labeled signal:
- Calculate entry (event close)
- Calculate SL (zone boundary + swing + buffer)
- Calculate TP (entry + RR * risk)
- Simulate future bars to determine hit type (tp/sl/none)

### 4. Build ML Dataset
```bash
python -m phase3_label_backtest.build_dataset
```
Combines sequences + labels + outcomes:
- `events_p2_labeled_10weeks.jsonl` - Human-readable labeled events
- `dataset_p2_labeled.pt` - PyTorch format for training

### 5. Run Baseline Backtest
```bash
python -m phase3_label_backtest.backtest_baseline
```
Analyzes rule-only performance:
- Winrate, Avg R, Expectancy
- Session breakdown (Asia/Europe/NY)
- Equity curve

Output: `backtest_report.txt`, `equity_curve.csv`

## Output Files

### events_p2_labeled_10weeks.jsonl
Each line contains:
```json
{
  "event_id": 0,
  "timestamp": "2025-09-01T10:30:00",
  "signal_side": "long",
  "signal_reason": "ext_trend_up+in_bull_fvg",
  "entry_price": 3500.5,
  "sl_price": 3496.0,
  "tp_price": 3514.0,
  "hit": "tp",
  "outcome_rr": 3.0,
  "hold_bars": 23
}
```

### dataset_p2_labeled.pt (PyTorch)
```python
{
    'sequences': torch.Tensor([20542, 60, 62]),  # Context features
    'labels': torch.Tensor([20542]),              # 0=long, 1=short, 2=skip
    'meta': [dict, ...]                           # Metadata per event
}
```

## Configuration

Edit `config.py` to adjust:
- `TICK_SIZE = 0.1` - GC tick size
- `BUFFER_TICKS = 2` - SL/TP buffer
- `TARGET_RR = 3.0` - Risk-reward ratio
- `MAX_HOLD_BARS = 100` - Max position hold time

## Expected Results

- **Signal Distribution**: ~40% actionable (long/short), ~60% skip
- **Baseline Metrics** (target):
  - Winrate: >50%
  - Expectancy E[R]: >0.5
  - Provides edge validation before ML training

## Risk Parameter Optimization

### Running the Risk Sweep

To find the optimal ATR multiplier and RR target:

```bash
cd phase3_label_backtest
python risk_sweep.py
```

This tests 16 configurations:
- ATR multipliers: 0.5, 1.0, 1.5, 2.0
- RR targets: 1.5, 2.0, 2.5, 3.0

**Outputs**:
- `output/phase3_labeled/risk_sweep_results.csv` - Full table
- `output/phase3_labeled/risk_sweep_results.txt` - Human-readable report

### Interpreting Results

The sweep ranks configurations by:
1. **Expectancy** (higher is better) - Average R per trade
2. **Max Drawdown** (lower is better) - Worst equity decline

**Key metrics**:
- **Winrate**: % of trades hitting TP
- **Expectancy**: Average profit per trade in R
- **Total R**: Cumulative profit
- **Max DD**: Maximum drawdown in R
- **Profit Factor**: Gross profit / Gross loss

**Example top result**:
```
ATR=0.5, RR=2.0
  Expectancy: +0.07R
  Winrate: 35.7%
  Total R: +473.7R
  Max DD: 97.0R
```

This means:
- Stops are 0.5× bar range from entry
- Targets are 2× the stop distance
- Win ~36% of trades
- Make +0.07R average per trade
- Max drawdown is 97R (vs 474R profit = 20% DD)

### Recommended Next Steps

1. **Review top 3-5 configs** based on your risk tolerance
2. **Consider tradeoffs**:
   - Tighter stops (0.5× ATR) = More losses but smaller DD
   - Wider stops (1.5× ATR) = Higher WR but larger DD
   - Lower RR (1.5-2.0) = Higher WR, lower per-trade profit
   - Higher RR (2.5-3.0) = Lower WR, higher per-trade profit

3. **Choose baseline** for Phase 4 ML training:
   - Prioritize expectancy > 0 (profitable)
   - Keep max DD reasonable (<50% of total R)
   - Balance winrate vs reward

4. **Rebuild dataset** with chosen config (optional):
   - Modify `outcome.py` to use chosen ATR/RR
   - Rerun `build_dataset.py`
   - This gives ML training data with optimized labels

---

## Next Phase

Phase 4 will use this dataset to train ML models that predict `signal_side` based on P2 event context, potentially improving upon the baseline rule performance.

