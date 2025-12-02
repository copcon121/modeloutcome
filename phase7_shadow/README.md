# Phase 7 - Shadow Trading Analysis

## Overview

Tools for analyzing shadow trading logs and comparing ML filter decisions against baseline rule outcomes.

**Shadow Trading**: System runs live, logs what it would do, but executes NO real trades. Used for validation before micro-lot deployment.

---

## What is Shadow Trading?

1. **Live Data**: NinjaTrader sends real-time features to API
2. **ML Decision**: API predicts KEEP/DROP and logs decision
3. **No Execution**: No actual trades placed
4. **Analysis**: Compare logged decisions vs what actually happened

**Purpose**: Validate that live performance matches backtest expectations.

---

## Module Files

```
phase7_shadow/
├── __init__.py
├── load_shadow_log.py           # Load & normalize JSONL logs
├── analyze_shadow_vs_baseline.py # Compare ML vs baseline
└── README.md                     # This file

output/phase7_shadow/
├── shadow_summary.txt            # Analysis report
└── shadow_results.csv            # Results by model/mode
```

---

## Usage

### 1. Collect Shadow Data

**Start API Server**:
```bash
uvicorn phase5_inference.server_fastapi:app --port 8000
```

**Configure NinjaTrader** to call `/predict_quality` for each P2 event.

**Let Run**: Collect data for 1-2 weeks (recommended minimum).

### 2. Analyze Shadow Log

```bash
python -m phase7_shadow.analyze_shadow_vs_baseline
```

**Output**:
- `output/phase7_shadow/shadow_summary.txt` - Human-readable report
- `output/phase7_shadow/shadow_results.csv` - Data for analysis

---

## Shadow Log Format

Each API request creates one log entry in `shadow_trading_log.jsonl`:

```json
{
  "timestamp_server": "2025-11-28T10:30:00.123",
  "model_type": "seq_v1",
  "mode": "seq_conservative",
  "threshold": 0.8,
  "p_keep": 0.873,
  "keep": true,
  "side": 1,
  "meta": {
    "symbol": "GC",
    "timeframe": "M1",
    "event_time": "2025-11-28T10:29:00",
    "bar_index": 12345
  }
}
```

---

## Analysis Metrics

### Baseline (Rule-Only)
- All candidates from SMC rule
- Winrate, expectancy, max DD
- No ML filter

### ML-Filtered
- Only trades where `keep == true`
- Same metrics as baseline
- Compare improvement

### Key Comparisons
1. **Expectancy**: Does ML filter improve vs baseline?
2. **Max DD**: Risk reduction from filtering?
3. **Trade Count**: How selective is the filter?
4. **Consistency**: Match backtest expectations?

---

## Go-Live Criteria

Before moving from shadow → micro-lot live:

- [ ] **Duration**: At least 1-2 weeks of shadow data
- [ ] **Sample Size**: Minimum 100 predictions logged
- [ ] **Expectancy**: Shadow results ≥ 80% of backtest expectancy
- [ ] **Max DD**: Shadow DD ≤ backtest DD threshold
- [ ] **Stability**: Consistent across sessions/market conditions
- [ ] **No Errors**: API responds reliably, no crashes

**If ALL criteria met** → Proceed to micro-lot (1 micro per trade, seq_conservative mode)

**If ANY fail** → Continue shadow, investigate discrepancies

---

## Notes

### Current Limitations

- Analysis script simplified (assumes shadow log structure)
- Full outcome matching requires production timestamp alignment
- Extend matching logic for your specific NinjaTrader integration

### Future Enhancements

- Real-time dashboard for shadow monitoring
- Automated alerts if shadow metrics diverge from backtest
- Multi-model comparison (seq vs tabular in shadow)

---

**Created**: 2025-11-28  
**Version**: 1.0  
**Status**: Ready for shadow deployment
