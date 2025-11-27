# 5-Day Export Validation Checklist

## Pre-Export (NinjaTrader)

- [ ] **Date Range**: Select 5 consecutive trading days
- [ ] **Instrument**: GC 12-25 (or current front month)
- [ ] **Timeframe**: M1 (1 minute)
- [ ] **Indicator**: SMCDeepSeekExporter_Enhanced
- [ ] **Verify Export Fields**:
  - `bar.volume` ✓
  - `bar.vwap_daily` ✓
  - `bar.delta` ✓
  - `tick_features.*` ✓

**Expected Bars**: ~7,200 (1440/day * 5 days)

---

## Post-Export Validation

### 1. File Check
```bash
# File location
data/raw/<filename>.jsonl

# Check size
# Expected: ~50-100 MB for 7,200 bars

# Check line count
Get-Content data/raw/<filename>.jsonl | Measure-Object -Line
# Expected: ~7,200 lines
```

### 2. Run Feature Pipeline
```python
# test_5day_validation.py
from layer2_feature_engine_v2.dataset_builder import DatasetBuilder
from layer2_feature_engine_v2.config import GC_M1_SMC_CONFIG
from layer2_feature_engine_v2.event_filter import EventFilter

builder = DatasetBuilder(GC_M1_SMC_CONFIG, tick_size=0.1)

# Load
raw_bars = builder.load_jsonl("data/raw/<filename>.jsonl")
print(f"Loaded: {len(raw_bars)} bars")

# Build features
feature_bars = builder.build_features(raw_bars)
print(f"Features: {len(feature_bars)} bars")

# Filter
event_filter = EventFilter()
flags = event_filter.compute_flags(feature_bars)
stats = event_filter.get_filter_stats(flags)

print("\nFilter Statistics:")
for phase, data in stats.items():
    if isinstance(data, dict):
        print(f"  {phase}: {data['count']} bars ({data['pct']:.1f}%)")
```

### 3. Validation Targets

**Total Bars**:
- [ ] 7,000 - 7,500 bars (acceptable range)

**Filter Statistics**:
- [ ] P1 (Strict): 0.5-2% (~35-150 bars)
- [ ] P2 (Moderate): 20-50% (~1,500-3,500 bars) ← **Target**
- [ ] P3 (Loose): 80-95% (~5,500-6,800 bars)

**Volume Profile**:
- [ ] 15 profiles (5 days * 3 sessions)
- [ ] POC values reasonable (around price range)
- [ ] Total volume > 0 for all profiles

**VWAP**:
- [ ] vwap_daily present in all bars
- [ ] Values within price range
- [ ] Distance calculations working

**SMC Features**:
- [ ] BOS/CHoCH detected (check P1 count)
- [ ] FVG/OB zones created
- [ ] PD zones calculated

### 4. Manual Inspection

**Check CSV Output**:
```python
# Export CSV for inspection
builder.export_csv(feature_bars, "output/features_5day.csv")
```

Open `features_5day.csv`:
- [ ] Scroll to bars with `int_bos_up=1` or `ext_bos_down=1`
- [ ] Verify these are at turning points
- [ ] Check `in_bull_fvg` / `in_bear_fvg` flags
- [ ] Verify `vwap_daily` has values
- [ ] Check `vp_poc_price` populated

### 5. Build Sequences

```python
sequences, indices = builder.build_sequences(feature_bars, window_size=60)
print(f"Sequences shape: {sequences.shape}")
# Expected: [~3,100, 60, 62]
```

**Targets**:
- [ ] Shape: `[N, 60, 62]`
- [ ] N ≈ (total_bars - 60) for stride=1
- [ ] No NaN values
- [ ] Reasonable value ranges

---

## Red Flags (Issues to Investigate)

⚠️ **P2 filter < 15%**: Thresholds too strict
⚠️ **P2 filter > 60%**: Thresholds too loose  
⚠️ **P1 = 0 bars**: No major BOS/CHoCH (check 5-day range includes volatility)
⚠️ **VWAP = 0**: Export missing vwap_daily field
⚠️ **Volume = 0**: Export missing volume field
⚠️ **All sequences identical**: Context manager not updating

---

## Success Criteria

✅ **All checks passed**  
✅ **P2 filter: 20-50%**  
✅ **CSV inspection looks good**  
✅ **Sequences shape correct**  
✅ **No errors in pipeline**

→ **READY FOR 30-DAY EXPORT**

---

## If Issues Found

1. Check NinjaTrader export settings
2. Verify indicator version (SMCDeepSeekExporter_Enhanced)
3. Inspect raw JSONL structure
4. Adjust filter thresholds if needed
5. Re-test with smaller sample (1 day)

---

## Next: 30-Day Export

Once 5-day validated:
- Export 30 consecutive trading days
- Expected: ~43,000 bars
- P2 filtered: ~19,000 bars
- Training sequences: ~18,940
- Validation split: 80/20
- **Ready for Phase 3: Labeling**
