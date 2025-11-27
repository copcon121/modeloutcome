# 🚧 SMC 2-Layer Refactor Status Report

**Date**: 2025-11-26
**Status**: ⚠️ **WORK IN PROGRESS** - 60% Complete
**Branch**: `claude/feature-engineering-layer-2-01Gg9TijgmkFJgjBCYCLNiCv`
**Commit**: `f960456`

---

## 📊 Progress Summary

### ✅ Completed (60%)

1. **SMCConfig** (`smc/config.py`) ✓
   - Internal/External swing parameters
   - Wave 5 (5 ticks) / Wave 50 (50 ticks) configuration
   - Presets for GC, ES, NQ
   - Helper methods: `ticks_to_price()`, `price_to_ticks()`

2. **2-Layer Swing Detection** (`smc/swing.py`) ✓
   - `InternalSwingState` / `ExternalSwingState` dataclasses
   - `InternalSwingDetector` (wave 5 logic)
   - `ExternalSwingDetector` (wave 50 logic built on internal)
   - Fractal pivot detection (backward compatible)

3. **SMC Engine** (`smc/smc_engine.py`) ✓
   - `SMCEngine` class orchestrates both layers
   - `SMCState` dataclass with int_*/ext_* fields
   - BOS/CHoCH detection for internal & external

4. **Debug Tool** (`smc/debug_smc.py`) ✓
   - Self-check utility `debug_smc_on_sample()`
   - Prints swing counts, BOS/CHoCH events
   - Invariant checking (ext/int ratio)

### ⚠️ Known Issues

**CRITICAL BUG**: Internal swing confirmation too strict

```
Test Results (300 bars GC M1):
❌ Internal swings: 0 highs, 0 lows (EXPECTED: ~40-60)
❌ External swings: 0 highs, 0 lows (Correct, built on internal)
⚠️  int_BOS: 19 up, 7 down (Invalid without swings)
⚠️  int_CHoCH: 18 up, 35 down (Invalid without swings)
```

**Root Cause**:
- `_confirm_internal_swing_high/low()` logic too restrictive
- Requires minimum 5 ticks move from LAST swing
- Not accumulating fractals properly

### 🔧 Pending Tasks (40%)

5. **Fix Swing Confirmation Logic** ⏳
   - Relax `min_int_move_ticks` requirement
   - Consider absolute move vs. move from last swing
   - Add debug logging to understand rejection reasons

6. **Update Context Manager** ⏳
   - Integrate `SMCEngine` into `ContextManager`
   - Replace old 1-layer swing with 2-layer

7. **Update Schema** ⏳
   - Add int_*/ext_* fields to `FeatureBar`
   - Backward compatibility with old features

8. **Full Pipeline Test** ⏳
   - End-to-end test with 1378 bars
   - Verify feature extraction works
   - Compare with NinjaTrader indicator

---

## 🏗️ Architecture Overview

### Current Implementation

```
RawBar → SMCEngine → SMCState
           │
           ├─ InternalSwingDetector (wave 5)
           │    ├─ Fractal Detection
           │    ├─ Swing Confirmation (❌ BUG HERE)
           │    └─ Trend Tracking
           │
           └─ ExternalSwingDetector (wave 50)
                ├─ Built on Internal Swings
                ├─ Wave 50 Confirmation
                └─ Trend Tracking
```

### SMCState Output

```python
@dataclass
class SMCState:
    # Internal (wave 5)
    int_trend_dir: int              # +1, -1, 0
    int_bos_up: bool
    int_bos_down: bool
    int_choch_up: bool
    int_choch_down: bool
    int_swing_high_price: float
    int_swing_low_price: float
    int_swing_high_bars_ago: int
    int_swing_low_bars_ago: int

    # External (wave 50)
    ext_trend_dir: int
    ext_bos_up: bool
    ext_bos_down: bool
    ext_choch_up: bool
    ext_choch_down: bool
    ext_swing_high_price: float
    ext_swing_low_price: float
    ext_swing_high_bars_ago: int
    ext_swing_low_bars_ago: int
```

---

## 🐛 Bug Analysis & Fix Plan

### Issue: Zero Swings Detected

**Current Logic** (BROKEN):
```python
def _confirm_internal_swing_high(self, bars, idx, price):
    # Check move from LAST swing low
    if self.state.swing_low_idx >= 0:
        move_ticks = self.config.price_to_ticks(price - self.state.swing_low_price)
        if move_ticks < self.config.min_int_move_ticks:  # ❌ Too strict!
            return False
```

**Problem**:
- First swing low never gets confirmed (no previous swing)
- Subsequent swings rejected if move < 5 ticks from LAST swing
- Should check move from price at time of fractal, not last confirmed swing

**Proposed Fix**:
```python
def _confirm_internal_swing_high(self, bars, idx, price):
    # Option 1: Always confirm first swing
    if self.state.swing_low_idx < 0:
        return True  # First swing, always accept

    # Option 2: Check move from price at fractal formation
    lookback_price = bars[idx - self.config.fractal_left].close
    move_ticks = self.config.price_to_ticks(abs(price - lookback_price))
    if move_ticks < self.config.min_int_move_ticks:
        return False

    # Option 3: Use ATR-based threshold instead of fixed ticks
    atr = compute_atr(bars[max(0, idx-14):idx+1])
    if abs(price - lookback_price) < atr * 0.5:
        return False

    return True
```

### Recommended Fix Steps

1. **Add Debug Logging**
   ```python
   # In _confirm_internal_swing_high/low
   logger.debug(f"Bar {idx}: Fractal @ {price:.2f}, move={move_ticks} ticks, threshold={self.config.min_int_move_ticks}")
   ```

2. **Relax First Swing Rule**
   - Always confirm first fractal as swing (bootstrap)

3. **Change Move Calculation**
   - Move from local range, not from last confirmed swing

4. **Test Incrementally**
   - Run debug tool after each fix
   - Target: 40-60 internal swings in 300 bars

---

## 📝 Files Created/Modified

### New Files
- `src/layer2_feature_engine/smc/config.py` (202 lines)
- `src/layer2_feature_engine/smc/smc_engine.py` (219 lines)
- `src/layer2_feature_engine/smc/debug_smc.py` (165 lines)

### Modified Files
- `src/layer2_feature_engine/smc/swing.py` (457 lines, +406 new)

### Total Code Added
~1,000 lines of new infrastructure

---

## 🧪 Testing

### Debug Tool Usage

```bash
python -m src.layer2_feature_engine.smc.debug_smc
```

**Current Output** (BROKEN):
```
Total bars processed: 300

Internal Structure (wave 5):
  Swing Highs:  0  ❌ (EXPECTED: ~50)
  Swing Lows:   0  ❌ (EXPECTED: ~50)
  BOS Up:       19 ⚠️  (Invalid without swings)
  BOS Down:     7  ⚠️
  CHoCH Up:     18 ⚠️
  CHoCH Down:   35 ⚠️

External Structure (wave 50):
  Swing Highs:  0  ✓ (Correct, no internal swings to build on)
  Swing Lows:   0  ✓
```

**Expected Output** (after fix):
```
Internal Structure (wave 5):
  Swing Highs:  45-55  ✓
  Swing Lows:   45-55  ✓
  BOS Up:       10-20  ✓
  BOS Down:     10-20  ✓
  CHoCH Up:     3-8    ✓
  CHoCH Down:   3-8    ✓

External Structure (wave 50):
  Swing Highs:  5-10   ✓
  Swing Lows:   5-10   ✓
  BOS Up:       2-5    ✓
  BOS Down:     2-5    ✓
```

---

## 🎯 Next Steps (Priority Order)

### Immediate (1-2 hours)

1. **Fix Internal Swing Confirmation**
   - [ ] Add debug logging to `_confirm_internal_swing_high/low()`
   - [ ] Change move calculation logic
   - [ ] Test with `debug_smc.py`
   - [ ] Verify 40-60 swings in 300 bars

2. **Verify BOS/CHoCH**
   - [ ] Ensure BOS/CHoCH only fire with confirmed swings
   - [ ] Check trend_dir updates correctly

### Short-term (2-4 hours)

3. **Integrate into ContextManager**
   - [ ] Replace old `detect_swings()` with `SMCEngine`
   - [ ] Update `_build_smc_structure()` method
   - [ ] Pass config from initialization

4. **Update FeatureBar Schema**
   - [ ] Add int_* fields (9 features)
   - [ ] Add ext_* fields (9 features)
   - [ ] Total: +18 SMC features (78 → 96 features)

5. **Full Pipeline Test**
   - [ ] Run `test_full_pipeline.py`
   - [ ] Verify [1319, 60, 96] output shape
   - [ ] Check for NaN/Inf values

### Long-term (4-8 hours)

6. **Optional: MTF M5 Bias**
   - [ ] Aggregate M1 → M5
   - [ ] Run SMCEngine on M5
   - [ ] Add htf_* fields to FeatureBar

7. **Documentation**
   - [ ] Update `SMC_VOLUME_PROFILE_LOGIC.md`
   - [ ] Add 2-layer swing explanation
   - [ ] Include wave 5 / wave 50 diagrams

8. **Validation**
   - [ ] Compare with NinjaTrader indicator
   - [ ] Visual inspection of swing points
   - [ ] Backtest on more data

---

## 💡 Recommendations

### For User

1. **Immediate Action**
   - Fix swing confirmation bug (see "Proposed Fix" above)
   - Test incrementally with debug tool

2. **Configuration Tuning**
   - May need to adjust `min_int_move_ticks` from 5 → 3 or 4
   - Test different `min_bars_between_swings` values
   - Consider using ATR-based thresholds instead of fixed ticks

3. **Validation Strategy**
   - Load same bars in NinjaTrader
   - Compare swing counts visually
   - Adjust config until counts match ±20%

### Code Quality

**Strengths**:
- ✓ Clean architecture (separation of concerns)
- ✓ Type hints throughout
- ✓ Backward compatible
- ✓ Self-documenting code
- ✓ Debug tool included

**Needs Improvement**:
- ⚠️ Swing confirmation logic needs refinement
- ⚠️ Missing unit tests
- ⚠️ No logging in production code
- ⚠️ Hard-coded parameters

---

## 📚 References

### Code Files
- `src/layer2_feature_engine/smc/config.py`
- `src/layer2_feature_engine/smc/swing.py`
- `src/layer2_feature_engine/smc/smc_engine.py`
- `src/layer2_feature_engine/smc/debug_smc.py`

### Documentation
- `docs/SMC_VOLUME_PROFILE_LOGIC.md` (needs update)
- `PHASE2_COMPLETION_REPORT.md`

### Related Issues
- Wave 5 / Wave 50 confirmation thresholds
- Multi-timeframe bias (MTF M5)
- Integration with existing pipeline

---

## ⚠️ Important Notes

1. **DO NOT MERGE** until swing confirmation bug is fixed
2. **DO NOT USE** in production trading yet
3. **Test thoroughly** with debug tool before integration
4. **Validate** against NinjaTrader indicator
5. **Document** any config changes you make

---

**Status**: 🟡 Work in Progress - Awaiting Bug Fix

**Prepared by**: Claude (Anthropic)
**Date**: 2025-11-26
**Commit**: `f960456`
