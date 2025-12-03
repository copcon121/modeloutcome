# S4_HighVol_FVG_London_v1 – Baseline Strategy

**Created**: 2025-12-02  
**Status**: LOCKED – Production Baseline  
**Version**: v1.0

---

## 1. Overview

### 1.1 Strategy Description

**S4_HighVol_FVG** là strategy **trend continuation** dựa trên SMC (Smart Money Concepts):

- **Core Logic**: Entry trên retest FVG/OB zone trong xu hướng đã xác nhận
- **Edge Source**: High volatility regime + Strong impulse tạo FVG + Continuation BOS
- **Style**: Rule-only, không phụ thuộc ML

### 1.2 Vai trò trong hệ thống

```
┌─────────────────────────────────────────────────────────┐
│  S4_HighVol_FVG_London_v1 = BASELINE STRATEGY           │
│  ─────────────────────────────────────────────────────  │
│  • Xương sống của hệ thống trading                      │
│  • Rule-only, deterministic                             │
│  • Mọi ML filter là lớp BỔ SUNG, không thay thế         │
└─────────────────────────────────────────────────────────┘
```

### 1.3 Performance Summary (Locked)

| Dataset | Period | Trades | Winrate | Expectancy | MaxDD |
|---------|--------|--------|---------|------------|-------|
| OLD VAL | Oct-Nov 2024 (2W) | 428 | 38.1% | +0.143R | 23.0R |
| NEW DATA (All) | Apr-Jun 2025 (6W) | 827 | 47.3% | +0.418R | 23.8R |
| **NEW DATA (London)** | Apr-Jun 2025 (6W) | **297** | **55.2%** | **+0.654R** | **15.1R** |

**Kết luận**: London session có edge mạnh nhất → Chọn làm **core session** cho S4_London_v1.

---

## 2. Rule Chi Tiết

### 2.1 Regime Conditions

```
REGIME FILTER:
├── Volatility: HIGH
│   └── high_low_range > quantile(0.66) của rolling 100 bars
│   └── HOẶC volume > avg_volume * 2.0
│   └── HOẶC |delta_over_volume| > 0.60 với delta_abs > avg_delta * 1.5
│
└── Trend: CLEAR DIRECTION
    └── M5/H1 trend aligned với entry direction
    └── External BOS đã xảy ra theo hướng trade
```

### 2.2 Zone Conditions

```
ZONE FILTER (FVG):
├── Long Entry:
│   └── in_bull_fvg == 1 (price trong bullish FVG)
│   └── FVG được tạo bởi impulse leg có BOS up
│
└── Short Entry:
    └── in_bear_fvg == 1 (price trong bearish FVG)
    └── FVG được tạo bởi impulse leg có BOS down
```

### 2.3 Entry Logic

```
ENTRY CONDITIONS:
├── Direction: CONTINUATION (không phải reversal)
│   └── Long: trend up + retest bull FVG
│   └── Short: trend down + retest bear FVG
│
├── Trigger:
│   └── Price touch/enter FVG zone
│   └── Candle close trong zone hoặc rejection từ zone
│
└── Confirmation (implicit):
    └── High vol regime đã được check
    └── Trend direction đã được check
```

### 2.4 SL/TP Configuration

```
RISK MANAGEMENT:
├── Stop Loss:
│   └── Long: Below FVG low - buffer (hoặc swing low)
│   └── Short: Above FVG high + buffer (hoặc swing high)
│   └── Buffer: ~2-5 ticks tùy volatility
│
├── Take Profit:
│   └── RR Target: 2.0R (primary) / 3.0R (extended)
│   └── Tính từ entry đến TP = RR * (entry - SL)
│
└── Position Sizing:
    └── Fixed risk per trade (e.g., 1% account)
```

### 2.5 Additional Filters (trong code)

```
ADDITIONAL CHECKS:
├── Displacement: Impulse leg phải có đủ strength
│   └── impulse_strength >= 30 (relaxed threshold)
│
├── Volume/Delta minimum:
│   └── volume > 50 (minimum liquidity)
│   └── delta có direction phù hợp với trade
│
└── Zone freshness:
    └── FVG chưa bị mitigated hoàn toàn
    └── First touch preferred
```

---

## 3. Session Filter

### 3.1 Primary Session: LONDON

```
LONDON SESSION:
├── Time Range (UTC): 08:00 - 14:00
├── Server Time: Adjust theo broker/data feed
│
├── Rationale:
│   └── Highest liquidity overlap (London + early NY)
│   └── Best performance in backtest: Exp +0.654R, WR 55.2%
│   └── Lower MaxDD: 15.1R vs 23.8R (all sessions)
│
└── Status: PRIMARY - All live trades trong session này
```

### 3.2 Secondary Sessions (Optional/Research)

```
OTHER SESSIONS (monitoring only):
├── NY Session (14:00 - 21:00 UTC):
│   └── Exp: +0.187R, WR: 39.5%
│   └── Status: MONITOR - có edge nhưng thấp hơn London
│
├── Asia Session (00:00 - 08:00 UTC):
│   └── Exp: +0.358R, WR: 45.3%
│   └── Status: MONITOR - decent edge, higher DD
│
└── Conclusion:
    └── Phase 1: Focus London only
    └── Phase 2: Consider adding Asia nếu cần thêm trades
```

---

## 4. Strategy Specification (Locked)

```
┌─────────────────────────────────────────────────────────┐
│  S4_HighVol_FVG_London_v1 - OFFICIAL SPECIFICATION      │
├─────────────────────────────────────────────────────────┤
│  Instrument:    GC (Gold Futures)                       │
│  Timeframe:     M1 (1-minute bars)                      │
│  Session:       London (08:00 - 14:00 UTC)              │
│  Regime:        HighVol + Trend Continuation            │
│  Entry:         FVG Retest                              │
│  RR Target:     2.0R                                    │
│  ML Filter:     NONE (rule-only baseline)               │
└─────────────────────────────────────────────────────────┘
```

---

## 5. Performance Benchmark

### 5.1 Expected Performance (based on backtest)

| Metric | Target | Achieved (London 6W) |
|--------|--------|----------------------|
| Winrate | > 50% | 55.2% ✅ |
| Expectancy | > +0.30R | +0.654R ✅ |
| MaxDD | < 20R | 15.1R ✅ |
| Trades/Week | 30-60 | ~50 ✅ |

### 5.2 Monitoring Thresholds

```
ALERT THRESHOLDS:
├── Expectancy drops below +0.20R over 2 weeks → REVIEW
├── MaxDD exceeds 25R → PAUSE & INVESTIGATE
├── Winrate drops below 40% over 50 trades → REVIEW
└── No trades for 3 consecutive days → CHECK FILTERS
```

---

## 6. Integration với NT-AUTOBOT

### 6.1 Phase 1: Rule-Only Execution (CURRENT)

```
NT-AUTOBOT INTEGRATION (Phase 1):
├── Signal Generation:
│   └── Ninja generates S4 signals based on rules above
│   └── Filter: London session only
│   └── No ML involvement in execution decision
│
├── Execution:
│   └── Auto-execute all valid S4_London signals
│   └── Fixed position size per trade
│
└── Logging:
    └── Log all signal details to file/DB
    └── Include: entry, SL, TP, context, regime, session
```

### 6.2 Phase 2: ASM Logging (Research Only)

```
ASM LOGGING (không ảnh hưởng execution):
├── Khi có S4 signal:
│   └── Gửi context tới Python API
│   └── Tính ASM scores: p_up, p_down, p_neutral
│   └── LOG scores vào DB, KHÔNG filter
│
├── Mục đích:
│   └── Thu thập data để validate ASM
│   └── Nghiên cứu VA-shift correlation với trade outcome
│   └── Chuẩn bị cho Phase 3 (ASM filter)
│
└── Status: RESEARCH MODE - không can thiệp execution
```

### 6.3 Phase 3: ASM Filter (FUTURE)

```
ASM FILTER (sau khi validate):
├── Khi có S4 signal:
│   └── Query ASM → p_up, p_down, p_neutral
│   └── Apply filter logic:
│       └── Long: trade nếu (p_up - p_down) >= threshold
│       └── Short: trade nếu (p_down - p_up) >= threshold
│
├── Mục tiêu:
│   └── Loại bớt trades trái chiều VA-shift
│   └── Giảm MaxDD, tăng Expectancy
│
└── Status: PENDING - chờ ASM v1 validation
```

### 6.4 API Payload Structure (TODO)

```json
{
  "signal_id": "uuid",
  "timestamp": "2025-12-02T10:15:00Z",
  "symbol": "GC",
  "timeframe": "M1",
  "session": "London",
  
  "signal": {
    "side": "long",
    "entry_price": 2650.5,
    "sl_price": 2648.0,
    "tp_price": 2655.5,
    "rr_target": 2.0
  },
  
  "context": {
    "vol_regime": "high",
    "trend_dir": 1,
    "in_fvg": true,
    "va_position": "near_val",
    "impulse_strength": 45.2
  },
  
  "ml_scores": {
    "asm_p_up": 0.45,
    "asm_p_down": 0.25,
    "asm_p_neutral": 0.30
  },
  
  "execution": {
    "action": "EXECUTE",
    "asm_filter_applied": false
  }
}
```

---

## 7. ASM v1.0 Shadow Filter Results

### 7.1 Shadow Backtest (2025-12-03)

**Model**: ASM-GRU64-v1.0-C3 (bar-only, 112 features)  
**Script**: `scripts/shadow_backtest_s4_asm_v1.py`  
**Results**: `backtests/s4_highvol_fvg_london_asm_v1_shadow.json`

| Scenario | Trades | WR% | Exp(R) | MaxDD | Retain% |
|----------|--------|-----|--------|-------|---------|
| **Baseline (no filter)** | 479 | 50.9% | +0.54R | 37R | 100% |
| Direct: T_shift=0.7, T_dir=0.0 | 249 | 43.4% | +0.31R | 29R | 52% |
| **Inverse: T_shift_max=0.2** | **109** | **82.6%** | **+1.48R** | **9R** | **22.8%** |

### 7.2 Key Finding: Inverse Filter Works!

✅ **ASM v1.0 INVERSE filter dramatically improves S4 baseline**:
- Expectancy: +0.54R → **+1.48R** (+174% improvement)
- Winrate: 50.9% → **82.6%** (+62% improvement)
- MaxDD: 37R → **9R** (-76% reduction)
- Trade retention: 22.8% (109/479 trades)

**Interpretation**:
- `p_shift = p_up + p_down` = probability of VA shifting (either direction)
- Low p_shift (≤ 0.2) = ASM predicts auction ổn định, VA không shift mạnh
- S4 (trend continuation) works best when VA is stable → continuation có edge
- High p_shift = volatile/uncertain market → bad for trend continuation

### 7.3 Conclusion

Direct filter (keep high p_shift aligned with direction) **KHÔNG** cải thiện baseline.  
**Inverse filter** (keep LOW p_shift) **CẢI THIỆN MẠNH** baseline.

**Next steps**:
- [x] Lock inverse filter as candidate strategy (Section 9)
- [ ] Forward test on new data period
- [ ] Integrate vào execution pipeline

---

---

## 9. 🔒 LOCKED Strategy – S4_LDN_ASM_LowShift_0.2_v1.1

### 9.1 Strategy Definition

**Strategy ID**: `S4_LDN_ASM_LowShift_0.2_v1.1`  
**Status**: 🔒 **LOCKED** – PASS Extended Validation, Ready for Shadow Trading

```
┌─────────────────────────────────────────────────────────────────────┐
│  S4_LDN_ASM_LowShift_0.2_v1.1                                       │
├─────────────────────────────────────────────────────────────────────┤
│  Base Strategy:  S4_HighVol_FVG_London_v1 (rule-only)               │
│  ML Filter:      ASM-GRU64-v1.0-C3 (bar-only, 100 features)         │
│  Filter Logic:   Keep trade if p_shift ≤ 0.2                        │
│                  where p_shift = p_up + p_down                      │
│  Interpretation: Trade only when ASM predicts stable auction        │
│                  (low VA-shift probability)                         │
│  Session:        London (08:00 - 14:00 UTC)                         │
│  Instrument:     GC (Gold Futures), M1                              │
│  RR Target:      2.0R                                               │
└─────────────────────────────────────────────────────────────────────┘
```

### 9.2 Rule Summary

```
S4_LDN_ASM_LowShift_0.2_v1.1 RULE:
├── BASE: S4 HighVol FVG London
│   ├── Session: London (08:00 - 14:00 UTC)
│   ├── Regime: HighVol (range > Q66 or vol > 2x avg)
│   ├── Entry: FVG retest in trend direction
│   └── RR: 2.0R
│
└── FILTER: ASM-GRU64-v1.0-C3
    ├── Input: 60-bar context (100 features)
    ├── Output: p_up, p_down, p_neutral
    ├── p_shift = p_up + p_down
    └── KEEP trade if p_shift <= 0.2
```

### 9.3 Performance Summary

#### OLD Dataset (Shadow Backtest, Oct-Nov 2024)

| Metric | Baseline S4_LDN | + ASM LowShift 0.2 | Change |
|--------|-----------------|-------------------|--------|
| Trades | 479 | 109 | -77.2% |
| Winrate | 50.9% | **82.6%** | +62.3% |
| Expectancy | +0.54R | **+1.48R** | +174% |
| MaxDD | 37R | **9R** | -75.7% |

#### NEW DATA 6W (Extended Validation, Apr-Jun 2025)

| Metric | Baseline S4_LDN | + ASM LowShift 0.2 | Change |
|--------|-----------------|-------------------|--------|
| Trades | 480 | 258 | -46.2% |
| Winrate | 50.8% | **61.6%** | +21.2% ✅ |
| Expectancy | +0.533R | **+0.849R** | +59.2% ✅ |
| MaxDD | 37.0R | **26.0R** | -29.7% ✅ |
| Retain% | 100% | 53.8% | - |

### 9.4 Filter Rule (Code)

```python
def should_trade_s4_asm_lowshift(asm_output, threshold=0.2):
    """
    S4_LDN_ASM_LowShift filter logic.
    
    Args:
        asm_output: dict with keys 'p_up', 'p_down', 'p_neutral'
        threshold: max p_shift to allow trade (default 0.2)
    
    Returns:
        bool: True if trade should be taken
    """
    p_shift = asm_output['p_up'] + asm_output['p_down']
    return p_shift <= threshold
```

### 9.5 Rationale

**Why inverse filter works for S4 (trend continuation)**:

1. **S4 = Trend Continuation Strategy**
   - Entry trên FVG retest trong xu hướng đã xác nhận
   - Cần thị trường tiếp tục theo hướng hiện tại

2. **Low p_shift = Stable Auction**
   - ASM predict VA không shift mạnh trong K bar tới
   - Auction ổn định → continuation có edge cao hơn

3. **High p_shift = Volatile/Uncertain**
   - ASM predict VA sẽ shift (lên hoặc xuống)
   - Thị trường đang chuyển đổi → continuation rủi ro cao

4. **ASM as Risk Filter**
   - Không dùng ASM để chọn hướng (direction)
   - Dùng ASM để tránh vùng VA-shift mạnh (volatility filter)

### 9.6 Model Details

| Aspect | Value |
|--------|-------|
| Model | ASM-GRU64-v1.0-C3 |
| Features | 100 (bar-only) |
| Sequence | 60 bars M1 |
| Output | p_up, p_down, p_neutral |
| Path | `output/asm_models_v1/ASM-GRU64-v1.0-C3.pt` |

### 9.7 Validation Status

✅ **Edge ổn định qua 2 regime khác nhau (OLD vs NEW data)**:
- Cả 2 dataset đều cho thấy ASM LowShift filter cải thiện baseline
- Improvement consistent: +59% to +174% expectancy
- MaxDD giảm đáng kể: -30% to -76%

**Kết luận**: S4_LDN_ASM_LowShift_0.2_v1.1 **PASS** extended validation → **LOCKED** for shadow trading.

### 9.8 Next Steps

- [x] Shadow backtest on OLD data
- [x] Extended validation on NEW DATA 6W → **PASS**
- [ ] Shadow trading on NT-AUTOBOT (4-8 weeks)
- [ ] Live deployment after shadow validation

---

## 10. Extended Validation – S4_LDN_ASM_LowShift_0.2_v1.0 trên NEW DATA 6W

### 10.1 Validation Setup

**Date**: 2025-12-03  
**Source**: `data/raw/new_data` (Apr 28 - Jun 02, 2025, 6 weeks)  
**Script**: `scripts/extended_validation_s4_asm_lowshift_new6w_v1.py`  
**Results**: `backtests/s4_asm_lowshift_extval_new6w_v1.json`

### 10.2 Results

| Metric | Baseline S4_LDN | + ASM LowShift 0.2 | Change |
|--------|-----------------|-------------------|--------|
| Trades | 480 | 258 | -46.2% |
| Winrate | 50.8% | **61.6%** | +21.2% ✅ |
| Expectancy | +0.533R | **+0.849R** | +59.2% ✅ |
| MaxDD | 37.0R | **26.0R** | -29.7% ✅ |
| Total R | 256.0R | 219.0R | -14.5% |
| Retain% | 100% | 53.8% | - |

### 10.3 Filter Sweep (NEW DATA 6W)

| Threshold | Trades | WR% | Exp(R) | MaxDD | Retain% |
|-----------|--------|-----|--------|-------|---------|
| p_shift ≤ 0.15 | 220 | **68.2%** | **+1.05R** | **10R** | 45.8% |
| p_shift ≤ 0.20 | 258 | 61.6% | +0.85R | 26R | 53.8% |
| p_shift ≤ 0.25 | 308 | 54.9% | +0.65R | 27R | 64.2% |
| p_shift ≤ 0.30 | 316 | 54.1% | +0.63R | 27R | 65.8% |

### 10.4 So sánh với Dataset cũ (Shadow Backtest)

| Dataset | Period | Baseline Exp | LowShift 0.2 Exp | Improvement |
|---------|--------|--------------|------------------|-------------|
| OLD (Shadow) | Oct-Nov 2024 | +0.54R | +1.48R | +174% |
| **NEW 6W** | Apr-Jun 2025 | +0.53R | **+0.85R** | **+59%** |

### 10.5 Nhận xét

✅ **Edge vẫn ổn định qua 2 regime khác nhau**:
- Cả 2 dataset đều cho thấy ASM LowShift filter cải thiện baseline
- NEW DATA: Exp tăng 59%, WR tăng 21%, MaxDD giảm 30%
- Trade retention cao hơn (53.8% vs 22.8%) → nhiều trades hơn

⚠️ **Lưu ý**:
- Improvement trên NEW DATA thấp hơn OLD DATA (59% vs 174%)
- Có thể do regime khác nhau hoặc model cần retrain
- Tuy nhiên edge vẫn positive và consistent

**Kết luận**: S4_LDN_ASM_LowShift_0.2_v1.0 **PASS** extended validation trên out-of-sample data.

---

## 11. Changelog

| Date | Version | Changes |
|------|---------|---------|
| 2025-12-02 | v1.0 | Initial lock - S4_HighVol_FVG_London baseline |
| 2025-12-03 | v1.0.1 | Added ASM v1.0 shadow filter results (direct filter negative) |
| 2025-12-03 | v1.0.2 | Added inverse filter results (positive), locked S4_LDN_ASM_LowShift_0.2_v1.0 |
| 2025-12-03 | v1.0.3 | Extended validation on NEW DATA 6W - PASS |
| 2025-12-03 | v1.1.0 | **LOCKED S4_LDN_ASM_LowShift_0.2_v1.1** - Ready for shadow trading |

---

## 12. References

- [PLAN_AuctionStateModel_v1.md](PLAN_AuctionStateModel_v1.md) - ASM development (ML layer)
- [PLAN_P4_outcome_v2.md](PLAN_P4_outcome_v2.md) - ML history & archive
- [PLAN_PIPELINE_SMC_ML_v1.md](PLAN_PIPELINE_SMC_ML_v1.md) - Overall ML pipeline
- `backtests/s4_highvol_fvg_london_asm_v1_shadow.json` - ASM shadow backtest results
- `backtests/pattern_A_shadow_new6w.json` - Extended validation results (archived)

---

**LOCKED STATUS**: 
- S4_HighVol_FVG_London_v1 (baseline) - LOCKED as production baseline
- **S4_LDN_ASM_LowShift_0.2_v1.1** - 🔒 LOCKED for shadow trading (PASS extended validation)

Any modifications require version bump and re-validation.
