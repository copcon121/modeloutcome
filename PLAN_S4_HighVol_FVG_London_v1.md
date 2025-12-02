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

## 7. Changelog

| Date | Version | Changes |
|------|---------|---------|
| 2025-12-02 | v1.0 | Initial lock - S4_HighVol_FVG_London baseline |

---

## 8. References

- [PLAN_AuctionStateModel_v1.md](PLAN_AuctionStateModel_v1.md) - ASM development (ML layer)
- [PLAN_P4_outcome_v2.md](PLAN_P4_outcome_v2.md) - ML history & archive
- [PLAN_PIPELINE_SMC_ML_v1.md](PLAN_PIPELINE_SMC_ML_v1.md) - Overall ML pipeline
- `backtests/pattern_A_shadow_new6w.json` - Extended validation results
- `scripts/extended_validation_pattern_a_v2.py` - Validation script

---

**LOCKED STATUS**: This strategy specification is locked as production baseline.  
Any modifications require version bump and re-validation.
