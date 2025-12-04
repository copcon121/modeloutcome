# STATE-ENC v1 — Feature Specification Document

> **Version**: 1.0.0  
> **Last Updated**: 2024-12-04  
> **Status**: OFFICIAL STANDARD  
> **Scope**: ASM v2, S4_LDN, STATE-ENC v1

---

## Table of Contents

1. [Overview](#1-overview)
2. [Full Feature List Per Bar](#2-full-feature-list-per-bar)
   - 2.1 [Group 1: OHLCV & Shape](#21-group-1-ohlcv--shape)
   - 2.2 [Group 2: Delta & Tick Microstructure](#22-group-2-delta--tick-microstructure)
   - 2.3 [Group 3: SMC Structure - External](#23-group-3-smc-structure---external)
   - 2.4 [Group 4: SMC Structure - Internal](#24-group-4-smc-structure---internal)
   - 2.5 [Group 5: Liquidity & Sweep](#25-group-5-liquidity--sweep)
   - 2.6 [Group 6: OB & FVG Proximity](#26-group-6-ob--fvg-proximity)
   - 2.7 [Group 7: VA / Auction / Session](#27-group-7-va--auction--session)
   - 2.8 [Group 8: ASM Regime Hint](#28-group-8-asm-regime-hint)
3. [Full Sample Schema](#3-full-sample-schema)
4. [Exact Feature Order](#4-exact-feature-order)
5. [Feature Grouping by Type](#5-feature-grouping-by-type)
6. [Normalization Rules](#6-normalization-rules)
7. [Field Mapping Table](#7-field-mapping-table)
8. [Data Validation Rules](#8-data-validation-rules)
9. [Aux Target Definition](#9-aux-target-definition)

---

## 1. Overview

### Purpose

Document này định nghĩa **TIÊU CHUẨN CHÍNH THỨC** cho tất cả features được sử dụng trong:
- **STATE-ENC v1**: Market State Encoder
- **ASM v2**: Auction State Model
- **S4_LDN**: Strategy 4 London Session

### Data Flow

```
┌─────────────────────┐
│  NinjaTrader 8      │
│  Raw Exporter       │
│  (OHLCV, Vol, Delta)│
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Python SMC Core    │
│  (BOS, CHoCH, OB,   │
│   FVG, VA, Swing)   │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  bars_enhanced.jsonl│
│  (Bar-level JSONL)  │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  STATE-ENC Dataset  │
│  Builder            │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  encoder_dataset    │
│  .jsonl             │
│  (Sequence samples) │
└─────────────────────┘
```

### Tensor Shape

- **Input**: `[B, N, D]` where:
  - `B` = Batch size
  - `N` = Sequence length (default: 128 bars)
  - `D` = Feature dimension (95 features)

---

## 2. Full Feature List Per Bar

### 2.1 Group 1: OHLCV & Shape

**Total: 15 features**

| # | Feature Name | Data Type | Description | Formula | Source |
|---|--------------|-----------|-------------|---------|--------|
| 1 | `o` | float32 | Open price (normalized) | Raw open price | raw_exporter |
| 2 | `h` | float32 | High price (normalized) | Raw high price | raw_exporter |
| 3 | `l` | float32 | Low price (normalized) | Raw low price | raw_exporter |
| 4 | `c` | float32 | Close price (normalized) | Raw close price | raw_exporter |
| 5 | `hl_range` | float32 | Bar range | `h - l` | derived |
| 6 | `body` | float32 | Candle body | `c - o` | derived |
| 7 | `upper_wick` | float32 | Upper wick length | `h - max(o, c)` | derived |
| 8 | `lower_wick` | float32 | Lower wick length | `min(o, c) - l` | derived |
| 9 | `bar_type` | int (cat) | Bar classification | 0=doji, 1=bull, 2=bear | derived |
| 10 | `volume` | float32 | Total volume | Raw volume | raw_exporter |
| 11 | `volume_vs_session_avg` | float32 | Volume relative to session | `volume / session_avg_volume` | derived |
| 12 | `volume_zscore` | float32 | Volume z-score | `(volume - mean) / std` | derived |
| 13 | `atr_m1_14` | float32 | ATR 14-period M1 | Standard ATR calculation | smc_python |
| 14 | `true_range` | float32 | True range | `max(h-l, |h-prev_c|, |l-prev_c|)` | derived |
| 15 | `range_vs_session_avg` | float32 | Range relative to session | `hl_range / session_avg_range` | derived |

**Bar Type Classification Logic:**
```python
if hl_range > 0 and abs(body) / hl_range < 0.1:
    bar_type = 0  # doji/small body
elif body > 0:
    bar_type = 1  # bullish
else:
    bar_type = 2  # bearish
```

---

### 2.2 Group 2: Delta & Tick Microstructure

**Total: 13 features**

| # | Feature Name | Data Type | Description | Formula | Source |
|---|--------------|-----------|-------------|---------|--------|
| 16 | `delta` | float32 | Delta (buy - sell volume) | `buy_volume - sell_volume` | raw_exporter |
| 17 | `delta_abs` | float32 | Absolute delta | `abs(delta)` | derived |
| 18 | `delta_sign` | int (cat) | Delta direction | -1, 0, +1 | derived |
| 19 | `delta_vs_volume` | float32 | Delta normalized by volume | `delta / max(volume, 1)` | derived |
| 20 | `cum_delta_session` | float32 | Cumulative delta in session | Running sum of delta | smc_python |
| 21 | `delta_zscore_session` | float32 | Delta z-score in session | `(delta - session_mean) / session_std` | derived |
| 22 | `tick_count` | float32 | Number of ticks in bar | Raw tick count | raw_exporter |
| 23 | `tick_speed` | float32 | Ticks per second | `tick_count / bar_duration_seconds` | derived |
| 24 | `buy_volume` | float32 | Buy volume (at ask) | Raw buy volume | raw_exporter |
| 25 | `sell_volume` | float32 | Sell volume (at bid) | Raw sell volume | raw_exporter |
| 26 | `buy_ratio` | float32 | Buy volume ratio | `buy_volume / max(volume, 1)` | derived |
| 27 | `sell_ratio` | float32 | Sell volume ratio | `sell_volume / max(volume, 1)` | derived |
| 28 | `imbalance_buy_sell` | float32 | Buy/sell imbalance | `(buy_volume - sell_volume) / max(volume, 1)` | derived |

**Delta Sign Logic:**
```python
if delta > 0:
    delta_sign = 1
elif delta < 0:
    delta_sign = -1
else:
    delta_sign = 0
```

---

### 2.3 Group 3: SMC Structure - External

**Total: 16 features**

| # | Feature Name | Data Type | Description | Formula | Source |
|---|--------------|-----------|-------------|---------|--------|
| 29 | `ext_trend_dir` | int (cat) | External trend direction | -1=down, 0=neutral, 1=up | smc_python |
| 30 | `int_trend_dir` | int (cat) | Internal trend direction | -1=down, 0=neutral, 1=up | smc_python |
| 31 | `ext_bos_up` | binary | External BOS up occurred | 0 or 1 | smc_python |
| 32 | `ext_bos_down` | binary | External BOS down occurred | 0 or 1 | smc_python |
| 33 | `ext_choch_up` | binary | External CHoCH up occurred | 0 or 1 | smc_python |
| 34 | `ext_choch_down` | binary | External CHoCH down occurred | 0 or 1 | smc_python |
| 35 | `bars_since_last_ext_bos` | float32 | Bars since last external BOS | Count, max 999 | smc_python |
| 36 | `bars_since_last_ext_choch` | float32 | Bars since last external CHoCH | Count, max 999 | smc_python |
| 37 | `int_bos_up` | binary | Internal BOS up occurred | 0 or 1 | smc_python |
| 38 | `int_bos_down` | binary | Internal BOS down occurred | 0 or 1 | smc_python |
| 39 | `int_choch_up` | binary | Internal CHoCH up occurred | 0 or 1 | smc_python |
| 40 | `int_choch_down` | binary | Internal CHoCH down occurred | 0 or 1 | smc_python |
| 41 | `bars_since_last_int_bos` | float32 | Bars since last internal BOS | Count, max 999 | smc_python |
| 42 | `bars_since_last_int_choch` | float32 | Bars since last internal CHoCH | Count, max 999 | smc_python |
| 43 | `swing_high` | float32 | Current swing high price | Price level | smc_python |
| 44 | `swing_low` | float32 | Current swing low price | Price level | smc_python |

---

### 2.4 Group 4: SMC Structure - Internal (Swing & Premium/Discount)

**Total: 8 features**

| # | Feature Name | Data Type | Description | Formula | Source |
|---|--------------|-----------|-------------|---------|--------|
| 45 | `price_vs_swing_mid` | float32 | Price position in swing range | `(c - swing_mid) / (swing_high - swing_low + eps)` | derived |
| 46 | `premium_zone` | binary | Price in premium zone (>50%) | 1 if `price_vs_swing_mid > 0` | derived |
| 47 | `discount_zone` | binary | Price in discount zone (<50%) | 1 if `price_vs_swing_mid < 0` | derived |
| 48 | `distance_to_swing_high` | float32 | Distance to swing high (ticks) | `(swing_high - c) / tick_size` | derived |
| 49 | `distance_to_swing_low` | float32 | Distance to swing low (ticks) | `(c - swing_low) / tick_size` | derived |
| 50 | `distance_to_swing_high_norm` | float32 | Normalized distance to swing high | `(swing_high - c) / (swing_high - swing_low + eps)` | derived |
| 51 | `distance_to_swing_low_norm` | float32 | Normalized distance to swing low | `(c - swing_low) / (swing_high - swing_low + eps)` | derived |

**Premium/Discount Zone Logic:**
```python
swing_mid = (swing_high + swing_low) / 2
swing_range = swing_high - swing_low + eps

price_vs_swing_mid = (close - swing_mid) / swing_range

# Premium zone: price > 50% of range (upper half)
premium_zone = 1 if price_vs_swing_mid > 0 else 0

# Discount zone: price < 50% of range (lower half)  
discount_zone = 1 if price_vs_swing_mid < 0 else 0
```

---

### 2.5 Group 5: Liquidity & Sweep

**Total: 4 features**

| # | Feature Name | Data Type | Description | Formula | Source |
|---|--------------|-----------|-------------|---------|--------|
| 52 | `sweep_prev_high` | binary | Swept previous swing high | 0 or 1 | smc_python |
| 53 | `sweep_prev_low` | binary | Swept previous swing low | 0 or 1 | smc_python |
| 54 | `sweep_type` | int (cat) | Type of sweep | 0=none, 1=high, 2=low, 3=both | smc_python |
| 55 | `bars_since_last_sweep` | float32 | Bars since last liquidity sweep | Count, max 999 | smc_python |

**Sweep Type Encoding:**
```python
SWEEP_TYPE = {
    0: "none",           # No sweep
    1: "sweep_high",     # Swept previous high only
    2: "sweep_low",      # Swept previous low only
    3: "sweep_both"      # Swept both (rare)
}
```

---

### 2.6 Group 6: OB & FVG Proximity

**Total: 12 features**

| # | Feature Name | Data Type | Description | Formula | Source |
|---|--------------|-----------|-------------|---------|--------|
| 56 | `near_ob_m1_bull` | binary | Near M1 bullish Order Block | 0 or 1 | smc_python |
| 57 | `near_ob_m1_bear` | binary | Near M1 bearish Order Block | 0 or 1 | smc_python |
| 58 | `near_ob_m5_bull` | binary | Near M5 bullish Order Block | 0 or 1 | smc_python |
| 59 | `near_ob_m5_bear` | binary | Near M5 bearish Order Block | 0 or 1 | smc_python |
| 60 | `ob_age_bars` | float32 | Age of nearest OB in bars | Count | smc_python |
| 61 | `distance_to_nearest_ob` | float32 | Signed distance to nearest OB | Ticks (+ above, - below) | smc_python |
| 62 | `near_fvg_m1_bull` | binary | Near M1 bullish FVG | 0 or 1 | smc_python |
| 63 | `near_fvg_m1_bear` | binary | Near M1 bearish FVG | 0 or 1 | smc_python |
| 64 | `near_fvg_m5_bull` | binary | Near M5 bullish FVG | 0 or 1 | smc_python |
| 65 | `near_fvg_m5_bear` | binary | Near M5 bearish FVG | 0 or 1 | smc_python |
| 66 | `fvg_age_bars` | float32 | Age of nearest FVG in bars | Count | smc_python |
| 67 | `distance_to_nearest_fvg` | float32 | Signed distance to nearest FVG | Ticks (+ above, - below) | smc_python |

**Proximity Detection Logic:**
```python
# "Near" is defined as price within X ticks of zone
PROXIMITY_THRESHOLD_TICKS = 20  # Configurable

def is_near_zone(price, zone_high, zone_low, tick_size):
    threshold = PROXIMITY_THRESHOLD_TICKS * tick_size
    return (zone_low - threshold <= price <= zone_high + threshold)
```

---

### 2.7 Group 7: VA / Auction / Session

**Total: 20 features**

| # | Feature Name | Data Type | Description | Formula | Source |
|---|--------------|-----------|-------------|---------|--------|
| 68 | `vah` | float32 | Value Area High | 70% volume distribution | smc_python |
| 69 | `val` | float32 | Value Area Low | 70% volume distribution | smc_python |
| 70 | `poc` | float32 | Point of Control | Price with highest volume | smc_python |
| 71 | `dist_to_vah` | float32 | Distance to VAH (ticks) | `(vah - c) / tick_size` | derived |
| 72 | `dist_to_val` | float32 | Distance to VAL (ticks) | `(c - val) / tick_size` | derived |
| 73 | `dist_to_poc` | float32 | Distance to POC (ticks) | `(poc - c) / tick_size` | derived |
| 74 | `inside_value` | binary | Price inside value area | 1 if `val <= c <= vah` | derived |
| 75 | `above_value` | binary | Price above value area | 1 if `c > vah` | derived |
| 76 | `below_value` | binary | Price below value area | 1 if `c < val` | derived |
| 77 | `session_high` | float32 | Session high price | Max high in session | derived |
| 78 | `session_low` | float32 | Session low price | Min low in session | derived |
| 79 | `pos_in_session_range` | float32 | Position in session range | `(c - session_low) / (session_high - session_low + eps)` | derived |
| 80 | `dist_to_session_high_norm` | float32 | Normalized dist to session high | `(session_high - c) / (session_high - session_low + eps)` | derived |
| 81 | `dist_to_session_low_norm` | float32 | Normalized dist to session low | `(c - session_low) / (session_high - session_low + eps)` | derived |
| 82 | `session_id` | int (cat) | Session identifier | 0=ASIA, 1=LDN, 2=NY | derived |
| 83 | `bar_index_in_session` | float32 | Bar index within session | 0, 1, 2, ... | derived |
| 84 | `bar_index_in_session_norm` | float32 | Normalized bar index | `bar_index / total_session_bars` | derived |
| 85 | `minute_of_day` | float32 | Minute of day (0-1439) | `hour * 60 + minute` | derived |
| 86 | `minute_of_day_norm` | float32 | Normalized minute of day | `minute_of_day / 1440` | derived |
| 87 | `day_of_week` | int (cat) | Day of week | 0=Mon, 1=Tue, ..., 6=Sun | derived |

**Session Time Definitions (EST/EDT):**
```python
SESSIONS = {
    "ASIA": {"start_hour": 18, "end_hour": 2},   # 6PM - 2AM
    "LDN":  {"start_hour": 2,  "end_hour": 8},   # 2AM - 8AM
    "NY":   {"start_hour": 8,  "end_hour": 17}   # 8AM - 5PM
}

SESSION_ID_MAP = {
    "ASIA": 0,
    "LDN": 1,
    "NY": 2
}
```

---

### 2.8 Group 8: ASM Regime Hint

**Total: 1 feature**

| # | Feature Name | Data Type | Description | Formula | Source |
|---|--------------|-----------|-------------|---------|--------|
| 88 | `asm_regime_hint` | int (cat) | ASM v1 rule-based regime | 0-5 classification | smc_python |

**Regime Classification:**
```python
ASM_REGIME = {
    0: "unknown",           # Cannot determine
    1: "range",             # Ranging/consolidation
    2: "trend_up",          # Uptrend
    3: "trend_down",        # Downtrend
    4: "opening_drive_up",  # Opening drive bullish
    5: "opening_drive_down" # Opening drive bearish
}
```

---

## 3. Full Sample Schema

### 3.1 Sample Structure

Mỗi sample trong `encoder_dataset.jsonl` có cấu trúc sau:

```json
{
  "symbol": "NQ",
  "tf": "M1",
  "date": "2024-01-15",
  "session": "LDN",
  "start_time": "2024-01-15T02:00:00",
  "end_time": "2024-01-15T04:07:00",
  "seq": [
    {
      "o": 17250.25,
      "h": 17252.50,
      "l": 17249.00,
      "c": 17251.75,
      "hl_range": 3.50,
      "body": 1.50,
      "upper_wick": 0.75,
      "lower_wick": 1.25,
      "bar_type": 1,
      "volume": 1250,
      "volume_vs_session_avg": 1.15,
      "volume_zscore": 0.45,
      "atr_m1_14": 4.25,
      "true_range": 3.50,
      "range_vs_session_avg": 0.92,
      "delta": 150,
      "delta_abs": 150,
      "delta_sign": 1,
      "delta_vs_volume": 0.12,
      "cum_delta_session": 2500,
      "delta_zscore_session": 0.35,
      "tick_count": 450,
      "tick_speed": 7.5,
      "buy_volume": 700,
      "sell_volume": 550,
      "buy_ratio": 0.56,
      "sell_ratio": 0.44,
      "imbalance_buy_sell": 0.12,
      "ext_trend_dir": 1,
      "int_trend_dir": 1,
      "ext_bos_up": 0,
      "ext_bos_down": 0,
      "ext_choch_up": 0,
      "ext_choch_down": 0,
      "bars_since_last_ext_bos": 45,
      "bars_since_last_ext_choch": 120,
      "int_bos_up": 1,
      "int_bos_down": 0,
      "int_choch_up": 0,
      "int_choch_down": 0,
      "bars_since_last_int_bos": 0,
      "bars_since_last_int_choch": 35,
      "swing_high": 17280.00,
      "swing_low": 17220.00,
      "price_vs_swing_mid": 0.053,
      "premium_zone": 1,
      "discount_zone": 0,
      "distance_to_swing_high": 113,
      "distance_to_swing_low": 127,
      "distance_to_swing_high_norm": 0.47,
      "distance_to_swing_low_norm": 0.53,
      "sweep_prev_high": 0,
      "sweep_prev_low": 0,
      "sweep_type": 0,
      "bars_since_last_sweep": 85,
      "near_ob_m1_bull": 0,
      "near_ob_m1_bear": 0,
      "near_ob_m5_bull": 1,
      "near_ob_m5_bear": 0,
      "ob_age_bars": 25,
      "distance_to_nearest_ob": -15,
      "near_fvg_m1_bull": 0,
      "near_fvg_m1_bear": 0,
      "near_fvg_m5_bull": 0,
      "near_fvg_m5_bear": 1,
      "fvg_age_bars": 40,
      "distance_to_nearest_fvg": 22,
      "vah": 17265.00,
      "val": 17235.00,
      "poc": 17250.00,
      "dist_to_vah": 53,
      "dist_to_val": 67,
      "dist_to_poc": -7,
      "inside_value": 1,
      "above_value": 0,
      "below_value": 0,
      "session_high": 17275.00,
      "session_low": 17230.00,
      "pos_in_session_range": 0.48,
      "dist_to_session_high_norm": 0.52,
      "dist_to_session_low_norm": 0.48,
      "session_id": 1,
      "bar_index_in_session": 67,
      "bar_index_in_session_norm": 0.28,
      "minute_of_day": 247,
      "minute_of_day_norm": 0.17,
      "day_of_week": 0,
      "asm_regime_hint": 2
    }
  ],
  "aux": {
    "future_return_5": 0.00035,
    "future_dir_5": 1,
    "future_range_5": 24.0
  }
}
```

### 3.2 Meta Fields

| Field | Type | Description |
|-------|------|-------------|
| `symbol` | string | Trading symbol (e.g., "NQ", "ES") |
| `tf` | string | Timeframe (always "M1") |
| `date` | string | Date in YYYY-MM-DD format |
| `session` | string | Session name: "ASIA", "LDN", "NY" |
| `start_time` | string | ISO8601 timestamp of first bar |
| `end_time` | string | ISO8601 timestamp of last bar |

### 3.3 Sequence Field

| Field | Type | Description |
|-------|------|-------------|
| `seq` | array | Array of N bar objects (default N=128) |

Mỗi bar object chứa tất cả 88+ features được định nghĩa ở Section 2.

### 3.4 Aux Targets

| Field | Type | Description |
|-------|------|-------------|
| `aux.future_return_5` | float | Return sau 5 bars |
| `aux.future_dir_5` | int | Direction: -1, 0, +1 |
| `aux.future_range_5` | float | Range trong 5 bars tiếp theo (ticks) |

---

## 4. Exact Feature Order

**OFFICIAL FEATURE ORDER** cho tensor `[N, D]` với D=95:

```python
FEATURE_ORDER = [
    # === Group 1: OHLCV & Shape (15 features) ===
    0:  "o",
    1:  "h",
    2:  "l",
    3:  "c",
    4:  "hl_range",
    5:  "body",
    6:  "upper_wick",
    7:  "lower_wick",
    8:  "bar_type",
    9:  "volume",
    10: "volume_vs_session_avg",
    11: "volume_zscore",
    12: "atr_m1_14",
    13: "true_range",
    14: "range_vs_session_avg",
    
    # === Group 2: Delta & Tick (13 features) ===
    15: "delta",
    16: "delta_abs",
    17: "delta_sign",
    18: "delta_vs_volume",
    19: "cum_delta_session",
    20: "delta_zscore_session",
    21: "tick_count",
    22: "tick_speed",
    23: "buy_volume",
    24: "sell_volume",
    25: "buy_ratio",
    26: "sell_ratio",
    27: "imbalance_buy_sell",
    
    # === Group 3: SMC External (16 features) ===
    28: "ext_trend_dir",
    29: "int_trend_dir",
    30: "ext_bos_up",
    31: "ext_bos_down",
    32: "ext_choch_up",
    33: "ext_choch_down",
    34: "bars_since_last_ext_bos",
    35: "bars_since_last_ext_choch",
    36: "int_bos_up",
    37: "int_bos_down",
    38: "int_choch_up",
    39: "int_choch_down",
    40: "bars_since_last_int_bos",
    41: "bars_since_last_int_choch",
    42: "swing_high",
    43: "swing_low",
    
    # === Group 4: Swing & Premium/Discount (7 features) ===
    44: "price_vs_swing_mid",
    45: "premium_zone",
    46: "discount_zone",
    47: "distance_to_swing_high",
    48: "distance_to_swing_low",
    49: "distance_to_swing_high_norm",
    50: "distance_to_swing_low_norm",
    
    # === Group 5: Liquidity & Sweep (4 features) ===
    51: "sweep_prev_high",
    52: "sweep_prev_low",
    53: "sweep_type",
    54: "bars_since_last_sweep",
    
    # === Group 6: OB & FVG (12 features) ===
    55: "near_ob_m1_bull",
    56: "near_ob_m1_bear",
    57: "near_ob_m5_bull",
    58: "near_ob_m5_bear",
    59: "ob_age_bars",
    60: "distance_to_nearest_ob",
    61: "near_fvg_m1_bull",
    62: "near_fvg_m1_bear",
    63: "near_fvg_m5_bull",
    64: "near_fvg_m5_bear",
    65: "fvg_age_bars",
    66: "distance_to_nearest_fvg",
    
    # === Group 7: VA & Session (20 features) ===
    67: "vah",
    68: "val",
    69: "poc",
    70: "dist_to_vah",
    71: "dist_to_val",
    72: "dist_to_poc",
    73: "inside_value",
    74: "above_value",
    75: "below_value",
    76: "session_high",
    77: "session_low",
    78: "pos_in_session_range",
    79: "dist_to_session_high_norm",
    80: "dist_to_session_low_norm",
    81: "session_id",
    82: "bar_index_in_session",
    83: "bar_index_in_session_norm",
    84: "minute_of_day",
    85: "minute_of_day_norm",
    86: "day_of_week",
    
    # === Group 8: Regime (1 feature) ===
    87: "asm_regime_hint",
]

# Total: 88 features (indices 0-87)
# Reserved indices 88-94 for future expansion
TOTAL_FEATURE_DIM = 95
```

---

## 5. Feature Grouping by Type

### 5.1 Numeric Float32 Features (56 features)

Features được normalize bằng z-score hoặc min-max:

```python
NUMERIC_FLOAT32_FEATURES = [
    # OHLCV
    "o", "h", "l", "c",
    "hl_range", "body", "upper_wick", "lower_wick",
    "volume", "volume_vs_session_avg", "volume_zscore",
    "atr_m1_14", "true_range", "range_vs_session_avg",
    
    # Delta & Tick
    "delta", "delta_abs", "delta_vs_volume",
    "cum_delta_session", "delta_zscore_session",
    "tick_count", "tick_speed",
    "buy_volume", "sell_volume",
    "buy_ratio", "sell_ratio", "imbalance_buy_sell",
    
    # SMC
    "bars_since_last_ext_bos", "bars_since_last_ext_choch",
    "bars_since_last_int_bos", "bars_since_last_int_choch",
    "swing_high", "swing_low",
    "price_vs_swing_mid",
    "distance_to_swing_high", "distance_to_swing_low",
    "distance_to_swing_high_norm", "distance_to_swing_low_norm",
    "bars_since_last_sweep",
    
    # OB/FVG
    "ob_age_bars", "distance_to_nearest_ob",
    "fvg_age_bars", "distance_to_nearest_fvg",
    
    # VA & Session
    "vah", "val", "poc",
    "dist_to_vah", "dist_to_val", "dist_to_poc",
    "session_high", "session_low",
    "pos_in_session_range",
    "dist_to_session_high_norm", "dist_to_session_low_norm",
    "bar_index_in_session", "bar_index_in_session_norm",
    "minute_of_day", "minute_of_day_norm",
]
# Total: 56 features
```

### 5.2 Binary Features (24 features)

Features với giá trị 0 hoặc 1, KHÔNG normalize:

```python
BINARY_FEATURES = [
    # SMC BOS/CHoCH
    "ext_bos_up", "ext_bos_down",
    "ext_choch_up", "ext_choch_down",
    "int_bos_up", "int_bos_down",
    "int_choch_up", "int_choch_down",
    
    # Premium/Discount
    "premium_zone", "discount_zone",
    
    # Sweep
    "sweep_prev_high", "sweep_prev_low",
    
    # OB Proximity
    "near_ob_m1_bull", "near_ob_m1_bear",
    "near_ob_m5_bull", "near_ob_m5_bear",
    
    # FVG Proximity
    "near_fvg_m1_bull", "near_fvg_m1_bear",
    "near_fvg_m5_bull", "near_fvg_m5_bear",
    
    # VA Position
    "inside_value", "above_value", "below_value",
]
# Total: 24 features
```

### 5.3 Integer Categorical Features (8 features)

Features với giá trị integer rời rạc, KHÔNG normalize:

```python
CATEGORICAL_FEATURES = [
    # Feature Name       | Values              | Num Categories
    "bar_type",          # 0, 1, 2             | 3
    "delta_sign",        # -1, 0, 1 → 0, 1, 2  | 3
    "ext_trend_dir",     # -1, 0, 1 → 0, 1, 2  | 3
    "int_trend_dir",     # -1, 0, 1 → 0, 1, 2  | 3
    "sweep_type",        # 0, 1, 2, 3          | 4
    "session_id",        # 0, 1, 2             | 3
    "day_of_week",       # 0-6                 | 7
    "asm_regime_hint",   # 0-5                 | 6
]
# Total: 8 features
```

**Categorical Encoding Note:**
- Các feature có giá trị âm (-1) được shift lên: `-1 → 0, 0 → 1, 1 → 2`
- Trong tensor, tất cả categorical được lưu dưới dạng float32 nhưng giữ nguyên giá trị integer

### 5.4 Derived Features (Computed at Runtime)

Features được tính từ raw features:

```python
DERIVED_FEATURES = {
    # From OHLCV
    "hl_range": "h - l",
    "body": "c - o",
    "upper_wick": "h - max(o, c)",
    "lower_wick": "min(o, c) - l",
    "bar_type": "classify(body, hl_range)",
    "volume_vs_session_avg": "volume / session_avg_volume",
    "volume_zscore": "(volume - mean) / std",
    "true_range": "max(h-l, |h-prev_c|, |l-prev_c|)",
    "range_vs_session_avg": "hl_range / session_avg_range",
    
    # From Delta
    "delta_abs": "abs(delta)",
    "delta_sign": "sign(delta)",
    "delta_vs_volume": "delta / max(volume, 1)",
    "delta_zscore_session": "(delta - session_mean) / session_std",
    "buy_ratio": "buy_volume / max(volume, 1)",
    "sell_ratio": "sell_volume / max(volume, 1)",
    "imbalance_buy_sell": "(buy_volume - sell_volume) / max(volume, 1)",
    
    # From Swing
    "price_vs_swing_mid": "(c - swing_mid) / swing_range",
    "premium_zone": "1 if price_vs_swing_mid > 0 else 0",
    "discount_zone": "1 if price_vs_swing_mid < 0 else 0",
    "distance_to_swing_high": "(swing_high - c) / tick_size",
    "distance_to_swing_low": "(c - swing_low) / tick_size",
    "distance_to_swing_high_norm": "(swing_high - c) / swing_range",
    "distance_to_swing_low_norm": "(c - swing_low) / swing_range",
    
    # From VA
    "dist_to_vah": "(vah - c) / tick_size",
    "dist_to_val": "(c - val) / tick_size",
    "dist_to_poc": "(poc - c) / tick_size",
    "inside_value": "1 if val <= c <= vah else 0",
    "above_value": "1 if c > vah else 0",
    "below_value": "1 if c < val else 0",
    
    # From Session
    "pos_in_session_range": "(c - session_low) / session_range",
    "dist_to_session_high_norm": "(session_high - c) / session_range",
    "dist_to_session_low_norm": "(c - session_low) / session_range",
    "bar_index_in_session_norm": "bar_index / total_session_bars",
    "minute_of_day": "hour * 60 + minute",
    "minute_of_day_norm": "minute_of_day / 1440",
}
```

---

## 6. Normalization Rules

### 6.1 Z-Score Normalization (Default)

Áp dụng cho hầu hết numeric features:

```python
normalized = (value - mean) / std
normalized = clip(normalized, -5.0, 5.0)  # Clip extreme values
```

**Features sử dụng Z-Score:**
```python
ZSCORE_FEATURES = [
    "o", "h", "l", "c",
    "hl_range", "body", "upper_wick", "lower_wick",
    "volume", "delta", "delta_abs",
    "cum_delta_session", "tick_count", "tick_speed",
    "buy_volume", "sell_volume",
    "swing_high", "swing_low",
    "distance_to_swing_high", "distance_to_swing_low",
    "ob_age_bars", "distance_to_nearest_ob",
    "fvg_age_bars", "distance_to_nearest_fvg",
    "vah", "val", "poc",
    "dist_to_vah", "dist_to_val", "dist_to_poc",
    "session_high", "session_low",
    "bar_index_in_session", "minute_of_day",
    "atr_m1_14", "true_range",
    "bars_since_last_ext_bos", "bars_since_last_ext_choch",
    "bars_since_last_int_bos", "bars_since_last_int_choch",
    "bars_since_last_sweep",
]
```

### 6.2 Keep Raw (No Normalization)

**Binary Features** - Giữ nguyên 0/1:
```python
RAW_BINARY = [
    "ext_bos_up", "ext_bos_down", "ext_choch_up", "ext_choch_down",
    "int_bos_up", "int_bos_down", "int_choch_up", "int_choch_down",
    "premium_zone", "discount_zone",
    "sweep_prev_high", "sweep_prev_low",
    "near_ob_m1_bull", "near_ob_m1_bear", "near_ob_m5_bull", "near_ob_m5_bear",
    "near_fvg_m1_bull", "near_fvg_m1_bear", "near_fvg_m5_bull", "near_fvg_m5_bear",
    "inside_value", "above_value", "below_value",
]
```

**Categorical Features** - Giữ nguyên integer:
```python
RAW_CATEGORICAL = [
    "bar_type",        # 0, 1, 2
    "delta_sign",      # 0, 1, 2 (shifted from -1, 0, 1)
    "ext_trend_dir",   # 0, 1, 2 (shifted from -1, 0, 1)
    "int_trend_dir",   # 0, 1, 2 (shifted from -1, 0, 1)
    "sweep_type",      # 0, 1, 2, 3
    "session_id",      # 0, 1, 2
    "day_of_week",     # 0-6
    "asm_regime_hint", # 0-5
]
```

**Already Normalized Features** - Đã trong range [0, 1] hoặc [-1, 1]:
```python
ALREADY_NORMALIZED = [
    "volume_vs_session_avg",      # Ratio, typically 0-3
    "volume_zscore",              # Already z-score
    "delta_vs_volume",            # Ratio [-1, 1]
    "delta_zscore_session",       # Already z-score
    "buy_ratio",                  # [0, 1]
    "sell_ratio",                 # [0, 1]
    "imbalance_buy_sell",         # [-1, 1]
    "range_vs_session_avg",       # Ratio, typically 0-3
    "price_vs_swing_mid",         # [-1, 1]
    "distance_to_swing_high_norm",# [0, 1]
    "distance_to_swing_low_norm", # [0, 1]
    "pos_in_session_range",       # [0, 1]
    "dist_to_session_high_norm",  # [0, 1]
    "dist_to_session_low_norm",   # [0, 1]
    "bar_index_in_session_norm",  # [0, 1]
    "minute_of_day_norm",         # [0, 1]
]
```

### 6.3 Normalization Config Format

```json
{
  "method": "zscore",
  "clip_zscore": 5.0,
  "eps": 1e-8,
  "stats": {
    "o": {"mean": 17250.5, "std": 50.2, "min_val": 17100.0, "max_val": 17400.0},
    "h": {"mean": 17252.3, "std": 50.5, "min_val": 17102.0, "max_val": 17405.0},
    "volume": {"mean": 1200.5, "std": 450.2, "min_val": 50, "max_val": 5000}
  }
}
```

---

## 7. Field Mapping Table

### 7.1 Complete Field Mapping: Raw Exporter → SMC Python → STATE-ENC v1

| Raw Exporter Field | SMC Python Field | STATE-ENC Feature | Notes |
|--------------------|------------------|-------------------|-------|
| `Open` | `open` / `o` | `o` | Direct mapping |
| `High` | `high` / `h` | `h` | Direct mapping |
| `Low` | `low` / `l` | `l` | Direct mapping |
| `Close` | `close` / `c` | `c` | Direct mapping |
| `Volume` | `volume` | `volume` | Direct mapping |
| `BuyVolume` | `buy_volume` | `buy_volume` | Direct mapping |
| `SellVolume` | `sell_volume` | `sell_volume` | Direct mapping |
| `Delta` | `delta` | `delta` | Direct mapping |
| `TickCount` | `tick_count` | `tick_count` | Direct mapping |
| `Time` / `Timestamp` | `time` / `timestamp` | (meta) | Used for session detection |
| - | `hl_range` | `hl_range` | Computed: `h - l` |
| - | `body` | `body` | Computed: `c - o` |
| - | `upper_wick` | `upper_wick` | Computed |
| - | `lower_wick` | `lower_wick` | Computed |
| - | `bar_type` | `bar_type` | Computed from body/range |
| - | `volume_vs_session_avg` | `volume_vs_session_avg` | Session-relative |
| - | `volume_zscore` | `volume_zscore` | Z-score |
| `ATR14` | `atr_m1_14` | `atr_m1_14` | May come from exporter or computed |
| - | `true_range` | `true_range` | Computed |
| - | `range_vs_session_avg` | `range_vs_session_avg` | Session-relative |
| - | `delta_abs` | `delta_abs` | Computed: `abs(delta)` |
| - | `delta_sign` | `delta_sign` | Computed: `sign(delta)` |
| - | `delta_vs_volume` | `delta_vs_volume` | Computed |
| - | `cum_delta_session` | `cum_delta_session` | Running sum |
| - | `delta_zscore_session` | `delta_zscore_session` | Z-score |
| - | `tick_speed` | `tick_speed` | Computed |
| - | `buy_ratio` | `buy_ratio` | Computed |
| - | `sell_ratio` | `sell_ratio` | Computed |
| - | `imbalance_buy_sell` | `imbalance_buy_sell` | Computed |
| - | `ext_trend_dir` | `ext_trend_dir` | SMC computed |
| - | `int_trend_dir` | `int_trend_dir` | SMC computed |
| - | `ext_bos_up` | `ext_bos_up` | SMC event |
| - | `ext_bos_down` | `ext_bos_down` | SMC event |
| - | `ext_choch_up` | `ext_choch_up` | SMC event |
| - | `ext_choch_down` | `ext_choch_down` | SMC event |
| - | `bars_since_last_ext_bos` | `bars_since_last_ext_bos` | Counter |
| - | `bars_since_last_ext_choch` | `bars_since_last_ext_choch` | Counter |
| - | `int_bos_up` | `int_bos_up` | SMC event |
| - | `int_bos_down` | `int_bos_down` | SMC event |
| - | `int_choch_up` | `int_choch_up` | SMC event |
| - | `int_choch_down` | `int_choch_down` | SMC event |
| - | `bars_since_last_int_bos` | `bars_since_last_int_bos` | Counter |
| - | `bars_since_last_int_choch` | `bars_since_last_int_choch` | Counter |
| - | `swing_high` | `swing_high` | SMC computed |
| - | `swing_low` | `swing_low` | SMC computed |
| - | `price_vs_swing_mid` | `price_vs_swing_mid` | Computed |
| - | `premium_zone` | `premium_zone` | Computed |
| - | `discount_zone` | `discount_zone` | Computed |
| - | `distance_to_swing_high` | `distance_to_swing_high` | Computed |
| - | `distance_to_swing_low` | `distance_to_swing_low` | Computed |
| - | `distance_to_swing_high_norm` | `distance_to_swing_high_norm` | Computed |
| - | `distance_to_swing_low_norm` | `distance_to_swing_low_norm` | Computed |
| - | `sweep_prev_high` | `sweep_prev_high` | SMC event |
| - | `sweep_prev_low` | `sweep_prev_low` | SMC event |
| - | `sweep_type` | `sweep_type` | SMC computed |
| - | `bars_since_last_sweep` | `bars_since_last_sweep` | Counter |
| - | `near_ob_m1_bull` | `near_ob_m1_bull` | SMC proximity |
| - | `near_ob_m1_bear` | `near_ob_m1_bear` | SMC proximity |
| - | `near_ob_m5_bull` | `near_ob_m5_bull` | SMC proximity |
| - | `near_ob_m5_bear` | `near_ob_m5_bear` | SMC proximity |
| - | `ob_age_bars` | `ob_age_bars` | Counter |
| - | `distance_to_nearest_ob` | `distance_to_nearest_ob` | Signed distance |
| - | `near_fvg_m1_bull` | `near_fvg_m1_bull` | SMC proximity |
| - | `near_fvg_m1_bear` | `near_fvg_m1_bear` | SMC proximity |
| - | `near_fvg_m5_bull` | `near_fvg_m5_bull` | SMC proximity |
| - | `near_fvg_m5_bear` | `near_fvg_m5_bear` | SMC proximity |
| - | `fvg_age_bars` | `fvg_age_bars` | Counter |
| - | `distance_to_nearest_fvg` | `distance_to_nearest_fvg` | Signed distance |
| - | `vah` | `vah` | Volume profile |
| - | `val` | `val` | Volume profile |
| - | `poc` | `poc` | Volume profile |
| - | `dist_to_vah` | `dist_to_vah` | Computed |
| - | `dist_to_val` | `dist_to_val` | Computed |
| - | `dist_to_poc` | `dist_to_poc` | Computed |
| - | `inside_value` | `inside_value` | Computed |
| - | `above_value` | `above_value` | Computed |
| - | `below_value` | `below_value` | Computed |
| - | `session_high` | `session_high` | Session computed |
| - | `session_low` | `session_low` | Session computed |
| - | `pos_in_session_range` | `pos_in_session_range` | Computed |
| - | `dist_to_session_high_norm` | `dist_to_session_high_norm` | Computed |
| - | `dist_to_session_low_norm` | `dist_to_session_low_norm` | Computed |
| - | `session_id` | `session_id` | From timestamp |
| - | `bar_index_in_session` | `bar_index_in_session` | Counter |
| - | `bar_index_in_session_norm` | `bar_index_in_session_norm` | Computed |
| - | `minute_of_day` | `minute_of_day` | From timestamp |
| - | `minute_of_day_norm` | `minute_of_day_norm` | Computed |
| - | `day_of_week` | `day_of_week` | From timestamp |
| - | `asm_regime_hint` | `asm_regime_hint` | ASM v1 rule-based |

### 7.2 Raw Exporter Required Fields

Minimum fields required from NinjaTrader exporter:

```python
RAW_EXPORTER_REQUIRED = [
    "Time",        # or "Timestamp", "DateTime"
    "Open",        # or "O"
    "High",        # or "H"
    "Low",         # or "L"
    "Close",       # or "C"
    "Volume",      # or "Vol"
    "Delta",       # Buy - Sell volume
    "BuyVolume",   # Volume at ask
    "SellVolume",  # Volume at bid
    "TickCount",   # Number of ticks
]

RAW_EXPORTER_OPTIONAL = [
    "ATR14",       # Pre-computed ATR
    "Symbol",      # Trading symbol
]
```

---

## 8. Data Validation Rules

### 8.1 Type Validation

```python
VALIDATION_RULES = {
    # OHLCV - must be positive
    "o": {"type": "float", "min": 0, "max": None},
    "h": {"type": "float", "min": 0, "max": None},
    "l": {"type": "float", "min": 0, "max": None},
    "c": {"type": "float", "min": 0, "max": None},
    "volume": {"type": "float", "min": 0, "max": None},
    
    # Derived OHLCV
    "hl_range": {"type": "float", "min": 0, "max": None},
    "body": {"type": "float", "min": None, "max": None},  # Can be negative
    "upper_wick": {"type": "float", "min": 0, "max": None},
    "lower_wick": {"type": "float", "min": 0, "max": None},
    "bar_type": {"type": "int", "values": [0, 1, 2]},
    
    # Volume ratios
    "volume_vs_session_avg": {"type": "float", "min": 0, "max": 100},
    "volume_zscore": {"type": "float", "min": -10, "max": 10},
    "buy_ratio": {"type": "float", "min": 0, "max": 1},
    "sell_ratio": {"type": "float", "min": 0, "max": 1},
    
    # Delta
    "delta": {"type": "float", "min": None, "max": None},
    "delta_abs": {"type": "float", "min": 0, "max": None},
    "delta_sign": {"type": "int", "values": [-1, 0, 1]},
    "delta_vs_volume": {"type": "float", "min": -1, "max": 1},
    "imbalance_buy_sell": {"type": "float", "min": -1, "max": 1},
    
    # Binary features
    "ext_bos_up": {"type": "int", "values": [0, 1]},
    "ext_bos_down": {"type": "int", "values": [0, 1]},
    "ext_choch_up": {"type": "int", "values": [0, 1]},
    "ext_choch_down": {"type": "int", "values": [0, 1]},
    "int_bos_up": {"type": "int", "values": [0, 1]},
    "int_bos_down": {"type": "int", "values": [0, 1]},
    "int_choch_up": {"type": "int", "values": [0, 1]},
    "int_choch_down": {"type": "int", "values": [0, 1]},
    "premium_zone": {"type": "int", "values": [0, 1]},
    "discount_zone": {"type": "int", "values": [0, 1]},
    "sweep_prev_high": {"type": "int", "values": [0, 1]},
    "sweep_prev_low": {"type": "int", "values": [0, 1]},
    "near_ob_m1_bull": {"type": "int", "values": [0, 1]},
    "near_ob_m1_bear": {"type": "int", "values": [0, 1]},
    "near_ob_m5_bull": {"type": "int", "values": [0, 1]},
    "near_ob_m5_bear": {"type": "int", "values": [0, 1]},
    "near_fvg_m1_bull": {"type": "int", "values": [0, 1]},
    "near_fvg_m1_bear": {"type": "int", "values": [0, 1]},
    "near_fvg_m5_bull": {"type": "int", "values": [0, 1]},
    "near_fvg_m5_bear": {"type": "int", "values": [0, 1]},
    "inside_value": {"type": "int", "values": [0, 1]},
    "above_value": {"type": "int", "values": [0, 1]},
    "below_value": {"type": "int", "values": [0, 1]},
    
    # Categorical
    "ext_trend_dir": {"type": "int", "values": [-1, 0, 1]},
    "int_trend_dir": {"type": "int", "values": [-1, 0, 1]},
    "sweep_type": {"type": "int", "values": [0, 1, 2, 3]},
    "session_id": {"type": "int", "values": [0, 1, 2]},
    "day_of_week": {"type": "int", "values": [0, 1, 2, 3, 4, 5, 6]},
    "asm_regime_hint": {"type": "int", "values": [0, 1, 2, 3, 4, 5]},
    
    # Counters
    "bars_since_last_ext_bos": {"type": "float", "min": 0, "max": 999},
    "bars_since_last_ext_choch": {"type": "float", "min": 0, "max": 999},
    "bars_since_last_int_bos": {"type": "float", "min": 0, "max": 999},
    "bars_since_last_int_choch": {"type": "float", "min": 0, "max": 999},
    "bars_since_last_sweep": {"type": "float", "min": 0, "max": 999},
    "ob_age_bars": {"type": "float", "min": 0, "max": 999},
    "fvg_age_bars": {"type": "float", "min": 0, "max": 999},
    "bar_index_in_session": {"type": "float", "min": 0, "max": 500},
    
    # Normalized [0, 1]
    "bar_index_in_session_norm": {"type": "float", "min": 0, "max": 1},
    "minute_of_day_norm": {"type": "float", "min": 0, "max": 1},
    "pos_in_session_range": {"type": "float", "min": 0, "max": 1},
    "dist_to_session_high_norm": {"type": "float", "min": 0, "max": 1},
    "dist_to_session_low_norm": {"type": "float", "min": 0, "max": 1},
    "distance_to_swing_high_norm": {"type": "float", "min": 0, "max": 1},
    "distance_to_swing_low_norm": {"type": "float", "min": 0, "max": 1},
    
    # Normalized [-1, 1]
    "price_vs_swing_mid": {"type": "float", "min": -1, "max": 1},
    
    # Time
    "minute_of_day": {"type": "float", "min": 0, "max": 1439},
}
```

### 8.2 Consistency Checks

```python
def validate_bar(bar: dict) -> list[str]:
    """Validate bar data consistency"""
    errors = []
    
    # OHLC consistency
    if bar["h"] < bar["l"]:
        errors.append("High < Low")
    if bar["h"] < max(bar["o"], bar["c"]):
        errors.append("High < max(Open, Close)")
    if bar["l"] > min(bar["o"], bar["c"]):
        errors.append("Low > min(Open, Close)")
    
    # Volume consistency
    if bar["buy_volume"] + bar["sell_volume"] != bar["volume"]:
        # Allow small tolerance
        if abs(bar["buy_volume"] + bar["sell_volume"] - bar["volume"]) > 1:
            errors.append("BuyVolume + SellVolume != Volume")
    
    # Delta consistency
    expected_delta = bar["buy_volume"] - bar["sell_volume"]
    if abs(bar["delta"] - expected_delta) > 1:
        errors.append("Delta != BuyVolume - SellVolume")
    
    # Ratio consistency
    if bar["buy_ratio"] + bar["sell_ratio"] > 1.01:
        errors.append("BuyRatio + SellRatio > 1")
    
    # Zone exclusivity
    if bar["premium_zone"] == 1 and bar["discount_zone"] == 1:
        errors.append("Both premium and discount zone = 1")
    
    # VA position exclusivity
    va_sum = bar["inside_value"] + bar["above_value"] + bar["below_value"]
    if va_sum != 1:
        errors.append(f"VA position sum = {va_sum}, expected 1")
    
    return errors
```

---

## 9. Aux Target Definition

### 9.1 Target Fields

| Field | Type | Description | Formula |
|-------|------|-------------|---------|
| `future_return_5` | float32 | Return sau 5 bars | `(close_{t+5} - close_t) / close_t` |
| `future_dir_5` | int | Direction classification | -1, 0, +1 based on thresholds |
| `future_range_5` | float32 | Range trong 5 bars tiếp theo | `(max_high - min_low) / tick_size` |

### 9.2 Future Direction Classification

```python
def compute_future_dir(future_return: float, 
                       threshold_up: float = 0.0005,
                       threshold_down: float = -0.0005) -> int:
    """
    Classify future direction based on return.
    
    Args:
        future_return: Return value (e.g., 0.0003 = 0.03%)
        threshold_up: Threshold for bullish classification
        threshold_down: Threshold for bearish classification
        
    Returns:
        -1: Bearish (return < threshold_down)
         0: Neutral (threshold_down <= return <= threshold_up)
        +1: Bullish (return > threshold_up)
    """
    if future_return > threshold_up:
        return 1
    elif future_return < threshold_down:
        return -1
    else:
        return 0
```

### 9.3 Label Encoding for Training

Trong PyTorch Dataset, `future_dir_5` được shift để phù hợp với CrossEntropyLoss:

```python
# Original values: -1, 0, +1
# Encoded values:   0, 1,  2

def encode_direction(dir_value: int) -> int:
    """Encode direction for CrossEntropyLoss"""
    return dir_value + 1  # -1 -> 0, 0 -> 1, 1 -> 2

def decode_direction(encoded: int) -> int:
    """Decode direction back to original"""
    return encoded - 1  # 0 -> -1, 1 -> 0, 2 -> 1
```

### 9.4 Future Return Calculation

```python
def compute_future_return(bars: list, 
                          current_idx: int, 
                          future_bars: int = 5) -> float:
    """
    Compute future return.
    
    Args:
        bars: List of bar dicts
        current_idx: Index of current bar
        future_bars: Number of bars to look ahead
        
    Returns:
        Future return as decimal (e.g., 0.0005 = 0.05%)
    """
    if current_idx + future_bars >= len(bars):
        return None
    
    current_close = bars[current_idx]["c"]
    future_close = bars[current_idx + future_bars]["c"]
    
    if current_close == 0:
        return 0.0
    
    return (future_close - current_close) / current_close
```

### 9.5 Future Range Calculation

```python
def compute_future_range(bars: list,
                         current_idx: int,
                         future_bars: int = 5,
                         tick_size: float = 0.25) -> float:
    """
    Compute future range in ticks.
    
    Args:
        bars: List of bar dicts
        current_idx: Index of current bar
        future_bars: Number of bars to look ahead
        tick_size: Tick size for the instrument
        
    Returns:
        Future range in ticks
    """
    if current_idx + future_bars >= len(bars):
        return None
    
    future_slice = bars[current_idx + 1: current_idx + 1 + future_bars]
    
    max_high = max(b["h"] for b in future_slice)
    min_low = min(b["l"] for b in future_slice)
    
    return (max_high - min_low) / tick_size
```

### 9.6 Default Thresholds

```python
# Default configuration for NQ (Nasdaq futures)
DEFAULT_THRESHOLDS = {
    "future_dir_threshold_up": 0.0005,    # +0.05% = bullish
    "future_dir_threshold_down": -0.0005, # -0.05% = bearish
    "tick_size": 0.25,                    # NQ tick size
    "future_bars": 5,                     # 5 bars lookahead
}

# For ES (S&P futures)
ES_THRESHOLDS = {
    "future_dir_threshold_up": 0.0003,
    "future_dir_threshold_down": -0.0003,
    "tick_size": 0.25,
    "future_bars": 5,
}
```

### 9.7 Class Distribution Monitoring

```python
def analyze_target_distribution(samples: list) -> dict:
    """
    Analyze distribution of target classes.
    
    Returns:
        Dict with class counts and percentages
    """
    dir_counts = {-1: 0, 0: 0, 1: 0}
    returns = []
    ranges = []
    
    for sample in samples:
        aux = sample["aux"]
        dir_counts[aux["future_dir_5"]] += 1
        returns.append(aux["future_return_5"])
        ranges.append(aux["future_range_5"])
    
    total = sum(dir_counts.values())
    
    return {
        "direction_counts": dir_counts,
        "direction_percentages": {
            k: v / total * 100 for k, v in dir_counts.items()
        },
        "return_stats": {
            "mean": np.mean(returns),
            "std": np.std(returns),
            "min": np.min(returns),
            "max": np.max(returns),
        },
        "range_stats": {
            "mean": np.mean(ranges),
            "std": np.std(ranges),
            "min": np.min(ranges),
            "max": np.max(ranges),
        }
    }
```

---

## Appendix A: Quick Reference

### A.1 Feature Counts by Group

| Group | Count | Type |
|-------|-------|------|
| OHLCV & Shape | 15 | Mixed |
| Delta & Tick | 13 | Mixed |
| SMC External | 16 | Mixed |
| Swing & Premium/Discount | 7 | Mixed |
| Liquidity & Sweep | 4 | Mixed |
| OB & FVG | 12 | Mixed |
| VA & Session | 20 | Mixed |
| ASM Regime | 1 | Categorical |
| **TOTAL** | **88** | - |
| Reserved | 7 | - |
| **TENSOR DIM** | **95** | - |

### A.2 Feature Type Summary

| Type | Count |
|------|-------|
| Numeric (float32) | 56 |
| Binary (0/1) | 24 |
| Categorical (int) | 8 |
| **TOTAL** | **88** |

### A.3 Normalization Summary

| Method | Count |
|--------|-------|
| Z-Score | ~40 |
| Keep Raw (Binary) | 24 |
| Keep Raw (Categorical) | 8 |
| Already Normalized | ~16 |

---

## Appendix B: Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2024-12-04 | Initial release |

---

## Appendix C: Usage Examples

### C.1 Loading Feature Config

```python
import json
from state_enc_v1.src.features_spec import FEATURE_SPEC, get_feature_names

# Get all feature names in order
feature_names = get_feature_names()
print(f"Total features: {len(feature_names)}")

# Load normalization config
with open("state_enc_v1/artifacts/feature_config.json") as f:
    config = json.load(f)

normalizer_stats = config["normalization"]["stats"]
```

### C.2 Validating Bar Data

```python
from state_enc_v1.src.features_spec import validate_bar_features

bar = {"o": 17250.0, "h": 17252.0, "l": 17248.0, "c": 17251.0, ...}
is_valid, missing = validate_bar_features(bar)

if not is_valid:
    print(f"Missing features: {missing}")
```

### C.3 Building Dataset

```bash
python state_enc_v1/scripts/build_state_enc_dataset.py \
    --config state_enc_v1/configs/state_enc_dataset_v1.json
```

### C.4 Training Model

```bash
python state_enc_v1/scripts/train_state_enc.py \
    --config state_enc_v1/configs/state_enc_train_v1.json
```

### C.5 Using Encoder in Other Modules

```python
import torch
from state_enc_v1.src.model.state_enc_model import load_state_enc_model

# Load trained model
model = load_state_enc_model(
    checkpoint_path="state_enc_v1/artifacts/final/state_enc_v1.pt",
    config_path="state_enc_v1/artifacts/final/model_config.json"
)

# Prepare input: [B, N, D] tensor
x = torch.randn(1, 128, 95)  # Example

# Get embedding
with torch.no_grad():
    z_t = model.encode(x)  # [1, 128] market state embedding

# Use z_t for ASM v2 or Meta S4
```

---

**END OF DOCUMENT**
