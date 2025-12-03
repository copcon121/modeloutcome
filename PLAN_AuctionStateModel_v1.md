# Auction State Model (ASM) v1

**Created**: 2025-12-02  
**Status**: Active Development  
**Version**: v1.x (with Weekly VA features)

---

## Baseline Model: ASM-GRU64-v1.0-C3

| Metric | Value |
|--------|-------|
| AUC_SHIFT | **0.712** |
| AUC_UP | **0.779** |
| AUC_DOWN | **0.857** |
| Macro F1 | 0.149 |
| Config | Focal Loss (γ=2.0) + inv class weights + oversample_shift (α=5.0) |
| Features | 112 (100 base + 12 Weekly VA) |
| Model | `output/asm_models_v1/ASM-GRU64-v1.0-C3.pt` |

---

## 1. Overview

### 1.1 Mục tiêu

**Auction State Model (ASM)** đo "trạng thái đấu giá" quanh Value Area (VA) và cấu trúc SMC:

- Phe nào đang nắm ưu thế (buyer / seller)?
- Khả năng VA sẽ shift lên / xuống / giữ nguyên trong K bar tới?

### 1.2 Vai trò trong hệ thống

```
┌─────────────────────────────────────────────────────────────────────┐
│  TRADING SYSTEM ARCHITECTURE                                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────────────┐                                           │
│  │  STRATEGY LAYER     │  ← Entry decisions                        │
│  │  (SMC + VA Rules)   │                                           │
│  │  - S4_London        │                                           │
│  │  - LS-CHoCH @ VA    │                                           │
│  │  - VA Breakout      │                                           │
│  └──────────┬──────────┘                                           │
│             │ candidate entry                                       │
│             ▼                                                       │
│  ┌─────────────────────┐                                           │
│  │  ASM LAYER          │  ← Context evaluation (CRITIC/FILTER)     │
│  │  (Auction State)    │                                           │
│  │  - p_up, p_down     │                                           │
│  │  - VA shift prob    │                                           │
│  └──────────┬──────────┘                                           │
│             │ filter decision                                       │
│             ▼                                                       │
│  ┌─────────────────────┐                                           │
│  │  EXECUTION          │  ← Final trade or skip                    │
│  └─────────────────────┘                                           │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.3 Core Principles

1. **ASM không tự nghĩ entry** - Entry luôn do SMC/VA rules quyết định
2. **ASM chỉ đóng vai critic/filter** - Đánh giá bối cảnh đấu giá
3. **Tư duy Auction/Wyckoff** - Cung-cầu, VA-shift, không phải pattern matching

### 1.4 So sánh với approach cũ

| Aspect | Old (Outcome/Pattern) | New (ASM) |
|--------|----------------------|-----------|
| Question | "Trade này WIN hay LOSS?" | "VA sẽ shift theo hướng nào?" |
| Label | WIN/LOSS per trade | UP/DOWN/NEUTRAL per bar |
| Scope | Only trade bars | All bars (or VA-edge bars) |
| Philosophy | Pattern matching | Auction/Supply-Demand |

---

## 2. Label Design (ASM v1)

### 2.1 Label Definition

```
va_shift_label ∈ {UP, DOWN, NEUTRAL}
```

Tại mỗi **reference bar** (t), nhìn trước **K bar** (K = 30 mặc định):

### 2.2 Label Rules

```python
# Parameters
K = 30                    # Lookahead bars
VA_SHIFT_THRESHOLD = 10   # Ticks để coi là VA shift đủ mạnh
BREAKOUT_HOLD_RATIO = 0.6 # Tỷ lệ bar giữ được breakout

# At reference bar t:
va_center_t = (VAH_t + VAL_t) / 2
va_center_t_plus_K = (VAH_{t+K} + VAL_{t+K}) / 2

va_shift = va_center_t_plus_K - va_center_t

# Count bars outside VA in direction
bars_above_vah = count(close[t+1:t+K] > VAH_t)
bars_below_val = count(close[t+1:t+K] < VAL_t)

# Label assignment
if va_shift >= VA_SHIFT_THRESHOLD:
    if bars_above_vah / K >= BREAKOUT_HOLD_RATIO:
        label = UP
    else:
        label = NEUTRAL  # Breakout failed
        
elif va_shift <= -VA_SHIFT_THRESHOLD:
    if bars_below_val / K >= BREAKOUT_HOLD_RATIO:
        label = DOWN
    else:
        label = NEUTRAL  # Breakout failed
        
else:
    label = NEUTRAL  # No significant VA shift
```

### 2.3 Label Interpretation

| Label | Meaning | Trading Implication |
|-------|---------|---------------------|
| UP | VA shifting up, buyers winning | Favor long setups |
| DOWN | VA shifting down, sellers winning | Favor short setups |
| NEUTRAL | VA stable, balanced auction | Caution, range-bound |

### 2.4 Parameters (Tunable)

| Parameter | Default | Description |
|-----------|---------|-------------|
| K | 30 | Lookahead bars for VA shift |
| VA_SHIFT_THRESHOLD | 10 ticks | Minimum shift to be significant |
| BREAKOUT_HOLD_RATIO | 0.6 | Min ratio of bars holding breakout |

---

## 3. Input Features (ASM)

### 3.1 Feature Groups

| Group | Features | Description |
|-------|----------|-------------|
| **Daily VA Position** | in_va, above_va, below_va, dist_to_vah, dist_to_val, dist_to_poc | Price position relative to Daily VA |
| **Daily VA Structure** | vah_price, val_price, poc_price, va_width, va_center | Current Daily VA parameters |
| **Weekly VA (v1.x)** | weekly_vah, weekly_val, weekly_va_center, in_weekly_va, dist_to_weekly_vah/val | Weekly VA parameters |
| **Daily-Weekly Relation (v1.x)** | daily_va_center_minus_weekly_va_center, daily_vah_minus_weekly_vah, daily_val_minus_weekly_val, daily_va_above/below/inside_weekly | Daily VA drift relative to Weekly VA |
| **SMC Structure** | ext_bos_up/down, ext_choch_up/down, int_bos_up/down, int_choch_up/down | Structure breaks |
| **Swing** | dist_to_swing_high, dist_to_swing_low, swing_high_price, swing_low_price | Swing levels |
| **Volume/Delta** | volume, delta, cum_delta_5/10/20, delta_over_volume | Orderflow |
| **Wave Strength** | impulse_strength, pullback_strength | Wave analysis |
| **Regime** | vol_regime, session (Asia/London/NY) | Market context |
| **Zones** | in_bull_fvg, in_bear_fvg, in_bull_ob, in_bear_ob, dist_to_nearest_fvg | Zone proximity |
| **Candle** | close, high_low_range, body, upper_wick, lower_wick | Price action |

**Note (v1.x)**: Daily VA drift ra khỏi Weekly VA được encode bằng các feature `daily_va_center_minus_weekly_va_center`, `daily_va_above_weekly`, `daily_va_below_weekly`, `daily_va_inside_weekly`. Điều này giúp model nhận biết khi Daily VA đang expand/contract so với Weekly context.

### 3.2 Context Window

- **Sequence length**: 60 bars (configurable)
- **Features per bar**: 112 (v1.x with Weekly VA, was 100 in v1.0)

---

## 4. Dataset Scope

### 4.1 Reference Bar Selection

**Option A: All bars** (recommended for v1)
- Train on every bar
- Pros: Maximum data, no selection bias
- Cons: Many neutral labels, potential imbalance

**Option B: VA-edge bars only**
- Train only on bars near VAH/VAL (within X ticks)
- Pros: More relevant samples
- Cons: Smaller dataset

**Decision for v1**: Start with **Option A** (all bars), handle imbalance with class weights.

### 4.2 Data Sources

```
Existing data:
├── data/processed_v2/*.csv          # 10 weeks training data
├── output/new_data_features/*.csv   # 6 weeks validation data
│
Required additions:
├── VA/VP features (need to verify availability)
└── Future VA calculation for labels (K-bar lookahead)
```

---

## 5. Roadmap

### Step 1: ASM Dataset v1 (Labeling & Build)

**Status**: TODO

**Tasks**:
- [ ] Create `scripts/build_asm_dataset_v1.py`
- [ ] Implement VA shift label calculation
- [ ] Handle K-bar lookahead (no leak - use future data only for labels)
- [ ] Export dataset as JSONL or .pt

**Output**:
```
output/asm_dataset_v1/
├── asm_dataset_v1.jsonl
├── asm_dataset_v1_stats.json
└── asm_dataset_v1_train.pt / _val.pt
```

**Pseudo-code**:
```python
for each bar t in data:
    # Get current VA
    va_t = get_va_at_bar(t)
    
    # Get future VA (K bars ahead)
    va_t_plus_K = get_va_at_bar(t + K)
    
    # Calculate shift
    va_shift = va_t_plus_K.center - va_t.center
    
    # Count breakout bars
    future_bars = data[t+1 : t+K+1]
    bars_above = sum(bar.close > va_t.vah for bar in future_bars)
    bars_below = sum(bar.close < va_t.val for bar in future_bars)
    
    # Assign label
    label = compute_label(va_shift, bars_above, bars_below, K)
    
    # Build context
    context = data[t-60 : t]  # 60 bars before
    
    # Save sample
    save_sample(context, label, meta)
```

### Step 2: ASM Training v1

**Status**: TODO

**Tasks**:
- [ ] Create `phase4_quality_tabular/train_asm_v1.py`
- [ ] Implement GRU/LSTM model for 3-class classification
- [ ] Time-based train/val split (no leak)
- [ ] Track metrics: macro F1, per-class F1, AUC

**Model Architecture**:
```
Input: (batch, seq_len=60, features=70)
    │
    ▼
┌─────────────────┐
│ GRU (hidden=64) │
│ bidirectional   │
│ 2 layers        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Linear (128→64) │
│ ReLU + Dropout  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Linear (64→3)   │
│ Softmax         │
└────────┬────────┘
         │
         ▼
Output: [p_up, p_down, p_neutral]
```

**Loss**: CrossEntropyLoss with class weights (handle imbalance)

**Metrics**:
- Macro F1 (primary)
- Per-class F1: F1_UP, F1_DOWN, F1_NEUTRAL
- AUC for UP vs rest, DOWN vs rest
- Calibration (reliability diagram)

### Step 3: ASM → Strategy Integration

**Status**: TODO

**Integration Flow**:
```
1. Strategy (rule) tạo candidate entry:
   - Ví dụ: LS-CHoCH ở biên VA, VA breakout + retest, FVG/OB + VA confluence
   - Rule tự xác định: side (long/short), entry price, SL, TP

2. ASM đánh giá bối cảnh:
   - Tại bar setup, feed context vào ASM
   - Nhận về: p_up, p_down, p_neutral

3. Filter logic (simple v1):
   - Candidate LONG:  trade nếu (p_up - p_down) >= T_up (e.g., 0.2)
   - Candidate SHORT: trade nếu (p_down - p_up) >= T_down (e.g., 0.2)
   - Otherwise: SKIP

4. Mục tiêu:
   - ASM loại bớt setup trái chiều với VA-shift probability
   - Hoặc trong trạng thái "neutral / choppy"
```

**Key Principle**: ASM KHÔNG thò tay vào entry/SL/TP logic, chỉ là filter/critic.

### Step 4: Shadow Backtest ASM + S4_London

**Status**: TODO

**Tasks**:
- [ ] Create `scripts/shadow_backtest_asm_on_S4.py`
- [ ] Load S4_London baseline trades
- [ ] For each trade: query ASM → get p_up, p_down
- [ ] Apply threshold filter
- [ ] Compare: Baseline vs ASM-filtered

**Output**:
```
| Filter | Trades | Winrate | Expectancy | MaxDD |
|--------|--------|---------|------------|-------|
| Baseline S4_London | xxx | xx.x% | +x.xxxR | xxR |
| ASM p_up >= 0.4 (long) | xxx | xx.x% | +x.xxxR | xxR |
| ASM p_down >= 0.4 (short) | xxx | xx.x% | +x.xxxR | xxR |
```

### Step 5: Production Integration

**Status**: FUTURE

- Integrate ASM into NT-AUTOBOT pipeline
- Real-time inference
- Logging and monitoring

---

## 6. Success Criteria

### 6.1 Model Performance

| Metric | Target |
|--------|--------|
| Macro F1 | > 0.45 |
| F1_UP | > 0.40 |
| F1_DOWN | > 0.40 |
| AUC (UP vs rest) | > 0.65 |
| AUC (DOWN vs rest) | > 0.65 |

### 6.2 Trading Performance (with ASM filter)

| Metric | Target |
|--------|--------|
| Expectancy improvement | > +0.10R vs baseline |
| MaxDD reduction | > 20% vs baseline |
| Trade retention | > 50% of baseline trades |

---

## 7. Files Structure

```
scripts/
├── build_asm_dataset_v1.py          # Dataset builder
├── shadow_backtest_asm_on_S4.py     # Shadow backtest

phase4_quality_tabular/
├── train_asm_v1.py                  # ASM training

output/asm_dataset_v1/
├── asm_dataset_v1.jsonl             # Dataset
├── asm_dataset_v1_stats.json        # Statistics
├── asm_v1_best.pt                   # Trained model
```

---

## 8. References

- [PLAN_S4_HighVol_FVG_London_v1.md](PLAN_S4_HighVol_FVG_London_v1.md) - Baseline strategy
- [PLAN_PIPELINE_SMC_ML_v1.md](PLAN_PIPELINE_SMC_ML_v1.md) - Overall pipeline (to be updated)

---

**Last Updated**: 2025-12-02
