# PLAN: Regime Model v1 (SMC Core + VA/VP + Volatility)

**Status**: 📋 **PLANNING**  
**Created**: 2025-12-03  
**Last Updated**: 2025-12-03

---

## 0. Objective

### Định nghĩa

Regime Model v1 gắn `regime_tag ∈ {trend_up, trend_down, range}` trên M1 dựa trên:
- **SMC core** (BOS/CHoCH structure)
- **VA/VP** (Daily + Weekly Value Area)
- **Volatility** (ATR, displacement)

**KHÔNG dùng MGannSwing** - chỉ dựa vào SMC structure thuần túy.

### Mục đích

1. **Bật/tắt chiến lược theo regime**:
   - `trend_up` / `trend_down` → Kích hoạt **Breakout / Trend continuation** (S4_HighVol_FVG_London, ...)
   - `range` → Kích hoạt **Fade / Range trades** (VA edge, Wyckoff-style) - thiết kế sau

2. **Feature macro cho các model khác**:
   - ASM: biết context "đang trend hay range" để đánh giá VA-shift
   - Pattern models: filter pattern theo regime phù hợp
   - Outcome models: regime là feature quan trọng

### Phân biệt với các model khác

| Model | Question | Output | Scope |
|-------|----------|--------|-------|
| **Regime Model** | "Thị trường đang trend hay range?" | trend_up / trend_down / range | Macro context |
| **ASM** | "VA sẽ shift hướng nào?" | p_up / p_down / p_neutral | Auction dynamics |
| **Pattern A** | "Có FVG exhaustion pattern?" | pattern_score | Entry timing |

---

## 1. Inputs & Features (thuần SMC core)

### 1.1 SMC Structure (M1/M5/H1)

**Flags external** (từ SMC core):
```
ext_bos_up, ext_bos_down
ext_choch_up, ext_choch_down
```

**Flags internal**:
```
int_bos_up, int_bos_down
int_choch_up, int_choch_down
```

**Counters / Density** (derived):
```python
# Số BOS cùng chiều trong N bar gần nhất
bos_up_count_N      # N = 100 bar M1 hoặc 20 bar H1
bos_down_count_N

# Tần suất đảo chiều
choch_mix_ratio_N   # = choch_count / (bos_count + choch_count)

# Bars since last event
bars_since_last_ext_bos_up
bars_since_last_ext_bos_down
bars_since_last_ext_choch
```

**Directional Persistence**:
```python
# Last dominant BOS direction
ext_trend_dir_h1    # +1 up, -1 down, 0 neutral (trên H1)
ext_trend_dir_m5    # +1 up, -1 down, 0 neutral (trên M5)

# Tính bằng: sign(bos_up_count - bos_down_count) trong N bar
```

### 1.2 VA / Volume Profile (Daily + Weekly)

**Daily VA**:
```python
VAH_D, VAL_D, VPOC_D

# Vị trí giá trong VA [0,1] từ VAL→VAH
pos_in_VA_D = (close - VAL_D) / (VAH_D - VAL_D)

# VPOC drift (so sánh với hôm qua)
drift_VPOC_D = VPOC_D_today - VPOC_D_yesterday
drift_VPOC_D_norm = drift_VPOC_D / atr_d  # normalized
```

**Weekly VA** (roll 5 ngày nếu chưa có VP tuần chuẩn):
```python
VAH_W, VAL_W, VPOC_W

# Vị trí giá trong VA_W
pos_in_VA_W = (close - VAL_W) / (VAH_W - VAL_W)
```

**Relationship features**:
```python
# VA_D nằm trong VA_W?
VA_D_inside_VA_W = (VAL_D >= VAL_W) and (VAH_D <= VAH_W)  # bool

# VPOC_D vs VPOC_W
VPOC_D_vs_W = sign(VPOC_D - VPOC_W)  # +1 trên, -1 dưới, 0 gần
VPOC_D_dist_W = (VPOC_D - VPOC_W) / (VAH_W - VAL_W)  # normalized distance
```

### 1.3 Directional Displacement & Volatility

**ATR-based**:
```python
atr_m1, atr_m15, atr_h1
atr_ratio = atr_m1 / atr_h1  # volatility compression/expansion
```

**Displacement**:
```python
# Net move trong N bar, normalized by ATR
net_move_N = (close - close_N_bars_ago) / atr_h1

# Tỉ lệ BOS up vs down
up_down_bos_ratio_N = bos_up_count_N / max(bos_down_count_N, 1)
```

**Overlap / Range-ness**:
```python
# Mức độ chồng lấp high/low trong N bar
bar_overlap_ratio_N = mean(overlap_i / range_i for i in N bars)
# overlap_i = max(0, min(high_i, high_{i-1}) - max(low_i, low_{i-1}))

# Tỷ lệ wick / body (range nhiều → wick nhiều, body nhỏ)
wickiness_ratio = (upper_wick + lower_wick) / max(body, epsilon)
avg_wickiness_N = mean(wickiness_ratio for N bars)
```

### 1.4 Session / Time-of-day (optional)

```python
session         # Asia=0, London=1, NY=2
hour_of_day     # 0-23 hoặc binned (0-7 Asia, 8-14 London, 15-21 NY)
```

---

## 2. Label Rules – Regime_tag v1 (Rule-based)

### Ngưỡng mặc định (v1, sẽ tinh chỉnh)

```python
# Lookback windows
N_M1 = 100      # 100 bar M1 (~1.5 giờ)
N_H1 = 20       # 20 bar H1 (~20 giờ)

# BOS thresholds
BOS_DOMINANT_THRESHOLD = 3      # Số BOS cùng chiều tối thiểu
BOS_OPPOSITE_MAX = 1            # Số BOS ngược chiều tối đa

# Displacement thresholds
NET_MOVE_TREND_THRESHOLD = 1.5  # ATR units
NET_MOVE_RANGE_THRESHOLD = 0.5  # ATR units

# VA position thresholds
POS_VA_UPPER = 0.6              # Nửa trên VA
POS_VA_LOWER = 0.4              # Nửa dưới VA
POS_VA_MID_LOW = 0.3            # Range zone lower
POS_VA_MID_HIGH = 0.7           # Range zone upper

# Overlap threshold
OVERLAP_RANGE_THRESHOLD = 0.5   # High overlap = range
```

### 2.1 trend_up (Macro Uptrend)

```python
def is_trend_up(features):
    # 1. SMC Structure: BOS up chiếm ưu thế
    bos_up_dominant = (
        features['bos_up_count_N'] >= BOS_DOMINANT_THRESHOLD and
        features['bos_down_count_N'] <= BOS_OPPOSITE_MAX
    )
    
    # 2. Không có CHoCH down gần đây (tránh đảo trend)
    no_recent_choch_down = features['bars_since_last_ext_choch_down'] > 20
    
    # 3. VA/VPOC drift dương
    vpoc_drifting_up = features['drift_VPOC_D_norm'] > 0
    
    # 4. Giá ở nửa trên VA_W hoặc trên VPOC_W
    price_position_bullish = (
        features['pos_in_VA_W'] >= POS_VA_UPPER or
        features['VPOC_D_vs_W'] >= 0
    )
    
    # 5. Displacement dương và đủ lớn
    displacement_up = features['net_move_N'] >= NET_MOVE_TREND_THRESHOLD
    
    return (
        bos_up_dominant and
        no_recent_choch_down and
        (vpoc_drifting_up or price_position_bullish) and
        displacement_up
    )
```

### 2.2 trend_down (Macro Downtrend)

```python
def is_trend_down(features):
    # 1. SMC Structure: BOS down chiếm ưu thế
    bos_down_dominant = (
        features['bos_down_count_N'] >= BOS_DOMINANT_THRESHOLD and
        features['bos_up_count_N'] <= BOS_OPPOSITE_MAX
    )
    
    # 2. Không có CHoCH up gần đây
    no_recent_choch_up = features['bars_since_last_ext_choch_up'] > 20
    
    # 3. VA/VPOC drift âm
    vpoc_drifting_down = features['drift_VPOC_D_norm'] < 0
    
    # 4. Giá ở nửa dưới VA_W hoặc dưới VPOC_W
    price_position_bearish = (
        features['pos_in_VA_W'] <= POS_VA_LOWER or
        features['VPOC_D_vs_W'] <= 0
    )
    
    # 5. Displacement âm và đủ lớn
    displacement_down = features['net_move_N'] <= -NET_MOVE_TREND_THRESHOLD
    
    return (
        bos_down_dominant and
        no_recent_choch_up and
        (vpoc_drifting_down or price_position_bearish) and
        displacement_down
    )
```

### 2.3 range (Consolidation / Sideways)

```python
def is_range(features):
    # Không thoả trend_up hay trend_down
    not_trending = not is_trend_up(features) and not is_trend_down(features)
    
    # Bổ sung điều kiện range-specific:
    
    # 1. BOS up/down xen kẽ
    bos_mixed = (
        0.5 <= features['up_down_bos_ratio_N'] <= 2.0
    )
    
    # 2. Bar overlap cao
    high_overlap = features['bar_overlap_ratio_N'] >= OVERLAP_RANGE_THRESHOLD
    
    # 3. Giá dao động trong vùng giữa VA_W
    price_in_mid_va = (
        POS_VA_MID_LOW <= features['pos_in_VA_W'] <= POS_VA_MID_HIGH
    )
    
    # 4. VPOC drift nhỏ
    vpoc_stable = abs(features['drift_VPOC_D_norm']) < 0.3
    
    # 5. Displacement nhỏ
    low_displacement = abs(features['net_move_N']) <= NET_MOVE_RANGE_THRESHOLD
    
    # Range nếu không trending VÀ có ít nhất 2/4 điều kiện range
    range_signals = sum([bos_mixed, high_overlap, price_in_mid_va, vpoc_stable])
    
    return not_trending and (range_signals >= 2 or low_displacement)
```

### 2.4 Final Label Assignment

```python
def assign_regime_tag(features):
    if is_trend_up(features):
        return 'trend_up'      # 0
    elif is_trend_down(features):
        return 'trend_down'    # 1
    else:
        return 'range'         # 2
```

---

## 3. Dataset & Scripts

### 3.1 Script: `scripts/build_regime_dataset_v1.py`

**Pipeline**:

```
1. Load feature stream M1
   └── Đã có: SMC core + VA + session từ SMCContextManager

2. Compute derived features
   └── bos_count_N, choch_mix_ratio, bars_since_*, overlap_ratio, etc.

3. Apply rule-based labeling (Section 2)
   └── Gán regime_tag cho mỗi bar

4. Create sequence samples
   └── X: (N_samples, T_regime=60, D_regime)
   └── y: {0=trend_up, 1=trend_down, 2=range}

5. Time-based split
   └── Train: 70% đầu
   └── Val: 30% sau

6. Save outputs
```

**Output files**:

```
output/regime_dataset_v1/
├── regime_v1_train.pt          # Training sequences
├── regime_v1_val.pt            # Validation sequences
├── regime_v1_stats.json        # Statistics
└── regime_v1_label_dist.png    # Label distribution plot
```

**Stats JSON format**:

```json
{
  "train": {
    "n_samples": 10000,
    "shape": [10000, 60, 25],
    "label_dist": {
      "trend_up": 3500,
      "trend_down": 3200,
      "range": 3300
    }
  },
  "val": {
    "n_samples": 3000,
    "shape": [3000, 60, 25],
    "label_dist": {
      "trend_up": 1000,
      "trend_down": 950,
      "range": 1050
    }
  },
  "feature_names": ["bos_up_count_N", "bos_down_count_N", ...],
  "thresholds": {
    "N_M1": 100,
    "BOS_DOMINANT_THRESHOLD": 3,
    ...
  }
}
```

### 3.2 Sequence Design

```python
# Sequence parameters
T_REGIME = 60           # 60 bar M1 context (~1 hour)
D_REGIME = ~25          # Feature dimension (estimated)
STRIDE = 10             # Sampling stride (mỗi 10 bar lấy 1 sample)

# Sample structure
X[i] = features[t-T_REGIME:t]   # Shape: (60, D_regime)
y[i] = regime_tag[t]            # Label tại bar cuối sequence
```

---

## 4. (Optional) RegimeModel_v1 – ML Classifier

> **Status**: 📋 ROADMAP - Chưa cần implement

### 4.1 Model Architecture

```python
class RegimeModelV1(nn.Module):
    """
    GRU-based 3-class classifier for regime detection.
    """
    def __init__(self, input_dim=25, hidden_dim=64, num_layers=2, num_classes=3):
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        # x: (batch, seq_len=60, input_dim)
        _, h_n = self.gru(x)
        out = self.fc(h_n[-1])
        return out  # (batch, 3) logits
```

### 4.2 Training Config

```yaml
model:
  type: GRU
  input_dim: 25
  hidden_dim: 64
  num_layers: 2
  num_classes: 3

training:
  epochs: 50
  batch_size: 64
  lr: 0.001
  weight_decay: 0.0001
  class_weights: auto  # Handle imbalance

loss: CrossEntropyLoss
optimizer: AdamW
scheduler: CosineAnnealingLR
```

### 4.3 Evaluation Metrics

| Metric | Target | Description |
|--------|--------|-------------|
| Macro F1 | > 0.50 | Balanced across 3 classes |
| Accuracy | > 0.55 | Overall accuracy |
| AUC (trend_up) | > 0.65 | One-vs-rest |
| AUC (trend_down) | > 0.65 | One-vs-rest |
| AUC (range) | > 0.60 | One-vs-rest |

### 4.4 Purpose of ML Model

- **Không thay thế 100% rule-based** - rule-based vẫn là baseline
- **Làm mượt** các vùng ranh giới (transition zones)
- **Học nuance** mà rule-based không capture được
- **Ensemble option**: `final_regime = vote(rule_based, ml_model)`

---

## 5. Integration với Pipeline Chiến Lược

### 5.1 Breakout / Trend Continuation Strategies

**S4_HighVol_FVG_London** (và các strategy tương tự):

```python
def should_activate_s4(regime_tag, entry_direction):
    """
    S4 chỉ kích hoạt khi regime phù hợp.
    """
    if regime_tag == 'trend_up':
        # Ưu tiên S4 long
        return entry_direction == 'long'
    
    elif regime_tag == 'trend_down':
        # Ưu tiên S4 short
        return entry_direction == 'short'
    
    elif regime_tag == 'range':
        # Không kích hoạt S4 trong range
        return False
    
    return False
```

**Integration flow**:

```
1. S4 generates candidate entry (long/short)
2. Check regime_tag at entry bar
3. If regime matches direction → proceed to ASM filter
4. If regime = range → skip entry
```

### 5.2 Range / Fade Strategies (Future)

> **Status**: 📋 TODO - Thiết kế sau

```python
def should_activate_fade(regime_tag, price_position):
    """
    Fade strategy chỉ kích hoạt trong range.
    """
    if regime_tag != 'range':
        return False
    
    # Fade tại VA edges
    if price_position == 'near_VAH':
        return 'short'  # Fade từ VAH
    elif price_position == 'near_VAL':
        return 'long'   # Fade từ VAL
    
    return False
```

**Kết hợp với ASM**:
- Fade khi `regime_tag = range` VÀ `ASM.p_neutral` cao
- Tức là: auction ổn định, VA không shift → mean reversion có edge

### 5.3 Regime as Feature cho các Model khác

**ASM (Auction State Model)**:
```python
# Thêm regime_tag vào feature set của ASM
asm_features = [
    ...existing_features...,
    'regime_tag_onehot',  # [is_trend_up, is_trend_down, is_range]
]
```

**Pattern Models**:
```python
# Filter pattern theo regime
if regime_tag in ['trend_up', 'trend_down']:
    # Pattern A (FVG continuation) có edge
    use_pattern_a = True
else:
    # Range → pattern A ít edge
    use_pattern_a = False
```

### 5.4 Liên kết với các PLAN khác

| PLAN | Relationship |
|------|--------------|
| **PLAN_PIPELINE_SMC_ML_v1.md** | Regime Model là parallel track, cung cấp macro context |
| **PLAN_S4_HighVol_FVG_London_v1.md** | S4 là trend continuation → cần regime = trend_up/down |
| **PLAN_AuctionStateModel_v1.md** | ASM đo VA-shift, Regime đo macro trend vs range |

---

## 6. Implementation Roadmap

### Phase 1: Rule-based Regime (Current Focus)

- [ ] Implement derived features (bos_count, overlap_ratio, etc.)
- [ ] Implement rule-based labeling functions
- [ ] Create `scripts/build_regime_dataset_v1.py`
- [ ] Validate label distribution trên historical data
- [ ] Visualize regime transitions

### Phase 2: Integration với S4

- [ ] Add regime check vào S4 entry logic
- [ ] Backtest S4 với regime filter
- [ ] Compare: S4 baseline vs S4 + regime filter

### Phase 3: ML Regime Model (Optional)

- [ ] Train GRU classifier
- [ ] Evaluate metrics (F1, AUC)
- [ ] Compare: rule-based vs ML vs ensemble

### Phase 4: Range Strategy

- [ ] Design fade/range strategy
- [ ] Integrate với regime_tag = range
- [ ] Backtest range strategy

---

## 7. Success Criteria

### Rule-based Regime v1

| Metric | Target |
|--------|--------|
| Label coverage | > 95% bars có regime_tag |
| Class balance | Không class nào < 20% |
| Visual validation | Regime transitions hợp lý trên chart |

### S4 + Regime Filter

| Metric | Target | Baseline (S4 alone) |
|--------|--------|---------------------|
| Expectancy | > +0.70R | +0.654R |
| Trade retention | > 60% | 100% |
| MaxDD | < 12R | 15.1R |

### ML Regime Model (nếu implement)

| Metric | Target |
|--------|--------|
| Macro F1 | > 0.50 |
| AUC (trend_up) | > 0.65 |
| AUC (trend_down) | > 0.65 |

---

## References

- [PLAN_PIPELINE_SMC_ML_v1.md](PLAN_PIPELINE_SMC_ML_v1.md) - Overall pipeline
- [PLAN_S4_HighVol_FVG_London_v1.md](PLAN_S4_HighVol_FVG_London_v1.md) - Baseline strategy
- [PLAN_AuctionStateModel_v1.md](PLAN_AuctionStateModel_v1.md) - Auction state model

---

**Last Updated**: 2025-12-03
