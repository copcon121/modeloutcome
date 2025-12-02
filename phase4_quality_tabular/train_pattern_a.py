"""
Train Pattern A Detector - HighVol FVG Retest Exhaustion

Binary classifier to detect Pattern A from pattern_dataset_v1.jsonl.
Target: AUC > 0.70 (pattern recognition should be easier than outcome prediction)

Usage:
    python phase4_quality_tabular/train_pattern_a.py [--model gru|xgb|mlp]
"""

import sys
from pathlib import Path
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix, accuracy_score, precision_recall_curve
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

print("="*70)
print("TRAIN PATTERN A DETECTOR - HighVol FVG Retest Exhaustion")
print("="*70)

# ============================================================
# 1. CONFIGURATION
# ============================================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Feature groups to use (matching build_pattern_dataset_v1.py)
FEATURE_GROUPS = {
    'price_candle': ['close', 'high_low_range', 'body', 'upper_wick', 'lower_wick', 'open', 'high', 'low'],
    'volume_delta': ['volume', 'delta', 'delta_over_volume', 'buy_volume', 'sell_volume', 
                     'impulse_strength', 'pullback_strength', 'cum_delta_5', 'cum_delta_10', 'cum_delta_20'],
    'smc_core': ['ext_bos_up', 'ext_bos_down', 'ext_choch_up', 'ext_choch_down',
                 'int_bos_up', 'int_bos_down', 'int_choch_up', 'int_choch_down',
                 'swept_prev_int_high', 'swept_prev_int_low', 'swept_prev_ext_high', 'swept_prev_ext_low',
                 'in_bull_fvg', 'in_bear_fvg', 'near_bull_fvg', 'near_bear_fvg',
                 'int_in_bull_ob', 'int_in_bear_ob', 'ext_in_bull_ob', 'ext_in_bear_ob',
                 'int_near_bull_ob', 'int_near_bear_ob', 'ext_near_bull_ob', 'ext_near_bear_ob',
                 'ext_trend_dir', 'int_trend_dir'],
    'macro': ['m5_trend_up', 'm5_trend_down', 'm5_premium', 'm5_discount',
              'dist_to_m5_swing_high', 'dist_to_m5_swing_low', 'near_m5_fvg',
              'h1_trend_up', 'h1_trend_down', 'h1_premium', 'h1_discount',
              'dist_to_h1_swing_high', 'dist_to_h1_swing_low', 'near_h1_fvg'],
    'volume_profile': ['vp_poc_price', 'vp_val_price', 'vp_vah_price',
                       'vp_in_value_area', 'vp_above_value_area', 'vp_below_value_area',
                       'vp_dist_to_poc', 'vp_dist_to_vah', 'vp_dist_to_val'],
}

# Flatten feature list
ALL_FEATURES = []
for group, features in FEATURE_GROUPS.items():
    ALL_FEATURES.extend(features)

# Context window
CTX_WINDOW = 59  # 59 context bars + 1 current = 60 total

# Training params
BATCH_SIZE = 64
MAX_EPOCHS = 50
LR = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 8

# ============================================================
# 2. LOAD DATA
# ============================================================
DATA_PATH = ROOT / "output/pattern_dataset_v1/pattern_dataset_v1.jsonl"

print(f"\n[1/5] Loading data from {DATA_PATH}...")

events = []
with open(DATA_PATH, 'r') as f:
    for line in f:
        events.append(json.loads(line))

print(f"  Loaded {len(events):,} events")

# ============================================================
# 3. FEATURE EXTRACTION
# ============================================================
print(f"\n[2/5] Extracting features...")

def extract_features(event: dict, feature_list: list) -> tuple:
    """
    Extract context sequence and current bar features from event.
    Returns (context_seq, current_vec, label)
    """
    context = event.get('context', [])
    current = event.get('current', {})
    label = event['pattern_labels']['P_A_RETEST_EXHAUST']
    
    # Build context sequence (CTX_WINDOW x num_features)
    ctx_seq = []
    for bar in context[-CTX_WINDOW:]:  # Take last CTX_WINDOW bars
        bar_vec = []
        for f in feature_list:
            val = bar.get(f, 0)
            bar_vec.append(float(val) if val is not None else 0.0)
        ctx_seq.append(bar_vec)
    
    # Pad if needed
    while len(ctx_seq) < CTX_WINDOW:
        ctx_seq.insert(0, [0.0] * len(feature_list))
    
    # Current bar features
    curr_vec = []
    for f in feature_list:
        val = current.get(f, 0)
        curr_vec.append(float(val) if val is not None else 0.0)
    
    # Add local stats from current
    local_stats = ['local_avg_vol', 'local_avg_delta', 'local_min_delta_z', 'local_max_delta_z',
                   'touch_fvg_bull', 'touch_fvg_bear', 'touch_ob_bull', 'touch_ob_bear', 'near_zone']
    for f in local_stats:
        val = current.get(f, 0)
        curr_vec.append(float(val) if val is not None else 0.0)
    
    return np.array(ctx_seq), np.array(curr_vec), label


# Determine available features from first event
first_ctx = events[0].get('context', [{}])
if first_ctx:
    available_features = [f for f in ALL_FEATURES if f in first_ctx[0]]
else:
    available_features = ALL_FEATURES

print(f"  Available features: {len(available_features)}")

# Extract all
X_ctx = []  # Context sequences
X_curr = []  # Current bar features
y = []
meta = []

for event in events:
    ctx_seq, curr_vec, label = extract_features(event, available_features)
    X_ctx.append(ctx_seq)
    X_curr.append(curr_vec)
    y.append(label)
    meta.append({
        'bar_index': event.get('bar_index', 0),
        'session': event.get('session', 'Unknown'),
        'time': event.get('time', ''),
        'outcome_rr': event['task_labels'].get('task_outcome_rr', 0),
        'outcome_winloss': event['task_labels'].get('task_outcome_winloss', -1),
    })

X_ctx = np.array(X_ctx, dtype=np.float32)
X_curr = np.array(X_curr, dtype=np.float32)
y = np.array(y, dtype=np.int64)

print(f"  Context shape: {X_ctx.shape}")  # (N, CTX_WINDOW, features)
print(f"  Current shape: {X_curr.shape}")  # (N, features + local_stats)
print(f"  Labels: {len(y)}, Positive: {y.sum()} ({y.mean()*100:.1f}%)")

# ============================================================
# 4. TRAIN/VAL SPLIT (TIME-BASED - ANTI-LEAK)
# ============================================================
print(f"\n[3/5] Splitting train/val (TIME-BASED 70/30)...")

# Sort by time to ensure no future leak
from datetime import datetime

def parse_time(time_str):
    """Parse timestamp string to datetime"""
    try:
        if not time_str:
            return datetime.min
        if 'T' in time_str:
            return datetime.fromisoformat(time_str.replace('Z', '').replace('+00:00', ''))
        return datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S')
    except:
        return datetime.min

# Get indices sorted by time
time_indices = [(i, parse_time(meta[i]['time'])) for i in range(len(meta))]
time_indices.sort(key=lambda x: x[1])
sorted_indices = [i for i, _ in time_indices]

# Time-based split: 70% train (earlier), 30% val (later)
n_train = int(len(sorted_indices) * 0.7)
train_idx = sorted_indices[:n_train]
val_idx = sorted_indices[n_train:]

# Get time ranges for logging
train_times = [parse_time(meta[i]['time']) for i in train_idx]
val_times = [parse_time(meta[i]['time']) for i in val_idx]

train_start = min(t for t in train_times if t != datetime.min)
train_end = max(t for t in train_times if t != datetime.min)
val_start = min(t for t in val_times if t != datetime.min)
val_end = max(t for t in val_times if t != datetime.min)

print(f"  TRAIN: {train_start.date()} to {train_end.date()}")
print(f"  VAL:   {val_start.date()} to {val_end.date()}")

X_ctx_train, X_ctx_val = X_ctx[train_idx], X_ctx[val_idx]
X_curr_train, X_curr_val = X_curr[train_idx], X_curr[val_idx]
y_train, y_val = y[train_idx], y[val_idx]
meta_val = [meta[i] for i in val_idx]

print(f"  Train: {len(y_train):,} samples, Positive: {y_train.sum()} ({y_train.mean()*100:.1f}%)")
print(f"  Val: {len(y_val):,} samples, Positive: {y_val.sum()} ({y_val.mean()*100:.1f}%)")

# Normalize features
def normalize_features(X_train, X_val, eps=1e-6):
    """Z-score normalization"""
    mean = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True) + eps
    return (X_train - mean) / std, (X_val - mean) / std, mean, std

# Normalize context (per feature across all bars)
X_ctx_train_flat = X_ctx_train.reshape(-1, X_ctx_train.shape[-1])
X_ctx_val_flat = X_ctx_val.reshape(-1, X_ctx_val.shape[-1])
X_ctx_train_norm, X_ctx_val_norm, ctx_mean, ctx_std = normalize_features(X_ctx_train_flat, X_ctx_val_flat)
X_ctx_train = X_ctx_train_norm.reshape(X_ctx_train.shape)
X_ctx_val = X_ctx_val_norm.reshape(X_ctx_val.shape)

# Normalize current
X_curr_train, X_curr_val, curr_mean, curr_std = normalize_features(X_curr_train, X_curr_val)

# Convert to tensors
X_ctx_train = torch.tensor(X_ctx_train, dtype=torch.float32)
X_ctx_val = torch.tensor(X_ctx_val, dtype=torch.float32)
X_curr_train = torch.tensor(X_curr_train, dtype=torch.float32)
X_curr_val = torch.tensor(X_curr_val, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32)

# ============================================================
# 5. MODEL DEFINITION
# ============================================================

class PatternGRU(nn.Module):
    """
    GRU-based Pattern Detector.
    Uses context sequence + current bar features.
    """
    def __init__(self, ctx_dim, curr_dim, hidden_dim=64, num_layers=2, dropout=0.3):
        super().__init__()
        
        # GRU for context sequence
        self.gru = nn.GRU(
            input_size=ctx_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Current bar encoder
        self.curr_encoder = nn.Sequential(
            nn.Linear(curr_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Classifier head
        # GRU output: hidden_dim * 2 (bidirectional) + hidden_dim (current)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2 + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, ctx, curr):
        # ctx: (batch, seq_len, ctx_dim)
        # curr: (batch, curr_dim)
        
        # GRU encoding
        out, h_n = self.gru(ctx)
        # h_n: (num_layers * 2, batch, hidden_dim) for bidirectional
        # Take last layer forward and backward
        h_forward = h_n[-2]  # (batch, hidden_dim)
        h_backward = h_n[-1]  # (batch, hidden_dim)
        ctx_enc = torch.cat([h_forward, h_backward], dim=1)  # (batch, hidden_dim * 2)
        
        # Current bar encoding
        curr_enc = self.curr_encoder(curr)  # (batch, hidden_dim)
        
        # Combine and classify
        combined = torch.cat([ctx_enc, curr_enc], dim=1)
        logit = self.classifier(combined)
        
        return logit.squeeze(-1)


class PatternMLP(nn.Module):
    """
    MLP-based Pattern Detector.
    Flattens context + current into single vector.
    """
    def __init__(self, ctx_dim, ctx_len, curr_dim, hidden_dim=256, dropout=0.3):
        super().__init__()
        
        input_dim = ctx_dim * ctx_len + curr_dim
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, ctx, curr):
        # Flatten context
        ctx_flat = ctx.view(ctx.size(0), -1)
        combined = torch.cat([ctx_flat, curr], dim=1)
        return self.net(combined).squeeze(-1)


# ============================================================
# 6. TRAINING
# ============================================================
print(f"\n[4/5] Training Pattern A Detector...")

# Model config
CTX_DIM = X_ctx_train.shape[2]
CURR_DIM = X_curr_train.shape[1]
HIDDEN_DIM = 64
NUM_LAYERS = 2
DROPOUT = 0.3

print(f"\n[CONFIG]")
print(f"  Device: {DEVICE}")
print(f"  Context dim: {CTX_DIM}, seq_len: {CTX_WINDOW}")
print(f"  Current dim: {CURR_DIM}")
print(f"  Hidden dim: {HIDDEN_DIM}")
print(f"  Dropout: {DROPOUT}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  LR: {LR}, Weight decay: {WEIGHT_DECAY}")

# Create model
model = PatternGRU(CTX_DIM, CURR_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT).to(DEVICE)
total_params = sum(p.numel() for p in model.parameters())
print(f"  Total params: {total_params:,}")

# Pos weight for imbalanced classes
n_pos = (y_train == 1).sum().item()
n_neg = (y_train == 0).sum().item()
pos_weight = torch.tensor([n_neg / n_pos]).to(DEVICE)
print(f"  Pos weight: {pos_weight.item():.2f}")

# Loss and optimizer
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

# Data loaders
class PatternDataset(torch.utils.data.Dataset):
    def __init__(self, ctx, curr, labels):
        self.ctx = ctx
        self.curr = curr
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.ctx[idx], self.curr[idx], self.labels[idx]

train_dataset = PatternDataset(X_ctx_train, X_curr_train, y_train)
val_dataset = PatternDataset(X_ctx_val, X_curr_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Training loop
print(f"\n{'='*70}")
print("TRAINING")
print(f"{'='*70}")

best_auc = 0
best_epoch = 0
patience_counter = 0
best_state = None

for epoch in range(MAX_EPOCHS):
    # Train
    model.train()
    train_loss = 0
    for ctx_batch, curr_batch, y_batch in train_loader:
        ctx_batch = ctx_batch.to(DEVICE)
        curr_batch = curr_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)
        
        optimizer.zero_grad()
        logits = model(ctx_batch, curr_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        train_loss += loss.item() * len(y_batch)
    
    train_loss /= len(train_dataset)
    
    # Validate
    model.eval()
    val_loss = 0
    all_logits = []
    all_labels = []
    
    with torch.no_grad():
        for ctx_batch, curr_batch, y_batch in val_loader:
            ctx_batch = ctx_batch.to(DEVICE)
            curr_batch = curr_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            
            logits = model(ctx_batch, curr_batch)
            loss = criterion(logits, y_batch)
            val_loss += loss.item() * len(y_batch)
            all_logits.append(logits.cpu())
            all_labels.append(y_batch.cpu())
    
    val_loss /= len(val_dataset)
    all_logits = torch.cat(all_logits).numpy()
    all_labels = torch.cat(all_labels).numpy()
    
    # Metrics
    probs = 1 / (1 + np.exp(-all_logits))  # sigmoid
    preds = (probs >= 0.5).astype(int)
    
    val_auc = roc_auc_score(all_labels, probs)
    val_acc = accuracy_score(all_labels, preds)
    val_f1 = f1_score(all_labels, preds, zero_division=0)
    
    # Update scheduler
    scheduler.step(val_auc)
    
    # Check improvement
    improved = ""
    if val_auc > best_auc:
        best_auc = val_auc
        best_epoch = epoch + 1
        best_state = model.state_dict().copy()
        patience_counter = 0
        improved = " *"
    else:
        patience_counter += 1
    
    print(f"Epoch {epoch+1:2d}/{MAX_EPOCHS} | "
          f"Train Loss: {train_loss:.4f} | "
          f"Val Loss: {val_loss:.4f} | "
          f"Val AUC: {val_auc:.4f} | "
          f"Val Acc: {val_acc:.4f} | "
          f"Val F1: {val_f1:.4f}{improved}")
    
    # Early stopping
    if patience_counter >= PATIENCE:
        print(f"\nEarly stopping at epoch {epoch+1} (patience={PATIENCE})")
        break

# ============================================================
# 7. FINAL EVALUATION
# ============================================================
print(f"\n{'='*70}")
print(f"BEST MODEL: Epoch {best_epoch}, Val AUC = {best_auc:.4f}")
print(f"{'='*70}")

model.load_state_dict(best_state)
model.eval()

# Get predictions on val
with torch.no_grad():
    all_logits = []
    for ctx_batch, curr_batch, _ in val_loader:
        ctx_batch = ctx_batch.to(DEVICE)
        curr_batch = curr_batch.to(DEVICE)
        logits = model(ctx_batch, curr_batch)
        all_logits.append(logits.cpu())
    
    all_logits = torch.cat(all_logits).numpy()

probs = 1 / (1 + np.exp(-all_logits))
labels = y_val.numpy()

# Metrics at different thresholds
print(f"\n[THRESHOLD ANALYSIS]")
print(f"{'Threshold':>10} | {'Pred_Pos':>8} | {'TP':>5} | {'FP':>5} | {'Precision':>10} | {'Recall':>8} | {'F1':>8}")
print("-" * 75)

for t in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
    preds = (probs >= t).astype(int)
    tp = ((preds == 1) & (labels == 1)).sum()
    fp = ((preds == 1) & (labels == 0)).sum()
    fn = ((preds == 0) & (labels == 1)).sum()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"{t:>10.1f} | {preds.sum():>8} | {tp:>5} | {fp:>5} | {precision:>10.4f} | {recall:>8.4f} | {f1:>8.4f}")

# Confusion matrix at 0.5
preds_05 = (probs >= 0.5).astype(int)
cm = confusion_matrix(labels, preds_05)
tn, fp, fn, tp = cm.ravel()

print(f"\n[CONFUSION MATRIX @ 0.5]")
print(f"  TP={tp}, FP={fp}")
print(f"  FN={fn}, TN={tn}")

# Final metrics
val_auc = roc_auc_score(labels, probs)
val_acc = accuracy_score(labels, preds_05)
val_f1 = f1_score(labels, preds_05, zero_division=0)

print(f"\n[FINAL METRICS]")
print(f"  Val AUC: {val_auc:.4f}")
print(f"  Val Acc: {val_acc:.4f}")
print(f"  Val F1: {val_f1:.4f}")
print(f"  Precision @ 0.5: {tp/(tp+fp):.4f}" if (tp+fp) > 0 else "  Precision: N/A")
print(f"  Recall @ 0.5: {tp/(tp+fn):.4f}" if (tp+fn) > 0 else "  Recall: N/A")

# ============================================================
# 8. SAVE MODEL
# ============================================================
OUTPUT_DIR = ROOT / "output/pattern_dataset_v1"
model_path = OUTPUT_DIR / "pattern_a_gru_best.pt"

torch.save({
    'model_state': best_state,
    'config': {
        'ctx_dim': CTX_DIM,
        'curr_dim': CURR_DIM,
        'hidden_dim': HIDDEN_DIM,
        'num_layers': NUM_LAYERS,
        'dropout': DROPOUT,
        'ctx_window': CTX_WINDOW,
    },
    'normalization': {
        'ctx_mean': ctx_mean,
        'ctx_std': ctx_std,
        'curr_mean': curr_mean,
        'curr_std': curr_std,
    },
    'metrics': {
        'best_epoch': best_epoch,
        'val_auc': val_auc,
        'val_acc': val_acc,
        'val_f1': val_f1,
    },
    'features': available_features,
}, model_path)

print(f"\n[SAVED] {model_path}")

# ============================================================
# 9. CONCLUSION
# ============================================================
print(f"\n{'='*70}")
print("CONCLUSION")
print(f"{'='*70}")

if val_auc >= 0.70:
    print(f"✓ Pattern A Detector shows GOOD signal!")
    print(f"  - Val AUC: {val_auc:.4f} (>= 0.70 target)")
    print(f"  → Recommend: Use as entry filter in trading system")
elif val_auc >= 0.60:
    print(f"~ Pattern A Detector shows MODERATE signal")
    print(f"  - Val AUC: {val_auc:.4f}")
    print(f"  → Recommend: Try adding more features or adjusting label rules")
else:
    print(f"✗ Pattern A Detector shows WEAK signal")
    print(f"  - Val AUC: {val_auc:.4f} (< 0.60)")
    print(f"  → Recommend: Review pattern definition or feature engineering")

print(f"\nDone!")
