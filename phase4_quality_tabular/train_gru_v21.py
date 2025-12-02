"""
P4 GRU64 v2.1 - Train GRU baseline on new SMC core dataset
"""

import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

print("="*70)
print("P4 GRU64 v2.1 - OUTCOME MODEL TRAINING")
print("="*70)

# ============================================================
# 1. LOAD DATA
# ============================================================
DATA_DIR = ROOT / "output/phase4_outcome_v2.1"
train_data = torch.load(DATA_DIR / "dataset_p4_outcome_v2.1_train.pt")
val_data = torch.load(DATA_DIR / "dataset_p4_outcome_v2.1_val.pt")

X_train, y_train = train_data['X'], train_data['y']
X_val, y_val = val_data['X'], val_data['y']
meta_val = val_data['meta']

print(f"\n[DATA]")
print(f"  Train: {X_train.shape[0]:,} samples, shape {X_train.shape}")
print(f"  Val: {X_val.shape[0]:,} samples, shape {X_val.shape}")
print(f"  Train WIN: {(y_train==1).sum().item():,} ({(y_train==1).float().mean()*100:.1f}%)")
print(f"  Val WIN: {(y_val==1).sum().item():,} ({(y_val==1).float().mean()*100:.1f}%)")

# ============================================================
# 2. MODEL DEFINITION
# ============================================================
class GRUClassifier(nn.Module):
    def __init__(self, input_dim=90, hidden_dim=64, num_layers=1, dropout=0.35):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=0 if num_layers == 1 else dropout
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        # x: (batch, seq_len, features)
        out, h_n = self.gru(x)
        # Use last hidden state
        last_hidden = h_n[-1]  # (batch, hidden_dim)
        last_hidden = self.dropout(last_hidden)
        logit = self.fc(last_hidden)  # (batch, 1)
        return logit.squeeze(-1)

# ============================================================
# 3. TRAINING CONFIG
# ============================================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n[CONFIG]")
print(f"  Device: {DEVICE}")

# Model params
INPUT_DIM = X_train.shape[2]  # 90
HIDDEN_DIM = 64
NUM_LAYERS = 1
DROPOUT = 0.35

# Training params
BATCH_SIZE = 64
MAX_EPOCHS = 30
LR = 1e-3
WEIGHT_DECAY = 1e-3
PATIENCE = 4

# Pos weight for imbalanced classes
n_pos = (y_train == 1).sum().item()
n_neg = (y_train == 0).sum().item()
pos_weight = torch.tensor([n_neg / n_pos]).to(DEVICE)

print(f"  Input dim: {INPUT_DIM}")
print(f"  Hidden dim: {HIDDEN_DIM}")
print(f"  Dropout: {DROPOUT}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  LR: {LR}, Weight decay: {WEIGHT_DECAY}")
print(f"  Pos weight: {pos_weight.item():.2f}")

# ============================================================
# 4. DATA LOADERS
# ============================================================
train_dataset = TensorDataset(X_train, y_train.float())
val_dataset = TensorDataset(X_val, y_val.float())

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ============================================================
# 5. MODEL, LOSS, OPTIMIZER
# ============================================================
model = GRUClassifier(INPUT_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT).to(DEVICE)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

total_params = sum(p.numel() for p in model.parameters())
print(f"  Total params: {total_params:,}")

# ============================================================
# 6. TRAINING LOOP
# ============================================================
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
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
        
        optimizer.zero_grad()
        logits = model(X_batch)
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
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            logits = model(X_batch)
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
    val_f1 = f1_score(all_labels, preds)
    
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
# 7. LOAD BEST MODEL & FINAL EVALUATION
# ============================================================
print(f"\n{'='*70}")
print(f"BEST MODEL: Epoch {best_epoch}, Val AUC = {best_auc:.4f}")
print(f"{'='*70}")

model.load_state_dict(best_state)
model.eval()

# Get predictions on val
with torch.no_grad():
    X_val_dev = X_val.to(DEVICE)
    logits = model(X_val_dev).cpu().numpy()

probs = 1 / (1 + np.exp(-logits))
preds = (probs >= 0.5).astype(int)
labels = y_val.numpy()

# Metrics
val_auc = roc_auc_score(labels, probs)
val_acc = accuracy_score(labels, preds)
val_f1 = f1_score(labels, preds)

print(f"\n[FINAL METRICS @ Best Epoch {best_epoch}]")
print(f"  Val Loss: (see above)")
print(f"  Val AUC: {val_auc:.4f}")
print(f"  Val Acc: {val_acc:.4f}")
print(f"  Val F1: {val_f1:.4f}")

# Confusion matrix
cm = confusion_matrix(labels, preds)
tn, fp, fn, tp = cm.ravel()
print(f"\n[CONFUSION MATRIX]")
print(f"  TP={tp}, FP={fp}")
print(f"  FN={fn}, TN={tn}")
print(f"  Precision: {tp/(tp+fp):.4f}" if (tp+fp) > 0 else "  Precision: N/A")
print(f"  Recall: {tp/(tp+fn):.4f}" if (tp+fn) > 0 else "  Recall: N/A")

# ============================================================
# 8. EXPECTANCY SWEEP
# ============================================================
print(f"\n{'='*70}")
print("EXPECTANCY SWEEP (Val Set)")
print(f"{'='*70}")

# Get outcome_rr from metadata
outcome_rr = np.array([m['outcome_rr'] for m in meta_val])

def compute_expectancy_metrics(mask, outcome_rr, labels):
    """Compute metrics for selected trades"""
    if mask.sum() == 0:
        return {'trades': 0, 'winrate': 0, 'expectancy_R': 0, 'maxDD_R': 0}
    
    selected_rr = outcome_rr[mask]
    selected_labels = labels[mask]
    
    trades = len(selected_rr)
    winrate = selected_labels.mean()
    expectancy = selected_rr.mean()
    
    # Max drawdown (cumulative)
    cumsum = np.cumsum(selected_rr)
    running_max = np.maximum.accumulate(cumsum)
    drawdown = cumsum - running_max
    maxDD = drawdown.min()
    
    return {
        'trades': trades,
        'winrate': winrate,
        'expectancy_R': expectancy,
        'maxDD_R': maxDD
    }

# Baseline (all trades)
baseline = compute_expectancy_metrics(np.ones(len(labels), dtype=bool), outcome_rr, labels)
print(f"\n[BASELINE - All Trades]")
print(f"  Trades: {baseline['trades']}")
print(f"  Winrate: {baseline['winrate']*100:.1f}%")
print(f"  Expectancy: {baseline['expectancy_R']:+.3f}R")
print(f"  MaxDD: {baseline['maxDD_R']:.2f}R")

# Threshold sweep
thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
print(f"\n[THRESHOLD SWEEP]")
print(f"{'Threshold':>10} | {'Trades':>7} | {'Winrate':>8} | {'Expect_R':>10} | {'MaxDD_R':>8}")
print("-" * 55)

best_expectancy = -999
best_threshold = 0

for t in thresholds:
    mask = probs >= t
    metrics = compute_expectancy_metrics(mask, outcome_rr, labels)
    
    print(f"{t:>10.1f} | {metrics['trades']:>7} | {metrics['winrate']*100:>7.1f}% | "
          f"{metrics['expectancy_R']:>+10.3f} | {metrics['maxDD_R']:>8.2f}")
    
    if metrics['expectancy_R'] > best_expectancy and metrics['trades'] >= 50:
        best_expectancy = metrics['expectancy_R']
        best_threshold = t

# ============================================================
# 9. SAVE MODEL
# ============================================================
OUTPUT_DIR = ROOT / "output/phase4_outcome_v2.1"
model_path = OUTPUT_DIR / "gru64_v21_best.pt"
torch.save({
    'model_state': best_state,
    'config': {
        'input_dim': INPUT_DIM,
        'hidden_dim': HIDDEN_DIM,
        'num_layers': NUM_LAYERS,
        'dropout': DROPOUT
    },
    'metrics': {
        'best_epoch': best_epoch,
        'val_auc': val_auc,
        'val_acc': val_acc,
        'val_f1': val_f1
    }
}, model_path)
print(f"\n[SAVED] {model_path}")

# ============================================================
# 10. CONCLUSION
# ============================================================
print(f"\n{'='*70}")
print("CONCLUSION")
print(f"{'='*70}")

if val_auc > 0.60 and best_expectancy >= 0.30:
    print(f"✓ P4_GRU64_v2.1 shows promising edge!")
    print(f"  - Val AUC: {val_auc:.4f} (> 0.60)")
    print(f"  - Best Expectancy: {best_expectancy:+.3f}R @ threshold {best_threshold}")
    print(f"  → Recommend: Keep for deeper backtest")
elif val_auc > 0.55 or best_expectancy >= 0.20:
    print(f"~ P4_GRU64_v2.1 shows moderate signal")
    print(f"  - Val AUC: {val_auc:.4f}")
    print(f"  - Best Expectancy: {best_expectancy:+.3f}R @ threshold {best_threshold}")
    print(f"  → Recommend: Consider adding macro features or adjusting labels")
else:
    print(f"✗ P4_GRU64_v2.1 shows weak/no edge")
    print(f"  - Val AUC: {val_auc:.4f} (~0.50-0.55)")
    print(f"  - Best Expectancy: {best_expectancy:+.3f}R")
    print(f"  → Recommend: Core SMC may not provide outcome signal")
    print(f"     Consider: catastrophic loss filter or macro M5/H1 features")

print(f"\nDone!")
