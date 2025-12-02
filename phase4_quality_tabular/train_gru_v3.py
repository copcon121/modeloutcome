"""
P4 GRU64 v3 - Train with Enhanced Macro Features (100 features)
"""

import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix, accuracy_score

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

print("="*70)
print("P4 GRU64 v3 - ENHANCED MACRO FEATURES (100 features)")
print("="*70)

# Load data
DATA_DIR = ROOT / "output/phase4_outcome_v3"
train_data = torch.load(DATA_DIR / "dataset_p4_outcome_v3_train.pt")
val_data = torch.load(DATA_DIR / "dataset_p4_outcome_v3_val.pt")

X_train, y_train = train_data['X'], train_data['y']
X_val, y_val = val_data['X'], val_data['y']
meta_val = val_data['meta']

print(f"\n[DATA] Train: {X_train.shape}, Val: {X_val.shape}")
print(f"  Train WIN: {(y_train==1).sum().item()} ({(y_train==1).float().mean()*100:.1f}%)")
print(f"  Val WIN: {(y_val==1).sum().item()} ({(y_val==1).float().mean()*100:.1f}%)")

# Model
class GRUClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, dropout=0.35):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        _, h_n = self.gru(x)
        return self.fc(self.dropout(h_n[-1])).squeeze(-1)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
INPUT_DIM = X_train.shape[2]
model = GRUClassifier(INPUT_DIM).to(DEVICE)

n_pos = (y_train == 1).sum().item()
n_neg = (y_train == 0).sum().item()
pos_weight = torch.tensor([n_neg / n_pos]).to(DEVICE)

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)

train_loader = DataLoader(TensorDataset(X_train, y_train.float()), batch_size=64, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val, y_val.float()), batch_size=64)

print(f"\n[CONFIG] Input: {INPUT_DIM}, Hidden: 64, Pos weight: {pos_weight.item():.2f}")

# Training
best_auc, best_epoch, best_state = 0, 0, None
patience_counter = 0

print(f"\n{'='*70}\nTRAINING\n{'='*70}")
for epoch in range(30):
    model.train()
    for X_b, y_b in train_loader:
        X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(X_b), y_b)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for X_b, y_b in val_loader:
            all_logits.append(model(X_b.to(DEVICE)).cpu())
            all_labels.append(y_b)
    
    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy()
    probs = 1 / (1 + np.exp(-logits))
    
    val_auc = roc_auc_score(labels, probs)
    val_acc = accuracy_score(labels, (probs >= 0.5).astype(int))
    
    improved = ""
    if val_auc > best_auc:
        best_auc, best_epoch, best_state = val_auc, epoch + 1, model.state_dict().copy()
        patience_counter = 0
        improved = " *"
    else:
        patience_counter += 1
    
    print(f"Epoch {epoch+1:2d} | Val AUC: {val_auc:.4f} | Val Acc: {val_acc:.4f}{improved}")
    
    if patience_counter >= 4:
        print(f"\nEarly stopping at epoch {epoch+1}")
        break

# Final evaluation
print(f"\n{'='*70}\nBEST MODEL: Epoch {best_epoch}, Val AUC = {best_auc:.4f}\n{'='*70}")
model.load_state_dict(best_state)
model.eval()

with torch.no_grad():
    logits = model(X_val.to(DEVICE)).cpu().numpy()
probs = 1 / (1 + np.exp(-logits))
labels = y_val.numpy()

# Expectancy sweep
outcome_rr = np.array([m['outcome_rr'] for m in meta_val])

print(f"\n[BASELINE] All trades: {len(labels)}, winrate={labels.mean()*100:.1f}%, expectancy={outcome_rr.mean():+.3f}R")

print(f"\n[THRESHOLD SWEEP]")
print(f"{'Thresh':>6} | {'Trades':>6} | {'Winrate':>8} | {'Expect_R':>10}")
print("-" * 45)

best_exp = -999
for t in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
    mask = probs >= t
    if mask.sum() > 0:
        wr = labels[mask].mean() * 100
        exp = outcome_rr[mask].mean()
        print(f"{t:>6.1f} | {mask.sum():>6} | {wr:>7.1f}% | {exp:>+10.3f}")
        if exp > best_exp and mask.sum() >= 50:
            best_exp = exp

# Conclusion
print(f"\n{'='*70}\nCONCLUSION\n{'='*70}")
if best_auc > 0.60 and best_exp >= 0.30:
    print(f"✓ P4_GRU64_v3 shows edge! AUC={best_auc:.4f}, Best Exp={best_exp:+.3f}R")
elif best_auc > 0.55 or best_exp >= 0.15:
    print(f"~ P4_GRU64_v3 moderate signal. AUC={best_auc:.4f}, Best Exp={best_exp:+.3f}R")
else:
    print(f"✗ P4_GRU64_v3 weak/no edge. AUC={best_auc:.4f}, Best Exp={best_exp:+.3f}R")
    print(f"  → Macro features không tạo thêm signal đáng kể")

# Save
torch.save({'model_state': best_state, 'best_auc': best_auc}, DATA_DIR / "gru64_v3_best.pt")
print(f"\nSaved to {DATA_DIR / 'gru64_v3_best.pt'}")
