"""
Train Baseline Model for P4 Outcome v2.1

Uses GRU with aggregated features for better signal detection
"""

import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, classification_report
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

print("="*60)
print("TRAIN P4 OUTCOME v2.1 MODEL")
print("="*60)

# Paths
ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "output/phase4_outcome_v2.1"
OUTPUT_DIR = DATA_DIR

# Load data
print("\n[1/5] Loading dataset...")
train_data = torch.load(DATA_DIR / "dataset_p4_outcome_v2.1_train.pt")
val_data = torch.load(DATA_DIR / "dataset_p4_outcome_v2.1_val.pt")

X_train = train_data['X']  # (N, 60, 90)
y_train = train_data['y']
X_val = val_data['X']
y_val = val_data['y']

print(f"  Train: {X_train.shape[0]:,} samples")
print(f"  Val: {X_val.shape[0]:,} samples")
print(f"  Sequence: {X_train.shape[1]} steps x {X_train.shape[2]} features")

# Create dataloaders
BATCH_SIZE = 64
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Model: Small GRU
class OutcomeGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=1, dropout=0.3):
        super().__init__()
        self.gru = nn.GRU(
            input_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, 1)
    
    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]  # Last timestep
        out = self.dropout(out)
        return self.fc(out).squeeze(-1)

print("\n[2/5] Building model...")
INPUT_DIM = X_train.shape[2]
model = OutcomeGRU(INPUT_DIM, hidden_dim=64, num_layers=1, dropout=0.3)
print(f"  Input dim: {INPUT_DIM}")
print(f"  Architecture: GRU(64, 1 layer, bidirectional)")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"  Device: {device}")
model = model.to(device)

# Class weights
n_pos = (y_train == 1).sum().item()
n_neg = (y_train == 0).sum().item()
pos_weight = torch.tensor([n_neg / n_pos]).to(device)
print(f"  Pos weight: {pos_weight.item():.2f}")

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5)

# Training
print("\n[3/5] Training...")
EPOCHS = 50
best_auc = 0
patience = 10
patience_counter = 0
history = []

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.float().to(device)
        
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        train_loss += loss.item()
    
    train_loss /= len(train_loader)
    
    # Validate
    model.eval()
    val_preds = []
    val_labels = []
    val_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.float().to(device)
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            val_loss += loss.item()
            probs = torch.sigmoid(logits).cpu().numpy()
            val_preds.extend(probs)
            val_labels.extend(y_batch.cpu().numpy())
    
    val_loss /= len(val_loader)
    val_preds = np.array(val_preds)
    val_labels = np.array(val_labels)
    
    val_auc = roc_auc_score(val_labels, val_preds)
    val_f1 = f1_score(val_labels, (val_preds > 0.5).astype(int))
    
    scheduler.step(val_auc)
    
    history.append({
        'epoch': epoch + 1,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'val_auc': val_auc,
        'val_f1': val_f1
    })
    
    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"  Epoch {epoch+1:3d}: loss={train_loss:.4f}/{val_loss:.4f}, AUC={val_auc:.4f}, F1={val_f1:.4f}")
    
    if val_auc > best_auc:
        best_auc = val_auc
        patience_counter = 0
        torch.save(model.state_dict(), OUTPUT_DIR / "model_outcome_v21_best.pt")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break

# Load best model
model.load_state_dict(torch.load(OUTPUT_DIR / "model_outcome_v21_best.pt"))

# Final evaluation
print("\n[4/5] Final Evaluation...")
model.eval()
val_preds = []
val_labels = []
val_sides = val_data['side'].numpy()
val_meta = val_data['meta']

with torch.no_grad():
    for X_batch, y_batch in val_loader:
        X_batch = X_batch.to(device)
        logits = model(X_batch)
        probs = torch.sigmoid(logits).cpu().numpy()
        val_preds.extend(probs)
        val_labels.extend(y_batch.numpy())

val_preds = np.array(val_preds)
val_labels = np.array(val_labels)

final_auc = roc_auc_score(val_labels, val_preds)
final_f1 = f1_score(val_labels, (val_preds > 0.5).astype(int))
final_acc = accuracy_score(val_labels, (val_preds > 0.5).astype(int))

print(f"\n  Best Val AUC: {final_auc:.4f}")
print(f"  Best Val F1: {final_f1:.4f}")
print(f"  Best Val Acc: {final_acc:.4f}")

print("\n  Classification Report (threshold=0.5):")
print(classification_report(val_labels, (val_preds > 0.5).astype(int), target_names=['LOSS', 'WIN']))


# Expectancy sweep
print("\n[5/5] Expectancy Sweep...")
print("\n  Threshold | Trades | WinRate | Expectancy | MaxDD")
print("  " + "-"*55)

thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
sweep_results = []

for th in thresholds:
    # Filter by threshold (predict WIN if prob > th)
    pred_win = val_preds > th
    
    if pred_win.sum() == 0:
        print(f"  {th:.1f}       | 0      | -       | -          | -")
        continue
    
    # Get outcome_rr for predicted wins
    outcome_rrs = []
    for i, is_win_pred in enumerate(pred_win):
        if is_win_pred:
            outcome_rrs.append(val_meta[i]['outcome_rr'])
    
    outcome_rrs = np.array(outcome_rrs)
    n_trades = len(outcome_rrs)
    win_rate = (outcome_rrs > 0).sum() / n_trades * 100
    expectancy = outcome_rrs.mean()
    
    # Max drawdown (cumulative)
    cumsum = np.cumsum(outcome_rrs)
    running_max = np.maximum.accumulate(cumsum)
    drawdowns = running_max - cumsum
    max_dd = drawdowns.max()
    
    print(f"  {th:.1f}       | {n_trades:4d}   | {win_rate:5.1f}%  | {expectancy:+.3f}R     | {max_dd:.1f}R")
    
    sweep_results.append({
        'threshold': th,
        'n_trades': n_trades,
        'win_rate': win_rate,
        'expectancy': expectancy,
        'max_dd': max_dd
    })

# Save results
results = {
    'model': 'GRU_v21',
    'best_val_auc': float(final_auc),
    'best_val_f1': float(final_f1),
    'best_val_acc': float(final_acc),
    'epochs_trained': len(history),
    'sweep_results': sweep_results,
    'history': history
}

with open(OUTPUT_DIR / "results_v21.json", 'w') as f:
    json.dump(results, f, indent=2)

print("\n" + "="*60)
print("TRAINING COMPLETE!")
print("="*60)
print(f"\nSaved:")
print(f"  - {OUTPUT_DIR / 'model_outcome_v21_best.pt'}")
print(f"  - {OUTPUT_DIR / 'results_v21.json'}")
