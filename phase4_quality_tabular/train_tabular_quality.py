"""
Train Tabular Quality Model

Trains MLP classifier on KEEP/DROP labels with class balancing.
"""

import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import csv

from phase4_quality_tabular.model import QualityMLP, prepare_features

print("="*80)
print("PHASE 4: TRAIN TABULAR QUALITY MODEL v1")
print("="*80)

# Config
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 50
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"\nDevice: {DEVICE}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Learning rate: {LEARNING_RATE}")
print(f"Epochs: {EPOCHS}")

# Paths
ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "output/phase4_quality"
OUTPUT_DIR = DATA_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load datasets
print(f"\n[1/6] Loading datasets...")
train_data = torch.load(DATA_DIR / "dataset_p2_quality_v1_train.pt")
val_data = torch.load(DATA_DIR / "dataset_p2_quality_v1_val.pt")

X_train, y_train, side_train = train_data['X'], train_data['y_quality'], train_data['side']
X_val, y_val, side_val = val_data['X'], val_data['y_quality'], val_data['side']

print(f"  Train: {len(X_train):,} samples")
print(f"  Val:   {len(X_val):,} samples")

# Feature normalization (z-score on train, apply to both)
print(f"\n[2/6] Normalizing features...")

# Prepare features
X_train_flat = prepare_features(X_train, side_train)  # [N, 3961]
X_val_flat = prepare_features(X_val, side_val)

# Compute mean/std on train
train_mean = X_train_flat.mean(dim=0, keepdim=True)
train_std = X_train_flat.std(dim=0, keepdim=True) + 1e-8  # Avoid div by zero

# Normalize
X_train_norm = (X_train_flat - train_mean) / train_std
X_val_norm = (X_val_flat - train_mean) / train_std

print(f"  Feature dim: {X_train_norm.shape[1]}")
print(f"  Train mean: {train_mean.mean().item():.4f}, std: {train_std.mean().item():.4f}")

# Save normalizer stats
normalizer = {
    'mean': train_mean,
    'std': train_std
}
torch.save(normalizer, OUTPUT_DIR / "normalizer_stats.pt")
print(f"  Saved normalizer stats")

# PyTorch Dataset
class QualityDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx].float()

train_dataset = QualityDataset(X_train_norm, y_train)
val_dataset = QualityDataset(X_val_norm, y_val)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Model
print(f"\n[3/6] Initializing model...")
model = QualityMLP(input_dim=X_train_norm.shape[1]).to(DEVICE)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  Total parameters: {total_params:,}")
print(f"  Trainable parameters: {trainable_params:,}")

# Loss with class balancing
num_neg = (y_train == 0).sum().item()
num_pos = (y_train == 1).sum().item()
pos_weight = torch.tensor([num_neg / num_pos]).to(DEVICE)

print(f"\n  Class balance:")
print(f"    KEEP (1): {num_pos:,}")
print(f"    DROP (0): {num_neg:,}")
print(f"    pos_weight: {pos_weight.item():.2f}")

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

# Training loop
print(f"\n[4/6] Training...")

best_val_f1 = 0.0
best_epoch = 0
training_log = []

for epoch in range(EPOCHS):
    # Train
    model.train()
    train_loss = 0.0
    
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
        
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * len(X_batch)
    
    train_loss /= len(train_dataset)
    
    # Validation
    model.eval()
    val_loss = 0.0
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            
            val_loss += loss.item() * len(X_batch)
            
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).long()
            
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().long().numpy())
    
    val_loss /= len(val_dataset)
    
    # Metrics
    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except:
        auc = 0.0
    
    # Log
    log_entry = {
        'epoch': epoch + 1,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'val_acc': acc,
        'val_precision': precision,
        'val_recall': recall,
        'val_f1': f1,
        'val_auc': auc
    }
    training_log.append(log_entry)
    
    # Print every 5 epochs
    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"Epoch {epoch+1:02d}/{EPOCHS} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val F1: {f1:.4f} | "
              f"Val AUC: {auc:.4f}")
    
    # Save best model
    if f1 > best_val_f1:
        best_val_f1 = f1
        best_epoch = epoch + 1
        torch.save(model.state_dict(), OUTPUT_DIR / "model_tabular_quality_v1_best.pt")

print(f"\n[5/6] Training complete!")
print(f"  Best Val F1: {best_val_f1:.4f} at epoch {best_epoch}")

# Save training log
log_path = OUTPUT_DIR / "training_log.csv"
with open(log_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=training_log[0].keys())
    writer.writeheader()
    writer.writerows(training_log)
print(f"  Saved training log: {log_path}")

# Final evaluation on best model
print(f"\n[6/6] Final evaluation (best model)...")
model.load_state_dict(torch.load(OUTPUT_DIR / "model_tabular_quality_v1_best.pt"))
model.eval()

all_preds = []
all_probs = []
all_labels = []

with torch.no_grad():
    for X_batch, y_batch in val_loader:
        X_batch = X_batch.to(DEVICE)
        
        logits = model(X_batch)
        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).long()
        
        all_probs.extend(probs.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y_batch.long().numpy())

# Metrics
acc = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, zero_division=0)
recall = recall_score(all_labels, all_preds, zero_division=0)
f1 = f1_score(all_labels, all_preds, zero_division=0)
auc = roc_auc_score(all_labels, all_probs)

print(f"\nFinal Validation Metrics:")
print(f"  Accuracy:  {acc:.4f}")
print(f"  Precision: {precision:.4f}")
print(f"  Recall:    {recall:.4f}")
print(f"  F1:        {f1:.4f}")
print(f"  AUC:       {auc:.4f}")

# Class-wise metrics
keep_mask = np.array(all_labels) == 1
drop_mask = np.array(all_labels) == 0

keep_preds = np.array(all_preds)[keep_mask]
drop_preds = np.array(all_preds)[drop_mask]

print(f"\nClass-wise performance:")
print(f"  KEEP class (1): {keep_preds.sum()}/{len(keep_preds)} predicted as KEEP ({keep_preds.sum()/len(keep_preds)*100:.1f}%)")
print(f"  DROP class (0): {(1-drop_preds).sum()}/{len(drop_preds)} predicted as DROP ({(1-drop_preds).sum()/len(drop_preds)*100:.1f}%)")

print("\n" + "="*80)
print("TRAINING COMPLETE!")
print("="*80)
print(f"\nSaved artifacts:")
print(f"  - model_tabular_quality_v1_best.pt (best model weights)")
print(f"  - normalizer_stats.pt (feature normalization)")
print(f"  - training_log.csv (epoch-by-epoch metrics)")
print(f"\nNext: Run eval_quality_backtest.py for trading metrics")
