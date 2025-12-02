"""
Train XGBoost for Outcome Model v2

Binary classification: WIN (1) vs LOSS (0)
"""

import sys
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
import xgboost as xgb
import json
import joblib

sys.path.insert(0, str(Path(__file__).parent.parent))

print("="*60)
print("TRAIN OUTCOME XGBOOST MODEL")
print("="*60)

# Paths
ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "output/phase4_outcome_v2"
OUTPUT_DIR = DATA_DIR

# Load data
print("\n[1/3] Loading dataset...")
train_data = torch.load(DATA_DIR / "dataset_outcome_v2_train.pt")
val_data = torch.load(DATA_DIR / "dataset_outcome_v2_val.pt")

X_train = train_data['X'].numpy()  # (N, 60, 90)
y_train = train_data['y'].numpy()
X_val = val_data['X'].numpy()
y_val = val_data['y'].numpy()

print(f"  Train: {X_train.shape[0]:,} samples")
print(f"  Val: {X_val.shape[0]:,} samples")

# Flatten sequences
X_train_flat = X_train.reshape(X_train.shape[0], -1)  # (N, 60*90)
X_val_flat = X_val.reshape(X_val.shape[0], -1)
print(f"  Flattened features: {X_train_flat.shape[1]:,}")

# Calculate scale_pos_weight for imbalanced data
n_pos = (y_train == 1).sum()
n_neg = (y_train == 0).sum()
scale_pos_weight = n_neg / n_pos
print(f"  Scale pos weight: {scale_pos_weight:.2f}")

# Train XGBoost
print("\n[2/3] Training XGBoost...")
params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'max_depth': 6,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'scale_pos_weight': scale_pos_weight,
    'seed': 42,
    'n_jobs': -1
}

dtrain = xgb.DMatrix(X_train_flat, label=y_train)
dval = xgb.DMatrix(X_val_flat, label=y_val)

evals = [(dtrain, 'train'), (dval, 'val')]
evals_result = {}

model = xgb.train(
    params,
    dtrain,
    num_boost_round=500,
    evals=evals,
    evals_result=evals_result,
    early_stopping_rounds=30,
    verbose_eval=50
)

# Evaluate
print("\n[3/3] Final Evaluation...")
val_preds = model.predict(dval)
train_preds = model.predict(dtrain)

train_auc = roc_auc_score(y_train, train_preds)
val_auc = roc_auc_score(y_val, val_preds)
val_acc = accuracy_score(y_val, (val_preds > 0.5).astype(int))

print(f"\n  Train AUC: {train_auc:.4f}")
print(f"  Val AUC: {val_auc:.4f}")
print(f"  Val Acc: {val_acc:.4f}")

# Classification report
print("\n  Classification Report (threshold=0.5):")
print(classification_report(y_val, (val_preds > 0.5).astype(int), target_names=['LOSS', 'WIN']))

# Feature importance (top 20)
print("\n  Top 20 Important Features:")
importance = model.get_score(importance_type='gain')
sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:20]
for feat, score in sorted_imp:
    # Parse feature index to get bar and feature name
    feat_idx = int(feat.replace('f', ''))
    bar_idx = feat_idx // 90
    feat_in_bar = feat_idx % 90
    print(f"    {feat}: bar={bar_idx}, feat={feat_in_bar}, gain={score:.2f}")

# Save model
model.save_model(str(OUTPUT_DIR / "model_outcome_xgb.json"))

# Save results
results = {
    'model': 'XGBoost',
    'train_auc': float(train_auc),
    'val_auc': float(val_auc),
    'val_acc': float(val_acc),
    'best_iteration': model.best_iteration,
    'params': params
}

with open(OUTPUT_DIR / "results_xgb.json", 'w') as f:
    json.dump(results, f, indent=2)

print("\n" + "="*60)
print("TRAINING COMPLETE!")
print("="*60)
print(f"\nSaved:")
print(f"  - {OUTPUT_DIR / 'model_outcome_xgb.json'}")
print(f"  - {OUTPUT_DIR / 'results_xgb.json'}")
