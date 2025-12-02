"""
Evaluate Quality Model with Backtest

Applies trained ML filter to validation events and computes trading metrics.
Compares against rule-only baseline.
"""

import sys
from pathlib import Path
import json
import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

from phase4_quality_tabular.model import QualityMLP, prepare_features

print("="*80)
print("PHASE 4: EVALUATE QUALITY MODEL - BACKTEST")
print("="*80)

# Paths
ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "output/phase4_quality"
OUTPUT_DIR = DATA_DIR

# Config
THRESHOLD = 0.5  # Default threshold for KEEP/DROP
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BASELINE_RR = 2.0  # From risk sweep

print(f"\nDevice: {DEVICE}")
print(f"Threshold: {THRESHOLD}")
print(f"Baseline RR target: {BASELINE_RR}")

# Load model
print(f"\n[1/6] Loading trained model...")
model = QualityMLP(input_dim=3961).to(DEVICE)
model.load_state_dict(torch.load(DATA_DIR / "model_tabular_quality_v1_best.pt", map_location=DEVICE))
model.eval()

# Load normalizer
normalizer = torch.load(DATA_DIR / "normalizer_stats.pt", map_location=DEVICE)
norm_mean, norm_std = normalizer['mean'], normalizer['std']

# Load val dataset
print(f"\n[2/6] Loading validation data...")
val_data = torch.load(DATA_DIR / "dataset_p2_quality_v1_val.pt")

X_val = val_data['X']
y_val = val_data['y_quality']
side_val = val_data['side']
meta_val = val_data['meta']

print(f"  Val events: {len(X_val):,}")

# Prepare and normalize features
X_val_flat = prepare_features(X_val, side_val)
X_val_norm = (X_val_flat - norm_mean) / norm_std

# Predict
print(f"\n[3/6] Running predictions...")
with torch.no_grad():
    logits = model(X_val_norm.to(DEVICE))
    probs = torch.sigmoid(logits).cpu().numpy()
    preds = (probs >= THRESHOLD).astype(int)

# Classification metrics
print(f"\n[4/6] Classification metrics...")
y_true = y_val.numpy()

acc = accuracy_score(y_true, preds)
precision = precision_score(y_true, preds, zero_division=0)
recall = recall_score(y_true, preds, zero_division=0)
f1 = f1_score(y_true, preds, zero_division=0)
auc = roc_auc_score(y_true, probs)

print(f"\n  Accuracy:  {acc:.4f}")
print(f"  Precision: {precision:.4f} (of predicted KEEP, how many are true KEEP)")
print(f"  Recall:    {recall:.4f} (of true KEEP, how many we captured)")
print(f"  F1:        {f1:.4f}")
print(f"  AUC:       {auc:.4f}")

# Confusion matrix
cm = confusion_matrix(y_true, preds)
print(f"\n  Confusion Matrix:")
print(f"                Predicted")
print(f"              DROP  KEEP")
print(f"  Actual DROP  {cm[0,0]:4d}  {cm[0,1]:4d}")
print(f"        KEEP  {cm[1,0]:4d}  {cm[1,1]:4d}")

# Trading metrics
print(f"\n[5/6] Trading metrics...")

def calculate_trading_metrics(events, filter_fn=None, name=""):
    """
    Calculate trading metrics for a set of events
    
    Args:
        events: List of meta dicts with hit, outcome_rr, etc.
        filter_fn: Optional function(event, idx) -> bool to filter events
        name: Name for this strategy
    
    Returns:
        dict of metrics
    """
    trades = []
    
    for i, event in enumerate(events):
        # Apply filter if provided
        if filter_fn and not filter_fn(event, i):
            continue
        
        trades.append({
            'hit': event['hit'],
            'outcome_rr': event['outcome_rr'],
            'signal_side': event['signal_side']
        })
    
    if len(trades) == 0:
        return {
            'name': name,
            'num_trades': 0,
            'winners': 0,
            'losers': 0,
            'winrate': 0.0,
            'avg_r': 0.0,
            'expectancy': 0.0,
            'total_r': 0.0,
            'max_dd_r': 0.0
        }
    
    # Compute metrics
    winners = sum(1 for t in trades if t['hit'] == 'tp')
    losers = sum(1 for t in trades if t['hit'] == 'sl')
    none_hits = sum(1 for t in trades if t['hit'] == 'none')
    
    winrate = winners / len(trades) if len(trades) > 0 else 0.0
    
    total_r = sum(t['outcome_rr'] for t in trades)
    avg_r = total_r / len(trades) if len(trades) > 0 else 0.0
    
    # Expectancy (same as avg_r for our setup)
    expectancy = avg_r
    
    # Max DD (simple cumulative)
    cumulative_r = 0.0
    peak_r = 0.0
    max_dd_r = 0.0
    
    for t in trades:
        cumulative_r += t['outcome_rr']
        peak_r = max(peak_r, cumulative_r)
        dd = peak_r - cumulative_r
        max_dd_r = max(max_dd_r, dd)
    
    return {
        'name': name,
        'num_trades': len(trades),
        'winners': winners,
        'losers': losers,
        'none_hits': none_hits,
        'winrate': winrate,
        'avg_r': avg_r,
        'expectancy': expectancy,
        'total_r': total_r,
        'max_dd_r': max_dd_r
    }

# Baseline (all val events, no ML filter)
baseline_metrics = calculate_trading_metrics(meta_val, name="Baseline (Rule Only)")

# ML Filtered (only events where p_keep >= threshold)
def ml_filter(event, idx):
    return preds[idx] == 1

ml_metrics = calculate_trading_metrics(meta_val, filter_fn=ml_filter, name="ML Filtered")

# Print comparison
print(f"\n{'='*80}")
print(f"BACKTEST COMPARISON")
print(f"{'='*80}")

def print_metrics(metrics):
    print(f"\n{metrics['name']}:")
    print(f"  Trades:     {metrics['num_trades']:,}")
    print(f"  Winners:    {metrics['winners']:,}")
    print(f"  Losers:     {metrics['losers']:,}")
    print(f"  Winrate:    {metrics['winrate']*100:.1f}%")
    print(f"  Avg R:      {metrics['avg_r']:+.4f}")
    print(f"  Expectancy: {metrics['expectancy']:+.4f}R")
    print(f"  Total R:    {metrics['total_r']:+.2f}R")
    print(f"  Max DD:     {metrics['max_dd_r']:.2f}R")

print_metrics(baseline_metrics)
print_metrics(ml_metrics)

# Improvement analysis
print(f"\n{'='*80}")
print(f"IMPROVEMENT ANALYSIS")
print(f"{'='*80}")

trades_change = ml_metrics['num_trades'] - baseline_metrics['num_trades']
exp_change = ml_metrics['expectancy'] - baseline_metrics['expectancy']
dd_change = ml_metrics['max_dd_r'] - baseline_metrics['max_dd_r']

print(f"\nML Filter vs Baseline:")
print(f"  Trades:     {ml_metrics['num_trades']:,} ({trades_change:+,}, {trades_change/baseline_metrics['num_trades']*100:+.1f}%)")
print(f"  Expectancy: {ml_metrics['expectancy']:+.4f}R ({exp_change:+.4f}R, {exp_change/abs(baseline_metrics['expectancy'])*100 if baseline_metrics['expectancy'] != 0 else 0:+.1f}%)")
print(f"  Max DD:     {ml_metrics['max_dd_r']:.2f}R ({dd_change:+.2f}R)")

# Verdict
print(f"\n{'='*80}")
print(f"VERDICT")
print(f"{'='*80}")

is_better = ml_metrics['expectancy'] > baseline_metrics['expectancy']
dd_better = ml_metrics['max_dd_r'] < baseline_metrics['max_dd_r']

print(f"\nExpectancy improvement: {'[YES]' if is_better else '[NO]'}")
print(f"Max DD improvement:     {'[YES]' if dd_better else '[NO]'}")

if is_better and dd_better:
    verdict = "[SUCCESS] ML filter improves both expectancy and drawdown!"
elif is_better:
    verdict = "[PARTIAL] ML filter improves expectancy but not drawdown"
elif dd_better:
    verdict = "[PARTIAL] ML filter reduces drawdown but hurts expectancy"
else:
    verdict = "[FAILED] ML filter does not improve trading performance"

print(f"\n{verdict}")

# Export results
print(f"\n[6/6] Exporting results...")

# Predictions CSV
predictions = []
for i in range(len(meta_val)):
    predictions.append({
        'event_id': meta_val[i]['event_id'],
        'timestamp': meta_val[i]['timestamp'],
        'signal_side': meta_val[i]['signal_side'],
        'p_keep': float(probs[i]),
        'predicted': int(preds[i]),
        'actual': int(y_true[i]),
        'hit': meta_val[i]['hit'],
        'outcome_rr': meta_val[i]['outcome_rr']
    })

import csv
predictions_path = OUTPUT_DIR / "predictions.csv"
with open(predictions_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=predictions[0].keys())
    writer.writeheader()
    writer.writerows(predictions)

print(f"  Saved predictions: {predictions_path}")

# Report
report_path = OUTPUT_DIR / "report_tabular_v1.txt"
with open(report_path, 'w') as f:
    f.write("="*80 + "\n")
    f.write("PHASE 4 QUALITY MODEL v1 - BACKTEST REPORT\n")
    f.write("="*80 + "\n\n")
    
    f.write("CLASSIFICATION METRICS\n")
    f.write("-" * 40 + "\n")
    f.write(f"Accuracy:  {acc:.4f}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall:    {recall:.4f}\n")
    f.write(f"F1:        {f1:.4f}\n")
    f.write(f"AUC:       {auc:.4f}\n\n")
    
    f.write("TRADING METRICS - BASELINE\n")
    f.write("-" * 40 + "\n")
    for key, val in baseline_metrics.items():
        if key != 'name':
            f.write(f"{key}: {val}\n")
    f.write("\n")
    
    f.write("TRADING METRICS - ML FILTERED\n")
    f.write("-" * 40 + "\n")
    for key, val in ml_metrics.items():
        if key != 'name':
            f.write(f"{key}: {val}\n")
    f.write("\n")
    
    f.write("VERDICT\n")
    f.write("-" * 40 + "\n")
    f.write(f"{verdict}\n")

print(f"  Saved report: {report_path}")

print("\n" + "="*80)
print("EVALUATION COMPLETE!")
print("="*80)
