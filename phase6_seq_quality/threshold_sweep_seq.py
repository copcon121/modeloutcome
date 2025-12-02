"""
Threshold Sweep for Sequence Model

Tests thresholds from 0.2 to 0.8 on val set.
"""

import sys
from pathlib import Path
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).parent.parent))

from phase6_seq_quality.model_seq import QualitySeqGRU
from phase6_seq_quality.dataset_seq import QualitySequenceDataset

print("="*60)
print("THRESHOLD SWEEP - SEQUENCE MODEL")
print("="*60)

ROOT = Path(__file__).parent.parent
MODEL_DIR = ROOT / "output/phase6_seq_quality"
DATA_DIR = ROOT / "output/phase4_quality"

# Load
model = QualitySeqGRU(input_dim=66, hidden_dim=128, num_layers=1, dropout=0.1)
model.load_state_dict(torch.load(MODEL_DIR / "model_seq_quality_v1_best.pt"))
model.eval()

normalizer = torch.load(MODEL_DIR / "normalizer_stats_seq.pt")
dataset = QualitySequenceDataset(DATA_DIR / "dataset_p2_quality_v1_val.pt", normalizer)

# Get predictions
print(f"\nComputing predictions...")
probs = []
labels = []
orig_data = torch.load(DATA_DIR / "dataset_p2_quality_v1_val.pt")
meta = orig_data.get('meta', None)

with torch.no_grad():
    for i in range(len(dataset)):
        sample = dataset[i]
        X = sample['X_seq'].unsqueeze(0)
        side = sample['side'].unsqueeze(0)
        
        logit = model(X, side).squeeze()
        prob = torch.sigmoid(logit).item()
        
        probs.append(prob)
        labels.append(sample['y'].item())

probs = np.array(probs)
labels = np.array(labels)

# Sweep
thresholds = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
results = []

print(f"\nSweeping thresholds...")
for threshold in thresholds:
    preds = (probs >= threshold).astype(int)
    f1 = f1_score(labels, preds, zero_division=0)
    
    # Trading
    ml_indices = np.where(preds == 1)[0]
    ml_trades = len(ml_indices)
    
    if ml_trades > 0 and meta:
        ml_meta = [meta[i] for i in ml_indices]
        ml_winners = sum(1 for m in ml_meta if m['hit'] == 'tp')
        ml_winrate = ml_winners / ml_trades
        ml_total_r = sum(m['outcome_rr'] for m in ml_meta)
        ml_avg_r = ml_total_r / ml_trades
        
        # Max DD
        cumulative, peak, maxdd = 0.0, 0.0, 0.0
        for m in ml_meta:
            cumulative += m['outcome_rr']
            peak = max(peak, cumulative)
            maxdd = max(maxdd, peak - cumulative)
    else:
        ml_winrate, ml_avg_r, maxdd = 0, 0, 0
    
    results.append({
        'threshold': threshold,
        'f1_keep': f1,
        'trades': ml_trades,
        'winrate': ml_winrate,
        'avg_r': ml_avg_r,
        'maxdd_r': maxdd
    })
    
    print(f"  t={threshold:.1f}: Trades={ml_trades:4d}, WR={ml_winrate:.1%}, Exp={ml_avg_r:+.4f}R, F1={f1:.4f}")

# Save
df = pd.DataFrame(results)
csv_path = MODEL_DIR / "threshold_sweep_seq_results.csv"
df.to_csv(csv_path, index=False)

print(f"\nSaved: {csv_path}")
print(f"\nBest threshold by expectancy: {df.loc[df['avg_r'].idxmax(), 'threshold']:.1f}")
print("="*60)
