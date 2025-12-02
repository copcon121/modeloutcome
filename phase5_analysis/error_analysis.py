"""
Error Analysis for Quality Model

Analyzes false positives and false negatives to understand model mistakes.
"""

import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

print("="*80)
print("PHASE 5: ERROR ANALYSIS")
print("="*80)

# Paths
ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "output/phase4_quality"
OUTPUT_DIR = ROOT / "output/phase5_quality"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load validation predictions and data
print(f"\n[1/5] Loading validation predictions...")
predictions_df = pd.read_csv(DATA_DIR / "predictions.csv")
print(f"  Loaded {len(predictions_df):,} predictions")

# Load enriched events for additional context
print(f"\n[2/5] Loading enriched event data...")
events_path = DATA_DIR / "events_p2_labeled_quality_v1.jsonl"
events = []
with open(events_path, 'r') as f:
    for line in f:
        events.append(json.loads(line))

# Create event lookup by event_id
events_dict = {e['event_id']: e for e in events if 'event_id' in e}
print(f"  Loaded {len(events_dict):,} events")

# Compute confusion matrix
print(f"\n[3/5] Computing confusion matrix...")
y_true = predictions_df['actual'].values
y_pred = predictions_df['predicted'].values

cm = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = cm.ravel()

print(f"\n  Confusion Matrix:")
print(f"                Predicted")
print(f"              DROP  KEEP")
print(f"  Actual DROP  {tn:4d}  {fp:4d}")
print(f"        KEEP  {fn:4d}  {tp:4d}")

print(f"\n  Breakdown:")
print(f"    True Negatives (DROP predicted as DROP):  {tn:,}")
print(f"    False Positives (DROP predicted as KEEP): {fp:,}")
print(f"    False Negatives (KEEP predicted as DROP): {fn:,}")
print(f"    True Positives (KEEP predicted as KEEP):  {tp:,}")

# Visualize confusion matrix
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['DROP (0)', 'KEEP (1)'],
            yticklabels=['DROP (0)', 'KEEP (1)'],
            ax=ax)
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.set_title('Confusion Matrix - Quality Model')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'confusion_matrix.png', dpi=150)
print(f"\n[OK] Saved confusion matrix plot")

# Extract False Positives (predicted KEEP, actually DROP)
print(f"\n[4/5] Analyzing False Positives...")
fp_mask = (predictions_df['predicted'] == 1) & (predictions_df['actual'] == 0)
fp_samples = predictions_df[fp_mask].copy()

# Sort by confidence (highest p_keep first - most confident wrong predictions)
fp_samples = fp_samples.sort_values('p_keep', ascending=False)

# Extract top 50
top_fp = fp_samples.head(50)

print(f"  Total False Positives: {len(fp_samples):,}")
print(f"  Extracting top 50 by confidence...")

# Enrich with event context
fp_enriched = []
for _, row in top_fp.iterrows():
    event_id = row['event_id']
    event = events_dict.get(event_id, {})
    
    fp_enriched.append({
        'event_id': int(event_id),
        'timestamp': row['timestamp'],
        'signal_side': row['signal_side'],
        'p_keep': float(row['p_keep']),
        'predicted': 1,
        'actual': 0,
        'hit': row['hit'],
        'outcome_rr': float(row['outcome_rr']),
        
        # SMC context
        'ext_trend_dir': event.get('ext_trend_dir', None),
        'in_bull_fvg': event.get('in_bull_fvg', None),
        'in_bear_fvg': event.get('in_bear_fvg', None),
        'int_in_bull_ob': event.get('int_in_bull_ob', None),
        'int_in_bear_ob': event.get('int_in_bear_ob', None),
        'ext_in_bull_ob': event.get('ext_in_bull_ob', None),
        'ext_in_bear_ob': event.get('ext_in_bear_ob', None),
        'session': event.get('session', None),
    })

# Save False Positives
fp_path = OUTPUT_DIR / "error_samples_fp.jsonl"
with open(fp_path, 'w') as f:
    for sample in fp_enriched:
        f.write(json.dumps(sample) + '\n')

print(f"  Saved {len(fp_enriched)} FP samples: {fp_path}")

# Analyze FP patterns
print(f"\n  FP Pattern Analysis:")
if len(fp_enriched) > 0:
    fp_df = pd.DataFrame(fp_enriched)
    
    # Hit distribution
    print(f"    Hit breakdown:")
    for hit_type, count in fp_df['hit'].value_counts().items():
        print(f"      {hit_type}: {count} ({count/len(fp_df)*100:.1f}%)")
    
    # Signal side
    print(f"    Signal side:")
    for side, count in fp_df['signal_side'].value_counts().items():
        print(f"      {side}: {count} ({count/len(fp_df)*100:.1f}%)")
    
    # Session distribution
    if 'session' in fp_df.columns and fp_df['session'].notna().any():
        print(f"    Session:")
        for session, count in fp_df['session'].value_counts().items():
            print(f"      {session}: {count} ({count/len(fp_df)*100:.1f}%)")

# Extract False Negatives (predicted DROP, actually KEEP)
print(f"\n[5/5] Analyzing False Negatives...")
fn_mask = (predictions_df['predicted'] == 0) & (predictions_df['actual'] == 1)
fn_samples = predictions_df[fn_mask].copy()

# Sort by margin (lowest p_keep first - missed the most obvious KEEPs)
fn_samples = fn_samples.sort_values('p_keep', ascending=True)

# Extract top 50
top_fn = fn_samples.head(50)

print(f"  Total False Negatives: {len(fn_samples):,}")
print(f"  Extracting top 50 by margin...")

# Enrich with event context
fn_enriched = []
for _, row in top_fn.iterrows():
    event_id = row['event_id']
    event = events_dict.get(event_id, {})
    
    fn_enriched.append({
        'event_id': int(event_id),
        'timestamp': row['timestamp'],
        'signal_side': row['signal_side'],
        'p_keep': float(row['p_keep']),
        'predicted': 0,
        'actual': 1,
        'hit': row['hit'],
        'outcome_rr': float(row['outcome_rr']),
        
        # SMC context
        'ext_trend_dir': event.get('ext_trend_dir', None),
        'in_bull_fvg': event.get('in_bull_fvg', None),
        'in_bear_fvg': event.get('in_bear_fvg', None),
        'int_in_bull_ob': event.get('int_in_bull_ob', None),
        'int_in_bear_ob': event.get('int_in_bear_ob', None),
        'ext_in_bull_ob': event.get('ext_in_bull_ob', None),
        'ext_in_bear_ob': event.get('ext_in_bear_ob', None),
        'session': event.get('session', None),
    })

# Save False Negatives
fn_path = OUTPUT_DIR / "error_samples_fn.jsonl"
with open(fn_path, 'w') as f:
    for sample in fn_enriched:
        f.write(json.dumps(sample) + '\n')

print(f"  Saved {len(fn_enriched)} FN samples: {fn_path}")

# Analyze FN patterns
print(f"\n  FN Pattern Analysis:")
if len(fn_enriched) > 0:
    fn_df = pd.DataFrame(fn_enriched)
    
    # Outcome RR distribution
    print(f"    Outcome RR stats:")
    print(f"      Mean: {fn_df['outcome_rr'].mean():.2f}R")
    print(f"      Median: {fn_df['outcome_rr'].median():.2f}R")
    print(f"      Max: {fn_df['outcome_rr'].max():.2f}R")
    
    # Signal side
    print(f"    Signal side:")
    for side, count in fn_df['signal_side'].value_counts().items():
        print(f"      {side}: {count} ({count/len(fn_df)*100:.1f}%)")
    
    # Session distribution
    if 'session' in fn_df.columns and fn_df['session'].notna().any():
        print(f"    Session:")
        for session, count in fn_df['session'].value_counts().items():
            print(f"      {session}: {count} ({count/len(fn_df)*100:.1f}%)")

# Summary report
print(f"\n{'='*80}")
print(f"ERROR ANALYSIS SUMMARY")
print(f"{'='*80}")

summary = f"""
CONFUSION MATRIX:
  True Negatives:  {tn:,} (correctly filtered bad trades)
  False Positives: {fp:,} (took bad trades - {fp/(fp+tn)*100:.1f}% of actual DROPs)
  False Negatives: {fn:,} (missed good trades - {fn/(fn+tp)*100:.1f}% of actual KEEPs)
  True Positives:  {tp:,} (correctly kept good trades)

ERROR RATES:
  FP Rate: {fp/(fp+tn)*100:.2f}% (of actual negatives)
  FN Rate: {fn/(fn+tp)*100:.2f}% (of actual positives)

IMPACT:
  False Positives: Took {fp} trades that hit SL/none
    - Cost in missed filtering
    - Review samples to understand why model was confident
  
  False Negatives: Missed {fn} trades that hit TP
    - Lost opportunities
    - Review samples to find patterns model doesn't capture

OUTPUT FILES:
  - {fp_path} ({len(fp_enriched)} samples)
  - {fn_path} ({len(fn_enriched)} samples)
  - {OUTPUT_DIR / 'confusion_matrix.png'}

NEXT STEPS:
  1. Manual review of FP/FN samples
  2. Look for common SMC patterns in errors
  3. Consider feature engineering to address blind spots
"""

print(summary)

# Save summary
summary_path = OUTPUT_DIR / "error_analysis_summary.txt"
with open(summary_path, 'w') as f:
    f.write("PHASE 5 ERROR ANALYSIS SUMMARY\n")
    f.write("="*80 + "\n\n")
    f.write(summary)

print(f"\nSaved summary: {summary_path}")
print(f"\n{'='*80}")
print("ERROR ANALYSIS COMPLETE!")
print("="*80)
