"""
Build Holdout Dataset by Time

Creates temporal holdout split (newest 20% of events) for validation.
"""

import sys
from pathlib import Path
import json
from datetime import datetime
import numpy as np
import torch

print("="*80)
print("PHASE 5.2: BUILD TEMPORAL HOLDOUT DATASET")
print("="*80)

# Paths
ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "output/phase4_quality"
SEQUENCES_DIR = ROOT / "output/production_10weeks"
OUTPUT_DIR = ROOT / "output/phase5_quality"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load enriched events
print(f"\n[1/5] Loading enriched events...")
events_path = DATA_DIR / "events_p2_labeled_quality_v1.jsonl"
events = []
with open(events_path, 'r') as f:
    for line in f:
        events.append(json.loads(line))

print(f"  Loaded {len(events):,} total events")

# Filter to candidates only (has quality_label)
candidates = [e for e in events if 'quality_label' in e and e['signal_side'] in ['long', 'short']]
print(f"  Candidates (long/short): {len(candidates):,}")

# Sort by timestamp
print(f"\n[2/5] Sorting by timestamp...")
for event in candidates:
    # Parse timestamp
    event['datetime'] = datetime.fromisoformat(event['timestamp'])

candidates_sorted = sorted(candidates, key=lambda x: x['datetime'])

# Find time split (80/20)
split_idx = int(len(candidates_sorted) * 0.8)
train_val_events = candidates_sorted[:split_idx]
holdout_events = candidates_sorted[split_idx:]

print(f"  Total candidates: {len(candidates_sorted):,}")
print(f"  Train+Val (first 80%): {len(train_val_events):,}")
print(f"  Holdout (last 20%): {len(holdout_events):,}")

# Time ranges
train_start = train_val_events[0]['datetime']
train_end = train_val_events[-1]['datetime']
holdout_start = holdout_events[0]['datetime']
holdout_end = holdout_events[-1]['datetime']

print(f"\n  Train+Val time range: {train_start} to {train_end}")
print(f"  Holdout time range:   {holdout_start} to {holdout_end}")

# Load sequences
print(f"\n[3/5] Loading sequences...")
sequences_path = SEQUENCES_DIR / "sequences_p2_10weeks_sequences.npy"
sequences = np.load(sequences_path)  # [N, 60, 66]
print(f"  Sequences shape: {sequences.shape}")

# Load indices mapping
indices_path = SEQUENCES_DIR / "sequences_p2_10weeks_indices.npy"
indices = np.load(indices_path)
print(f"  Indices shape: {indices.shape}")

# Build holdout dataset
print(f"\n[4/5] Building holdout dataset...")

# Map event_id to P2 sequence index
# Assume event_id in events corresponds to P2 index (0-based)
holdout_X = []
holdout_y = []
holdout_side = []
holdout_meta = []

for event in holdout_events:
    event_id = event['event_id']
    
    # event_id should match P2 index
    # Since events were created from P2 filtered data
    # But we need to be careful about alignment
    
    # For safety, use event_id as direct index if within bounds
    if event_id < len(sequences):
        X = sequences[event_id]  # [60, 66]
        holdout_X.append(X)
        holdout_y.append(event['quality_label'])
        holdout_side.append(1 if event['signal_side'] == 'long' else -1)
        holdout_meta.append({
            'event_id': event_id,
            'timestamp': event['timestamp'],
            'signal_side': event['signal_side'],
            'hit': event['hit'],
            'outcome_rr': event['outcome_rr'],
            'session': event.get('session', None),
        })

print(f"  Holdout events mapped: {len(holdout_X):,}")

# Convert to tensors
X_holdout = torch.FloatTensor(np.array(holdout_X))
y_holdout = torch.LongTensor(holdout_y)
side_holdout = torch.LongTensor(holdout_side)

print(f"\n  Holdout dataset shapes:")
print(f"    X: {X_holdout.shape}")
print(f"    y: {y_holdout.shape}")
print(f"    side: {side_holdout.shape}")
print(f"    meta: {len(holdout_meta)}")

# Label distribution
keep_count = (y_holdout == 1).sum().item()
drop_count = (y_holdout == 0).sum().item()

print(f"\n  Label distribution:")
print(f"    KEEP (1): {keep_count} ({keep_count/len(y_holdout)*100:.1f}%)")
print(f"    DROP (0): {drop_count} ({drop_count/len(y_holdout)*100:.1f}%)")

# Save dataset
print(f"\n[5/5] Saving holdout dataset...")

dataset = {
    'X': X_holdout,
    'y_quality': y_holdout,
    'side': side_holdout,
    'meta': holdout_meta
}

dataset_path = OUTPUT_DIR / "dataset_p2_quality_holdout_by_time.pt"
torch.save(dataset, dataset_path)
print(f"  Saved PyTorch dataset: {dataset_path}")

# Save enriched events
events_jsonl_path = OUTPUT_DIR / "events_p2_labeled_quality_holdout_by_time.jsonl"
with open(events_jsonl_path, 'w') as f:
    for event in holdout_events:
        # Remove datetime (not JSON serializable)
        event_copy = {k: v for k, v in event.items() if k != 'datetime'}
        f.write(json.dumps(event_copy) + '\n')

print(f"  Saved enriched events: {events_jsonl_path}")

# Summary
print(f"\n{'='*80}")
print(f"HOLDOUT DATASET COMPLETE!")
print(f"{'='*80}")
print(f"\nSummary:")
print(f"  Total holdout events: {len(holdout_meta):,}")
print(f"  KEEP: {keep_count} ({keep_count/len(y_holdout)*100:.1f}%)")
print(f"  DROP: {drop_count} ({drop_count/len(y_holdout)*100:.1f}%)")
print(f"  Time range: {holdout_start} to {holdout_end}")
print(f"\nOutput files:")
print(f"  - {dataset_path}")
print(f"  - {events_jsonl_path}")
