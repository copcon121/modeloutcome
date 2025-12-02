"""
Phase 8 - Shadow Replay on Historical Data

Replays historical events through API to test end-to-end pipeline
before live deployment.
"""

import sys
from pathlib import Path
import json
import argparse
import requests
import numpy as np
import torch
from tqdm import tqdm

print("="*80)
print("PHASE 8: SHADOW REPLAY - HISTORICAL DATA")
print("="*80)

# Parse arguments
parser = argparse.ArgumentParser(description="Replay historical events through API for shadow testing")
parser.add_argument('--events-jsonl', type=str, required=True, help='Path to events JSONL file')
parser.add_argument('--sequences-npy', type=str, required=True, help='Path to sequences .npy file')
parser.add_argument('--server-url', type=str, default='http://localhost:8000/predict_quality', help='API endpoint URL')
parser.add_argument('--model-type', type=str, default='seq_v1', help='Model type')
parser.add_argument('--mode', type=str, default='seq_conservative', help='Mode')
parser.add_argument('--shadow-only', action='store_true', default=True, help='Shadow mode (default: True)')
parser.add_argument('--max-events', type=int, default=None, help='Max events to replay (for testing)')

args = parser.parse_args()

# Setup paths
ROOT = Path(__file__).parent.parent
OUTPUT_DIR = ROOT / "output/phase8_shadow_history"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load events
print(f"\n[1/4] Loading events from {args.events_jsonl}...")
events = []
with open(args.events_jsonl, 'r') as f:
    for line in f:
        if line.strip():
            events.append(json.loads(line))

if args.max_events:
    events = events[:args.max_events]
    print(f"  Limited to {args.max_events} events for testing")

print(f"  Loaded {len(events):,} events")

# Load sequences
print(f"\n[2/4] Loading sequences from {args.sequences_npy}...")
sequences = np.load(args.sequences_npy)
print(f"  Sequences shape: {sequences.shape}")

# Open output log
log_path = OUTPUT_DIR / "shadow_replay_log.jsonl"
log_file = open(log_path, 'w')
print(f"\n[3/4] Replaying events through API...")
print(f"  Server: {args.server_url}")
print(f"  Model: {args.model_type} / {args.mode}")
print(f"  Shadow only: {args.shadow_only}")
print(f"  Output: {log_path}")

# Replay
success_count = 0
error_count = 0

for event in tqdm(events, desc="Replaying"):
    try:
        event_id = event['event_id']
        
        # Get sequence
        if event_id >= len(sequences):
            print(f"\n  WARNING: event_id {event_id} out of range, skipping")
            continue
        
        X_seq = sequences[event_id]  # [60, 66]
        side = 1 if event['signal_side'] == 'long' else -1
        
        # Build request
        request_data = {
            "X": X_seq.tolist(),
            "side": side,
            "model_type": args.model_type,
            "mode": args.mode,
            "shadow_only": args.shadow_only,
            "meta": {
                "symbol": event.get('symbol_root', 'UNKNOWN'),
                "timeframe": event.get('timeframe', 'M1'),
                "event_time": event['timestamp'],
                "event_id": event_id,
                "session": event.get('session', None),
                # Include outcome for later analysis
                "hit": event.get('hit', None),
                "outcome_rr": event.get('outcome_rr', None)
            }
        }
        
        # Send request
        response = requests.post(args.server_url, json=request_data, timeout=5)
        
        if response.status_code == 200:
            result = response.json()
            
            # Log entry (same format as live shadow log)
            log_entry = {
                "timestamp": result['timestamp'],
                "model_type": result['model_type'],
                "mode": result['mode'],
                "threshold": result['threshold'],
                "shadow_only": result['shadow_only'],
                "p_keep": result['p_keep'],
                "keep": result['keep'],
                "side": result['side'],
                "meta": request_data['meta']
            }
            
            log_file.write(json.dumps(log_entry) + '\n')
            success_count += 1
        else:
            print(f"\n  ERROR: API returned {response.status_code} for event {event_id}")
            error_count += 1
    
    except Exception as e:
        print(f"\n  ERROR: Exception for event {event_id}: {str(e)}")
        error_count += 1

log_file.close()

# Summary
print(f"\n[4/4] Replay complete!")
print(f"  Success: {success_count:,}")
print(f"  Errors: {error_count:,}")
print(f"  Log saved: {log_path}")

print(f"\n{'='*80}")
print("NEXT STEPS:")
print("="*80)
print(f"\n1. Analyze shadow replay results:")
print(f"   python -m phase7_shadow.analyze_shadow_vs_baseline \\")
print(f"     --shadow-log {log_path}")
print(f"\n2. Compare to backtest expectations")
print(f"\n3. If results match -> proceed to live shadow period")
print("="*80)
