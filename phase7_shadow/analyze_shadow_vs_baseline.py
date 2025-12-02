"""
Analyze Shadow Trading vs Baseline

Compares ML-filtered decisions (from shadow log) against baseline rule outcomes.
"""

import sys
from pathlib import Path
import json
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from phase7_shadow.load_shadow_log import load_shadow_log

print("="*80)
print("PHASE 7: SHADOW TRADING ANALYSIS")
print("="*80)

ROOT = Path(__file__).parent.parent
OUTPUT_DIR = ROOT / "output/phase7_shadow"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load shadow log
print(f"\n[1/4] Loading shadow trading log...")
shadow_log_path = ROOT / "output/phase5_quality/shadow_trading_log.jsonl"

if not shadow_log_path.exists():
    print(f"  Shadow log not found: {shadow_log_path}")
    print(f"  Please run API server and make requests first.")
    sys.exit(0)

shadow_df = load_shadow_log(shadow_log_path)

if len(shadow_df) == 0:
    print(f"  Shadow log is empty. No analysis possible.")
    sys.exit(0)

print(f"  Loaded {len(shadow_df):,} shadow entries")

# Load Phase 3 labeled events
print(f"\n[2/4] Loading Phase 3 baseline outcomes...")
events_path = ROOT / "output/phase4_quality/events_p2_labeled_quality_v1.jsonl"

events = []
with open(events_path, 'r') as f:
    for line in f:
        events.append(json.loads(line))

events_df = pd.DataFrame(events)
print(f"  Loaded {len(events_df):,} labeled events")

# Join shadow log with outcomes
print(f"\n[3/4] Matching shadow decisions to outcomes...")

# For simplicity, assume we can match by index or timestamp
# In production, you'd match by (symbol, timeframe, event_time)

# Here we assume shadow log entries correspond to validation set
# Load val dataset meta for matching
val_data = torch.load(ROOT / "output/phase4_quality/dataset_p2_quality_v1_val.pt")
if 'meta' in val_data:
    val_meta = val_data['meta']
    print(f"  Val meta: {len(val_meta)} events")
    
    # Create mapping (simplified - match by order for now)
    # In production: match by timestamp/symbol/timeframe
    
# Simplified analysis using available data
# Group by (model_type, mode)
print(f"\n[4/4] Computing metrics by model/mode...")

grouped = shadow_df.groupby(['model_type', 'mode'])

results = []

for (model_type, mode), group in grouped:
    print(f"\n  {model_type} / {mode}:")
    print(f"    Total predictions: {len(group):,}")
    print(f"    Keep decisions: {group['keep'].sum():,} ({group['keep'].mean()*100:.1f}%)")
    print(f"    Avg p_keep: {group['p_keep'].mean():.4f}")
    
    results.append({
        'model_type': model_type,
        'mode': mode,
        'total_predictions': len(group),
        'keep_count': group['keep'].sum(),
        'keep_rate': group['keep'].mean(),
        'avg_p_keep': group['p_keep'].mean()
    })

# Save results
results_df = pd.DataFrame(results)
results_path = OUTPUT_DIR / "shadow_results.csv"
results_df.to_csv(results_path, index=False)

print(f"\n  Saved: {results_path}")

# Generate summary report
summary = f"""
================================================================================
SHADOW TRADING ANALYSIS SUMMARY
================================================================================

Shadow Log: {shadow_log_path}
Total Entries: {len(shadow_df):,}

ANALYSIS BY MODEL/MODE:

"""

for _, row in results_df.iterrows():
    summary += f"""
{row['model_type'].upper()} / {row['mode']}:
  Total Predictions: {row['total_predictions']:,}
  Keep Decisions: {row['keep_count']} ({row['keep_rate']*100:.1f}%)
  Average p_keep: {row['avg_p_keep']:.4f}
"""

summary += f"""

================================================================================
INTERPRETATION
================================================================================

The shadow log shows ML filter decisions without executing trades.

To validate against baseline:
  1. Match shadow entries to Phase 3 outcomes (by timestamp/symbol)
  2. Compute trading metrics (WR, expectancy, DD) for keep==True subset
  3. Compare to baseline (all candidates) and backtest results

Current analysis shows model usage distribution only.
Full outcome matching requires production event timestamps.

================================================================================
NEXT STEPS
================================================================================

1. Run system in shadow mode for 1-2 weeks
2. Collect sufficient shadow log entries
3. Match to actual market outcomes
4. Compare to Phase 5.2/6 backtest expectations
5. If consistent → proceed to micro-lot live testing

================================================================================
"""

summary_path = OUTPUT_DIR / "shadow_summary.txt"
with open(summary_path, 'w') as f:
    f.write(summary)

print(f"\n{summary}")
print(f"Saved summary: {summary_path}")

print(f"\n{'='*80}")
print("SHADOW ANALYSIS COMPLETE!")
print("="*80)
