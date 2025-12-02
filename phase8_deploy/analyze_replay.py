"""
Quick analysis of shadow replay log
"""
import json
import numpy as np

# Load data
with open('output/phase8_shadow_history/shadow_replay_log.jsonl', 'r') as f:
    data = [json.loads(line) for line in f if line.strip()]

print("="*80)
print("SHADOW REPLAY ANALYSIS - 1000 EVENTS")
print("="*80)

# Overall stats
print(f"\n[1] OVERALL STATS")
print(f"  Total predictions: {len(data)}")
print(f"  Keep (t>=0.8): {sum(1 for d in data if d['keep'])} ({sum(1 for d in data if d['keep'])/len(data)*100:.1f}%)")
print(f"  Drop: {sum(1 for d in data if not d['keep'])} ({sum(1 for d in data if not d['keep'])/len(data)*100:.1f}%)")

# p_keep distribution
p_keeps = [d['p_keep'] for d in data]
print(f"\n[2] P_KEEP DISTRIBUTION")
print(f"  Min: {min(p_keeps):.4f}")
print(f"  Max: {max(p_keeps):.4f}")
print(f"  Mean: {np.mean(p_keeps):.4f}")
print(f"  Median: {np.median(p_keeps):.4f}")

# Trading metrics for KEEP trades
keep_data = [d for d in data if d['keep']]
print(f"\n[3] TRADING METRICS (KEEP ONLY)")
print(f"  Kept trades: {len(keep_data)}")

if keep_data:
    outcomes = [d.get('meta', {}).get('outcome_rr', 0) for d in keep_data]
    winners = sum(1 for d in keep_data if d.get('meta', {}).get('outcome_rr', 0) > 0)
    losers = sum(1 for d in keep_data if d.get('meta', {}).get('outcome_rr', 0) < 0)
    
    print(f"  Winners: {winners}")
    print(f"  Losers: {losers}")
    print(f"  Winrate: {winners/len(keep_data)*100:.1f}%")
    print(f"  Avg R per trade: {np.mean(outcomes):.4f}R")
    print(f"  Total R: {sum(outcomes):.2f}R")
    print(f"  Max win: {max(outcomes):.2f}R")
    print(f"  Max loss: {min(outcomes):.2f}R")

# Compare to baseline print(f"\n[4] BASELINE COMPARISON (all 1000 events)")
all_outcomes = [d.get('meta', {}).get('outcome_rr', 0) for d in data]
all_winners = sum(1 for d in data if d.get('meta', {}).get('outcome_rr', 0) > 0)
print(f"  Baseline trades: {len(data)}")
print(f"  Baseline winrate: {all_winners/len(data)*100:.1f}%")
print(f"  Baseline avg R: {np.mean(all_outcomes):.4f}R")
print(f"  Baseline total R: {sum(all_outcomes):.2f}R")

# Improvement
if keep_data:
    print(f"\n[5] ML FILTER IMPROVEMENT")
    print(f"  Expectancy improvement: {(np.mean(outcomes) - np.mean(all_outcomes)):.4f}R")
    print(f"  Winrate improvement: {(winners/len(keep_data) - all_winners/len(data))*100:.1f}%")
    print(f"  Trade reduction: {(1 - len(keep_data)/len(data))*100:.1f}%")

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)
print(f"\nModel: seq_v1 / seq_conservative (t=0.8)")
print(f"Filter selectivity: {len(keep_data)/len(data)*100:.1f}% kept")
if keep_data:
    print(f"Expected performance: {np.mean(outcomes):.4f}R per trade")
    print(f"\nVs Backtest expectation: +1.19R (validation)")
    if abs(np.mean(outcomes) - 1.19) < 0.5:
        print("[OK] CLOSE TO BACKTEST - Replay validates model!")
    else:
        print("[WARNING] DIFFERENT FROM BACKTEST - Investigate")
print("="*80)
