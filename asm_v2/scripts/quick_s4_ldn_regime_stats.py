#!/usr/bin/env python
"""
Quick stats for S4_LDN enriched trades by regime

Usage:
    python asm_v2/scripts/quick_s4_ldn_regime_stats.py
"""

import json
import sys
from pathlib import Path
from collections import defaultdict


def main():
    enriched_path = "asm_v2/artifacts/gc_m1/s4_ldn_trades_enriched_v1.jsonl"
    
    if not Path(enriched_path).exists():
        print(f"File not found: {enriched_path}")
        print("Run enrich_s4_ldn_with_context.py first")
        return 1
    
    # Load enriched trades
    trades = []
    with open(enriched_path, "r") as f:
        for line in f:
            if line.strip():
                trades.append(json.loads(line))
    
    print(f"Loaded {len(trades)} enriched trades")
    print()
    
    # Aggregate by regime
    regime_stats = defaultdict(lambda: {
        "count": 0,
        "wins": 0,
        "losses": 0,
        "total_rr": 0.0
    })
    
    for trade in trades:
        context = trade.get("context", {})
        regime_name = context.get("asm_regime_name", "unknown")
        
        regime_stats[regime_name]["count"] += 1
        
        label = trade.get("label", "flat")
        if label == "win":
            regime_stats[regime_name]["wins"] += 1
        elif label == "loss":
            regime_stats[regime_name]["losses"] += 1
        
        rr = trade.get("outcome_rr", 0.0)
        regime_stats[regime_name]["total_rr"] += rr
    
    # Print table
    print("=" * 70)
    print(f"{'Regime':<20} {'Count':>8} {'Wins':>8} {'Losses':>8} {'WR%':>8} {'Avg RR':>10}")
    print("=" * 70)
    
    for regime, stats in sorted(regime_stats.items()):
        count = stats["count"]
        wins = stats["wins"]
        losses = stats["losses"]
        winrate = wins / count * 100 if count > 0 else 0
        avg_rr = stats["total_rr"] / count if count > 0 else 0
        
        print(f"{regime:<20} {count:>8} {wins:>8} {losses:>8} {winrate:>7.1f}% {avg_rr:>10.2f}")
    
    print("=" * 70)
    
    # Overall
    total = sum(s["count"] for s in regime_stats.values())
    total_wins = sum(s["wins"] for s in regime_stats.values())
    total_losses = sum(s["losses"] for s in regime_stats.values())
    total_rr = sum(s["total_rr"] for s in regime_stats.values())
    
    overall_wr = total_wins / total * 100 if total > 0 else 0
    overall_avg_rr = total_rr / total if total > 0 else 0
    
    print(f"{'TOTAL':<20} {total:>8} {total_wins:>8} {total_losses:>8} {overall_wr:>7.1f}% {overall_avg_rr:>10.2f}")
    print("=" * 70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
