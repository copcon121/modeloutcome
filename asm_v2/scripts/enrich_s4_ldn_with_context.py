#!/usr/bin/env python
"""
Enrich S4_LDN trades with STATE-ENC embeddings and ASM regime predictions

Usage:
    python asm_v2/scripts/enrich_s4_ldn_with_context.py \
        --bars-glob "data/raw/smc_export_gc_m1_v3_*.jsonl" \
        --s4-file "data/s4_ldn/gc_m1_trades_rule_only.jsonl"
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from asm_v2.src.s4_context_enricher import S4ContextEnricher


def main():
    parser = argparse.ArgumentParser(description="Enrich S4_LDN trades with context")
    parser.add_argument("--bars-glob", type=str, required=True, help="Glob pattern for bar data files")
    parser.add_argument("--s4-file", type=str, required=True, help="Path to S4 trades JSONL")
    parser.add_argument("--output", type=str, default="asm_v2/artifacts/gc_m1/s4_ldn_trades_enriched_v1.jsonl",
                        help="Output path for enriched trades")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    args = parser.parse_args()
    
    print("=" * 60)
    print("S4_LDN Context Enricher")
    print("=" * 60)
    print(f"Bars: {args.bars_glob}")
    print(f"S4 trades: {args.s4_file}")
    print(f"Output: {args.output}")
    print()
    
    # Check if S4 file exists
    if not Path(args.s4_file).exists():
        print(f"Warning: S4 file not found: {args.s4_file}")
        print("Creating mock S4 trades file for testing...")
        
        # Create mock S4 trades
        mock_trades = [
            {
                "symbol": "GC 12-25",
                "tf": "M1",
                "time": "2025-10-13T01:51:00.000Z",
                "bar_index": 170,
                "session": "Asia",
                "signal": "short",
                "signal_type": "s4_ldn_short",
                "entry": 4066.4,
                "sl": 4070.5,
                "tp": 4054.1,
                "hit": "tp",
                "outcome_rr": 3.0,
                "label": "win"
            },
            {
                "symbol": "GC 12-25",
                "tf": "M1",
                "time": "2025-10-14T02:30:00.000Z",
                "bar_index": 210,
                "session": "LDN",
                "signal": "long",
                "signal_type": "s4_ldn_long",
                "entry": 4080.0,
                "sl": 4075.0,
                "tp": 4095.0,
                "hit": "sl",
                "outcome_rr": -1.0,
                "label": "loss"
            }
        ]
        
        Path(args.s4_file).parent.mkdir(parents=True, exist_ok=True)
        with open(args.s4_file, "w") as f:
            for trade in mock_trades:
                f.write(json.dumps(trade) + "\n")
        print(f"Created mock file with {len(mock_trades)} trades")
        print()
    
    # Initialize enricher
    enricher = S4ContextEnricher(
        state_enc_model_path="state_enc_v1/artifacts/v1_2/final/state_enc_v1.2.pt",
        state_enc_config_path="state_enc_v1/artifacts/v1_2/final/model_config_v1.2.json",
        state_enc_feature_config_path="state_enc_v1/artifacts/v1_2/final/feature_config_v1.2.json",
        asm_model_path="asm_v2/artifacts/final/asm_v2_gc_m1_v1.pt",
        asm_config_path="asm_v2/artifacts/final/asm_model_config_v1.json",
        asm_feature_config_path="asm_v2/artifacts/final/asm_feature_config_v1.json",
        device=args.device
    )
    
    # Load bar data
    bars_by_date = enricher.load_bar_data(args.bars_glob)
    
    # Load S4 trades
    trades = enricher.load_s4_trades(args.s4_file)
    
    # Enrich trades
    stats = enricher.enrich_trades(bars_by_date, trades, args.output)
    
    # Print summary
    print()
    print("=" * 60)
    print("Enrichment Summary")
    print("=" * 60)
    print(f"Total trades: {stats['total_trades']}")
    print(f"Enriched trades: {stats['enriched_trades']}")
    print(f"Skipped (no bars): {stats['skipped_no_bars']}")
    print()
    print("Regime distribution by outcome:")
    for regime, counts in stats.get("regime_distribution", {}).items():
        total = counts["total"]
        wins = counts["win"]
        losses = counts["loss"]
        winrate = wins / total * 100 if total > 0 else 0
        print(f"  {regime}: {total} trades, {wins} wins, {losses} losses ({winrate:.1f}% WR)")
    print("=" * 60)


if __name__ == "__main__":
    main()
