#!/usr/bin/env python3
"""
S4_LDN Trade Enrichment Pipeline for GC M1

This script:
1. Builds/loads standardized trade dataset
2. Enriches trades with z_t embeddings and regime predictions
3. Outputs enriched trades for policy backtest

Usage:
    python asm_v2/scripts/run_s4_ldn_enrich_gc_m1.py --config asm_v2/configs/s4_enrich_gc_m1_v1.json
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from asm_v2.src.s4_trade_dataset_builder import (
    build_trade_dataset,
    load_raw_trades,
    get_trade_stats,
    S4TradeRaw,
)
from asm_v2.src.s4_context_enricher_v2 import (
    S4ContextEnricherV2,
    save_enriched_trades,
)


def load_config(config_path: str) -> dict:
    """Load config from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description='S4_LDN Trade Enrichment')
    parser.add_argument('--config', type=str,
                        default='asm_v2/configs/s4_enrich_gc_m1_v1.json',
                        help='Path to config file')
    args = parser.parse_args()
    
    print("=" * 80)
    print("S4_LDN Trade Enrichment Pipeline (GC M1)")
    print("=" * 80)
    
    # Load config
    config = load_config(args.config)
    paths = config['paths']
    builder_cfg = config.get('trade_builder', {})
    enricher_cfg = config.get('enricher', {})
    
    # Step 1: Build/load standardized trades
    print("\n" + "=" * 80)
    print("Step 1: Building Trade Dataset")
    print("=" * 80)
    
    raw_path = paths.get('raw_trades')
    std_path = paths['standardized_trades']
    
    trades = build_trade_dataset(
        input_path=raw_path if raw_path and Path(raw_path).exists() else None,
        output_path=std_path,
        generate_if_missing=builder_cfg.get('generate_if_missing', True),
        n_synthetic=builder_cfg.get('n_synthetic_trades', 200),
    )
    
    # Print stats
    stats = get_trade_stats(trades)
    print(f"\nTrade Dataset Stats:")
    print(f"  Total trades: {stats['n_trades']}")
    print(f"  Date range: {stats['date_range']}")
    print(f"  Days: {stats['n_days']}")
    print(f"  Sessions: {stats['sessions']}")
    print(f"  Directions: {stats['directions']}")
    print(f"  Labels: {stats['labels']}")
    print(f"  Win rate: {stats['win_rate']:.1%}")
    print(f"  Avg RR: {stats['avg_rr']:.2f}")
    print(f"  Total R: {stats['total_r']:.2f}")
    
    # Step 2: Initialize enricher
    print("\n" + "=" * 80)
    print("Step 2: Initializing Enricher")
    print("=" * 80)
    
    enricher = S4ContextEnricherV2(
        state_enc_path=paths['state_enc_model'],
        state_enc_config_path=paths['state_enc_config'],
        feature_config_path=paths['feature_config'],
        asm_model_path=paths['asm_model'],
        asm_config_path=paths['asm_config'],
        encoder_dataset_path=paths['encoder_dataset'],
        device=enricher_cfg.get('device', 'cpu'),
    )
    
    # Step 3: Enrich trades
    print("\n" + "=" * 80)
    print("Step 3: Enriching Trades")
    print("=" * 80)
    
    # Convert S4TradeRaw to dict for enricher
    trade_dicts = [t.to_dict() for t in trades]
    
    enriched_trades = enricher.enrich_trades(
        trade_dicts,
        show_progress=enricher_cfg.get('show_progress', True),
    )
    
    print(f"\nEnriched {len(enriched_trades)} trades")
    
    # Step 4: Save enriched trades
    enriched_path = paths['enriched_trades']
    save_enriched_trades(enriched_trades, enriched_path)
    print(f"Saved to: {enriched_path}")
    
    # Print enrichment summary
    print("\n" + "=" * 80)
    print("Enrichment Summary")
    print("=" * 80)
    
    regime_dist = {}
    for t in enriched_trades:
        regime_dist[t.regime_name] = regime_dist.get(t.regime_name, 0) + 1
    
    print(f"Regime distribution:")
    for regime, count in sorted(regime_dist.items()):
        pct = count / len(enriched_trades) * 100
        print(f"  {regime}: {count} ({pct:.1f}%)")
    
    # Sample enriched trade
    if enriched_trades:
        sample = enriched_trades[0]
        print(f"\nSample enriched trade:")
        print(f"  trade_id: {sample.trade_id}")
        print(f"  time: {sample.time}")
        print(f"  direction: {sample.direction}")
        print(f"  regime: {sample.regime_name} (conf={sample.regime_confidence:.3f})")
        print(f"  z_t dim: {len(sample.z_t)}")
        print(f"  outcome: {sample.label} ({sample.outcome_rr:.2f}R)")
    
    print("\n" + "=" * 80)
    print("✅ Enrichment complete!")
    print(f"   Output: {enriched_path}")
    print("=" * 80)


if __name__ == '__main__':
    main()
