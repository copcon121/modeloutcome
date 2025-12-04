#!/usr/bin/env python3
"""
S4_LDN Real Trade Enrichment Pipeline for GC M1

Usage:
    python asm_v2/scripts/run_s4_ldn_enrich_gc_m1_real.py --config asm_v2/configs/s4_enrich_gc_m1_real_v1.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from asm_v2.src.s4_trade_dataset_builder_real import load_raw_trades_real
from asm_v2.src.s4_context_enricher_v3 import (
    S4ContextEnricherV3,
    save_enriched_trades_real,
)


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description='S4_LDN Real Trade Enrichment')
    parser.add_argument('--config', type=str,
                        default='asm_v2/configs/s4_enrich_gc_m1_real_v1.json',
                        help='Path to config file')
    args = parser.parse_args()
    
    print("=" * 80)
    print("S4_LDN Real Trade Enrichment Pipeline (GC M1)")
    print("=" * 80)
    
    config = load_config(args.config)
    paths = config['paths']
    enricher_cfg = config.get('enricher', {})
    
    # Load trades
    trades_path = paths['trades_input']
    print(f"\nLoading trades from: {trades_path}")
    
    try:
        trades = load_raw_trades_real(trades_path)
        print(f"Loaded {len(trades)} trades")
    except FileNotFoundError:
        print(f"ERROR: File not found: {trades_path}")
        print("Please run build_s4_ldn_trades_real_gc_m1.py first")
        sys.exit(1)
    
    # Convert to dict for enricher
    trade_dicts = [t.to_dict() for t in trades]
    
    # Initialize enricher
    print(f"\n" + "=" * 80)
    print("Initializing Enricher V3")
    print("=" * 80)
    
    enricher = S4ContextEnricherV3(
        state_enc_path=paths['state_enc_model'],
        state_enc_config_path=paths['state_enc_config'],
        feature_config_path=paths['feature_config'],
        asm_model_path=paths['asm_model'],
        asm_config_path=paths['asm_config'],
        encoder_dataset_path=paths['encoder_dataset'],
        time_tolerance_minutes=enricher_cfg.get('time_tolerance_minutes', 5),
        device=enricher_cfg.get('device', 'cpu'),
    )
    
    # Enrich trades
    print(f"\n" + "=" * 80)
    print("Enriching Trades")
    print("=" * 80)
    
    enriched, unmatched = enricher.enrich_trades(
        trade_dicts,
        show_progress=enricher_cfg.get('show_progress', True),
    )
    
    print(f"\nEnrichment Results:")
    print(f"  Input trades: {len(trades)}")
    print(f"  Enriched: {len(enriched)}")
    print(f"  Unmatched: {len(unmatched)}")
    print(f"  Match rate: {len(enriched)/len(trades)*100:.1f}%")
    
    # Save enriched trades
    output_path = paths['enriched_output']
    save_enriched_trades_real(enriched, output_path)
    print(f"\nSaved to: {output_path}")
    
    # Print summary
    print(f"\n" + "=" * 80)
    print("Enrichment Summary")
    print("=" * 80)
    
    # Regime distribution
    regime_dist = {}
    for t in enriched:
        regime_dist[t.regime_name] = regime_dist.get(t.regime_name, 0) + 1
    
    print(f"\nRegime distribution:")
    for regime, count in sorted(regime_dist.items()):
        pct = count / len(enriched) * 100 if enriched else 0
        print(f"  {regime}: {count} ({pct:.1f}%)")
    
    # Session distribution
    session_dist = {}
    for t in enriched:
        session_dist[t.session] = session_dist.get(t.session, 0) + 1
    
    print(f"\nSession distribution:")
    for session, count in sorted(session_dist.items()):
        pct = count / len(enriched) * 100 if enriched else 0
        print(f"  {session}: {count} ({pct:.1f}%)")
    
    # Sample enriched trades
    if enriched:
        print(f"\nSample enriched trades:")
        for i, t in enumerate(enriched[:3]):
            print(f"  [{i+1}] {t.trade_id}: {t.direction} @ {t.entry_time[:19]}")
            print(f"      regime={t.regime_name} (conf={t.regime_confidence:.3f})")
            print(f"      hit={t.hit}, rr={t.rr_realized:.2f}R")
    
    # Log unmatched trades
    if unmatched:
        print(f"\nUnmatched trades (first 5):")
        for t in unmatched[:5]:
            print(f"  - {t.get('trade_id', '?')}: {t.get('entry_time', '?')}")
    
    print(f"\n" + "=" * 80)
    print(f"✅ Enrichment complete!")
    print(f"   Output: {output_path}")
    print("=" * 80)


if __name__ == '__main__':
    main()
