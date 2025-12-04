#!/usr/bin/env python3
"""
S4_LDN Policy Sanity Tests v2 - Enhanced leak/sanity tests

This script:
1. Loads enriched trades
2. Runs 4 sanity tests (time split, label shuffle, future guard, stability)
3. Outputs detailed report

Usage:
    python asm_v2/scripts/eval_s4_ldn_policy_sanity_v2.py --config asm_v2/configs/s4_policy_sanity_gc_m1_v1.json
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from asm_v2.src.s4_policy.policy_dataset import S4TradeEnriched
from asm_v2.src.s4_policy.eval_policy_sanity_v2 import (
    run_all_sanity_tests_v2,
    save_sanity_report_v2,
    print_sanity_summary_v2,
)
from asm_v2.src.s4_context_enricher_v2 import load_enriched_trades_v2


def load_config(config_path: str) -> dict:
    """Load config from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def convert_enriched_v2_to_policy_format(enriched_trades):
    """Convert S4TradeEnrichedV2 to S4TradeEnriched for policy dataset."""
    from datetime import datetime
    
    converted = []
    for t in enriched_trades:
        try:
            time = datetime.fromisoformat(t.time.replace('Z', '+00:00'))
        except:
            time = datetime.now()
        
        trade = S4TradeEnriched(
            symbol=t.symbol,
            tf=t.tf,
            time=time,
            direction=t.direction,
            entry_price=t.entry,
            sl_price=t.sl,
            tp_price=t.tp,
            rr_outcome=t.outcome_rr,
            label=t.label,
            regime=t.regime_id,
            regime_name=t.regime_name,
            z_t=t.z_t,
            regime_confidence=t.regime_confidence,
            session=t.session,
            session_id=t.session_id,
            pos_in_session_range=t.pos_in_session_range,
            inside_value=t.inside_value,
            above_value=t.above_value,
            below_value=t.below_value,
            bar_index=t.bar_index,
            raw={},
        )
        converted.append(trade)
    
    return converted


def time_split_trades(trades, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2):
    """Split trades by date."""
    trades = sorted(trades, key=lambda t: t.time)
    dates = sorted(set(t.get_date() for t in trades))
    n_dates = len(dates)
    
    n_train = int(n_dates * train_ratio)
    n_val = int(n_dates * val_ratio)
    
    train_dates = set(dates[:n_train])
    val_dates = set(dates[n_train:n_train + n_val])
    test_dates = set(dates[n_train + n_val:])
    
    train = [t for t in trades if t.get_date() in train_dates]
    val = [t for t in trades if t.get_date() in val_dates]
    test = [t for t in trades if t.get_date() in test_dates]
    
    return train, val, test


def main():
    parser = argparse.ArgumentParser(description='S4_LDN Policy Sanity Tests V2')
    parser.add_argument('--config', type=str,
                        default='asm_v2/configs/s4_policy_sanity_gc_m1_v1.json',
                        help='Path to config file')
    args = parser.parse_args()
    
    print("=" * 80)
    print("S4_LDN Policy Sanity Tests v2")
    print("=" * 80)
    
    # Load config
    config = load_config(args.config)
    paths = config['paths']
    splits_cfg = config.get('splits', {})
    
    # Load enriched trades
    trades_path = paths['enriched_trades']
    print(f"\nLoading enriched trades from: {trades_path}")
    
    try:
        enriched_trades = load_enriched_trades_v2(trades_path)
        print(f"Loaded {len(enriched_trades)} trades")
    except FileNotFoundError:
        print(f"ERROR: File not found: {trades_path}")
        print("Please run run_s4_ldn_enrich_gc_m1.py first")
        sys.exit(1)
    
    if not enriched_trades:
        print("ERROR: No trades loaded")
        sys.exit(1)
    
    # Convert to policy format
    trades = convert_enriched_v2_to_policy_format(enriched_trades)
    
    # Split trades
    train_trades, val_trades, test_trades = time_split_trades(
        trades,
        train_ratio=splits_cfg.get('train_ratio', 0.6),
        val_ratio=splits_cfg.get('val_ratio', 0.2),
        test_ratio=splits_cfg.get('test_ratio', 0.2),
    )
    
    print(f"\nSplit summary:")
    print(f"  Train: {len(train_trades)} trades")
    print(f"  Val: {len(val_trades)} trades")
    print(f"  Test: {len(test_trades)} trades")
    
    # Run sanity tests
    print("\nRunning sanity tests...")
    results = run_all_sanity_tests_v2(train_trades, val_trades, test_trades)
    
    # Print summary
    print_sanity_summary_v2(results)
    
    # Save report
    output_path = paths['output_report']
    save_sanity_report_v2(results, output_path)
    print(f"\nSaved report to: {output_path}")
    
    # Exit with error code if critical failures
    critical_failed = [r for r in results if not r.passed and r.severity == 'critical']
    if critical_failed:
        print(f"\n🔴 CRITICAL FAILURES detected!")
        sys.exit(1)
    
    all_passed = all(r.passed for r in results)
    sys.exit(0 if all_passed else 0)  # Warnings don't fail


if __name__ == '__main__':
    main()
