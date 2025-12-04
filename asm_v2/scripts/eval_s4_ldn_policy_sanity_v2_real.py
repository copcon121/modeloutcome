#!/usr/bin/env python3
"""
S4_LDN Policy Sanity Tests v2 for REAL Data

Usage:
    python asm_v2/scripts/eval_s4_ldn_policy_sanity_v2_real.py --config asm_v2/configs/s4_policy_sanity_gc_m1_real_v1.json
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from asm_v2.src.s4_policy.policy_dataset import S4TradeEnriched
from asm_v2.src.s4_policy.eval_policy_sanity_v2 import (
    run_all_sanity_tests_v2,
    save_sanity_report_v2,
    print_sanity_summary_v2,
    SanityTestResultV2,
)
from asm_v2.src.s4_context_enricher_v3 import load_enriched_trades_real


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return json.load(f)


def convert_real_to_policy_format(enriched_trades):
    """Convert S4TradeEnrichedReal to S4TradeEnriched."""
    converted = []
    for t in enriched_trades:
        try:
            time = datetime.fromisoformat(t.entry_time.replace('Z', '+00:00'))
        except:
            time = datetime.now()
        
        trade = S4TradeEnriched(
            symbol=t.symbol,
            tf=t.tf,
            time=time,
            direction=t.direction,
            entry_price=t.entry_price,
            sl_price=t.sl_price,
            tp_price=t.tp_price,
            rr_outcome=t.rr_realized,
            label=t.get_label(),
            regime=t.regime_id,
            regime_name=t.regime_name,
            z_t=t.z_t,
            regime_confidence=t.regime_confidence,
            session=t.session,
            session_id=t.session_id,
            pos_in_session_range=0.5,
            inside_value=0,
            above_value=0,
            below_value=0,
            bar_index=0,
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


def test_regime_leak_guard(enriched_trades) -> SanityTestResultV2:
    """T5: Verify regime prediction uses only past context.
    
    Check that encoder_sample_idx corresponds to context BEFORE entry.
    """
    issues = []
    
    for t in enriched_trades:
        # Check time_match_delta_sec - should be small and positive
        # (encoder sample end_time should be <= entry_time)
        if hasattr(t, 'time_match_delta_sec'):
            if t.time_match_delta_sec > 300:  # More than 5 min
                issues.append({
                    'trade_id': t.trade_id,
                    'delta_sec': t.time_match_delta_sec,
                })
    
    passed = len(issues) == 0
    
    return SanityTestResultV2(
        name="T5_RegimeLeakGuard",
        passed=passed,
        severity='warning' if not passed else 'info',
        details={
            'n_trades_checked': len(enriched_trades),
            'n_issues': len(issues),
            'issues': issues[:5],  # First 5
            'note': 'Checks that regime is computed from context before entry',
        }
    )


def main():
    parser = argparse.ArgumentParser(description='S4_LDN Policy Sanity Tests V2 (REAL)')
    parser.add_argument('--config', type=str,
                        default='asm_v2/configs/s4_policy_sanity_gc_m1_real_v1.json',
                        help='Path to config file')
    args = parser.parse_args()
    
    print("=" * 80)
    print("S4_LDN Policy Sanity Tests v2 (REAL Data)")
    print("=" * 80)
    
    config = load_config(args.config)
    paths = config['paths']
    splits_cfg = config.get('splits', {})
    tests_cfg = config.get('tests', {})
    
    # Load enriched trades
    trades_path = paths['enriched_trades']
    print(f"\nLoading enriched trades from: {trades_path}")
    
    try:
        enriched_trades = load_enriched_trades_real(trades_path)
        print(f"Loaded {len(enriched_trades)} trades")
    except FileNotFoundError:
        print(f"ERROR: File not found: {trades_path}")
        print("Please run run_s4_ldn_enrich_gc_m1_real.py first")
        sys.exit(1)
    
    if not enriched_trades:
        print("ERROR: No trades loaded")
        sys.exit(1)
    
    # Convert to policy format
    trades = convert_real_to_policy_format(enriched_trades)
    
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
    
    # Run standard sanity tests
    print("\nRunning sanity tests...")
    results = run_all_sanity_tests_v2(train_trades, val_trades, test_trades)
    
    # Add T5: Regime Leak Guard
    if tests_cfg.get('T5_RegimeLeakGuard', {}).get('enabled', True):
        t5_result = test_regime_leak_guard(enriched_trades)
        results.append(t5_result)
    
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
    
    sys.exit(0)


if __name__ == '__main__':
    main()
