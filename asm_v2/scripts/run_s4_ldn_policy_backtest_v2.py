#!/usr/bin/env python3
"""
S4_LDN Policy Backtest v2 - Full backtest with train/val/test splits

This script:
1. Loads enriched trades
2. Splits by time (train/val/test)
3. Evaluates rule-based and ML policies
4. Outputs league tables and best policy

Usage:
    python asm_v2/scripts/run_s4_ldn_policy_backtest_v2.py --config asm_v2/configs/s4_policy_config_gc_m1_v1.json
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from asm_v2.src.s4_policy.policy_dataset import S4PolicyDataset
from asm_v2.src.s4_policy.policy_rules import create_policies_from_config
from asm_v2.src.s4_policy.backtester_v2 import BacktesterV2
from asm_v2.src.s4_context_enricher_v2 import load_enriched_trades_v2


def load_config(config_path: str) -> dict:
    """Load config from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def convert_enriched_v2_to_policy_format(enriched_trades):
    """Convert S4TradeEnrichedV2 to S4TradeEnriched for policy dataset."""
    from asm_v2.src.s4_policy.policy_dataset import S4TradeEnriched
    from datetime import datetime
    
    converted = []
    for t in enriched_trades:
        # Parse time
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


def main():
    parser = argparse.ArgumentParser(description='S4_LDN Policy Backtest V2')
    parser.add_argument('--config', type=str,
                        default='asm_v2/configs/s4_policy_config_gc_m1_v1.json',
                        help='Path to config file')
    args = parser.parse_args()
    
    print("=" * 100)
    print("S4_LDN Policy Backtest v2 (Full)")
    print("=" * 100)
    
    # Load config
    config = load_config(args.config)
    paths = config['paths']
    splits_cfg = config.get('splits', {})
    backtest_cfg = config.get('backtest', {})
    meta_model_cfg = config.get('meta_model', {})
    
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
    
    # Print dataset summary
    print(f"\nDataset Summary:")
    print(f"  Total trades: {len(trades)}")
    dates = sorted(set(t.get_date() for t in trades))
    print(f"  Date range: {dates[0]} to {dates[-1]}")
    print(f"  Days: {len(dates)}")
    
    # Regime distribution
    regime_dist = {}
    for t in trades:
        regime_dist[t.regime_name] = regime_dist.get(t.regime_name, 0) + 1
    print(f"  Regime distribution: {regime_dist}")
    
    # Label distribution
    label_dist = {}
    for t in trades:
        label_dist[t.label] = label_dist.get(t.label, 0) + 1
    print(f"  Label distribution: {label_dist}")
    
    # Create backtester
    print(f"\nCreating backtester with splits:")
    print(f"  Train: {splits_cfg.get('train_ratio', 0.6)*100:.0f}%")
    print(f"  Val: {splits_cfg.get('val_ratio', 0.2)*100:.0f}%")
    print(f"  Test: {splits_cfg.get('test_ratio', 0.2)*100:.0f}%")
    
    bt = BacktesterV2(
        trades=trades,
        train_ratio=splits_cfg.get('train_ratio', 0.6),
        val_ratio=splits_cfg.get('val_ratio', 0.2),
        test_ratio=splits_cfg.get('test_ratio', 0.2),
        min_trades=backtest_cfg.get('min_trades_for_valid_policy', 5),
    )
    
    # Create policies
    print(f"\nCreating {len(config['policies'])} policies...")
    policies = create_policies_from_config(config['policies'])
    
    # Evaluate policies
    print(f"\nEvaluating policies...")
    bt.evaluate_policies(
        policies=policies,
        include_ml_policies=meta_model_cfg.get('enabled', True),
        ml_thresholds=meta_model_cfg.get('thresholds', [0.5]),
    )
    
    # Print summary
    bt.print_summary(top_n=10)
    
    # Save results
    output_dir = paths['output_dir']
    bt.save_results(output_dir)
    
    print(f"\n" + "=" * 100)
    print(f"Results saved to: {output_dir}/")
    print(f"  - s4_policy_league_gc_m1_v1.json")
    print(f"  - s4_policy_league_gc_m1_v1.csv")
    print(f"  - s4_policy_best_gc_m1_v1.json")
    print("=" * 100)


if __name__ == '__main__':
    main()
