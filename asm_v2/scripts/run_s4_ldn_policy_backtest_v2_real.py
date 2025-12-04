#!/usr/bin/env python3
"""
S4_LDN Policy Backtest v2 for REAL Data

Usage:
    python asm_v2/scripts/run_s4_ldn_policy_backtest_v2_real.py --config asm_v2/configs/s4_policy_config_gc_m1_real_v1.json
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from asm_v2.src.s4_policy.policy_dataset import S4TradeEnriched
from asm_v2.src.s4_policy.policy_rules import create_policies_from_config
from asm_v2.src.s4_policy.backtester_v2 import BacktesterV2
from asm_v2.src.s4_context_enricher_v3 import load_enriched_trades_real


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return json.load(f)


def convert_real_to_policy_format(enriched_trades):
    """Convert S4TradeEnrichedReal to S4TradeEnriched for policy dataset."""
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


def main():
    parser = argparse.ArgumentParser(description='S4_LDN Policy Backtest V2 (REAL)')
    parser.add_argument('--config', type=str,
                        default='asm_v2/configs/s4_policy_config_gc_m1_real_v1.json',
                        help='Path to config file')
    args = parser.parse_args()
    
    print("=" * 100)
    print("S4_LDN Policy Backtest v2 (REAL Data)")
    print("=" * 100)
    
    config = load_config(args.config)
    paths = config['paths']
    splits_cfg = config.get('splits', {})
    backtest_cfg = config.get('backtest', {})
    meta_model_cfg = config.get('meta_model', {})
    output_files = config.get('output_files', {})
    
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
    
    # Print dataset summary
    print(f"\nDataset Summary:")
    print(f"  Total trades: {len(trades)}")
    dates = sorted(set(t.get_date() for t in trades))
    print(f"  Date range: {dates[0]} to {dates[-1]}")
    print(f"  Days: {len(dates)}")
    
    # Distributions
    regime_dist = {}
    label_dist = {}
    session_dist = {}
    for t in trades:
        regime_dist[t.regime_name] = regime_dist.get(t.regime_name, 0) + 1
        label_dist[t.label] = label_dist.get(t.label, 0) + 1
        session_dist[t.session] = session_dist.get(t.session, 0) + 1
    
    print(f"  Regime: {regime_dist}")
    print(f"  Labels: {label_dist}")
    print(f"  Sessions: {session_dist}")
    
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
        min_trades=backtest_cfg.get('min_trades_for_valid_policy', 3),
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
    
    # Save results with custom filenames
    output_dir = paths['output_dir']
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Full results JSON
    results_data = {
        'n_total_trades': len(trades),
        'n_train': len(bt.train_trades),
        'n_val': len(bt.val_trades),
        'n_test': len(bt.test_trades),
        'results': [r.to_dict() for r in bt.results],
    }
    league_json_path = f"{output_dir}/{output_files.get('league_json', 's4_policy_league_gc_m1_real_v1.json')}"
    with open(league_json_path, 'w') as f:
        json.dump(results_data, f, indent=2)
    
    # League CSV
    league_csv_path = f"{output_dir}/{output_files.get('league_csv', 's4_policy_league_gc_m1_real_v1.csv')}"
    bt._save_league_csv(league_csv_path)
    
    # Best policy
    best = bt.get_best_policy('test')
    if best:
        from dataclasses import asdict
        best_data = {
            'policy_name': best.name,
            'train': asdict(best.train) if best.train else None,
            'val': asdict(best.val) if best.val else None,
            'test': asdict(best.test) if best.test else None,
        }
        best_path = f"{output_dir}/{output_files.get('best_policy', 's4_policy_best_gc_m1_real_v1.json')}"
        with open(best_path, 'w') as f:
            json.dump(best_data, f, indent=2)
    
    print(f"\n" + "=" * 100)
    print(f"Results saved to: {output_dir}/")
    print(f"  - {output_files.get('league_json', 's4_policy_league_gc_m1_real_v1.json')}")
    print(f"  - {output_files.get('league_csv', 's4_policy_league_gc_m1_real_v1.csv')}")
    print(f"  - {output_files.get('best_policy', 's4_policy_best_gc_m1_real_v1.json')}")
    print("=" * 100)


if __name__ == '__main__':
    main()
