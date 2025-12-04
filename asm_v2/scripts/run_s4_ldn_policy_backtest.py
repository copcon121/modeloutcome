#!/usr/bin/env python3
"""
S4_LDN Policy Backtest Runner

Runs backtest on enriched S4 trades with multiple policies.
Outputs league table and detailed results.

Usage:
    python asm_v2/scripts/run_s4_ldn_policy_backtest.py --config asm_v2/configs/s4_policy_config_v1.json
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from asm_v2.src.s4_policy.policy_dataset import S4PolicyDataset, load_enriched_trades
from asm_v2.src.s4_policy.policy_rules import (
    Policy, PolicyConfig, create_policy, create_policies_from_config
)
from asm_v2.src.s4_policy.backtester import Backtester, run_backtest
from asm_v2.src.s4_policy.policy_model import S4MetaModel, train_meta_model
from asm_v2.src.s4_policy.metrics import compute_metrics_by_regime


def load_config(config_path: str) -> dict:
    """Load config from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description='S4_LDN Policy Backtest')
    parser.add_argument('--config', type=str, 
                        default='asm_v2/configs/s4_policy_config_v1.json',
                        help='Path to config file')
    args = parser.parse_args()
    
    print("=" * 80)
    print("S4_LDN Policy Backtest v1")
    print("=" * 80)
    
    # Load config
    config = load_config(args.config)
    paths = config['paths']
    backtest_cfg = config['backtest']
    
    # Load enriched trades
    trades_path = paths['enriched_trades']
    print(f"\nLoading enriched trades from: {trades_path}")
    
    try:
        trades = load_enriched_trades(trades_path)
        print(f"Loaded {len(trades)} trades")
    except FileNotFoundError:
        print(f"ERROR: File not found: {trades_path}")
        print("Please run enrich_s4_ldn_with_context.py first")
        sys.exit(1)
    
    if not trades:
        print("ERROR: No trades loaded")
        sys.exit(1)
    
    # Create dataset
    dataset = S4PolicyDataset(trades)
    
    # Print dataset summary
    print(f"\nDataset Summary:")
    print(f"  Total trades: {len(dataset)}")
    print(f"  Date range: {trades[0].get_date()} to {trades[-1].get_date()}")
    print(f"  Regime distribution: {dataset.get_regime_distribution()}")
    print(f"  Label distribution: {dataset.get_label_distribution()}")
    
    # Create policies from config
    print(f"\nCreating {len(config['policies'])} policies...")
    policies = create_policies_from_config(config['policies'])
    
    # Add ML-based policies if enabled
    meta_model_cfg = config.get('meta_model', {})
    if meta_model_cfg.get('enabled', False) and len(trades) >= 20:
        print("\nTraining meta-model...")
        train_ds, val_ds = dataset.time_split(0.8)
        
        if len(train_ds) >= 10:
            model, results = train_meta_model(train_ds, val_ds)
            print(f"  Train accuracy: {results['train']['accuracy']:.3f}")
            if 'val' in results:
                print(f"  Val accuracy: {results['val']['accuracy']:.3f}")
            
            # Create ML policies with different thresholds
            for threshold in meta_model_cfg.get('thresholds', [0.5]):
                from asm_v2.src.s4_policy.policy_rules import MLPredictionPolicy
                ml_config = PolicyConfig(
                    name=f"P_ML_thresh_{threshold}",
                    policy_type="ml",
                    params={"threshold": threshold}
                )
                ml_policy = MLPredictionPolicy(ml_config)
                ml_policy.set_model(model)
                policies.append(ml_policy)
    
    # Run backtest
    print(f"\nRunning backtest with {len(policies)} policies...")
    output_dir = paths['output_dir']
    
    bt = run_backtest(
        trades=trades,
        policies=policies,
        min_trades=backtest_cfg.get('min_trades_for_valid_policy', 1),
        output_dir=output_dir,
    )
    
    # Print summary
    bt.print_summary(top_n=10, sort_by=backtest_cfg.get('sort_by', 'expectancy'))
    
    # Save selected policy (best by expectancy with min trades)
    league = bt.get_league_table(sort_by='expectancy')
    if league:
        best = league[0]
        selected = {
            'name': best.name,
            'expectancy': best.expectancy,
            'win_rate': best.win_rate,
            'n_trades': best.n_trades,
            'max_drawdown_r': best.max_drawdown_r,
            'profit_factor': best.profit_factor,
        }
        selected_path = f"{output_dir}/s4_policy_selected_v1.json"
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        with open(selected_path, 'w') as f:
            json.dump(selected, f, indent=2)
        print(f"\nSaved selected policy to: {selected_path}")
    
    # Print regime breakdown for baseline
    print("\n" + "=" * 80)
    print("Regime Breakdown (Baseline - All Trades)")
    print("=" * 80)
    regime_metrics = compute_metrics_by_regime(trades)
    for regime, metrics in sorted(regime_metrics.items()):
        print(f"  {regime}: n={metrics['n_trades']}, WR={metrics['win_rate']:.1%}, Exp={metrics['expectancy']:.2f}R")
    
    print("\n" + "=" * 80)
    print(f"Results saved to: {output_dir}/")
    print("  - s4_policy_results_v1.json")
    print("  - s4_policy_league_table_v1.csv")
    print("  - s4_policy_selected_v1.json")
    print("=" * 80)


if __name__ == '__main__':
    main()
