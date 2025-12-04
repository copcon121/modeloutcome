#!/usr/bin/env python3
"""
S4_LDN Policy Sanity & Leak Tests

Runs sanity tests to ensure no data leakage in policy layer.

Usage:
    python asm_v2/scripts/eval_s4_ldn_policy_sanity.py --config asm_v2/configs/s4_policy_sanity_v1.json
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from asm_v2.src.s4_policy.policy_dataset import S4PolicyDataset, load_enriched_trades
from asm_v2.src.s4_policy.policy_rules import create_policies_from_config, DEFAULT_POLICIES
from asm_v2.src.s4_policy.eval_policy_sanity import (
    run_all_sanity_tests,
    save_sanity_report,
    print_sanity_summary,
)


def load_config(config_path: str) -> dict:
    """Load config from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description='S4_LDN Policy Sanity Tests')
    parser.add_argument('--config', type=str,
                        default='asm_v2/configs/s4_policy_sanity_v1.json',
                        help='Path to config file')
    args = parser.parse_args()
    
    print("=" * 60)
    print("S4_LDN Policy Sanity & Leak Tests")
    print("=" * 60)
    
    # Load config
    config = load_config(args.config)
    paths = config['paths']
    
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
    
    # Create some policies for consistency testing
    policies = create_policies_from_config(DEFAULT_POLICIES[:3])
    
    # Run all sanity tests
    print("\nRunning sanity tests...")
    results = run_all_sanity_tests(dataset, policies)
    
    # Print summary
    print_sanity_summary(results)
    
    # Save report
    output_path = paths['output_report']
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    save_sanity_report(results, output_path)
    print(f"\nSaved report to: {output_path}")
    
    # Exit with error code if any test failed
    all_passed = all(r.passed for r in results)
    sys.exit(0 if all_passed else 1)


if __name__ == '__main__':
    main()
