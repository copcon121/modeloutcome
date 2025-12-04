#!/usr/bin/env python
"""
STATE-ENC v1.2 — Leak & Integrity Test Suite CLI

Usage:
    python state_enc_v1/scripts/eval_state_enc_leaks_gc_m1.py --config state_enc_v1/configs/eval_leak_gc_m1_v1.2.json

Tests:
    L1: Index and Future Boundaries Check
    L2: Time-based Split Validation
    L3: Label Shuffle Sanity Test
    L4: Future Cheat Upper Bound Test
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from state_enc_v1.src.eval.leak_eval import run_leak_tests, LeakEvaluator


def main():
    parser = argparse.ArgumentParser(
        description='STATE-ENC v1.2 Leak & Integrity Test Suite',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        required=True,
        help='Path to leak evaluation config JSON file'
    )
    
    parser.add_argument(
        '--max-samples', '-n',
        type=int,
        default=None,
        help='Override max samples to check'
    )
    
    parser.add_argument(
        '--test', '-t',
        type=str,
        choices=['L1', 'L2', 'L3', 'L4', 'all'],
        default='all',
        help='Run specific test or all tests'
    )
    
    args = parser.parse_args()
    
    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Override with CLI args
    if args.max_samples:
        config['max_samples'] = args.max_samples

    print("=" * 70)
    print("STATE-ENC v1.2 — LEAK & INTEGRITY TEST SUITE")
    print("=" * 70)
    print(f"\nConfig: {config_path}")
    print(f"Dataset: {config.get('dataset_path', 'N/A')}")
    print(f"Max samples: {config.get('max_samples', 'all')}")
    print(f"Output: {config.get('output_dir', 'N/A')}")
    print()
    
    start_time = datetime.now()
    
    try:
        # Save config for reference
        output_dir = Path(config.get('output_dir', 'state_enc_v1/artifacts/v1_2/eval_gc_m1'))
        output_dir.mkdir(parents=True, exist_ok=True)
        
        config_used_path = output_dir / 'leak_eval_config_used.json'
        with open(config_used_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        # Run tests
        results = run_leak_tests(str(config_path))
        
        elapsed = (datetime.now() - start_time).total_seconds()
        
        print(f"\n⏱️ Total time: {elapsed:.1f}s")
        print(f"📁 Results saved to: {output_dir / 'leak_eval_full_report_v1.2.json'}")
        
        # Exit code based on results
        if results.get('all_passed', False):
            print("\n✅ All leak tests passed!")
            sys.exit(0)
        else:
            print("\n⚠️ Some leak tests failed - review results")
            sys.exit(1)
            
    except FileNotFoundError as e:
        print(f"\n❌ Error: File not found - {e}")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
