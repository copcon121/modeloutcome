#!/usr/bin/env python
"""
STATE-ENC v1.2 — Semantic Evaluation CLI

Usage:
    python state_enc_v1/scripts/eval_state_enc_semantic.py --config configs/eval_semantic_v1.2.json

Tests:
    S1: Linear Probe Future_Dir_5 (baseline vs z_t)
    S2: Regime Probe (regime_hint classification)
    S3: Cluster & Market Behavior Map (KMeans analysis)
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from state_enc_v1.src.eval.semantic_eval import run_all_semantic_tests


def main():
    parser = argparse.ArgumentParser(
        description='STATE-ENC v1.2 Semantic Evaluation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python state_enc_v1/scripts/eval_state_enc_semantic.py --config state_enc_v1/configs/eval_semantic_v1.2.json
    
    # With custom parameters
    python state_enc_v1/scripts/eval_state_enc_semantic.py \\
        --config state_enc_v1/configs/eval_semantic_v1.2.json \\
        --max-samples 10000 \\
        --kmeans-k 6
        """
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        required=True,
        help='Path to evaluation config JSON file'
    )
    
    parser.add_argument(
        '--max-samples', '-n',
        type=int,
        default=None,
        help='Override max samples to evaluate'
    )
    
    parser.add_argument(
        '--kmeans-k', '-k',
        type=int,
        default=None,
        help='Override number of KMeans clusters'
    )

    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default=None,
        help='Override output directory'
    )
    
    parser.add_argument(
        '--test', '-t',
        type=str,
        choices=['S1', 'S2', 'S3', 'all'],
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
    if args.kmeans_k:
        config['kmeans_k'] = args.kmeans_k
    if args.output_dir:
        config['output_dir'] = args.output_dir
    
    # Save modified config to temp file
    temp_config_path = Path(config.get('output_dir', 'state_enc_v1/artifacts/v1_2/eval')) / 'eval_config_used.json'
    temp_config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(temp_config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print("=" * 70)
    print("STATE-ENC v1.2 — SEMANTIC EVALUATION")
    print("=" * 70)
    print(f"\nConfig: {config_path}")
    print(f"Dataset: {config.get('dataset_path', 'N/A')}")
    print(f"Max samples: {config.get('max_samples', 'all')}")
    print(f"KMeans K: {config.get('kmeans_k', 8)}")
    print(f"Output: {config.get('output_dir', 'N/A')}")
    print()
    
    # Run evaluation
    try:
        if args.test == 'all':
            results = run_all_semantic_tests(str(temp_config_path))
        else:
            # Run specific test
            from state_enc_v1.src.eval.semantic_eval import (
                SemanticEvaluator,
                run_future_dir_probe,
                run_regime_probe,
                run_cluster_analysis
            )
            
            evaluator = SemanticEvaluator(config)
            
            if args.test == 'S1':
                results = {'S1': run_future_dir_probe(evaluator)}
            elif args.test == 'S2':
                results = {'S2': run_regime_probe(evaluator)}
            elif args.test == 'S3':
                results = {'S3': run_cluster_analysis(evaluator)}
        
        print("\n✅ Evaluation completed successfully!")
        print(f"Results saved to: {config.get('output_dir', 'state_enc_v1/artifacts/v1_2/eval')}")
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: File not found - {e}")
        print("\nMake sure the following files exist:")
        print(f"  - Model: {config.get('model_path', 'N/A')}")
        print(f"  - Model config: {config.get('model_config_path', 'N/A')}")
        print(f"  - Feature config: {config.get('feature_config_path', 'N/A')}")
        print(f"  - Dataset: {config.get('dataset_path', 'N/A')}")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
