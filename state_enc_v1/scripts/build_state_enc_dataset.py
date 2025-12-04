#!/usr/bin/env python
"""
Script to build STATE-ENC dataset from raw bar data.

Usage:
    python scripts/build_state_enc_dataset.py --config configs/state_enc_dataset_v1.json
"""

import argparse
import json
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from state_enc_v1.src.config import StateEncDatasetConfig
from state_enc_v1.src.dataset_builder import build_state_enc_dataset


def main():
    parser = argparse.ArgumentParser(
        description="Build STATE-ENC dataset from raw bar data"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to dataset config JSON file"
    )
    parser.add_argument(
        "--raw-path",
        type=str,
        default=None,
        help="Override raw bars path from config"
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Override output path from config"
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=None,
        help="Override sequence length"
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help="Override stride"
    )
    
    args = parser.parse_args()
    
    # Load config
    print(f"Loading config from: {args.config}")
    config = StateEncDatasetConfig.from_json(args.config)
    
    # Apply overrides
    if args.raw_path:
        config.raw_bars_path = args.raw_path
    if args.output_path:
        config.output_path = args.output_path
    if args.sequence_length:
        config.sequence_length = args.sequence_length
    if args.stride:
        config.stride = args.stride
    
    print("\nConfiguration:")
    print(f"  Raw bars path: {config.raw_bars_path}")
    print(f"  Output path: {config.output_path}")
    print(f"  Sequence length: {config.sequence_length}")
    print(f"  Stride: {config.stride}")
    print(f"  Future bars: {config.future_bars}")
    print()
    
    # Build dataset
    try:
        summary = build_state_enc_dataset(
            raw_bars_path=config.raw_bars_path,
            output_path=config.output_path,
            config=config
        )
        
        print("\n" + "=" * 50)
        print("BUILD COMPLETE")
        print("=" * 50)
        print(json.dumps(summary, indent=2))
        
        return 0
        
    except FileNotFoundError as e:
        print(f"\nERROR: File not found - {e}")
        return 1
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
