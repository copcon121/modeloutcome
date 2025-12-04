#!/usr/bin/env python
"""
Script to train STATE-ENC model.

Usage:
    python scripts/train_state_enc.py --config configs/state_enc_train_v1.json
"""

import argparse
import json
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from state_enc_v1.training.trainer import train_state_enc


def main():
    parser = argparse.ArgumentParser(
        description="Train STATE-ENC v1 model"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to training config JSON file"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override device (cuda/cpu)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override max epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override learning rate"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("STATE-ENC v1 Training")
    print("=" * 60)
    print(f"Config: {args.config}")
    
    # Load and potentially modify config
    with open(args.config, "r") as f:
        config_dict = json.load(f)
    
    # Apply overrides
    if args.device:
        config_dict["device"] = args.device
    if args.epochs:
        config_dict["max_epochs"] = args.epochs
    if args.batch_size:
        config_dict["batch_size"] = args.batch_size
    if args.lr:
        config_dict["learning_rate"] = args.lr
    
    # Save modified config to temp file if needed
    if any([args.device, args.epochs, args.batch_size, args.lr]):
        temp_config_path = Path(args.config).parent / "temp_train_config.json"
        with open(temp_config_path, "w") as f:
            json.dump(config_dict, f, indent=2)
        config_path = str(temp_config_path)
        print(f"Using modified config: {config_path}")
    else:
        config_path = args.config
    
    print()
    
    try:
        summary = train_state_enc(config_path)
        
        print("\n" + "=" * 60)
        print("TRAINING COMPLETE")
        print("=" * 60)
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
