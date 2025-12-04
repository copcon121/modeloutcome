#!/usr/bin/env python
"""
Build ASM v2 dataset from encoder_dataset_gc_m1_v1.2.jsonl

Usage:
    python asm_v2/scripts/build_asm_dataset_gc_m1.py --config asm_v2/configs/asm_dataset_gc_m1_v1.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from asm_v2.src.dataset_builder import AsmDatasetBuilder
from asm_v2.src.config import AsmDatasetConfig


def main():
    parser = argparse.ArgumentParser(description="Build ASM v2 dataset")
    parser.add_argument("--config", type=str, required=True, help="Path to config JSON")
    args = parser.parse_args()
    
    # Load config
    config = AsmDatasetConfig.from_json(args.config)
    
    print("=" * 60)
    print("Building ASM v2 Dataset")
    print("=" * 60)
    print(f"Encoder dataset: {config.encoder_dataset_path}")
    print(f"STATE-ENC model: {config.state_enc_model_path}")
    print(f"Output: {config.output_path}")
    print()
    
    # Build dataset
    builder = AsmDatasetBuilder(
        encoder_dataset_path=config.encoder_dataset_path,
        state_enc_model_path=config.state_enc_model_path,
        state_enc_model_config_path=config.state_enc_model_config_path,
        meta_features=config.meta_features,
        regime_mapping=config.regime_mapping,
        device="cuda"
    )
    
    stats = builder.build(
        output_path=config.output_path,
        splits_output_path=config.splits_output_path,
        feature_config_output_path=config.feature_config_output_path,
        train_ratio=config.train_ratio
    )
    
    print()
    print("=" * 60)
    print("Build Summary")
    print("=" * 60)
    print(f"Total samples: {stats['num_samples']}")
    print(f"Regime classes: {stats['num_classes']}")
    print(f"Train samples: {stats['train_samples']}")
    print(f"Val samples: {stats['val_samples']}")
    print(f"Regime distribution: {stats['regime_counts']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
