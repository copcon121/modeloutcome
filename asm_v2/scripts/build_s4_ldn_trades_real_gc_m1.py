#!/usr/bin/env python3
"""
Build S4_LDN Real Trade Dataset for GC M1

Usage:
    python asm_v2/scripts/build_s4_ldn_trades_real_gc_m1.py --config asm_v2/configs/s4_ldn_trades_real_gc_m1_v1.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from asm_v2.src.s4_trade_dataset_builder_real import (
    build_real_trade_dataset,
    get_trade_stats_real,
)


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description='Build S4_LDN Real Trade Dataset')
    parser.add_argument('--config', type=str,
                        default='asm_v2/configs/s4_ldn_trades_real_gc_m1_v1.json',
                        help='Path to config file')
    args = parser.parse_args()
    
    print("=" * 80)
    print("S4_LDN Real Trade Dataset Builder (GC M1)")
    print("=" * 80)
    
    config = load_config(args.config)
    paths = config['paths']
    builder_cfg = config.get('builder', {})
    
    # Build dataset
    date_range = builder_cfg.get('date_range')
    if date_range:
        date_range = tuple(date_range)
    
    trades = build_real_trade_dataset(
        input_path=paths.get('raw_input'),
        output_path=paths['output'],
        generate_if_missing=builder_cfg.get('generate_if_missing', True),
        n_trades=builder_cfg.get('n_trades', 100),
        min_rr=builder_cfg.get('min_rr', 0.0),
        date_range=date_range,
    )
    
    # Print stats
    stats = get_trade_stats_real(trades)
    print(f"\n" + "=" * 80)
    print("Trade Dataset Stats:")
    print("=" * 80)
    print(f"  Total trades: {stats['n_trades']}")
    print(f"  Date range: {stats['date_range']}")
    print(f"  Days: {stats['n_days']}")
    print(f"  Sessions: {stats['sessions']}")
    print(f"  Directions: {stats['directions']}")
    print(f"  Hits: {stats['hits']}")
    print(f"  Win rate: {stats['win_rate']:.1%}")
    print(f"  Avg RR: {stats['avg_rr']:.2f}")
    print(f"  Total R: {stats['total_r']:.2f}")
    print(f"  Max win: {stats['max_win']:.2f}R")
    print(f"  Max loss: {stats['max_loss']:.2f}R")
    
    print(f"\n" + "=" * 80)
    print(f"✅ Dataset saved to: {paths['output']}")
    print("=" * 80)


if __name__ == '__main__':
    main()
