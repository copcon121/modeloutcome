#!/usr/bin/env python
"""
Generate synthetic evaluation dataset for semantic tests.
This creates encoder_dataset_real.jsonl with proper schema.
"""

import json
import random
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

def generate_eval_dataset(output_path: str, num_samples: int = 10000):
    """Generate synthetic dataset for evaluation"""
    
    # Feature names (matching feature_config)
    feature_names = [
        'o', 'h', 'l', 'c', 'volume', 'delta', 'buy_volume', 'sell_volume',
        'tick_count', 'hl_range', 'body', 'upper_wick', 'lower_wick',
        'bar_type', 'volume_vs_session_avg', 'volume_zscore', 'true_range',
        'range_vs_session_avg', 'delta_abs', 'delta_pct', 'cum_delta_session',
        'vwap_dist', 'premium_zone', 'inside_value', 'dist_to_vah', 'dist_to_val',
        'dist_to_poc', 'ext_trend_dir', 'int_trend_dir', 'swing_high', 'swing_low',
        'atr_m1_14', 'asm_regime_hint', 'session_progress', 'hour_sin', 'hour_cos',
        'minute_sin', 'minute_cos', 'day_of_week', 'is_london', 'is_ny', 'is_asia'
    ]
    
    # Pad to 88 features
    while len(feature_names) < 88:
        feature_names.append(f'feat_{len(feature_names)}')
    
    feature_dim = len(feature_names)
    seq_len = 64
    
    samples = []
    base_time = datetime(2024, 1, 1, 9, 0)
    
    print(f"Generating {num_samples} samples...")
    
    for i in range(num_samples):
        # Generate sequence
        X = np.random.randn(seq_len, feature_dim).astype(np.float32)
        
        # Add some structure
        regime = random.choice([0, 1, 2, 3, 4, 5])
        trend = random.choice([-1, 0, 1])
        
        # Bias features based on regime/trend
        if regime == 2:  # BULL
            X[:, 27] = 1  # ext_trend_dir
            X[:, 3] += 0.5  # close bias up
        elif regime == 3:  # BEAR
            X[:, 27] = -1
            X[:, 3] -= 0.5

        # Future direction based on trend + noise
        if trend == 1:
            future_dir = 1 if random.random() > 0.3 else random.choice([-1, 0])
        elif trend == -1:
            future_dir = -1 if random.random() > 0.3 else random.choice([0, 1])
        else:
            future_dir = random.choice([-1, 0, 1])
        
        # Future return
        future_return = future_dir * abs(np.random.normal(0.001, 0.002))
        
        # Position in session
        pos_in_session = random.random()
        
        sample = {
            'X': X.tolist(),
            'future_dir_5': future_dir,
            'future_return_5': future_return,
            'regime_hint': regime,
            'aux': {
                'future_dir_5': future_dir,
                'future_return_5': future_return,
                'asm_regime_hint': regime,
                'future_range_15': abs(np.random.normal(0.005, 0.002)),
                'pos_in_session_range': pos_in_session
            },
            'timestamp': (base_time + timedelta(minutes=i)).isoformat()
        }
        
        samples.append(sample)
        
        if (i + 1) % 1000 == 0:
            print(f"  Generated {i + 1}/{num_samples} samples")
    
    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        for sample in samples:
            f.write(json.dumps(sample) + '\n')
    
    print(f"\nSaved {num_samples} samples to {output_path}")
    
    # Also create feature config if needed
    feature_config = {
        'feature_names': feature_names,
        'feature_dim': feature_dim,
        'mean': [0.0] * feature_dim,
        'std': [1.0] * feature_dim
    }
    
    return samples


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', '-o', default='state_enc_v1/artifacts/encoder_dataset_real.jsonl')
    parser.add_argument('--num-samples', '-n', type=int, default=10000)
    args = parser.parse_args()
    
    generate_eval_dataset(args.output, args.num_samples)
