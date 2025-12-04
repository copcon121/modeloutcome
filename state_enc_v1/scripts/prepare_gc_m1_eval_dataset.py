#!/usr/bin/env python
"""
Prepare GC M1 real data for STATE-ENC v1.2 semantic evaluation.

Steps:
1. Load all processed CSV files from data/processed_v2/
2. Merge into single dataset
3. Create encoder dataset with sequences
4. Save to state_enc_v1/artifacts/gc_m1/
"""

import json
import glob
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Feature mapping from CSV columns to STATE-ENC features
FEATURE_MAPPING = {
    # OHLCV
    'open': 'o',
    'high': 'h', 
    'low': 'l',
    'close': 'c',
    'volume': 'volume',
    'delta': 'delta',
    'buy_volume': 'buy_volume',
    'sell_volume': 'sell_volume',
    
    # Derived
    'high_low_range': 'hl_range',
    'body': 'body',
    'upper_wick': 'upper_wick',
    'lower_wick': 'lower_wick',
    'tick_speed': 'tick_count',
    
    # Trend
    'int_trend_dir': 'int_trend_dir',
    'ext_trend_dir': 'ext_trend_dir',
    
    # Value Area
    'vp_poc_price': 'poc',
    'vp_vah_price': 'vah',
    'vp_val_price': 'val',
    'vp_dist_to_poc': 'dist_to_poc',
    'vp_dist_to_vah': 'dist_to_vah',
    'vp_dist_to_val': 'dist_to_val',
    'vp_in_value_area': 'inside_value',
}


def load_all_csv_files(data_dir: str, pattern: str = "smc_export_gc_m1_v3_*_features.csv") -> pd.DataFrame:
    """Load and merge all CSV files"""
    csv_files = sorted(glob.glob(f"{data_dir}/{pattern}"))
    
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found matching {data_dir}/{pattern}")
    
    logger.info(f"Found {len(csv_files)} CSV files")
    
    dfs = []
    for f in csv_files:
        logger.info(f"  Loading {Path(f).name}...")
        df = pd.read_csv(f)
        dfs.append(df)
    
    merged = pd.concat(dfs, ignore_index=True)
    
    # Sort by timestamp
    if 'timestamp' in merged.columns:
        merged['timestamp'] = pd.to_datetime(merged['timestamp'])
        merged = merged.sort_values('timestamp').reset_index(drop=True)
    
    logger.info(f"Total bars: {len(merged)}")
    return merged


def compute_future_labels(df: pd.DataFrame, future_bars: int = 5) -> pd.DataFrame:
    """Compute future direction and return labels"""
    df = df.copy()
    
    # Future close
    df['future_close'] = df['close'].shift(-future_bars)
    
    # Future return
    df['future_return_5'] = (df['future_close'] - df['close']) / df['close']
    
    # Future direction: -1 (down), 0 (flat), 1 (up)
    threshold = 0.0005  # 0.05% threshold
    df['future_dir_5'] = 0
    df.loc[df['future_return_5'] > threshold, 'future_dir_5'] = 1
    df.loc[df['future_return_5'] < -threshold, 'future_dir_5'] = -1
    
    # Regime hint (simplified based on trend)
    df['regime_hint'] = 0  # UNKNOWN
    df.loc[(df['ext_trend_dir'] > 0) & (df['int_trend_dir'] > 0), 'regime_hint'] = 2  # BULL
    df.loc[(df['ext_trend_dir'] < 0) & (df['int_trend_dir'] < 0), 'regime_hint'] = 3  # BEAR
    df.loc[(df['ext_trend_dir'] == 0) | (df['int_trend_dir'] == 0), 'regime_hint'] = 1  # CHOP
    
    # Position in session (simplified)
    if 'vp_dist_to_vah' in df.columns and 'vp_dist_to_val' in df.columns:
        range_size = df['vp_dist_to_vah'].abs() + df['vp_dist_to_val'].abs()
        df['pos_in_session_range'] = df['vp_dist_to_val'].abs() / (range_size + 1e-8)
        df['pos_in_session_range'] = df['pos_in_session_range'].clip(0, 1)
    else:
        df['pos_in_session_range'] = 0.5
    
    return df


def extract_features(df: pd.DataFrame, feature_config_path: str) -> np.ndarray:
    """Extract features matching feature_config order"""
    
    # Load feature config
    with open(feature_config_path, 'r') as f:
        feat_cfg = json.load(f)
    
    feature_names = feat_cfg.get('feature_names', [])
    feature_dim = len(feature_names)
    
    logger.info(f"Extracting {feature_dim} features...")
    
    # Build feature matrix
    features = np.zeros((len(df), feature_dim), dtype=np.float32)
    
    # Map columns
    col_mapping = {}
    for csv_col, feat_name in FEATURE_MAPPING.items():
        if csv_col in df.columns and feat_name in feature_names:
            col_mapping[csv_col] = feature_names.index(feat_name)
    
    # Also try direct column names
    for col in df.columns:
        if col in feature_names:
            col_mapping[col] = feature_names.index(col)
    
    logger.info(f"  Mapped {len(col_mapping)} columns to features")
    
    for csv_col, feat_idx in col_mapping.items():
        features[:, feat_idx] = df[csv_col].fillna(0).values
    
    return features, feature_names


def create_sequences(features: np.ndarray, labels: Dict[str, np.ndarray],
                     seq_len: int = 64, stride: int = 16) -> List[Dict]:
    """Create sequences for encoder dataset"""
    n_bars = len(features)
    samples = []
    
    for i in range(0, n_bars - seq_len, stride):
        end_idx = i + seq_len
        
        # Skip if future labels not available
        if np.isnan(labels['future_return_5'][end_idx - 1]):
            continue
        
        sample = {
            'X': features[i:end_idx].tolist(),
            'future_dir_5': int(labels['future_dir_5'][end_idx - 1]),
            'future_return_5': float(labels['future_return_5'][end_idx - 1]),
            'regime_hint': int(labels['regime_hint'][end_idx - 1]),
            'aux': {
                'future_dir_5': int(labels['future_dir_5'][end_idx - 1]),
                'future_return_5': float(labels['future_return_5'][end_idx - 1]),
                'asm_regime_hint': int(labels['regime_hint'][end_idx - 1]),
                'pos_in_session_range': float(labels['pos_in_session_range'][end_idx - 1]),
                'future_range_15': 0.0  # Not available
            }
        }
        samples.append(sample)
    
    return samples


def prepare_gc_m1_dataset(
    data_dir: str = "data/processed_v2",
    feature_config_path: str = "state_enc_v1/artifacts/v1_2/final/feature_config_v1.2.json",
    output_dir: str = "state_enc_v1/artifacts/gc_m1",
    seq_len: int = 64,
    stride: int = 16
) -> str:
    """Main function to prepare GC M1 evaluation dataset"""
    
    logger.info("=" * 60)
    logger.info("PREPARING GC M1 EVALUATION DATASET")
    logger.info("=" * 60)
    
    # Create output dir
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Load all CSV files
    logger.info("\n[Step 1] Loading CSV files...")
    df = load_all_csv_files(data_dir)
    
    # Step 2: Compute future labels
    logger.info("\n[Step 2] Computing future labels...")
    df = compute_future_labels(df, future_bars=5)
    
    # Step 3: Extract features
    logger.info("\n[Step 3] Extracting features...")
    features, feature_names = extract_features(df, feature_config_path)
    
    # Normalize features
    logger.info("\n[Step 4] Normalizing features...")
    mean = np.nanmean(features, axis=0)
    std = np.nanstd(features, axis=0) + 1e-8
    features_norm = (features - mean) / std
    features_norm = np.nan_to_num(features_norm, nan=0.0)
    
    # Step 5: Create sequences
    logger.info("\n[Step 5] Creating sequences...")
    labels = {
        'future_dir_5': df['future_dir_5'].values,
        'future_return_5': df['future_return_5'].values,
        'regime_hint': df['regime_hint'].values,
        'pos_in_session_range': df['pos_in_session_range'].values
    }
    
    samples = create_sequences(features_norm, labels, seq_len, stride)
    logger.info(f"  Created {len(samples)} samples")
    
    # Step 6: Save dataset
    logger.info("\n[Step 6] Saving dataset...")
    dataset_path = output_path / "encoder_dataset_gc_m1_v1.2.jsonl"
    with open(dataset_path, 'w') as f:
        for sample in samples:
            f.write(json.dumps(sample) + '\n')
    
    # Save feature config for this dataset
    feature_config = {
        'feature_names': feature_names,
        'feature_dim': len(feature_names),
        'mean': mean.tolist(),
        'std': std.tolist(),
        'source': 'gc_m1_processed_v2',
        'seq_len': seq_len,
        'stride': stride,
        'n_samples': len(samples)
    }
    
    config_path = output_path / "feature_config_gc_m1_v1.2.json"
    with open(config_path, 'w') as f:
        json.dump(feature_config, f, indent=2)
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("DATASET PREPARATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"  Total bars: {len(df)}")
    logger.info(f"  Total samples: {len(samples)}")
    logger.info(f"  Sequence length: {seq_len}")
    logger.info(f"  Feature dim: {len(feature_names)}")
    logger.info(f"  Dataset: {dataset_path}")
    logger.info(f"  Config: {config_path}")
    
    # Label distribution
    dir_counts = pd.Series([s['future_dir_5'] for s in samples]).value_counts().sort_index()
    logger.info(f"\n  Future dir distribution:")
    for d, c in dir_counts.items():
        logger.info(f"    {d}: {c} ({c/len(samples)*100:.1f}%)")
    
    regime_counts = pd.Series([s['regime_hint'] for s in samples]).value_counts().sort_index()
    logger.info(f"\n  Regime distribution:")
    for r, c in regime_counts.items():
        logger.info(f"    {r}: {c} ({c/len(samples)*100:.1f}%)")
    
    return str(dataset_path)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='data/processed_v2')
    parser.add_argument('--feature-config', default='state_enc_v1/artifacts/v1_2/final/feature_config_v1.2.json')
    parser.add_argument('--output-dir', default='state_enc_v1/artifacts/gc_m1')
    parser.add_argument('--seq-len', type=int, default=64)
    parser.add_argument('--stride', type=int, default=16)
    args = parser.parse_args()
    
    prepare_gc_m1_dataset(
        data_dir=args.data_dir,
        feature_config_path=args.feature_config,
        output_dir=args.output_dir,
        seq_len=args.seq_len,
        stride=args.stride
    )
