"""
Outcome Labeler
Generates R-based outcome labels for training data
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import torch
import argparse
import sys
import os

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from layer2_feature_engine.core.schema import RawBar, Record
from layer2_feature_engine.core.context_manager import ContextManager
from layer2_feature_engine.utils.config_loader import load_config
from layer2_feature_engine.utils.logging_utils import setup_logger

logger = setup_logger('labeler', log_file='logs/labeler.log')


def load_historical_ohlc(filepath: str) -> pd.DataFrame:
    """
    Load historical OHLC data from CSV

    Expected columns: timestamp, open, high, low, close, volume

    Args:
        filepath: Path to CSV file

    Returns:
        DataFrame with OHLC data
    """
    logger.info(f"Loading historical data from {filepath}")

    df = pd.read_csv(filepath)

    # Ensure required columns exist
    required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Sort by timestamp
    df = df.sort_values('timestamp').reset_index(drop=True)

    logger.info(f"Loaded {len(df)} bars from {df['timestamp'].min()} to {df['timestamp'].max()}")

    return df


def compute_atr(df: pd.DataFrame, period: int = 14) -> np.ndarray:
    """
    Compute Average True Range

    Args:
        df: DataFrame with OHLC data
        period: ATR period

    Returns:
        Array of ATR values
    """
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift(1))
    low_close = np.abs(df['low'] - df['close'].shift(1))

    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(window=period).mean()

    return atr.fillna(method='bfill').values


def compute_outcome_for_bar(
    index: int,
    df: pd.DataFrame,
    atr_values: np.ndarray,
    config: Dict
) -> Tuple[str, float, float]:
    """
    Compute R-based outcome for a specific bar

    Args:
        index: Bar index in DataFrame
        df: DataFrame with OHLC data
        atr_values: Array of ATR values
        config: Labeling configuration

    Returns:
        Tuple of (label, max_up_R, max_down_R)
        label: 'long', 'short', or 'skip'
    """
    future_window = config['future_window']
    stop_R = config['stop_R']
    target_R = config['target_R']

    # Get entry price and ATR
    entry_price = df.iloc[index]['close']
    atr = atr_values[index]

    if atr < 1e-8:
        return 'skip', 0.0, 0.0

    # Get future prices
    end_idx = min(index + future_window, len(df))
    future_prices = df.iloc[index+1:end_idx]

    if len(future_prices) < 10:
        return 'skip', 0.0, 0.0

    # Compute max excursions in R terms
    max_up = future_prices['high'].max() - entry_price
    max_down = entry_price - future_prices['low'].min()

    max_up_R = max_up / atr
    max_down_R = max_down / atr

    # Determine label based on which target is hit first
    # Check bar by bar
    for i, row in future_prices.iterrows():
        high_R = (row['high'] - entry_price) / atr
        low_R = (entry_price - row['low']) / atr

        # Check if hit target or stop first
        if high_R >= target_R:
            return 'long', max_up_R, -max_down_R

        if low_R >= stop_R:
            # Stop hit first, check if going other direction
            if low_R >= target_R:
                return 'short', max_up_R, -max_down_R
            else:
                return 'skip', max_up_R, -max_down_R

    # Neither target hit
    return 'skip', max_up_R, -max_down_R


def build_dataset(
    df: pd.DataFrame,
    config: Dict,
    context_manager: ContextManager
) -> List[Record]:
    """
    Build labeled dataset from historical data

    Args:
        df: DataFrame with OHLC data
        config: Configuration dict
        context_manager: ContextManager for feature extraction

    Returns:
        List of Record objects
    """
    context_len = config['context_len']
    min_bars_after = config['min_bars_after']
    atr_values = compute_atr(df, period=config.get('atr_period', 14))

    records = []
    label_counts = {'long': 0, 'short': 0, 'skip': 0}

    # Convert DataFrame to RawBar objects
    raw_bars = []
    for idx, row in df.iterrows():
        bar = RawBar(
            ts=row['timestamp'],
            open=row['open'],
            high=row['high'],
            low=row['low'],
            close=row['close'],
            volume=row['volume']
        )
        raw_bars.append(bar)

    logger.info(f"Building dataset from {len(raw_bars)} bars")

    # Add bars to context manager in batches
    for i in range(0, len(raw_bars), 100):
        batch = raw_bars[i:i+100]
        context_manager.add_bars_batch(batch)

    # Generate labels for each candidate bar
    start_idx = context_len
    end_idx = len(df) - min_bars_after

    for idx in range(start_idx, end_idx):
        if idx % 1000 == 0:
            logger.info(f"Processed {idx}/{end_idx} bars")

        # Compute outcome label
        label_str, max_up_R, max_down_R = compute_outcome_for_bar(
            idx, df, atr_values, config
        )

        # Map label to index
        label_map = {'long': 0, 'short': 1, 'skip': 2}
        label_idx = label_map[label_str]
        label_counts[label_str] += 1

        # Extract features for this context
        # Get context bars (previous context_len bars up to idx)
        context_bars = raw_bars[max(0, idx-context_len):idx]

        if len(context_bars) < context_len:
            continue

        # Build features using ContextManager
        temp_cm = ContextManager(context_len=context_len)
        temp_cm.add_bars_batch(context_bars)
        feature_bars = temp_cm.get_latest_context()

        if len(feature_bars) < context_len:
            continue

        # Create Record
        record = Record(
            context=feature_bars,
            label=label_idx,
            max_up_R=max_up_R,
            max_down_R=max_down_R,
            entry_price=df.iloc[idx]['close'],
            atr=atr_values[idx]
        )

        records.append(record)

    logger.info(f"Generated {len(records)} labeled records")
    logger.info(f"Label distribution: {label_counts}")
    logger.info(f"  Long: {label_counts['long']/len(records)*100:.1f}%")
    logger.info(f"  Short: {label_counts['short']/len(records)*100:.1f}%")
    logger.info(f"  Skip: {label_counts['skip']/len(records)*100:.1f}%")

    return records


def save_dataset(records: List[Record], output_path: str, feature_names: List[str]):
    """
    Save dataset to file

    Args:
        records: List of Record objects
        output_path: Path to save dataset
        feature_names: List of feature names (for ordering)
    """
    logger.info(f"Saving dataset to {output_path}")

    # Convert records to tensors
    features_list = []
    labels_list = []
    metadata_list = []

    for record in records:
        # Convert context to feature matrix
        feature_matrix = record.to_feature_matrix(feature_names)
        features_list.append(feature_matrix)
        labels_list.append(record.label)
        metadata_list.append({
            'max_up_R': record.max_up_R,
            'max_down_R': record.max_down_R,
            'entry_price': record.entry_price,
            'atr': record.atr
        })

    # Save as PyTorch tensors
    dataset = {
        'features': torch.tensor(features_list, dtype=torch.float32),
        'labels': torch.tensor(labels_list, dtype=torch.long),
        'metadata': metadata_list,
        'feature_names': feature_names
    }

    torch.save(dataset, output_path)
    logger.info(f"Dataset saved: {dataset['features'].shape}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Generate labeled dataset for outcome model')
    parser.add_argument('--config', type=str, default='config/model_config.yaml',
                        help='Path to config file')
    parser.add_argument('--input', type=str, default='data/raw/NQ_M1_6months.csv',
                        help='Path to input CSV file')
    parser.add_argument('--output', type=str, default='data/datasets/outcome_train.pt',
                        help='Path to output dataset file')

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    labeling_config = config['labeling']

    # Load historical data
    df = load_historical_ohlc(args.input)

    # Create context manager
    context_manager = ContextManager(
        context_len=labeling_config['context_len'],
        max_history=200
    )

    # Build dataset
    records = build_dataset(df, labeling_config, context_manager)

    if not records:
        logger.error("No records generated!")
        return

    # Get feature names from first record
    feature_names = list(records[0].context[0].features.keys())

    # Save dataset
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    save_dataset(records, args.output, feature_names)

    logger.info("Labeling complete!")


if __name__ == '__main__':
    main()
