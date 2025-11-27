"""
Dataset Builder - Build training datasets from raw JSONL
Converts raw bars → feature contexts → training samples
"""

from typing import List, Tuple, Optional
from pathlib import Path
import numpy as np
import logging

from .schema import RawBar, FeatureBar, Record
from .data_loader import load_raw_bars
from .context_manager import ContextManager
from .normalizer import Normalizer

logger = logging.getLogger(__name__)


class FeatureContext:
    """
    A context window of features ready for ML
    Contains: context_features + metadata
    """
    def __init__(
        self,
        context_features: np.ndarray,  # [context_len, feature_dim]
        feature_names: List[str],
        timestamp: str,
        symbol: str,
        timeframe: str,
        entry_price: float,
        atr: float
    ):
        self.context_features = context_features
        self.feature_names = feature_names
        self.timestamp = timestamp
        self.symbol = symbol
        self.timeframe = timeframe
        self.entry_price = entry_price
        self.atr = atr

    @property
    def context_len(self) -> int:
        return self.context_features.shape[0]

    @property
    def feature_dim(self) -> int:
        return self.context_features.shape[1]

    def __repr__(self) -> str:
        return (f"FeatureContext(shape={self.context_features.shape}, "
                f"timestamp={self.timestamp}, price={self.entry_price:.2f})")


def build_context_dataset(
    raw_jsonl_path: str,
    ctx_len: int = 60,
    enable_normalization: bool = True,
    normalizer: Optional[Normalizer] = None,
    stride: int = 1
) -> Tuple[List[FeatureContext], Normalizer]:
    """
    Build a dataset of feature contexts from raw JSONL

    Args:
        raw_jsonl_path: Path to raw .jsonl file
        ctx_len: Context window length (number of bars)
        enable_normalization: Whether to apply z-score normalization
        normalizer: Pre-fitted normalizer (if None, will fit on this data)
        stride: Step size for sliding window (1 = every bar, 10 = every 10th bar)

    Returns:
        Tuple of (list of FeatureContext objects, fitted Normalizer)

    Example:
        >>> contexts, normalizer = build_context_dataset(
        ...     "data/raw/smc_export_gc_m1_v3.jsonl",
        ...     ctx_len=60
        ... )
        >>> print(f"Built {len(contexts)} contexts")
        >>> print(f"Shape: [{contexts[0].context_len}, {contexts[0].feature_dim}]")
    """
    logger.info(f"Building context dataset from: {raw_jsonl_path}")
    logger.info(f"  Context length: {ctx_len}")
    logger.info(f"  Normalization:  {enable_normalization}")
    logger.info(f"  Stride:         {stride}")

    # 1. Load raw bars
    bars = load_raw_bars(raw_jsonl_path)
    if len(bars) < ctx_len:
        raise ValueError(f"Not enough bars: {len(bars)} < {ctx_len}")

    logger.info(f"Loaded {len(bars)} bars")

    # 2. Initialize Context Manager (without normalizer for now)
    context_mgr = ContextManager(
        context_len=ctx_len,
        max_history=len(bars),  # Keep all bars
        normalizer=None  # We'll normalize later
    )

    # 3. Add all bars
    context_mgr.add_bars_batch(bars)
    logger.info(f"Added {len(context_mgr.bars)} bars to context")

    # 4. Build features for all bars
    feature_bars = context_mgr.build_features()
    logger.info(f"Built features for {len(feature_bars)} bars")

    if not feature_bars:
        raise RuntimeError("No feature bars generated!")

    # 5. Get feature names (consistent ordering)
    feature_names = sorted(feature_bars[0].features.keys())
    feature_dim = len(feature_names)
    logger.info(f"Feature dimension: {feature_dim}")

    # 6. Fit normalizer if needed
    if enable_normalization:
        if normalizer is None:
            logger.info("Fitting normalizer on all features...")
            normalizer = Normalizer(method='zscore')
            all_features = [fb.features for fb in feature_bars]
            normalizer.fit(all_features)
            logger.info(f"✓ Normalizer fitted on {len(all_features)} samples")
        else:
            logger.info("Using pre-fitted normalizer")

        # Normalize all feature bars
        logger.info("Normalizing features...")
        for fb in feature_bars:
            fb.features = normalizer.transform(fb.features)
        logger.info("✓ Features normalized")
    else:
        normalizer = None

    # 7. Build sliding windows
    logger.info(f"Building sliding windows (stride={stride})...")

    contexts = []
    num_possible_windows = len(feature_bars) - ctx_len + 1

    for i in range(0, num_possible_windows, stride):
        # Get context window
        window = feature_bars[i:i+ctx_len]

        if len(window) < ctx_len:
            continue  # Skip incomplete windows

        # Convert to numpy array [ctx_len, feature_dim]
        context_matrix = np.array([
            [fb.features[name] for name in feature_names]
            for fb in window
        ], dtype=np.float32)

        # Get metadata from last bar in context
        last_bar = window[-1]

        # Create FeatureContext
        fc = FeatureContext(
            context_features=context_matrix,
            feature_names=feature_names,
            timestamp=last_bar.ts.isoformat(),
            symbol="GC 02-26",  # TODO: Extract from JSONL metadata
            timeframe="M1",
            entry_price=bars[i+ctx_len-1].close,
            atr=context_mgr._compute_atr(list(context_mgr.bars)[i:i+ctx_len])
        )

        contexts.append(fc)

    logger.info(f"✓ Built {len(contexts)} feature contexts")
    logger.info(f"  Context shape: [{ctx_len}, {feature_dim}]")

    return contexts, normalizer


def save_dataset(
    contexts: List[FeatureContext],
    output_path: str,
    format: str = 'numpy'
) -> None:
    """
    Save dataset to disk

    Args:
        contexts: List of FeatureContext objects
        output_path: Path to save dataset
        format: 'numpy' (npz) or 'torch' (pt)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if format == 'numpy':
        # Stack all contexts into single array
        X = np.stack([ctx.context_features for ctx in contexts])

        # Save metadata
        metadata = {
            'feature_names': contexts[0].feature_names,
            'timestamps': [ctx.timestamp for ctx in contexts],
            'entry_prices': [ctx.entry_price for ctx in contexts],
            'atrs': [ctx.atr for ctx in contexts],
        }

        np.savez_compressed(
            output_path,
            X=X,
            feature_names=metadata['feature_names'],
            timestamps=metadata['timestamps'],
            entry_prices=metadata['entry_prices'],
            atrs=metadata['atrs']
        )

        logger.info(f"✓ Saved dataset to {output_path}")
        logger.info(f"  Shape: {X.shape}")
        logger.info(f"  Size:  {output_path.stat().st_size / 1024 / 1024:.2f} MB")

    elif format == 'torch':
        try:
            import torch

            X = torch.tensor(np.stack([ctx.context_features for ctx in contexts]))

            torch.save({
                'X': X,
                'feature_names': contexts[0].feature_names,
                'timestamps': [ctx.timestamp for ctx in contexts],
                'entry_prices': [ctx.entry_price for ctx in contexts],
                'atrs': [ctx.atr for ctx in contexts],
            }, output_path)

            logger.info(f"✓ Saved dataset to {output_path}")
            logger.info(f"  Shape: {X.shape}")

        except ImportError:
            logger.error("PyTorch not installed. Use format='numpy' instead.")
            raise

    else:
        raise ValueError(f"Unknown format: {format}")


def load_dataset(dataset_path: str) -> Tuple[np.ndarray, dict]:
    """
    Load saved dataset

    Args:
        dataset_path: Path to .npz or .pt file

    Returns:
        Tuple of (X array, metadata dict)
    """
    dataset_path = Path(dataset_path)

    if dataset_path.suffix == '.npz':
        data = np.load(dataset_path, allow_pickle=True)
        X = data['X']
        metadata = {
            'feature_names': data['feature_names'].tolist(),
            'timestamps': data['timestamps'].tolist(),
            'entry_prices': data['entry_prices'].tolist(),
            'atrs': data['atrs'].tolist(),
        }
        return X, metadata

    elif dataset_path.suffix == '.pt':
        import torch
        data = torch.load(dataset_path)
        X = data['X'].numpy()
        metadata = {
            'feature_names': data['feature_names'],
            'timestamps': data['timestamps'],
            'entry_prices': data['entry_prices'],
            'atrs': data['atrs'],
        }
        return X, metadata

    else:
        raise ValueError(f"Unknown file format: {dataset_path.suffix}")


if __name__ == "__main__":
    # Test dataset builder
    import sys

    print("\n" + "="*80)
    print("Testing Dataset Builder")
    print("="*80 + "\n")

    # Build dataset
    jsonl_path = "/home/user/modeloutcome/data/raw/smc_export_gc_m1_v3.jsonl"

    contexts, normalizer = build_context_dataset(
        raw_jsonl_path=jsonl_path,
        ctx_len=60,
        enable_normalization=True,
        stride=10  # Use stride=10 for faster testing
    )

    print(f"\n✓ Built {len(contexts)} contexts")
    print(f"  Context shape: [{contexts[0].context_len}, {contexts[0].feature_dim}]")
    print(f"  Feature names: {len(contexts[0].feature_names)}")

    # Show first context
    print(f"\nFirst context:")
    print(f"  {contexts[0]}")

    # Save normalizer
    normalizer_path = Path("data/processed/normalizer_stats.json")
    normalizer_path.parent.mkdir(parents=True, exist_ok=True)
    normalizer.save(normalizer_path)
    print(f"\n✓ Saved normalizer to {normalizer_path}")

    # Save dataset
    dataset_path = Path("data/processed/feature_contexts.npz")
    save_dataset(contexts, dataset_path, format='numpy')
    print(f"\n✓ Saved dataset to {dataset_path}")

    # Load back
    X, metadata = load_dataset(dataset_path)
    print(f"\n✓ Loaded dataset:")
    print(f"  Shape: {X.shape}")
    print(f"  Feature names: {len(metadata['feature_names'])}")

    print("\n" + "="*80)
    print("✓ Dataset Builder Test PASSED!")
    print("="*80 + "\n")
