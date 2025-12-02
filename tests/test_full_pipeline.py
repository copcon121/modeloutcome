"""
Test Full Feature Engineering Pipeline - End-to-End
Phase 2: Feature Engineering Layer Complete Test
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from pathlib import Path
import numpy as np

from layer2_feature_engine.core.data_loader import load_raw_bars
from layer2_feature_engine.core.dataset_builder import build_context_dataset, save_dataset
from layer2_feature_engine.core.normalizer import Normalizer


def test_full_pipeline():
    """
    Full end-to-end test of Feature Engineering Layer
    Phase 1 → Phase 2 → Ready for Phase 3 (Labeling)
    """
    print("\n" + "="*80)
    print("PHASE 2 - FEATURE ENGINEERING LAYER")
    print("Full Pipeline Test - End-to-End")
    print("="*80 + "\n")

    # ========== Configuration ==========
    raw_jsonl_path = "data/raw/smc_export_gc_m1_v3_20250901.jsonl"
    context_len = 60
    stride = 1  # Every bar (production mode)

    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = output_dir / "feature_dataset_gc_m1.npz"
    normalizer_path = output_dir / "normalizer_zscore.json"

    print(f"Configuration:")
    print(f"  Raw data:     {raw_jsonl_path}")
    print(f"  Context len:  {context_len}")
    print(f"  Stride:       {stride} (every bar)")
    print(f"  Output dir:   {output_dir}")
    print()

    # ========== Step 1: Load Raw Data ==========
    print("Step 1: Loading raw bars from JSONL...")
    print("-" * 80)

    bars = load_raw_bars(raw_jsonl_path)

    print(f"✓ Loaded {len(bars)} bars")
    print(f"  Symbol:    GC 02-26")
    print(f"  Timeframe: M1")
    print(f"  Start:     {bars[0].ts}")
    print(f"  End:       {bars[-1].ts}")
    print()

    # ========== Step 2: Build Feature Dataset ==========
    print("Step 2: Building feature contexts...")
    print("-" * 80)

    contexts, normalizer = build_context_dataset(
        raw_jsonl_path=raw_jsonl_path,
        ctx_len=context_len,
        enable_normalization=True,
        normalizer=None,  # Fit new normalizer
        stride=stride
    )

    print(f"✓ Built {len(contexts)} feature contexts")
    print(f"  Context shape: [{contexts[0].context_len}, {contexts[0].feature_dim}]")
    print(f"  Total features: {contexts[0].feature_dim}")
    print()

    # Feature breakdown
    feature_names = contexts[0].feature_names

    ohlcv_features = [f for f in feature_names if any(x in f for x in ['open', 'high', 'low', 'close', 'range', 'body', 'wick', 'volume', 'bullish'])]
    tick_features = [f for f in feature_names if any(x in f for x in ['tick', 'aggr', 'buy_sell', 'delta', 'price_speed', 'buying_pressure', 'activity'])]
    smc_features = [f for f in feature_names if any(x in f for x in ['swing', 'bos', 'choch', 'ob', 'fvg'])]
    vp_features = [f for f in feature_names if any(x in f for x in ['vah', 'val', 'poc', 'value_area', 'hvn', 'lvn'])]
    l2_features = [f for f in feature_names if 'l2' in f]
    time_features = [f for f in feature_names if any(x in f for x in ['time', 'session', 'day', 'weekend', 'morning', 'afternoon', 'evening', 'night'])]

    print("Feature Categories:")
    print(f"  OHLCV:       {len(ohlcv_features):2d} features")
    print(f"  Tick/Flow:   {len(tick_features):2d} features")
    print(f"  SMC:         {len(smc_features):2d} features")
    print(f"  Vol Profile: {len(vp_features):2d} features")
    print(f"  Level 2:     {len(l2_features):2d} features")
    print(f"  Time:        {len(time_features):2d} features")
    print(f"  {'─'*40}")
    print(f"  TOTAL:       {len(feature_names):2d} features")
    print()

    # ========== Step 3: Data Quality Check ==========
    print("Step 3: Data quality check...")
    print("-" * 80)

    # Check for NaN/Inf
    X = np.stack([ctx.context_features for ctx in contexts])

    nan_count = np.isnan(X).sum()
    inf_count = np.isinf(X).sum()

    if nan_count > 0:
        print(f"✗ Found {nan_count} NaN values!")
        return False
    else:
        print(f"✓ No NaN values")

    if inf_count > 0:
        print(f"✗ Found {inf_count} Inf values!")
        return False
    else:
        print(f"✓ No Inf values")

    # Check value ranges (after z-score normalization, most should be in [-3, 3])
    mean_val = np.mean(X)
    std_val = np.std(X)
    min_val = np.min(X)
    max_val = np.max(X)

    print(f"\nValue statistics:")
    print(f"  Mean:  {mean_val:.4f}")
    print(f"  Std:   {std_val:.4f}")
    print(f"  Min:   {min_val:.4f}")
    print(f"  Max:   {max_val:.4f}")

    # Check for extreme outliers (beyond ±10 sigma)
    outliers = np.abs(X) > 10.0
    outlier_pct = 100.0 * outliers.sum() / X.size

    if outlier_pct > 1.0:
        print(f"⚠ Warning: {outlier_pct:.2f}% of values are extreme outliers (>10 sigma)")
    else:
        print(f"✓ Outlier percentage: {outlier_pct:.4f}% (acceptable)")

    print()

    # ========== Step 4: Save Dataset ==========
    print("Step 4: Saving dataset and normalizer...")
    print("-" * 80)

    # Save normalizer
    normalizer.save(normalizer_path)
    print(f"✓ Saved normalizer to {normalizer_path}")
    print(f"  Method: {normalizer.method}")
    print(f"  Features: {len(normalizer.stats)}")

    # Save dataset
    save_dataset(contexts, dataset_path, format='numpy')
    print(f"✓ Saved dataset to {dataset_path}")
    print(f"  Shape: {X.shape}")
    print(f"  Size:  {dataset_path.stat().st_size / 1024 / 1024:.2f} MB")
    print()

    # ========== Step 5: Verification ==========
    print("Step 5: Verifying saved files...")
    print("-" * 80)

    # Reload normalizer
    normalizer_reloaded = Normalizer()
    normalizer_reloaded.load(normalizer_path)
    print(f"✓ Reloaded normalizer: {normalizer_reloaded}")

    # Reload dataset
    from layer2_feature_engine.core.dataset_builder import load_dataset
    X_reloaded, metadata = load_dataset(dataset_path)
    print(f"✓ Reloaded dataset: shape={X_reloaded.shape}")
    print(f"  Feature names: {len(metadata['feature_names'])}")
    print(f"  Timestamps:    {len(metadata['timestamps'])}")
    print()

    # ========== Summary ==========
    print("="*80)
    print("PHASE 2 - FEATURE ENGINEERING LAYER: ✅ COMPLETE")
    print("="*80)
    print()
    print("📊 Dataset Summary:")
    print(f"   • Total contexts:  {len(contexts)}")
    print(f"   • Context length:  {context_len} bars")
    print(f"   • Feature dim:     {contexts[0].feature_dim}")
    print(f"   • Array shape:     {X.shape}")
    print(f"   • Data quality:    ✓ No NaN/Inf")
    print()
    print("📁 Output Files:")
    print(f"   • Dataset:    {dataset_path}")
    print(f"   • Normalizer: {normalizer_path}")
    print()
    print("🎯 Next Steps:")
    print("   Phase 3: Data Labeling (compute outcomes)")
    print("   Phase 4: Model Training (Transformer/MLP)")
    print()
    print("="*80)
    print()

    return True


if __name__ == "__main__":
    success = test_full_pipeline()
    sys.exit(0 if success else 1)
