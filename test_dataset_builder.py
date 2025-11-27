"""
Test Dataset Builder - Full Pipeline
"""

from pathlib import Path
import sys
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Add src to path
src_dir = Path(__file__).parent / 'src'
sys.path.insert(0, str(src_dir))

from layer2_feature_engine_v2.config import GC_M1_SMC_CONFIG
from layer2_feature_engine_v2.dataset_builder import build_context_dataset

# Configuration
JSONL_FILE = r"data\raw\deepseek_enhanced_GC 12-25_M1_20251111.jsonl"
OUTPUT_DIR = r"output\test_dataset"
WINDOW_SIZE = 60  # 60 bars = 1 hour on M1
STRIDE = 1        # Every bar
MAX_BARS = 500    # Test with 500 bars

print("\n" + "="*80)
print("TESTING DATASET BUILDER - FULL PIPELINE")
print("="*80)
print(f"Input: {JSONL_FILE}")
print(f"Output: {OUTPUT_DIR}")
print(f"Window: {WINDOW_SIZE} bars")
print(f"Max bars: {MAX_BARS}")
print("="*80 + "\n")

# Build dataset
sequences, indices = build_context_dataset(
    jsonl_path=JSONL_FILE,
    output_dir=OUTPUT_DIR,
    config=GC_M1_SMC_CONFIG,
    tick_size=0.1,
    window_size=WINDOW_SIZE,
    stride=STRIDE,
    max_bars=MAX_BARS
)

print("\n" + "="*80)
print("RESULTS")
print("="*80)
print(f"Sequences shape: {sequences.shape}")
print(f"  - Number of sequences: {sequences.shape[0]}")
print(f"  - Window size: {sequences.shape[1]}")
print(f"  - Number of features: {sequences.shape[2]}")
print(f"\nEnding indices: {len(indices)}")
print(f"  First 5: {indices[:5]}")
print(f"  Last 5: {indices[-5:]}")

# Check output files
output_path = Path(OUTPUT_DIR)
files = list(output_path.glob("*"))
print(f"\nOutput files created:")
for f in files:
    size_kb = f.stat().st_size / 1024
    print(f"  - {f.name} ({size_kb:.1f} KB)")

# Sample data inspection
print(f"\nFirst sequence (first 5 features):")
print(sequences[0, :5, :5])  # First 5 bars, first 5 features

print("\n" + "="*80)
print("✅ DATASET BUILDER TEST COMPLETE!")
print("="*80)
