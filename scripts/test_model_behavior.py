#!/usr/bin/env python3
"""Test ASM model behavior with different inputs"""

import numpy as np
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.asm_feature_spec import ASM_FEATURE_COLS
from asm_inference_v1 import ASMModelV1Loader

# Load model
print("Loading ASM model...")
loader = ASMModelV1Loader()

# Test 1: zeros
print("\n=== Test 1: All zeros ===")
zeros = np.zeros((60, 100), dtype=np.float32)
probs = loader.predict_proba(zeros)
print(f"  p_up: {probs['p_up']:.4f}")
print(f"  p_down: {probs['p_down']:.4f}")
print(f"  p_neutral: {probs['p_neutral']:.4f}")
print(f"  p_shift: {probs['p_shift']:.4f}")

# Test 2: random
print("\n=== Test 2: Random noise ===")
random_input = np.random.randn(60, 100).astype(np.float32)
probs = loader.predict_proba(random_input)
print(f"  p_up: {probs['p_up']:.4f}")
print(f"  p_down: {probs['p_down']:.4f}")
print(f"  p_neutral: {probs['p_neutral']:.4f}")
print(f"  p_shift: {probs['p_shift']:.4f}")

# Test 3: realistic values
print("\n=== Test 3: Realistic values (close~3300, vol~100) ===")
realistic = np.zeros((60, 100), dtype=np.float32)
for i in range(60):
    realistic[i, 0] = 3300 + np.random.randn() * 5  # close
    realistic[i, 1] = 2 + np.random.rand()  # high_low_range
    realistic[i, 6] = 100 + np.random.randn() * 20  # volume
    realistic[i, 8] = np.random.randn() * 10  # delta

probs = loader.predict_proba(realistic)
print(f"  p_up: {probs['p_up']:.4f}")
print(f"  p_down: {probs['p_down']:.4f}")
print(f"  p_neutral: {probs['p_neutral']:.4f}")
print(f"  p_shift: {probs['p_shift']:.4f}")

# Test 4: Load actual training data and test
print("\n=== Test 4: Actual training data sample ===")
import torch
train_data = torch.load(ROOT / "output/asm_dataset_v1/asm_dataset_v1_train.pt", weights_only=False)
X_train = train_data["X"].numpy()
y_train = train_data["y"].numpy()

# Get samples from each class
for label, name in [(0, "UP"), (1, "DOWN"), (2, "NEUTRAL")]:
    indices = np.where(y_train == label)[0]
    if len(indices) > 0:
        sample = X_train[indices[0]]
        probs = loader.predict_proba(sample)
        print(f"\n  Sample with label {name}:")
        print(f"    p_up: {probs['p_up']:.4f}")
        print(f"    p_down: {probs['p_down']:.4f}")
        print(f"    p_neutral: {probs['p_neutral']:.4f}")
        print(f"    p_shift: {probs['p_shift']:.4f}")

# Test 5: Distribution of p_shift on training data
print("\n=== Test 5: p_shift distribution on training data (first 1000 samples) ===")
p_shifts = []
for i in range(min(1000, len(X_train))):
    probs = loader.predict_proba(X_train[i])
    p_shifts.append(probs['p_shift'])

p_shifts = np.array(p_shifts)
print(f"  Min: {p_shifts.min():.4f}")
print(f"  Max: {p_shifts.max():.4f}")
print(f"  Mean: {p_shifts.mean():.4f}")
print(f"  Median: {np.median(p_shifts):.4f}")
print(f"  Samples with p_shift <= 0.2: {(p_shifts <= 0.2).sum()}")
print(f"  Samples with p_shift <= 0.3: {(p_shifts <= 0.3).sum()}")
print(f"  Samples with p_shift <= 0.5: {(p_shifts <= 0.5).sum()}")
