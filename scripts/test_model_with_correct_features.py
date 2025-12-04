#!/usr/bin/env python3
"""Test ASM model with correct 100 features from training data"""

import torch
import numpy as np
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from asm_inference_v1 import ASMModelV1Loader

# Load model
print("Loading ASM model...")
loader = ASMModelV1Loader()

# Load training data and truncate to 100 features
print("\nLoading training data...")
train_data = torch.load(ROOT / "output/asm_dataset_v1/asm_dataset_v1_train.pt", weights_only=False)
X_train = train_data["X"].numpy()
y_train = train_data["y"].numpy()

print(f"Original X_train shape: {X_train.shape}")

# Truncate to 100 features
X_train_100 = X_train[:, :, :100]
print(f"Truncated X_train shape: {X_train_100.shape}")

# Test with samples from each class
print("\n=== Test with actual training samples (truncated to 100 features) ===")
for label, name in [(0, "UP"), (1, "DOWN"), (2, "NEUTRAL")]:
    indices = np.where(y_train == label)[0]
    if len(indices) > 0:
        sample = X_train_100[indices[0]]
        probs = loader.predict_proba(sample)
        print(f"\n  Sample with label {name}:")
        print(f"    p_up: {probs['p_up']:.4f}")
        print(f"    p_down: {probs['p_down']:.4f}")
        print(f"    p_neutral: {probs['p_neutral']:.4f}")
        print(f"    p_shift: {probs['p_shift']:.4f}")

# Test distribution of p_shift
print("\n=== p_shift distribution on training data (first 200 samples) ===")
p_shifts = []
for i in range(min(200, len(X_train_100))):
    probs = loader.predict_proba(X_train_100[i])
    p_shifts.append(probs['p_shift'])

p_shifts = np.array(p_shifts)
print(f"  Min: {p_shifts.min():.4f}")
print(f"  Max: {p_shifts.max():.4f}")
print(f"  Mean: {p_shifts.mean():.4f}")
print(f"  Median: {np.median(p_shifts):.4f}")
print(f"  Samples with p_shift <= 0.2: {(p_shifts <= 0.2).sum()} ({100*(p_shifts <= 0.2).sum()/len(p_shifts):.1f}%)")
print(f"  Samples with p_shift <= 0.3: {(p_shifts <= 0.3).sum()} ({100*(p_shifts <= 0.3).sum()/len(p_shifts):.1f}%)")
print(f"  Samples with p_shift <= 0.5: {(p_shifts <= 0.5).sum()} ({100*(p_shifts <= 0.5).sum()/len(p_shifts):.1f}%)")

# Check label distribution in test samples
print("\n=== Label distribution in test samples ===")
y_test = y_train[:200]
for label, name in [(0, "UP"), (1, "DOWN"), (2, "NEUTRAL")]:
    count = (y_test == label).sum()
    print(f"  {name}: {count} ({100*count/len(y_test):.1f}%)")
