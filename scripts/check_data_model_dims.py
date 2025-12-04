#!/usr/bin/env python3
"""Check training data and model dimensions"""

import torch
import json

# Check training data shape
train_data = torch.load('output/asm_dataset_v1/asm_dataset_v1_train.pt', weights_only=False)
X_train = train_data['X']
print(f'Training data X shape: {X_train.shape}')

# Check stats
with open('output/asm_dataset_v1/asm_dataset_v1_stats.json') as f:
    stats = json.load(f)
print(f'Stats n_features: {stats["features"]["n_features"]}')
print(f'Stats feature_names count: {len(stats["features"]["feature_names"])}')

# Check model input dim
checkpoint = torch.load('output/asm_models_v1/ASM-GRU64-v1.0-C3.pt', map_location='cpu', weights_only=False)
for key in checkpoint['model_state_dict'].keys():
    if 'gru.weight_ih_l0' in key:
        print(f'Model input dim: {checkpoint["model_state_dict"][key].shape[1]}')
        break

# Check config in checkpoint
config = checkpoint.get('config', {})
print(f'Checkpoint config: {config}')
