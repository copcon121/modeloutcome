#!/usr/bin/env python3
"""
Build ASM Dataset for GC M1 NEW DATA (OOS Phase 3)

Builds ASM dataset using frozen STATE-ENC v1.2 - NO retraining.

Usage:
    python asm_v2/scripts/build_asm_dataset_gc_m1_newdata.py
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

import torch
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return json.load(f)


def load_encoder_dataset(path: str) -> list:
    """Load encoder dataset."""
    samples = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                samples.append(json.loads(line))
            except:
                continue
    return samples


def load_state_enc(model_path: str, config_path: str, device: str):
    """Load STATE-ENC model."""
    from state_enc_v1.src.model.state_enc_model import StateEncModel
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    model = StateEncModel(
        input_dim=config['input_dim'],
        d_model=config['d_model'],
        num_heads=config.get('num_heads', 4),
        num_layers=config.get('num_layers', 4),
        dim_feedforward=config.get('dim_feedforward', 256),
        dropout=config.get('dropout', 0.1),
        sequence_length=config.get('sequence_length', 64),
        pooling=config.get('pooling', 'last'),
        heads_config=config.get('heads', {}),
    )
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    return model, config


def compute_z_t(model, sample: dict, feature_config: dict, device: str) -> np.ndarray:
    """Compute z_t embedding for a sample."""
    seq = sample.get('seq', [])
    if not seq:
        return np.zeros(64)
    
    # Use feature_names from feature_config
    feature_names = feature_config.get('feature_names', feature_config.get('features', []))
    
    if not feature_names:
        print("WARNING: No feature names found in feature_config!")
        return np.zeros(64)
    
    seq_data = []
    for bar in seq:
        bar_features = []
        for fname in feature_names:
            val = bar.get(fname, 0.0)
            if val is None:
                val = 0.0
            try:
                bar_features.append(float(val))
            except:
                bar_features.append(0.0)
        seq_data.append(bar_features)
    
    # Pad/truncate
    seq_len = 64
    n_features = len(feature_names)
    while len(seq_data) < seq_len:
        seq_data.insert(0, [0.0] * n_features)
    seq_data = seq_data[-seq_len:]
    
    x = torch.tensor([seq_data], dtype=torch.float32, device=device)
    
    with torch.no_grad():
        outputs = model(x)
        z_t = outputs['z_t']
    
    return z_t.cpu().numpy()[0]


def main():
    print("=" * 80)
    print("Build ASM Dataset for GC M1 NEW DATA (OOS Phase 3)")
    print("=" * 80)
    
    # Load config
    config_path = "asm_v2/configs/asm_dataset_gc_m1_newdata_v1.json"
    config = load_config(config_path)
    paths = config['paths']
    device = config.get('device', 'cpu')
    
    # Load encoder dataset
    print(f"\nLoading encoder dataset: {paths['encoder_dataset']}")
    samples = load_encoder_dataset(paths['encoder_dataset'])
    print(f"Loaded {len(samples)} samples")
    
    if not samples:
        print("ERROR: No samples loaded!")
        sys.exit(1)
    
    # Load STATE-ENC
    print(f"\nLoading STATE-ENC v1.2...")
    state_enc, state_enc_config = load_state_enc(
        paths['state_enc_model'],
        paths['state_enc_config'],
        device,
    )
    print(f"  z_dim: {state_enc_config['d_model']}")
    
    # Load feature config
    with open(paths['feature_config'], 'r') as f:
        feature_config = json.load(f)
    
    # Build ASM dataset
    print(f"\nComputing z_t embeddings...")
    asm_samples = []
    
    for sample in tqdm(samples, desc="Processing"):
        meta = sample.get('meta', {})
        
        # Compute z_t
        z_t = compute_z_t(state_enc, sample, feature_config, device)
        
        # Get regime hint (if available)
        seq = sample.get('seq', [])
        regime_hint = 0
        if seq:
            regime_hint = seq[-1].get('asm_regime_hint', 0)
        
        asm_sample = {
            'symbol': meta.get('symbol', 'GC'),
            'tf': meta.get('tf', 'M1'),
            'date': meta.get('date', ''),
            'session': meta.get('session', ''),
            'end_time': meta.get('end_time', ''),
            'z_t': z_t.tolist(),
            'regime_hint': regime_hint,
            'meta': {
                'session_id': {'ASIA': 0, 'LDN': 1, 'NY': 2}.get(meta.get('session', 'LDN'), 1),
                'pos_in_session_range': 0.5,
                'inside_value': 0,
                'above_value': 0,
                'below_value': 0,
                'minute_of_day_norm': 0.5,
            }
        }
        asm_samples.append(asm_sample)
    
    # Save ASM dataset
    output_path = paths['asm_dataset']
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for s in asm_samples:
            f.write(json.dumps(s) + '\n')
    
    print(f"\nSaved ASM dataset to: {output_path}")
    
    # Stats
    dates = sorted(set(s['date'] for s in asm_samples))
    sessions = defaultdict(int)
    for s in asm_samples:
        sessions[s['session']] += 1
    
    print(f"\n" + "=" * 80)
    print("ASM Dataset Statistics:")
    print("=" * 80)
    print(f"  Samples: {len(asm_samples)}")
    print(f"  Date range: {dates[0]} to {dates[-1]}")
    print(f"  Sessions: {dict(sessions)}")
    
    print(f"\n" + "=" * 80)
    print("✅ ASM dataset for NEW DATA built successfully!")
    print("=" * 80)


if __name__ == '__main__':
    main()
