#!/usr/bin/env python3
"""
Run ASM v2 Regime Inference on GC M1 NEW DATA (OOS Phase 3)

Uses frozen ASM v2 model - NO retraining.

Usage:
    python asm_v2/scripts/run_asm_infer_gc_m1_newdata.py
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

import torch
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


REGIME_NAMES = {0: 'trend_up', 1: 'trend_down', 2: 'balance'}


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return json.load(f)


def load_asm_dataset(path: str) -> list:
    """Load ASM dataset."""
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


def load_asm_model(model_path: str, config_path: str, device: str):
    """Load ASM v2 model."""
    from asm_v2.src.model.asm_model import AsmModel
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    model = AsmModel(
        z_dim=config['z_dim'],
        meta_dim=config['meta_dim'],
        hidden_dim=config['hidden_dim'],
        num_classes=config['num_classes'],
        dropout=config.get('dropout', 0.1),
    )
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    return model, config


def run_inference(model, samples: list, device: str) -> list:
    """Run regime inference on all samples."""
    results = []
    
    for sample in tqdm(samples, desc="Inference"):
        z_t = np.array(sample['z_t'], dtype=np.float32)
        meta = sample.get('meta', {})
        
        # Build meta features
        meta_features = np.array([
            meta.get('session_id', 1),
            meta.get('pos_in_session_range', 0.5),
            meta.get('inside_value', 0),
            meta.get('above_value', 0),
            meta.get('below_value', 0),
            meta.get('minute_of_day_norm', 0.5),
        ], dtype=np.float32)
        
        # Concatenate
        x = np.concatenate([z_t, meta_features])
        x_tensor = torch.tensor([x], dtype=torch.float32, device=device)
        
        # Forward
        with torch.no_grad():
            outputs = model(x_tensor)
            probs = outputs['probs'][0].cpu().numpy()
            pred_class = int(np.argmax(probs))
        
        result = {
            'symbol': sample.get('symbol', 'GC'),
            'date': sample.get('date', ''),
            'time': sample.get('end_time', ''),
            'session': sample.get('session', ''),
            'regime_pred': REGIME_NAMES.get(pred_class, 'unknown'),
            'regime_id': pred_class,
            'regime_prob': probs.tolist(),
            'regime_confidence': float(probs[pred_class]),
        }
        results.append(result)
    
    return results


def main():
    print("=" * 80)
    print("ASM v2 Regime Inference on GC M1 NEW DATA (OOS Phase 3)")
    print("=" * 80)
    
    # Load config
    config_path = "asm_v2/configs/asm_dataset_gc_m1_newdata_v1.json"
    config = load_config(config_path)
    paths = config['paths']
    device = config.get('device', 'cpu')
    
    # Load ASM dataset
    print(f"\nLoading ASM dataset: {paths['asm_dataset']}")
    samples = load_asm_dataset(paths['asm_dataset'])
    print(f"Loaded {len(samples)} samples")
    
    if not samples:
        print("ERROR: No samples loaded!")
        sys.exit(1)
    
    # Load ASM model
    print(f"\nLoading ASM v2 model...")
    asm_model, asm_config = load_asm_model(
        paths['asm_model'],
        paths['asm_config'],
        device,
    )
    print(f"  Classes: {asm_config['num_classes']}")
    
    # Run inference
    print(f"\nRunning regime inference...")
    results = run_inference(asm_model, samples, device)
    
    # Save results
    output_path = paths['regime_pred']
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for r in results:
            f.write(json.dumps(r) + '\n')
    
    print(f"\nSaved regime predictions to: {output_path}")
    
    # Stats
    regime_dist = defaultdict(int)
    for r in results:
        regime_dist[r['regime_pred']] += 1
    
    print(f"\n" + "=" * 80)
    print("Regime Distribution:")
    print("=" * 80)
    for regime, count in sorted(regime_dist.items()):
        pct = count / len(results) * 100
        print(f"  {regime}: {count} ({pct:.1f}%)")
    
    print(f"\n" + "=" * 80)
    print("✅ ASM regime inference for NEW DATA completed!")
    print("=" * 80)


if __name__ == '__main__':
    main()
