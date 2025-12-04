#!/usr/bin/env python
"""
STATE-ENC v1.1 Full Pipeline Test

1. Build dataset (synthetic)
2. Train model
3. Export artifacts
4. Run validation tests (Stability, Shock, Missing, Regime Bias)
5. Generate validation report
"""

import argparse
import json
import sys
import shutil
import time
import random
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torch.nn.functional as F

print("=" * 70)
print("STATE-ENC v1.1 — FULL PIPELINE TEST")
print("=" * 70)


def generate_synthetic_bars(output_path: str, num_bars: int = 1000) -> str:
    """Generate synthetic bar data"""
    print(f"\n[SYNTHETIC] Generating {num_bars} bars...")
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    base_price = 17250.0
    base_time = datetime(2024, 1, 15, 2, 0, 0)
    
    bars = []
    price = base_price
    cum_delta = 0
    
    for i in range(num_bars):
        change = random.gauss(0, 2)
        price += change
        
        o = price
        h = price + abs(random.gauss(0, 1.5))
        l = price - abs(random.gauss(0, 1.5))
        c = price + random.gauss(0, 1)
        h = max(h, o, c)
        l = min(l, o, c)
        
        volume = int(abs(random.gauss(1000, 300)))
        buy_vol = int(volume * random.uniform(0.3, 0.7))
        sell_vol = volume - buy_vol
        delta = buy_vol - sell_vol
        cum_delta += delta
        
        bar_time = base_time + timedelta(minutes=i)
        hour = bar_time.hour
        session = "LDN" if 2 <= hour < 8 else ("NY" if 8 <= hour < 17 else "ASIA")
        
        bar = {
            "time": bar_time.isoformat(),
            "o": round(o, 2), "h": round(h, 2), "l": round(l, 2), "c": round(c, 2),
            "volume": volume, "delta": delta,
            "buy_volume": buy_vol, "sell_volume": sell_vol,
            "tick_count": int(volume * 0.8),
            "session": session, "symbol": "NQ",
            "ext_trend_dir": random.choice([-1, 0, 1]),
            "int_trend_dir": random.choice([-1, 0, 1]),
            "ext_bos_up": random.choice([0, 0, 0, 1]),
            "ext_bos_down": random.choice([0, 0, 0, 1]),
            "ext_choch_up": random.choice([0, 0, 0, 0, 1]),
            "ext_choch_down": random.choice([0, 0, 0, 0, 1]),
            "int_bos_up": random.choice([0, 0, 1]),
            "int_bos_down": random.choice([0, 0, 1]),
            "int_choch_up": random.choice([0, 0, 0, 1]),
            "int_choch_down": random.choice([0, 0, 0, 1]),
            "swing_high": round(price + 30, 2),
            "swing_low": round(price - 30, 2),
            "sweep_prev_high": random.choice([0, 0, 0, 0, 1]),
            "sweep_prev_low": random.choice([0, 0, 0, 0, 1]),
            "near_ob_m1_bull": random.choice([0, 0, 1]),
            "near_ob_m1_bear": random.choice([0, 0, 1]),
            "near_ob_m5_bull": random.choice([0, 0, 1]),
            "near_ob_m5_bear": random.choice([0, 0, 1]),
            "near_fvg_m1_bull": random.choice([0, 0, 1]),
            "near_fvg_m1_bear": random.choice([0, 0, 1]),
            "near_fvg_m5_bull": random.choice([0, 0, 1]),
            "near_fvg_m5_bear": random.choice([0, 0, 1]),
            "vah": round(price + 15, 2),
            "val": round(price - 15, 2),
            "poc": round(price, 2),
            "cum_delta_session": cum_delta,
            "atr_m1_14": round(abs(random.gauss(4, 1)), 2),
            "asm_regime_hint": random.choice([0, 1, 1, 2, 2, 3, 3]),
        }
        bars.append(bar)
    
    with open(output_path, "w") as f:
        for bar in bars:
            f.write(json.dumps(bar) + "\n")
    
    print(f"[SYNTHETIC] Saved: {output_path}")
    return output_path


def cosine_distance(z1: torch.Tensor, z2: torch.Tensor) -> float:
    """Compute cosine distance between two embeddings"""
    sim = F.cosine_similarity(z1, z2, dim=-1)
    return (1 - sim).item()


def run_validation_tests(model, normalizer, sample_seq, device) -> dict:
    """Run 4 validation tests"""
    results = {}
    model.eval()
    
    # Prepare base sequence
    X_base = normalizer.transform_sequence(sample_seq)
    X_base_tensor = torch.tensor(X_base, dtype=torch.float32).unsqueeze(0).to(device)
    
    with torch.no_grad():
        z_base = model.encode(X_base_tensor)
    
    # TEST 1: Stability Test
    print("\n  TEST 1: Stability Test...")
    distances = []
    for _ in range(10):
        # Create slightly perturbed sequence
        perturbed_seq = []
        for bar in sample_seq:
            new_bar = bar.copy()
            for key in ["o", "h", "l", "c"]:
                if key in new_bar:
                    new_bar[key] = new_bar[key] * (1 + random.gauss(0, 0.001))
            perturbed_seq.append(new_bar)
        
        X_pert = normalizer.transform_sequence(perturbed_seq)
        X_pert_tensor = torch.tensor(X_pert, dtype=torch.float32).unsqueeze(0).to(device)
        
        with torch.no_grad():
            z_pert = model.encode(X_pert_tensor)
        
        dist = cosine_distance(z_base, z_pert)
        distances.append(dist)
    
    stable_pairs = sum(1 for d in distances if d < 0.25)
    test1_pass = stable_pairs >= 8  # 80%
    results["test1_stability"] = {
        "pass": test1_pass,
        "stable_pairs": stable_pairs,
        "total_pairs": 10,
        "avg_distance": float(np.mean(distances))
    }
    print(f"    Stable pairs: {stable_pairs}/10, Pass: {test1_pass}")
    
    # TEST 2: Shock Test (volume * 3)
    print("\n  TEST 2: Shock Test (volume x3)...")
    shocked_seq = []
    for bar in sample_seq:
        new_bar = bar.copy()
        if "volume" in new_bar:
            new_bar["volume"] = new_bar["volume"] * 3
        shocked_seq.append(new_bar)
    
    X_shock = normalizer.transform_sequence(shocked_seq)
    X_shock_tensor = torch.tensor(X_shock, dtype=torch.float32).unsqueeze(0).to(device)
    
    with torch.no_grad():
        z_shock = model.encode(X_shock_tensor)
    
    shock_dist = cosine_distance(z_base, z_shock)
    test2_pass = shock_dist < 0.3
    results["test2_shock"] = {
        "pass": test2_pass,
        "distance": float(shock_dist)
    }
    print(f"    Distance: {shock_dist:.4f}, Pass: {test2_pass}")
    
    # TEST 3: Missing Data Test (mask 3 bars)
    print("\n  TEST 3: Missing Data Test (mask 3 bars)...")
    missing_seq = sample_seq.copy()
    mask_indices = random.sample(range(len(missing_seq)), min(3, len(missing_seq)))
    for idx in mask_indices:
        missing_seq[idx] = {k: 0.0 for k in missing_seq[idx].keys()}
    
    X_missing = normalizer.transform_sequence(missing_seq)
    X_missing_tensor = torch.tensor(X_missing, dtype=torch.float32).unsqueeze(0).to(device)
    
    with torch.no_grad():
        z_missing = model.encode(X_missing_tensor)
    
    missing_dist = cosine_distance(z_base, z_missing)
    test3_pass = missing_dist < 0.35
    results["test3_missing"] = {
        "pass": test3_pass,
        "distance": float(missing_dist),
        "masked_bars": len(mask_indices)
    }
    print(f"    Distance: {missing_dist:.4f}, Pass: {test3_pass}")
    
    # TEST 4: Regime Bias Test
    print("\n  TEST 4: Regime Bias Test (100 inferences)...")
    regime_counts = defaultdict(int)
    
    for _ in range(100):
        # Random perturbation
        rand_seq = []
        for bar in sample_seq:
            new_bar = bar.copy()
            for key in ["o", "h", "l", "c", "volume"]:
                if key in new_bar:
                    new_bar[key] = new_bar[key] * (1 + random.gauss(0, 0.05))
            new_bar["asm_regime_hint"] = random.randint(0, 5)
            rand_seq.append(new_bar)
        
        X_rand = normalizer.transform_sequence(rand_seq)
        X_rand_tensor = torch.tensor(X_rand, dtype=torch.float32).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(X_rand_tensor)
            if "regime_logits" in outputs:
                pred = outputs["regime_logits"].argmax(dim=-1).item()
                regime_counts[pred] += 1
    
    max_pct = max(regime_counts.values()) / 100 * 100 if regime_counts else 0
    test4_pass = max_pct <= 60
    results["test4_regime_bias"] = {
        "pass": test4_pass,
        "distribution": dict(regime_counts),
        "max_class_pct": float(max_pct)
    }
    print(f"    Max class: {max_pct:.1f}%, Pass: {test4_pass}")
    
    return results


def run_full_pipeline():
    """Run complete v1.1 pipeline"""
    start_time = time.time()
    
    # Setup
    debug_dir = Path("state_enc_v1/artifacts/v1_1")
    debug_dir.mkdir(parents=True, exist_ok=True)
    
    # Clean previous
    for item in debug_dir.iterdir():
        if item.is_file():
            item.unlink()
        elif item.is_dir():
            shutil.rmtree(item)
    
    # Step 1: Generate synthetic data
    print("\n" + "=" * 70)
    print("STEP 1: Generate Synthetic Data")
    print("=" * 70)
    
    synthetic_path = str(debug_dir / "synthetic_bars.jsonl")
    generate_synthetic_bars(synthetic_path, num_bars=800)
    
    # Step 2: Build dataset
    print("\n" + "=" * 70)
    print("STEP 2: Build Dataset")
    print("=" * 70)
    
    from state_enc_v1.src.config import StateEncDatasetConfig
    from state_enc_v1.src.dataset_builder import build_state_enc_dataset
    
    dataset_config = StateEncDatasetConfig(
        raw_bars_path=synthetic_path,
        output_path=str(debug_dir / "encoder_dataset.jsonl"),
        feature_config_path=str(debug_dir / "feature_config_v1.1.json"),
        sequence_length=64,
        stride=16,
        future_bars=5
    )
    
    summary = build_state_enc_dataset(
        raw_bars_path=dataset_config.raw_bars_path,
        output_path=dataset_config.output_path,
        config=dataset_config
    )
    
    print(f"  Samples: {summary['total_samples']}")
    print(f"  Dir distribution: {summary['future_dir_distribution']}")
    
    # Step 3: Train model
    print("\n" + "=" * 70)
    print("STEP 3: Train Model (v1.1)")
    print("=" * 70)
    
    from state_enc_v1.src.config import StateEncTrainConfig, StateEncModelConfig
    from state_enc_v1.src.dataset_encoder import create_dataloaders
    from state_enc_v1.src.model.state_enc_model import StateEncModel
    from state_enc_v1.training.losses import MultiHeadLossV11
    from state_enc_v1.training.eval_metrics import MetricsAccumulator
    
    # Get actual input dim from feature config
    with open(dataset_config.feature_config_path, "r") as f:
        feature_cfg = json.load(f)
    input_dim = feature_cfg.get("feature_dim", 88)
    
    # Model config
    model_config = StateEncModelConfig(
        input_dim=input_dim,
        d_model=64,
        num_heads=4,
        num_layers=4,
        dim_feedforward=256,
        dropout=0.05,
        sequence_length=64
    )
    
    # Create dataloaders
    train_loader, val_loader, _ = create_dataloaders(
        dataset_path=dataset_config.output_path,
        feature_config_path=dataset_config.feature_config_path,
        batch_size=16,
        val_split=0.15,
        test_split=0.1,
        num_workers=0,
        seed=42
    )
    
    print(f"  Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}")
    
    # Create model
    model = StateEncModel.from_config(model_config.__dict__)
    device = torch.device("cpu")
    model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,}")
    print(f"  z_t dim: {model.get_embedding_dim()}")
    print(f"  Heads: {model.get_head_names()}")
    
    # Training
    loss_fn = MultiHeadLossV11(loss_weights={
        "future_dir": 1.0, "future_return": 0.2, "regime": 0.3, "mask_ssl": 0.5
    })
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    
    best_val_loss = float('inf')
    best_state = None
    epochs = 3
    
    print("\n  Training...")
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            X = batch["X"].to(device)
            targets = {k: v.to(device) for k, v in batch.items() if k != "X"}
            
            optimizer.zero_grad()
            outputs = model(X, augment=True)
            losses = loss_fn(outputs, targets)
            losses["loss_total"].backward()
            optimizer.step()
            train_loss += losses["loss_total"].item()
        
        train_loss /= len(train_loader)
        
        # Validate
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch in val_loader:
                X = batch["X"].to(device)
                targets = {k: v.to(device) for k, v in batch.items() if k != "X"}
                outputs = model(X, augment=False)
                losses = loss_fn(outputs, targets)
                val_loss += losses["loss_total"].item()
                
                preds = outputs["dir_logits"].argmax(dim=-1)
                val_correct += (preds == targets["future_dir_5"]).sum().item()
                val_total += len(targets["future_dir_5"])
        
        val_loss /= len(val_loader)
        val_acc = val_correct / val_total if val_total > 0 else 0
        
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            best_state = model.state_dict().copy()
        
        print(f"    Epoch {epoch}: train={train_loss:.4f}, val={val_loss:.4f}, acc={val_acc:.4f} {'*BEST*' if is_best else ''}")
    
    # Step 4: Export artifacts
    print("\n" + "=" * 70)
    print("STEP 4: Export Artifacts")
    print("=" * 70)
    
    final_dir = debug_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model
    model_path = final_dir / "state_enc_v1.1.pt"
    torch.save(best_state, model_path)
    print(f"  Model: {model_path}")
    
    # Save configs
    model_config_path = final_dir / "model_config_v1.1.json"
    with open(model_config_path, "w") as f:
        json.dump(model_config.__dict__, f, indent=2)
    print(f"  Model config: {model_config_path}")
    
    feature_config_dst = final_dir / "feature_config_v1.1.json"
    shutil.copy(dataset_config.feature_config_path, feature_config_dst)
    print(f"  Feature config: {feature_config_dst}")
    
    # Step 5: Validation Tests
    print("\n" + "=" * 70)
    print("STEP 5: Validation Tests")
    print("=" * 70)
    
    # Load model for testing
    model.load_state_dict(best_state)
    model.eval()
    
    # Load normalizer
    from state_enc_v1.src.normalization import FeatureNormalizer
    normalizer = FeatureNormalizer.from_file(str(feature_config_dst))
    
    # Load sample sequence
    with open(dataset_config.output_path, "r") as f:
        sample = json.loads(f.readline())
    sample_seq = sample["seq"]
    
    # Run tests
    test_results = run_validation_tests(model, normalizer, sample_seq, device)
    
    # Save validation report
    validation_report = {
        "version": "1.1",
        "timestamp": datetime.now().isoformat(),
        "training": {
            "total_samples": summary["total_samples"],
            "epochs": epochs,
            "best_val_loss": best_val_loss,
            "parameters": total_params,
            "z_t_dim": model.get_embedding_dim()
        },
        "tests": test_results,
        "all_tests_passed": all(t.get("pass", False) for t in test_results.values())
    }
    
    report_path = final_dir / "validation_report_v1.1.json"
    with open(report_path, "w") as f:
        json.dump(validation_report, f, indent=2)
    
    # Final summary
    total_time = time.time() - start_time
    
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"  Total time: {total_time:.1f}s")
    print(f"  Samples: {summary['total_samples']}")
    print(f"  Best val loss: {best_val_loss:.4f}")
    print(f"  Parameters: {total_params:,}")
    print(f"  z_t dimension: {model.get_embedding_dim()}")
    print(f"\n  Test Results:")
    for name, result in test_results.items():
        status = "✅ PASS" if result.get("pass") else "❌ FAIL"
        print(f"    {name}: {status}")
    
    all_passed = validation_report["all_tests_passed"]
    print(f"\n  Overall: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")
    
    print(f"\n  Artifacts:")
    for f in final_dir.iterdir():
        print(f"    {f.name}: {f.stat().st_size:,} bytes")
    
    return validation_report


if __name__ == "__main__":
    report = run_full_pipeline()
    sys.exit(0 if report["all_tests_passed"] else 1)
