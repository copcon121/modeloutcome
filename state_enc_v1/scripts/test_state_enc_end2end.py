#!/usr/bin/env python
"""
End-to-End Test Script for STATE-ENC v1

Chạy toàn bộ pipeline:
1. Build dataset (debug config)
2. Train model (debug config)
3. Export artifacts
4. Load model và test inference

Usage:
    python state_enc_v1/scripts/test_state_enc_end2end.py
    
    # Với synthetic data (không cần file thật):
    python state_enc_v1/scripts/test_state_enc_end2end.py --synthetic
    
    # Chỉ định input file:
    python state_enc_v1/scripts/test_state_enc_end2end.py --input data/bars_enhanced.jsonl
"""

import argparse
import json
import sys
import os
import shutil
import time
from pathlib import Path
from datetime import datetime, timedelta
import random

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np

print("=" * 70)
print("STATE-ENC v1 — END-TO-END TEST")
print("=" * 70)


def generate_synthetic_bars(output_path: str, num_bars: int = 1000) -> str:
    """Generate synthetic bar data for testing"""
    print(f"\n[SYNTHETIC] Generating {num_bars} synthetic bars...")
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    base_price = 17250.0
    base_time = datetime(2024, 1, 15, 2, 0, 0)  # LDN session start
    
    bars = []
    price = base_price
    cum_delta = 0
    
    for i in range(num_bars):
        # Random walk price
        change = random.gauss(0, 2)
        price += change
        
        o = price
        h = price + abs(random.gauss(0, 1.5))
        l = price - abs(random.gauss(0, 1.5))
        c = price + random.gauss(0, 1)
        
        # Ensure OHLC consistency
        h = max(h, o, c)
        l = min(l, o, c)
        
        volume = int(abs(random.gauss(1000, 300)))
        buy_vol = int(volume * random.uniform(0.3, 0.7))
        sell_vol = volume - buy_vol
        delta = buy_vol - sell_vol
        cum_delta += delta
        
        bar_time = base_time + timedelta(minutes=i)
        
        # Determine session
        hour = bar_time.hour
        if 2 <= hour < 8:
            session = "LDN"
        elif 8 <= hour < 17:
            session = "NY"
        else:
            session = "ASIA"
        
        bar = {
            "time": bar_time.isoformat(),
            "o": round(o, 2),
            "h": round(h, 2),
            "l": round(l, 2),
            "c": round(c, 2),
            "volume": volume,
            "delta": delta,
            "buy_volume": buy_vol,
            "sell_volume": sell_vol,
            "tick_count": int(volume * 0.8),
            "session": session,
            "symbol": "NQ",
            # SMC features (simplified)
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
    
    # Write to file
    with open(output_path, "w") as f:
        for bar in bars:
            f.write(json.dumps(bar) + "\n")
    
    print(f"[SYNTHETIC] Saved to: {output_path}")
    return output_path


def print_checklist(title: str, items: dict):
    """Print formatted checklist"""
    print(f"\n{'─' * 50}")
    print(f"📋 {title}")
    print(f"{'─' * 50}")
    for key, value in items.items():
        print(f"  • {key}: {value}")


def run_end2end_test(input_path: str = None, use_synthetic: bool = False):
    """Run complete end-to-end test"""
    
    start_time = time.time()
    
    # =========================================================================
    # STEP 0: Setup paths
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 0: Setup")
    print("=" * 70)
    
    debug_dir = Path("state_enc_v1/artifacts/debug")
    debug_dir.mkdir(parents=True, exist_ok=True)
    
    # Clean previous debug artifacts
    if debug_dir.exists():
        for item in debug_dir.iterdir():
            if item.is_file():
                item.unlink()
            elif item.is_dir() and item.name == "runs":
                shutil.rmtree(item)
    
    # Determine input path
    if use_synthetic:
        synthetic_path = str(debug_dir / "synthetic_bars.jsonl")
        input_path = generate_synthetic_bars(synthetic_path, num_bars=800)
    elif input_path is None:
        input_path = "data/bars_enhanced.jsonl"
    
    # Check input exists
    if not Path(input_path).exists():
        print(f"\n❌ ERROR: Input file not found: {input_path}")
        print("   Use --synthetic flag to generate test data")
        return False
    
    print_checklist("Configuration", {
        "Input file": input_path,
        "Debug directory": str(debug_dir),
        "Synthetic data": use_synthetic,
    })
    
    # =========================================================================
    # STEP 1: Build Dataset
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 1: Build Dataset")
    print("=" * 70)
    
    from state_enc_v1.src.config import StateEncDatasetConfig
    from state_enc_v1.src.dataset_builder import build_state_enc_dataset
    
    # Load and modify config
    dataset_config = StateEncDatasetConfig.from_json(
        "state_enc_v1/configs/state_enc_dataset_debug.json"
    )
    dataset_config.raw_bars_path = input_path
    
    try:
        summary = build_state_enc_dataset(
            raw_bars_path=dataset_config.raw_bars_path,
            output_path=dataset_config.output_path,
            config=dataset_config
        )
        
        print_checklist("Dataset Build Results", {
            "Total samples": summary["total_samples"],
            "Total bars": summary["total_bars"],
            "Sequence length": summary["sequence_length"],
            "Stride": summary["stride"],
            "Feature dim": summary["feature_dim"],
        })
        
        # Future direction distribution
        dir_dist = summary.get("future_dir_distribution", {})
        print("\n  📊 Future Direction Distribution:")
        total = sum(dir_dist.values())
        for k, v in sorted(dir_dist.items()):
            pct = v / total * 100 if total > 0 else 0
            label = {-1: "DOWN", 0: "NEUTRAL", 1: "UP"}.get(k, str(k))
            print(f"     {label}: {v} ({pct:.1f}%)")
        
        if summary["total_samples"] < 10:
            print("\n⚠️  WARNING: Very few samples generated. Training may not be meaningful.")
        
    except Exception as e:
        print(f"\n❌ ERROR in dataset build: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # =========================================================================
    # STEP 2: Train Model
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 2: Train Model")
    print("=" * 70)
    
    # Need to import torch here
    import torch
    
    from state_enc_v1.src.config import StateEncTrainConfig, StateEncModelConfig
    from state_enc_v1.src.dataset_encoder import create_dataloaders
    from state_enc_v1.src.model.state_enc_model import StateEncModel
    from state_enc_v1.training.losses import MultiHeadLoss
    from state_enc_v1.training.eval_metrics import MetricsAccumulator
    
    # Load configs
    train_config = StateEncTrainConfig.from_json(
        "state_enc_v1/configs/state_enc_train_debug.json"
    )
    model_config = StateEncModelConfig.from_json(
        train_config.model_config_path
    )
    
    # Get actual input_dim from feature config
    with open(train_config.feature_config_path, "r") as f:
        feature_cfg = json.load(f)
    actual_input_dim = feature_cfg.get("feature_dim", model_config.input_dim)
    model_config.input_dim = actual_input_dim
    model_config.sequence_length = dataset_config.sequence_length
    
    print_checklist("Training Config", {
        "Batch size": train_config.batch_size,
        "Max epochs": train_config.max_epochs,
        "Learning rate": train_config.learning_rate,
        "Device": train_config.device,
        "d_model": model_config.d_model,
        "num_layers": model_config.num_layers,
    })
    
    try:
        # Create dataloaders
        train_loader, val_loader, test_loader = create_dataloaders(
            dataset_path=train_config.dataset_path,
            feature_config_path=train_config.feature_config_path,
            batch_size=train_config.batch_size,
            val_split=train_config.val_split,
            test_split=train_config.test_split,
            num_workers=0,
            seed=train_config.seed
        )
        
        print(f"\n  Train samples: {len(train_loader.dataset)}")
        print(f"  Val samples: {len(val_loader.dataset)}")
        print(f"  Test samples: {len(test_loader.dataset)}")
        
        # Create model
        model = StateEncModel.from_config(model_config.__dict__)
        device = torch.device(train_config.device)
        model.to(device)
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  Model parameters: {total_params:,}")
        
        # Loss and optimizer
        loss_fn = MultiHeadLoss(loss_weights=train_config.loss_weights)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=train_config.learning_rate,
            weight_decay=train_config.weight_decay
        )
        
        # Simple training loop
        print("\n  Training...")
        best_val_acc = 0.0
        best_state = None
        
        for epoch in range(1, train_config.max_epochs + 1):
            # Train
            model.train()
            train_loss = 0.0
            for batch in train_loader:
                X = batch["X"].to(device)
                targets = {k: v.to(device) for k, v in batch.items() if k != "X"}
                
                optimizer.zero_grad()
                outputs = model(X)
                losses = loss_fn(outputs, targets)
                losses["loss_total"].backward()
                optimizer.step()
                
                train_loss += losses["loss_total"].item()
            
            train_loss /= len(train_loader)
            
            # Validate
            model.eval()
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for batch in val_loader:
                    X = batch["X"].to(device)
                    targets = batch["future_dir_5"].to(device)
                    
                    outputs = model(X)
                    preds = outputs["dir_logits"].argmax(dim=-1)
                    val_correct += (preds == targets).sum().item()
                    val_total += len(targets)
            
            val_acc = val_correct / val_total if val_total > 0 else 0
            
            print(f"    Epoch {epoch}: train_loss={train_loss:.4f}, val_acc={val_acc:.4f}")
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = model.state_dict().copy()
        
        # Save best model
        run_dir = Path(train_config.output_dir) / "debug_run"
        run_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            "model_state_dict": best_state or model.state_dict(),
            "config": model_config.__dict__,
            "metrics": {"best_val_acc": best_val_acc}
        }
        checkpoint_path = run_dir / "best_model.pt"
        torch.save(checkpoint, checkpoint_path)
        
        print(f"\n  ✅ Best model saved: {checkpoint_path}")
        print(f"  ✅ Best val accuracy: {best_val_acc:.4f}")
        
    except Exception as e:
        print(f"\n❌ ERROR in training: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # =========================================================================
    # STEP 3: Export Artifacts
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 3: Export Artifacts")
    print("=" * 70)
    
    try:
        final_dir = debug_dir / "final"
        final_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model weights
        model_path = final_dir / "state_enc_v1.pt"
        torch.save(best_state or model.state_dict(), model_path)
        
        # Save model config
        model_config_path = final_dir / "model_config.json"
        with open(model_config_path, "w") as f:
            json.dump(model_config.__dict__, f, indent=2)
        
        # Copy feature config
        feature_config_src = Path(train_config.feature_config_path)
        feature_config_dst = final_dir / "feature_config.json"
        if feature_config_src.exists():
            shutil.copy(feature_config_src, feature_config_dst)
        
        print_checklist("Exported Artifacts", {
            "Model weights": str(model_path),
            "Model config": str(model_config_path),
            "Feature config": str(feature_config_dst),
        })
        
    except Exception as e:
        print(f"\n❌ ERROR in export: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # =========================================================================
    # STEP 4: Test Inference
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 4: Test Inference")
    print("=" * 70)
    
    try:
        # Load model
        model_loaded = StateEncModel.from_config(model_config.__dict__)
        model_loaded.load_state_dict(torch.load(model_path, map_location="cpu"))
        model_loaded.eval()
        
        print("  ✅ Model loaded successfully")
        
        # Load one sample
        dataset_path = train_config.dataset_path
        with open(dataset_path, "r") as f:
            sample_line = f.readline()
            sample = json.loads(sample_line)
        
        print(f"\n  Sample info:")
        print(f"    Symbol: {sample.get('symbol', 'N/A')}")
        print(f"    Date: {sample.get('date', 'N/A')}")
        print(f"    Session: {sample.get('session', 'N/A')}")
        print(f"    Sequence length: {len(sample.get('seq', []))}")
        
        # Load normalizer and transform
        from state_enc_v1.src.normalization import FeatureNormalizer
        
        normalizer = FeatureNormalizer.from_file(str(feature_config_dst))
        
        # Transform sequence
        seq = sample["seq"]
        X = normalizer.transform_sequence(seq)
        X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(0)  # [1, N, D]
        
        print(f"\n  Input tensor shape: {X_tensor.shape}")
        
        # Inference
        with torch.no_grad():
            outputs = model_loaded(X_tensor)
        
        z_t = outputs["z_t"]
        z_seq = outputs["z_seq"]
        dir_logits = outputs["dir_logits"]
        
        print(f"\n  📊 Output shapes:")
        print(f"    z_t (state embedding): {z_t.shape}")
        print(f"    z_seq (sequence): {z_seq.shape}")
        print(f"    dir_logits: {dir_logits.shape}")
        
        print(f"\n  📊 z_t sample values (first 10):")
        print(f"    {z_t[0, :10].numpy()}")
        
        print(f"\n  📊 Direction prediction:")
        probs = torch.softmax(dir_logits, dim=-1)[0]
        pred_class = dir_logits.argmax(dim=-1).item()
        pred_label = {0: "DOWN", 1: "NEUTRAL", 2: "UP"}[pred_class]
        print(f"    Probabilities: DOWN={probs[0]:.3f}, NEUTRAL={probs[1]:.3f}, UP={probs[2]:.3f}")
        print(f"    Prediction: {pred_label}")
        
        # Compare with actual
        actual_dir = sample["aux"]["future_dir_5"]
        actual_label = {-1: "DOWN", 0: "NEUTRAL", 1: "UP"}[actual_dir]
        print(f"    Actual: {actual_label}")
        
    except Exception as e:
        print(f"\n❌ ERROR in inference: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    elapsed = time.time() - start_time
    
    print("\n" + "=" * 70)
    print("✅ END-TO-END TEST COMPLETE")
    print("=" * 70)
    
    print_checklist("Final Summary", {
        "Total time": f"{elapsed:.1f} seconds",
        "Dataset samples": summary["total_samples"],
        "Best val accuracy": f"{best_val_acc:.4f}",
        "Model parameters": f"{total_params:,}",
        "z_t dimension": z_t.shape[-1],
    })
    
    print("\n  📁 Generated files:")
    for f in sorted(debug_dir.rglob("*")):
        if f.is_file() and f.name != ".gitkeep":
            size = f.stat().st_size
            print(f"    {f.relative_to(debug_dir)}: {size:,} bytes")
    
    print("\n" + "=" * 70)
    print("🎉 All tests passed! STATE-ENC v1 pipeline is working correctly.")
    print("=" * 70)
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="End-to-end test for STATE-ENC v1"
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to input bars file (default: data/bars_enhanced.jsonl)"
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Generate synthetic data for testing"
    )
    
    args = parser.parse_args()
    
    success = run_end2end_test(
        input_path=args.input,
        use_synthetic=args.synthetic
    )
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
