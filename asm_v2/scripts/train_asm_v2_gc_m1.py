#!/usr/bin/env python
"""
Train ASM v2 model on GC M1 data

Usage:
    python asm_v2/scripts/train_asm_v2_gc_m1.py --config asm_v2/configs/asm_train_v1.json
"""

import argparse
import json
import sys
import torch
import numpy as np
import random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from asm_v2.src.config import AsmTrainConfig, AsmModelConfig
from asm_v2.src.model.asm_model import AsmModel
from asm_v2.src.dataset_asm import create_asm_dataloaders
from asm_v2.src.training.trainer import AsmTrainer


def set_seed(seed: int):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description="Train ASM v2")
    parser.add_argument("--config", type=str, required=True, help="Path to train config JSON")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    args = parser.parse_args()
    
    # Load config
    config = AsmTrainConfig.from_json(args.config)
    if args.epochs:
        config.epochs = args.epochs
    
    # Set seed
    set_seed(config.seed)
    
    # Device
    device = config.device if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    print("=" * 60)
    print("Training ASM v2")
    print("=" * 60)
    print(f"Dataset: {config.dataset_path}")
    print(f"Epochs: {config.epochs}")
    print(f"Batch size: {config.batch_size}")
    print(f"Learning rate: {config.learning_rate}")
    print()
    
    # Load feature config
    with open(config.feature_config_path, "r") as f:
        feature_config = json.load(f)
    
    z_dim = feature_config.get("z_dim", 64)
    meta_dim = feature_config.get("meta_dim", 6)
    num_classes = feature_config.get("num_classes", 5)
    
    print(f"z_dim: {z_dim}, meta_dim: {meta_dim}, num_classes: {num_classes}")
    
    # Load model config
    model_config = AsmModelConfig.from_json(config.model_config_path)
    model_config.z_dim = z_dim
    model_config.meta_dim = meta_dim
    model_config.num_classes = num_classes
    
    # Create model
    model = AsmModel(
        z_dim=model_config.z_dim,
        meta_dim=model_config.meta_dim,
        hidden_dim=model_config.hidden_dim,
        num_layers=model_config.num_layers,
        dropout=model_config.dropout,
        num_classes=model_config.num_classes,
        use_grn=model_config.use_grn
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create dataloaders
    train_loader, val_loader = create_asm_dataloaders(
        dataset_path=config.dataset_path,
        splits_path=config.splits_path,
        batch_size=config.batch_size,
        z_dim=z_dim
    )
    
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Create trainer
    trainer = AsmTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_classes=num_classes,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        label_smoothing=config.label_smoothing,
        device=device,
        output_dir=config.output_dir
    )
    
    # Train
    print()
    print("Starting training...")
    history = trainer.train(epochs=config.epochs, save_best=config.save_best)
    
    # Save eval report
    final_val = history["val"][-1] if history["val"] else {}
    eval_report = {
        "final_val_accuracy": final_val.get("accuracy", 0),
        "final_val_macro_f1": final_val.get("macro_f1", 0),
        "best_val_f1": trainer.best_val_f1,
        "best_epoch": trainer.best_epoch,
        "num_epochs": config.epochs,
        "per_class_f1": final_val.get("per_class_f1", {}),
        "confusion_matrix": final_val.get("confusion_matrix", [])
    }
    
    report_path = Path(config.output_dir) / "asm_eval_report_gc_m1_v1.json"
    with open(report_path, "w") as f:
        json.dump(eval_report, f, indent=2)
    
    print()
    print("=" * 60)
    print("Training Complete")
    print("=" * 60)
    print(f"Best Val F1: {trainer.best_val_f1:.4f} at epoch {trainer.best_epoch}")
    print(f"Final Val Accuracy: {final_val.get('accuracy', 0):.4f}")
    print(f"Saved model to: {config.output_dir}/asm_v2_gc_m1_v1.pt")
    print(f"Saved report to: {report_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
