#!/usr/bin/env python
"""
Evaluate ASM v2 model on GC M1 data

Usage:
    python asm_v2/scripts/eval_asm_v2_gc_m1.py --config asm_v2/configs/asm_eval_v1.json
"""

import argparse
import json
import sys
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from asm_v2.src.config import AsmEvalConfig
from asm_v2.src.model.asm_model import load_asm_model
from asm_v2.src.dataset_asm import create_asm_dataloaders
from asm_v2.src.training.eval_metrics import AsmMetrics
from asm_v2.src.eval.semantic_eval_asm import AsmSemanticEvaluator


def main():
    parser = argparse.ArgumentParser(description="Evaluate ASM v2")
    parser.add_argument("--config", type=str, required=True, help="Path to eval config JSON")
    args = parser.parse_args()
    
    # Load config
    config = AsmEvalConfig.from_json(args.config)
    
    device = config.device if torch.cuda.is_available() else "cpu"
    
    print("=" * 60)
    print("Evaluating ASM v2")
    print("=" * 60)
    print(f"Model: {config.model_path}")
    print(f"Dataset: {config.dataset_path}")
    print()
    
    # Load model
    model = load_asm_model(
        config.model_path,
        config.model_config_path,
        device=device
    )
    
    # Load feature config
    with open(config.feature_config_path, "r") as f:
        feature_config = json.load(f)
    
    num_classes = feature_config.get("num_classes", 5)
    z_dim = feature_config.get("z_dim", 64)
    
    # Create val dataloader
    _, val_loader = create_asm_dataloaders(
        dataset_path=config.dataset_path,
        splits_path=config.splits_path,
        batch_size=64,
        z_dim=z_dim
    )
    
    # Evaluate
    model.eval()
    metrics = AsmMetrics(num_classes)
    
    with torch.no_grad():
        for batch in val_loader:
            x = batch["x"].to(device)
            labels = batch["label"].to(device)
            
            outputs = model(x)
            preds = outputs["logits"].argmax(dim=-1)
            
            metrics.update(preds, labels)
    
    results = metrics.compute()
    
    print("Validation Results:")
    print(f"  Accuracy: {results['accuracy']:.4f}")
    print(f"  Macro F1: {results['macro_f1']:.4f}")
    print(f"  Samples: {results['num_samples']}")
    print()
    print("Per-class F1:")
    for cls, f1 in results["per_class_f1"].items():
        regime_name = feature_config.get("regime_names", {}).get(str(cls), f"class_{cls}")
        print(f"  {regime_name}: {f1:.4f}")
    
    # Semantic evaluation
    print()
    print("Running semantic evaluation...")
    semantic_eval = AsmSemanticEvaluator(
        model=model,
        dataset_path=config.dataset_path,
        feature_config_path=config.feature_config_path,
        device=device
    )
    semantic_results = semantic_eval.evaluate()
    
    print()
    print("Per-session accuracy:")
    for session, stats in semantic_results.get("per_session_accuracy", {}).items():
        print(f"  {session}: {stats['accuracy']:.4f} ({stats['total']} samples)")
    
    # Save results
    output_path = Path(config.output_dir) / "asm_eval_full_report_gc_m1_v1.json"
    full_report = {
        "validation": results,
        "semantic": semantic_results
    }
    with open(output_path, "w") as f:
        json.dump(full_report, f, indent=2, default=str)
    
    print()
    print(f"Saved full report to: {output_path}")


if __name__ == "__main__":
    main()
