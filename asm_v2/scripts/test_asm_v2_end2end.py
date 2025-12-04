#!/usr/bin/env python
"""
End-to-end test for ASM v2

Runs:
1. Build ASM dataset
2. Train ASM v2 (debug epochs)
3. Run leak tests

Usage:
    python asm_v2/scripts/test_asm_v2_end2end.py
"""

import subprocess
import sys
import json
from pathlib import Path


def run_command(cmd: list, description: str) -> bool:
    """Run a command and return success status"""
    print(f"\n{'='*60}")
    print(f"Step: {description}")
    print(f"Command: {' '.join(cmd)}")
    print("="*60)
    
    result = subprocess.run(cmd, capture_output=False)
    
    if result.returncode != 0:
        print(f"❌ Failed: {description}")
        return False
    
    print(f"✅ Completed: {description}")
    return True


def main():
    print("="*60)
    print("ASM v2 End-to-End Test")
    print("="*60)
    
    # Step 1: Build ASM dataset
    success = run_command(
        [sys.executable, "asm_v2/scripts/build_asm_dataset_gc_m1.py",
         "--config", "asm_v2/configs/asm_dataset_gc_m1_v1.json"],
        "Build ASM Dataset"
    )
    
    if not success:
        print("\n❌ End-to-end test failed at dataset building")
        return 1
    
    # Step 2: Train ASM v2 (debug: 3 epochs)
    success = run_command(
        [sys.executable, "asm_v2/scripts/train_asm_v2_gc_m1.py",
         "--config", "asm_v2/configs/asm_train_v1.json",
         "--epochs", "3"],
        "Train ASM v2 (debug)"
    )
    
    if not success:
        print("\n❌ End-to-end test failed at training")
        return 1
    
    # Step 3: Run leak tests
    success = run_command(
        [sys.executable, "asm_v2/scripts/eval_asm_v2_leaks_gc_m1.py",
         "--config", "asm_v2/configs/asm_eval_v1.json"],
        "Leak Evaluation"
    )
    
    if not success:
        print("\n❌ End-to-end test failed at leak evaluation")
        return 1
    
    # Load and print summary
    print("\n" + "="*60)
    print("End-to-End Test Summary")
    print("="*60)
    
    # Load dataset stats
    feature_config_path = "asm_v2/artifacts/final/asm_feature_config_v1.json"
    if Path(feature_config_path).exists():
        with open(feature_config_path, "r") as f:
            feature_config = json.load(f)
        print(f"Regime classes: {feature_config.get('num_classes', 'N/A')}")
    
    # Load eval report
    eval_report_path = "asm_v2/artifacts/final/asm_eval_report_gc_m1_v1.json"
    if Path(eval_report_path).exists():
        with open(eval_report_path, "r") as f:
            eval_report = json.load(f)
        print(f"Val Accuracy: {eval_report.get('final_val_accuracy', 'N/A'):.4f}")
        print(f"Val Macro F1: {eval_report.get('final_val_macro_f1', 'N/A'):.4f}")
        print(f"Best Val F1: {eval_report.get('best_val_f1', 'N/A'):.4f}")
    
    # Load leak report
    leak_report_path = "asm_v2/artifacts/gc_m1/asm_leak_report_gc_m1_v1.json"
    if Path(leak_report_path).exists():
        with open(leak_report_path, "r") as f:
            leak_report = json.load(f)
        all_passed = leak_report.get("all_passed", False)
        print(f"Leak tests: {'✅ All passed' if all_passed else '❌ Some failed'}")
    
    print("="*60)
    print("✅ End-to-end test completed successfully!")
    print("="*60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
