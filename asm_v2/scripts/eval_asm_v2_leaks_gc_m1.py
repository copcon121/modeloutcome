#!/usr/bin/env python
"""
Run leak evaluation tests for ASM v2

Usage:
    python asm_v2/scripts/eval_asm_v2_leaks_gc_m1.py --config asm_v2/configs/asm_eval_v1.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from asm_v2.src.eval.leak_eval_asm import run_leak_evaluation


def main():
    parser = argparse.ArgumentParser(description="Run ASM v2 leak tests")
    parser.add_argument("--config", type=str, required=True, help="Path to eval config JSON")
    args = parser.parse_args()
    
    # Load config
    with open(args.config, "r") as f:
        config = json.load(f)
    
    print("=" * 60)
    print("ASM v2 Leak Evaluation")
    print("=" * 60)
    
    # Run leak tests
    output_path = Path(config["output_dir"]) / "asm_leak_report_gc_m1_v1.json"
    
    results = run_leak_evaluation(
        dataset_path=config["dataset_path"],
        splits_path=config["splits_path"],
        feature_config_path=config["feature_config_path"],
        output_path=str(output_path)
    )
    
    # Print results
    print()
    for test_name, test_result in results.items():
        if test_name == "all_passed":
            continue
        
        status = "✅ PASS" if test_result.get("passed", False) else "❌ FAIL"
        print(f"{test_result.get('test', test_name)}: {status}")
        
        # Print details
        if test_name == "L1_TimeSplit":
            print(f"  Train dates: {test_result.get('train_dates_count', 0)}")
            print(f"  Val dates: {test_result.get('val_dates_count', 0)}")
            print(f"  Overlap: {test_result.get('overlap_count', 0)}")
        
        elif test_name == "L2_LabelShuffle":
            print(f"  Num classes: {test_result.get('num_classes', 0)}")
            print(f"  Label distribution: {test_result.get('label_distribution', {})}")
        
        elif test_name == "L3_FutureLeakGuard":
            leaked = test_result.get("leaked_features", [])
            if leaked:
                print(f"  Leaked features: {leaked}")
            else:
                print(f"  No future leaks detected")
        
        print()
    
    # Overall result
    all_passed = results.get("all_passed", False)
    print("=" * 60)
    if all_passed:
        print("✅ All leak tests passed!")
    else:
        print("❌ Some leak tests failed!")
    print("=" * 60)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
