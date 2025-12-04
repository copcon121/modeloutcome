#!/usr/bin/env python3
"""
PHASE 3 - GC M1 NEW DATA OOS Pipeline Runner

Runs the complete OOS evaluation pipeline on 6 weeks of new data.
Uses FROZEN models: STATE-ENC v1.2, ASM v2, P7_direction_aligned policy.

Usage:
    python asm_v2/scripts/run_phase3_newdata_pipeline.py [--step STEP]
    
Steps:
    0. build_features     - Build SMC features from raw data
    1. encoder_dataset    - Build encoder dataset from bars_enhanced
    2. asm_dataset        - Build ASM dataset with z_t embeddings
    3. asm_inference      - Run regime inference
    4. s4_trades          - Build S4 trades from new data
    5. s4_enrich          - Enrich trades with z_t + regime
    6. policy_backtest    - Run policy backtest
    7. sanity_tests       - Run sanity/leak tests
    8. shadow_report      - Generate final report
    all (default)
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_command(cmd: str, description: str) -> bool:
    """Run a command and handle errors."""
    print(f"\n{'='*80}")
    print(f"STEP: {description}")
    print(f"CMD: {cmd}")
    print(f"{'='*80}")
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=False)
        print(f"\n✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ {description} failed with exit code {e.returncode}")
        return False


def main():
    parser = argparse.ArgumentParser(description='PHASE 3 - GC M1 NEW DATA OOS Pipeline')
    parser.add_argument('--step', type=str, default='all',
                        choices=['build_features', 'encoder_dataset', 'asm_dataset', 
                                'asm_inference', 's4_trades', 's4_enrich', 
                                'policy_backtest', 'sanity_tests', 'shadow_report', 'all'],
                        help='Specific step to run (default: all)')
    args = parser.parse_args()
    
    print("=" * 100)
    print("PHASE 3 - GC M1 NEW DATA OOS EVALUATION PIPELINE")
    print("=" * 100)
    print(f"Running step: {args.step}")
    
    steps = {
        'build_features': (
            'python scripts/build_gc_m1_features_newdata.py',
            'Build SMC Features for NEW DATA'
        ),
        'encoder_dataset': (
            'python state_enc_v1/scripts/build_encoder_dataset_gc_m1_newdata.py',
            'Build Encoder Dataset for NEW DATA'
        ),
        'asm_dataset': (
            'python asm_v2/scripts/build_asm_dataset_gc_m1_newdata.py',
            'Build ASM Dataset for NEW DATA'
        ),
        'asm_inference': (
            'python asm_v2/scripts/run_asm_infer_gc_m1_newdata.py',
            'Run ASM Regime Inference on NEW DATA'
        ),
        's4_trades': (
            'python asm_v2/scripts/build_s4_ldn_trades_real_gc_m1_newdata.py',
            'Build S4 Trades for NEW DATA'
        ),
        's4_enrich': (
            'python asm_v2/scripts/run_s4_ldn_enrich_gc_m1_real_newdata.py',
            'Enrich S4 Trades for NEW DATA'
        ),
        'policy_backtest': (
            'python asm_v2/scripts/run_s4_ldn_policy_backtest_v2_real_newdata.py',
            'Run Policy Backtest on NEW DATA'
        ),
        'sanity_tests': (
            'python asm_v2/scripts/eval_s4_ldn_policy_sanity_v2_real_newdata.py',
            'Run Sanity Tests on NEW DATA'
        ),
        'shadow_report': (
            'python asm_v2/scripts/s4_ldn_shadow_report_gc_m1_real_newdata.py',
            'Generate Shadow Report for NEW DATA'
        ),
    }
    
    if args.step == 'all':
        steps_to_run = list(steps.keys())
    else:
        steps_to_run = [args.step]
    
    success_count = 0
    total_steps = len(steps_to_run)
    
    for step in steps_to_run:
        cmd, description = steps[step]
        success = run_command(cmd, description)
        if success:
            success_count += 1
        else:
            print(f"\n❌ Pipeline failed at step: {step}")
            sys.exit(1)
    
    print(f"\n" + "=" * 100)
    print(f"PHASE 3 PIPELINE COMPLETED")
    print(f"=" * 100)
    print(f"✅ {success_count}/{total_steps} steps completed successfully")
    
    if args.step == 'all' or args.step == 'shadow_report':
        print(f"\n📊 Final report generated. Check:")
        print(f"   asm_v2/artifacts/gc_m1_new/s4_policy_league_gc_m1_real_newdata_v1.json")
        print(f"   asm_v2/artifacts/gc_m1_new/s4_policy_best_gc_m1_real_newdata_v1.json")
    
    print(f"\n" + "=" * 100)


if __name__ == '__main__':
    main()
