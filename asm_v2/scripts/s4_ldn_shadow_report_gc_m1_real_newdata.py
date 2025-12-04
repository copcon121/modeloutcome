#!/usr/bin/env python3
"""
S4_LDN Shadow-Run Report for GC M1 NEW DATA (OOS Phase 3)

Generates final OOS evaluation report comparing with original REAL data.

Usage:
    python asm_v2/scripts/s4_ldn_shadow_report_gc_m1_real_newdata.py
"""

import json
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def load_json(path: str) -> dict:
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def main():
    print("=" * 100)
    print("S4_LDN SHADOW-RUN REPORT - GC M1 NEW DATA (OOS Phase 3)")
    print("=" * 100)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Paths
    base_dir = "asm_v2/artifacts/gc_m1_new"
    league_path = f"{base_dir}/s4_policy_league_gc_m1_real_newdata_v1.json"
    best_path = f"{base_dir}/s4_policy_best_gc_m1_real_newdata_v1.json"
    sanity_path = f"{base_dir}/s4_policy_sanity_gc_m1_real_newdata_v1.json"
    
    # Original REAL data paths for comparison
    orig_base_dir = "asm_v2/artifacts/gc_m1"
    orig_league_path = f"{orig_base_dir}/s4_policy_league_gc_m1_real_v1.json"
    orig_best_path = f"{orig_base_dir}/s4_policy_best_gc_m1_real_v1.json"
    
    # Load data
    league_data = load_json(league_path)
    best_data = load_json(best_path)
    sanity_data = load_json(sanity_path)
    
    # Load original for comparison
    orig_league_data = load_json(orig_league_path)
    orig_best_data = load_json(orig_best_path)
    
    if not league_data:
        print(f"\nERROR: Required file not found: {league_path}")
        print("Please run the full NEW DATA pipeline first:")
        print("  python asm_v2/scripts/run_phase3_newdata_pipeline.py")
        sys.exit(1)
    
    # ========================================
    # SECTION 1: DATASET SUMMARY
    # ========================================
    print("\n" + "=" * 100)
    print("1. DATASET SUMMARY (NEW DATA OOS)")
    print("=" * 100)
    
    n_total = league_data.get('n_total_trades', 0)
    n_train = league_data.get('n_train', 0)
    n_val = league_data.get('n_val', 0)
    n_test = league_data.get('n_test', 0)
    
    print(f"  Total trades: {n_total}")
    print(f"  Train: {n_train} ({n_train/n_total*100:.1f}%)" if n_total > 0 else "  Train: 0")
    print(f"  Val: {n_val} ({n_val/n_total*100:.1f}%)" if n_total > 0 else "  Val: 0")
    print(f"  Test: {n_test} ({n_test/n_total*100:.1f}%)" if n_total > 0 else "  Test: 0")
    
    # Baseline metrics
    results = league_data.get('results', [])
    baseline = None
    for r in results:
        if r['name'] == 'P0_baseline_all':
            baseline = r
            break
    
    if baseline and baseline.get('test'):
        test_baseline = baseline['test']
        print(f"\n  Baseline (P0_baseline_all) on TEST:")
        print(f"    Win Rate: {test_baseline['win_rate']*100:.1f}%")
        print(f"    Expectancy: {test_baseline['expectancy']:.3f}R")
        print(f"    Max DD: {test_baseline['max_drawdown_r']:.2f}R")
    
    # ========================================
    # SECTION 2: POLICY SUMMARY (TOP 5 on TEST)
    # ========================================
    print("\n" + "=" * 100)
    print("2. POLICY SUMMARY (Top 5 on TEST by Expectancy) - NEW DATA")
    print("=" * 100)
    
    valid_results = [r for r in results if r.get('test') and r['test']['n_trades'] >= 3]
    valid_results.sort(key=lambda r: r['test']['expectancy'], reverse=True)
    
    print(f"\n{'Rank':<5} {'Policy':<30} {'N':<6} {'WR%':<8} {'Exp(R)':<10} {'MaxDD':<8} {'PF':<8}")
    print("-" * 85)
    
    for i, r in enumerate(valid_results[:5], 1):
        t = r['test']
        pf_str = f"{t['profit_factor']:.2f}" if t['profit_factor'] != float('inf') else "inf"
        print(f"{i:<5} {r['name']:<30} {t['n_trades']:<6} {t['win_rate']*100:>5.1f}%  {t['expectancy']:>8.3f}  {t['max_drawdown_r']:>6.2f}  {pf_str:>6}")
    
    # ========================================
    # SECTION 3: P7_DIRECTION_ALIGNED DETAIL
    # ========================================
    print("\n" + "=" * 100)
    print("3. P7_DIRECTION_ALIGNED DETAIL (NEW DATA vs ORIGINAL)")
    print("=" * 100)
    
    p7_new = None
    for r in results:
        if r['name'] == 'P7_direction_aligned':
            p7_new = r
            break
    
    # Find P7 in original data
    p7_orig = None
    if orig_league_data:
        orig_results = orig_league_data.get('results', [])
        for r in orig_results:
            if r['name'] == 'P7_direction_aligned':
                p7_orig = r
                break
    
    if p7_new:
        print(f"\n  NEW DATA (OOS):")
        for split_name in ['train', 'val', 'test']:
            split_data = p7_new.get(split_name)
            if split_data:
                print(f"    {split_name.upper()}: N={split_data['n_trades']}, WR={split_data['win_rate']*100:.1f}%, Exp={split_data['expectancy']:.3f}R")
    
    if p7_orig:
        print(f"\n  ORIGINAL DATA:")
        for split_name in ['train', 'val', 'test']:
            split_data = p7_orig.get(split_name)
            if split_data:
                print(f"    {split_name.upper()}: N={split_data['n_trades']}, WR={split_data['win_rate']*100:.1f}%, Exp={split_data['expectancy']:.3f}R")
    
    # Comparison
    if p7_new and p7_orig and p7_new.get('test') and p7_orig.get('test'):
        new_test = p7_new['test']
        orig_test = p7_orig['test']
        
        wr_delta = new_test['win_rate'] - orig_test['win_rate']
        exp_delta = new_test['expectancy'] - orig_test['expectancy']
        
        print(f"\n  COMPARISON (NEW vs ORIGINAL TEST):")
        print(f"    Win Rate: {new_test['win_rate']*100:.1f}% vs {orig_test['win_rate']*100:.1f}% (Δ{wr_delta*100:+.1f}%)")
        print(f"    Expectancy: {new_test['expectancy']:.3f}R vs {orig_test['expectancy']:.3f}R (Δ{exp_delta:+.3f}R)")
    
    # ========================================
    # SECTION 4: SANITY TESTS
    # ========================================
    print("\n" + "=" * 100)
    print("4. SANITY & LEAK TESTS (NEW DATA)")
    print("=" * 100)
    
    if sanity_data:
        all_passed = sanity_data.get('all_passed', False)
        n_tests = sanity_data.get('n_tests', 0)
        n_passed = sanity_data.get('n_passed', 0)
        critical_failures = sanity_data.get('critical_failures', [])
        
        print(f"\n  Tests: {n_passed}/{n_tests} passed")
        
        for test in sanity_data.get('tests', []):
            status = "✅ PASS" if test['passed'] else "❌ FAIL"
            severity = test.get('severity', 'info')
            icon = {'critical': '🔴', 'warning': '🟡', 'info': '🟢'}.get(severity, '⚪')
            print(f"  {icon} {test['name']}: {status}")
        
        if critical_failures:
            print(f"\n  🔴 CRITICAL FAILURES: {critical_failures}")
    else:
        print("\n  No sanity test data available")
    
    # ========================================
    # SECTION 5: CONCLUSION
    # ========================================
    print("\n" + "=" * 100)
    print("5. OOS EVALUATION CONCLUSION")
    print("=" * 100)
    
    # Decision logic
    oos_performance_ok = True
    reasons = []
    
    # Check sanity
    if sanity_data and sanity_data.get('critical_failures'):
        oos_performance_ok = False
        reasons.append("Critical sanity test failures")
    
    # Check P7 performance
    if p7_new and p7_new.get('test'):
        test_p7 = p7_new['test']
        
        if test_p7['n_trades'] < 5:
            reasons.append(f"Low sample size on TEST ({test_p7['n_trades']} trades)")
        
        if test_p7['expectancy'] < 0.1:
            oos_performance_ok = False
            reasons.append(f"Low expectancy on TEST ({test_p7['expectancy']:.3f}R)")
        
        # Compare with original if available
        if p7_orig and p7_orig.get('test'):
            orig_exp = p7_orig['test']['expectancy']
            exp_degradation = orig_exp - test_p7['expectancy']
            if exp_degradation > 0.5:  # Significant degradation
                reasons.append(f"Significant expectancy degradation ({exp_degradation:.3f}R)")
    else:
        oos_performance_ok = False
        reasons.append("P7_direction_aligned not found or no test data")
    
    # Print conclusion
    if oos_performance_ok:
        print(f"\n  ✅ OOS EVALUATION: P7_direction_aligned shows ACCEPTABLE performance on NEW DATA")
        if reasons:
            print(f"\n  Notes:")
            for r in reasons:
                print(f"    - {r}")
    else:
        print(f"\n  ⚠️ OOS EVALUATION: P7_direction_aligned shows DEGRADED performance on NEW DATA")
        print(f"\n  Issues:")
        for r in reasons:
            print(f"    - {r}")
    
    # ========================================
    # SUMMARY BLOCK (Copy-paste friendly)
    # ========================================
    print("\n" + "=" * 100)
    print("SUMMARY FOR REVIEW (Copy-paste friendly)")
    print("=" * 100)
    
    if p7_new and p7_new.get('test'):
        t = p7_new['test']
        comparison_str = ""
        if p7_orig and p7_orig.get('test'):
            orig_t = p7_orig['test']
            wr_delta = t['win_rate'] - orig_t['win_rate']
            exp_delta = t['expectancy'] - orig_t['expectancy']
            comparison_str = f"""
COMPARISON vs ORIGINAL:
- Win Rate: {t['win_rate']*100:.1f}% vs {orig_t['win_rate']*100:.1f}% (Δ{wr_delta*100:+.1f}%)
- Expectancy: {t['expectancy']:.3f}R vs {orig_t['expectancy']:.3f}R (Δ{exp_delta:+.3f}R)"""
        
        sanity_summary = "N/A"
        if sanity_data:
            n_passed = sanity_data.get('n_passed', 0)
            n_tests = sanity_data.get('n_tests', 0)
            critical_failures = len(sanity_data.get('critical_failures', []))
            sanity_summary = f"{n_passed}/{n_tests} passed, {critical_failures} critical failures"
        
        print(f"""
POLICY: P7_direction_aligned
DATASET: GC M1 NEW DATA (OOS Phase 3) - {n_total} trades
TEST SET: {t['n_trades']} trades

TEST METRICS (NEW DATA):
- Win Rate: {t['win_rate']*100:.1f}%
- Expectancy: {t['expectancy']:.3f}R
- Max Drawdown: {t['max_drawdown_r']:.2f}R
- Profit Factor: {t['profit_factor']:.2f}
- Cumulative R: {t['cum_r_final']:.2f}R{comparison_str}

SANITY TESTS: {sanity_summary}

OOS EVALUATION: {"ACCEPTABLE" if oos_performance_ok else "DEGRADED"}
""")
    
    print("=" * 100)


if __name__ == '__main__':
    main()
