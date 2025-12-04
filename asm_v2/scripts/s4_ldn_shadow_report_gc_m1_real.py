#!/usr/bin/env python3
"""
S4_LDN Shadow-Run Report Generator for REAL Data

Generates final summary report for shadow-run decision.

Usage:
    python asm_v2/scripts/s4_ldn_shadow_report_gc_m1_real.py
"""

import json
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def load_json(path: str) -> dict:
    with open(path, 'r') as f:
        return json.load(f)


def main():
    print("=" * 100)
    print("S4_LDN SHADOW-RUN REPORT (REAL GC M1 Data)")
    print("=" * 100)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Paths
    base_dir = "asm_v2/artifacts/gc_m1"
    league_path = f"{base_dir}/s4_policy_league_gc_m1_real_v1.json"
    best_path = f"{base_dir}/s4_policy_best_gc_m1_real_v1.json"
    sanity_path = f"{base_dir}/s4_policy_sanity_gc_m1_real_v1.json"
    
    # Load data
    try:
        league_data = load_json(league_path)
        best_data = load_json(best_path)
        sanity_data = load_json(sanity_path)
    except FileNotFoundError as e:
        print(f"\nERROR: Required file not found: {e}")
        print("Please run the full pipeline first:")
        print("  1. build_s4_ldn_trades_real_gc_m1.py")
        print("  2. run_s4_ldn_enrich_gc_m1_real.py")
        print("  3. run_s4_ldn_policy_backtest_v2_real.py")
        print("  4. eval_s4_ldn_policy_sanity_v2_real.py")
        sys.exit(1)
    
    # ========================================
    # SECTION 1: DATASET SUMMARY
    # ========================================
    print("\n" + "=" * 100)
    print("1. DATASET SUMMARY")
    print("=" * 100)
    
    n_total = league_data.get('n_total_trades', 0)
    n_train = league_data.get('n_train', 0)
    n_val = league_data.get('n_val', 0)
    n_test = league_data.get('n_test', 0)
    
    print(f"  Total trades: {n_total}")
    print(f"  Train: {n_train} ({n_train/n_total*100:.1f}%)" if n_total > 0 else "  Train: 0")
    print(f"  Val: {n_val} ({n_val/n_total*100:.1f}%)" if n_total > 0 else "  Val: 0")
    print(f"  Test: {n_test} ({n_test/n_total*100:.1f}%)" if n_total > 0 else "  Test: 0")
    
    # Find baseline (P0) metrics
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
    print("2. POLICY SUMMARY (Top 5 on TEST by Expectancy)")
    print("=" * 100)
    
    # Sort by test expectancy
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
    print("3. P7_DIRECTION_ALIGNED DETAIL (Candidate Policy)")
    print("=" * 100)
    
    p7 = None
    for r in results:
        if r['name'] == 'P7_direction_aligned':
            p7 = r
            break
    
    if p7:
        print(f"\n  Policy Logic: Trade aligned with regime direction")
        print(f"    - Long trades: only when regime = trend_up")
        print(f"    - Short trades: only when regime = trend_down")
        
        for split_name in ['train', 'val', 'test']:
            split_data = p7.get(split_name)
            if split_data:
                print(f"\n  {split_name.upper()}:")
                print(f"    N trades: {split_data['n_trades']}")
                print(f"    Win Rate: {split_data['win_rate']*100:.1f}%")
                print(f"    Expectancy: {split_data['expectancy']:.3f}R")
                print(f"    Max DD: {split_data['max_drawdown_r']:.2f}R")
                print(f"    Profit Factor: {split_data['profit_factor']:.2f}")
                print(f"    Cum R: {split_data['cum_r_final']:.2f}R")
                print(f"    Skip Rate: {split_data['skip_rate']*100:.1f}%")
    else:
        print("  P7_direction_aligned not found in results")
    
    # ========================================
    # SECTION 4: SANITY TESTS
    # ========================================
    print("\n" + "=" * 100)
    print("4. SANITY & LEAK TESTS")
    print("=" * 100)
    
    all_passed = sanity_data.get('all_passed', False)
    n_tests = sanity_data.get('n_tests', 0)
    n_passed = sanity_data.get('n_passed', 0)
    n_failed = sanity_data.get('n_failed', 0)
    critical_failures = sanity_data.get('critical_failures', [])
    
    print(f"\n  Tests: {n_passed}/{n_tests} passed")
    
    for test in sanity_data.get('tests', []):
        status = "✅ PASS" if test['passed'] else "❌ FAIL"
        severity = test.get('severity', 'info')
        icon = {'critical': '🔴', 'warning': '🟡', 'info': '🟢'}.get(severity, '⚪')
        print(f"  {icon} {test['name']}: {status}")
    
    if critical_failures:
        print(f"\n  🔴 CRITICAL FAILURES: {critical_failures}")
    
    # ========================================
    # SECTION 5: CONCLUSION
    # ========================================
    print("\n" + "=" * 100)
    print("5. CONCLUSION & RECOMMENDATION")
    print("=" * 100)
    
    # Decision logic
    recommend_shadow = True
    reasons = []
    
    # Check sanity
    if critical_failures:
        recommend_shadow = False
        reasons.append("Critical sanity test failures detected")
    
    # Check P7 performance
    if p7 and p7.get('test'):
        test_p7 = p7['test']
        
        if test_p7['n_trades'] < 10:
            reasons.append(f"Low sample size on TEST ({test_p7['n_trades']} trades)")
        
        if test_p7['expectancy'] < 0.3:
            recommend_shadow = False
            reasons.append(f"Low expectancy on TEST ({test_p7['expectancy']:.3f}R)")
        
        if test_p7['max_drawdown_r'] > 10:
            reasons.append(f"High max drawdown ({test_p7['max_drawdown_r']:.2f}R)")
        
        if test_p7['win_rate'] < 0.4:
            reasons.append(f"Low win rate ({test_p7['win_rate']*100:.1f}%)")
    else:
        recommend_shadow = False
        reasons.append("P7_direction_aligned not found or no test data")
    
    # Print recommendation
    if recommend_shadow:
        print(f"\n  ✅ RECOMMENDATION: P7_direction_aligned is APPROVED for SHADOW-RUN on NT8")
        if reasons:
            print(f"\n  Notes:")
            for r in reasons:
                print(f"    - {r}")
    else:
        print(f"\n  ❌ RECOMMENDATION: P7_direction_aligned is NOT APPROVED for shadow-run")
        print(f"\n  Reasons:")
        for r in reasons:
            print(f"    - {r}")
    
    # Summary for copy-paste
    print("\n" + "=" * 100)
    print("SUMMARY FOR REVIEW (Copy-paste friendly)")
    print("=" * 100)
    
    if p7 and p7.get('test'):
        t = p7['test']
        print(f"""
Policy: P7_direction_aligned
Dataset: GC M1 REAL ({n_total} trades)
Test Set: {t['n_trades']} trades

TEST METRICS:
- Win Rate: {t['win_rate']*100:.1f}%
- Expectancy: {t['expectancy']:.3f}R
- Max Drawdown: {t['max_drawdown_r']:.2f}R
- Profit Factor: {t['profit_factor']:.2f}
- Cumulative R: {t['cum_r_final']:.2f}R

SANITY TESTS: {n_passed}/{n_tests} passed
CRITICAL FAILURES: {len(critical_failures)}

RECOMMENDATION: {"APPROVED for SHADOW-RUN" if recommend_shadow else "NOT APPROVED"}
""")
    
    print("=" * 100)


if __name__ == '__main__':
    main()
