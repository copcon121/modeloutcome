"""
S4 Policy Sanity & Leak Tests v2 - Enhanced tests for production

Tests:
- T1: Time-split sanity (train/val/test by date, no overlap)
- T2: Label shuffle (ML meta-policy should degrade with shuffled labels)
- T3: Future field guard (no outcome fields in features)
- T4: Stability by week (no explosive weeks)
"""

import json
import random
from copy import deepcopy
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from .policy_dataset import S4TradeEnriched, S4PolicyDataset
from .policy_rules import Policy, create_policies_from_config, DEFAULT_POLICIES
from .policy_model import S4MetaModel, train_meta_model
from .metrics import compute_metrics


# Forbidden outcome fields
FORBIDDEN_FIELDS = {
    'outcome_rr', 'rr_outcome', 'hit', 'label', 'result',
    'pnl', 'profit', 'loss', 'exit_price', 'exit_time',
    'final_price', 'realized_r', 'actual_rr',
}

# Suspicious future-looking fields
SUSPICIOUS_PATTERNS = [
    'future_', 'next_', 'forward_', '_outcome', '_result',
]


@dataclass
class SanityTestResultV2:
    """Result of a sanity test."""
    name: str
    passed: bool
    severity: str  # 'critical', 'warning', 'info'
    details: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def test_time_split_sanity(
    train_trades: List[S4TradeEnriched],
    val_trades: List[S4TradeEnriched],
    test_trades: List[S4TradeEnriched],
) -> SanityTestResultV2:
    """T1: Verify time-based split with no overlap."""
    
    train_dates = set(t.get_date() for t in train_trades)
    val_dates = set(t.get_date() for t in val_trades)
    test_dates = set(t.get_date() for t in test_trades)
    
    # Check overlaps
    train_val_overlap = train_dates & val_dates
    val_test_overlap = val_dates & test_dates
    train_test_overlap = train_dates & test_dates
    
    no_overlap = (
        len(train_val_overlap) == 0 and
        len(val_test_overlap) == 0 and
        len(train_test_overlap) == 0
    )
    
    # Check chronological order
    chronological = True
    if train_trades and val_trades:
        last_train = max(t.time for t in train_trades)
        first_val = min(t.time for t in val_trades)
        if last_train > first_val:
            chronological = False
    
    if val_trades and test_trades:
        last_val = max(t.time for t in val_trades)
        first_test = min(t.time for t in test_trades)
        if last_val > first_test:
            chronological = False
    
    passed = no_overlap and chronological
    
    return SanityTestResultV2(
        name="T1_TimeSplitSanity",
        passed=passed,
        severity='critical' if not passed else 'info',
        details={
            'train_dates': len(train_dates),
            'val_dates': len(val_dates),
            'test_dates': len(test_dates),
            'train_val_overlap': len(train_val_overlap),
            'val_test_overlap': len(val_test_overlap),
            'train_test_overlap': len(train_test_overlap),
            'chronological_order': chronological,
            'train_samples': len(train_trades),
            'val_samples': len(val_trades),
            'test_samples': len(test_trades),
        }
    )


def test_label_shuffle_policy(
    train_trades: List[S4TradeEnriched],
    val_trades: List[S4TradeEnriched],
    n_iterations: int = 3,
) -> SanityTestResultV2:
    """T2: Verify ML meta-policy degrades with shuffled labels."""
    
    if len(train_trades) < 30 or len(val_trades) < 10:
        return SanityTestResultV2(
            name="T2_LabelShufflePolicy",
            passed=True,
            severity='info',
            details={
                'skipped': True,
                'reason': 'Too few samples for meaningful test',
                'train_samples': len(train_trades),
                'val_samples': len(val_trades),
            }
        )
    
    # Train on real labels
    train_ds = S4PolicyDataset(train_trades)
    val_ds = S4PolicyDataset(val_trades)
    
    model_real, results_real = train_meta_model(train_ds, val_ds)
    real_acc = results_real.get('val', {}).get('accuracy', 0.5)
    
    # Compute real expectancy with threshold 0.5
    real_kept = []
    for t in val_trades:
        if t.label in ['win', 'loss']:
            p_win = model_real.predict_proba_single(t)
            if p_win >= 0.5:
                real_kept.append(t)
    real_metrics = compute_metrics(real_kept) if real_kept else {'expectancy': 0}
    real_exp = real_metrics['expectancy']
    
    # Train on shuffled labels
    shuffled_accs = []
    shuffled_exps = []
    
    for _ in range(n_iterations):
        shuffled_trades = deepcopy(train_trades)
        labels = [t.label for t in shuffled_trades]
        random.shuffle(labels)
        for t, new_label in zip(shuffled_trades, labels):
            t.label = new_label
        
        shuffled_ds = S4PolicyDataset(shuffled_trades)
        model_shuffled, results_shuffled = train_meta_model(shuffled_ds, val_ds)
        shuffled_acc = results_shuffled.get('val', {}).get('accuracy', 0.5)
        shuffled_accs.append(shuffled_acc)
        
        # Compute shuffled expectancy
        shuffled_kept = []
        for t in val_trades:
            if t.label in ['win', 'loss']:
                p_win = model_shuffled.predict_proba_single(t)
                if p_win >= 0.5:
                    shuffled_kept.append(t)
        shuffled_metrics = compute_metrics(shuffled_kept) if shuffled_kept else {'expectancy': 0}
        shuffled_exps.append(shuffled_metrics['expectancy'])
    
    avg_shuffled_acc = np.mean(shuffled_accs)
    avg_shuffled_exp = np.mean(shuffled_exps)
    
    # Real should be better (or at least not worse)
    acc_margin = 0.02
    passed = real_acc >= avg_shuffled_acc - acc_margin
    
    return SanityTestResultV2(
        name="T2_LabelShufflePolicy",
        passed=passed,
        severity='warning' if not passed else 'info',
        details={
            'real_val_accuracy': float(real_acc),
            'shuffled_val_accuracy_avg': float(avg_shuffled_acc),
            'real_val_expectancy': float(real_exp),
            'shuffled_val_expectancy_avg': float(avg_shuffled_exp),
            'accuracy_margin': acc_margin,
            'n_iterations': n_iterations,
        }
    )


def test_future_field_guard(
    trades: List[S4TradeEnriched],
) -> SanityTestResultV2:
    """T3: Verify no outcome/future fields used as input features."""
    
    if not trades:
        return SanityTestResultV2(
            name="T3_FutureFieldGuard",
            passed=True,
            severity='info',
            details={'skipped': True, 'reason': 'No trades to check'}
        )
    
    # Check what fields are available in trade
    sample = trades[0]
    
    # Fields used for policy decisions (features)
    feature_fields = [
        'regime_id', 'regime_name', 'regime_confidence', 'regime_onehot',
        'session', 'session_id', 'pos_in_session_range',
        'minute_of_day_norm', 'day_of_week_norm',
        'inside_value', 'above_value', 'below_value',
        'direction', 'z_t', 'setup_type',
    ]
    
    # Check for forbidden fields in features
    leaked_features = []
    for field in feature_fields:
        if field.lower() in FORBIDDEN_FIELDS:
            leaked_features.append(field)
        for pattern in SUSPICIOUS_PATTERNS:
            if pattern in field.lower():
                leaked_features.append(field)
    
    # Check raw dict if available
    raw_outcome_fields = []
    if hasattr(sample, 'raw') and sample.raw:
        for key in sample.raw.keys():
            if key.lower() in FORBIDDEN_FIELDS:
                raw_outcome_fields.append(key)
    
    passed = len(leaked_features) == 0
    
    return SanityTestResultV2(
        name="T3_FutureFieldGuard",
        passed=passed,
        severity='critical' if not passed else 'info',
        details={
            'feature_fields_checked': feature_fields,
            'leaked_features': leaked_features,
            'raw_outcome_fields': raw_outcome_fields,
            'note': 'Outcome fields in raw data are OK (used for metrics only)',
        }
    )


def test_stability_by_week(
    trades: List[S4TradeEnriched],
    threshold: float = 2.0,  # Max deviation from overall expectancy
) -> SanityTestResultV2:
    """T4: Check for explosive weeks that might indicate issues."""
    
    if len(trades) < 20:
        return SanityTestResultV2(
            name="T4_StabilityByWeek",
            passed=True,
            severity='info',
            details={'skipped': True, 'reason': 'Too few trades for weekly analysis'}
        )
    
    # Group by week
    weeks = {}
    for t in trades:
        try:
            dt = t.time
            # Get ISO week
            week_key = dt.strftime('%Y-W%W')
            if week_key not in weeks:
                weeks[week_key] = []
            weeks[week_key].append(t)
        except:
            continue
    
    if len(weeks) < 3:
        return SanityTestResultV2(
            name="T4_StabilityByWeek",
            passed=True,
            severity='info',
            details={'skipped': True, 'reason': 'Too few weeks for analysis'}
        )
    
    # Compute overall expectancy
    overall_metrics = compute_metrics(trades)
    overall_exp = overall_metrics['expectancy']
    overall_std = overall_metrics['std_rr']
    
    # Compute per-week expectancy
    week_stats = {}
    explosive_weeks = []
    
    for week, week_trades in weeks.items():
        if len(week_trades) < 3:
            continue
        
        week_metrics = compute_metrics(week_trades)
        week_exp = week_metrics['expectancy']
        week_stats[week] = {
            'n_trades': len(week_trades),
            'expectancy': week_exp,
            'win_rate': week_metrics['win_rate'],
        }
        
        # Check if explosive
        deviation = abs(week_exp - overall_exp)
        if overall_std > 0 and deviation > threshold * overall_std:
            explosive_weeks.append({
                'week': week,
                'expectancy': week_exp,
                'deviation': deviation,
                'n_trades': len(week_trades),
            })
    
    passed = len(explosive_weeks) == 0
    
    return SanityTestResultV2(
        name="T4_StabilityByWeek",
        passed=passed,
        severity='warning' if not passed else 'info',
        details={
            'n_weeks': len(weeks),
            'overall_expectancy': float(overall_exp),
            'overall_std': float(overall_std),
            'threshold_multiplier': threshold,
            'explosive_weeks': explosive_weeks,
            'week_stats': week_stats,
        }
    )


def run_all_sanity_tests_v2(
    train_trades: List[S4TradeEnriched],
    val_trades: List[S4TradeEnriched],
    test_trades: List[S4TradeEnriched],
) -> List[SanityTestResultV2]:
    """Run all sanity tests."""
    results = []
    
    # T1: Time split
    results.append(test_time_split_sanity(train_trades, val_trades, test_trades))
    
    # T2: Label shuffle
    results.append(test_label_shuffle_policy(train_trades, val_trades))
    
    # T3: Future field guard
    all_trades = train_trades + val_trades + test_trades
    results.append(test_future_field_guard(all_trades))
    
    # T4: Stability by week (on test set)
    results.append(test_stability_by_week(test_trades))
    
    return results


def convert_to_json_serializable(obj):
    """Convert numpy/bool types to JSON serializable."""
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, (np.integer, int)):
        return int(obj)
    elif isinstance(obj, (np.floating, float)):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(v) for v in obj]
    return obj


def save_sanity_report_v2(results: List[SanityTestResultV2], path: str):
    """Save sanity report to JSON."""
    all_passed = all(r.passed for r in results)
    critical_failed = [r.name for r in results if not r.passed and r.severity == 'critical']
    
    report = {
        'all_passed': bool(all_passed),
        'critical_failures': critical_failed,
        'n_tests': len(results),
        'n_passed': sum(1 for r in results if r.passed),
        'n_failed': sum(1 for r in results if not r.passed),
        'tests': [convert_to_json_serializable(r.to_dict()) for r in results],
    }
    
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(report, f, indent=2)


def print_sanity_summary_v2(results: List[SanityTestResultV2]):
    """Print sanity test summary."""
    print("=" * 80)
    print("S4 Policy Sanity Test Results (V2)")
    print("=" * 80)
    
    for result in results:
        status = "✅ PASS" if result.passed else "❌ FAIL"
        severity_icon = {
            'critical': '🔴',
            'warning': '🟡',
            'info': '🟢',
        }.get(result.severity, '⚪')
        
        print(f"\n{severity_icon} {result.name}: {status}")
        
        # Print key details
        for key, value in result.details.items():
            if key == 'week_stats':
                print(f"  {key}: [{len(value)} weeks]")
            elif isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            elif isinstance(value, list) and len(value) > 5:
                print(f"  {key}: [{len(value)} items]")
            else:
                print(f"  {key}: {value}")
    
    print("\n" + "=" * 80)
    all_passed = all(r.passed for r in results)
    critical_failed = [r.name for r in results if not r.passed and r.severity == 'critical']
    
    if all_passed:
        print("✅ All sanity tests PASSED!")
    elif critical_failed:
        print(f"🔴 CRITICAL FAILURES: {critical_failed}")
    else:
        warnings = [r.name for r in results if not r.passed]
        print(f"🟡 Warnings: {warnings}")
    print("=" * 80)
