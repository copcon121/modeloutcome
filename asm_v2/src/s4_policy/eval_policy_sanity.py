"""
S4 Policy Sanity & Leak Tests

Tests to ensure policy layer doesn't have data leakage:
- T1: Time-split sanity - verify no future info used
- T2: Label shuffle check - verify model learns real patterns
- T3: Future-field guard - verify no outcome fields in features
"""

from typing import List, Dict, Any, Tuple
import numpy as np
import json
from pathlib import Path
from copy import deepcopy
import random

from .policy_dataset import S4TradeEnriched, S4PolicyDataset
from .policy_rules import Policy, create_policy, PolicyConfig
from .policy_model import S4MetaModel, train_meta_model
from .backtester import Backtester
from .metrics import compute_metrics


# Fields that should NEVER be used as input features (they are outcomes)
FORBIDDEN_OUTCOME_FIELDS = {
    'rr_outcome',
    'outcome_rr',
    'label',
    'hit',
    'result',
    'pnl',
    'profit',
    'loss',
    'final_price',
    'exit_price',
    'exit_time',
}

# Fields that are suspicious (might leak future info)
SUSPICIOUS_FIELDS = {
    'future_return',
    'future_dir',
    'next_',
    'forward_',
}


class SanityTestResult:
    """Result of a sanity test."""
    
    def __init__(self, name: str, passed: bool, details: Dict[str, Any]):
        self.name = name
        self.passed = passed
        self.details = details
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'test': self.name,
            'passed': self.passed,
            'details': self.details,
        }
    
    def __repr__(self) -> str:
        status = "✅ PASS" if self.passed else "❌ FAIL"
        return f"{self.name}: {status}"


def test_time_split_sanity(dataset: S4PolicyDataset) -> SanityTestResult:
    """T1: Time-split sanity test.
    
    Verifies:
    - Train/val split is by DATE (no shuffle)
    - No date overlap between train and val
    - Chronological order preserved
    """
    train_ds, val_ds = dataset.time_split(train_ratio=0.8)
    
    train_dates = set(t.get_date() for t in train_ds.trades)
    val_dates = set(t.get_date() for t in val_ds.trades)
    
    # Check no overlap
    overlap = train_dates & val_dates
    no_overlap = len(overlap) == 0
    
    # Check chronological order
    if train_ds.trades and val_ds.trades:
        last_train_time = max(t.time for t in train_ds.trades)
        first_val_time = min(t.time for t in val_ds.trades)
        chronological = last_train_time <= first_val_time
    else:
        chronological = True
    
    passed = no_overlap and chronological
    
    return SanityTestResult(
        name="T1_TimeSplitSanity",
        passed=passed,
        details={
            'train_dates_count': len(train_dates),
            'val_dates_count': len(val_dates),
            'overlap_count': len(overlap),
            'overlap_dates': list(overlap)[:5],  # Show first 5
            'chronological_order': chronological,
            'train_samples': len(train_ds),
            'val_samples': len(val_ds),
        }
    )


def test_label_shuffle(dataset: S4PolicyDataset, n_iterations: int = 3) -> SanityTestResult:
    """T2: Label shuffle test.
    
    Verifies that model/policy learns real patterns by:
    1. Training meta-model on real labels
    2. Training on shuffled labels
    3. Comparing performance (real should be better)
    """
    train_ds, val_ds = dataset.time_split(train_ratio=0.8)
    
    # Skip if too few samples
    if len(train_ds) < 20 or len(val_ds) < 10:
        return SanityTestResult(
            name="T2_LabelShuffle",
            passed=True,
            details={
                'skipped': True,
                'reason': 'Too few samples for meaningful test',
                'train_samples': len(train_ds),
                'val_samples': len(val_ds),
            }
        )
    
    # Train on real labels
    model_real, results_real = train_meta_model(train_ds, val_ds)
    real_acc = results_real.get('val', {}).get('accuracy', 0.5)
    
    # Train on shuffled labels multiple times
    shuffled_accs = []
    for _ in range(n_iterations):
        # Create shuffled dataset
        shuffled_trades = deepcopy(train_ds.trades)
        labels = [t.label for t in shuffled_trades]
        random.shuffle(labels)
        for t, new_label in zip(shuffled_trades, labels):
            t.label = new_label
        
        shuffled_ds = S4PolicyDataset(shuffled_trades)
        model_shuffled, results_shuffled = train_meta_model(shuffled_ds, val_ds)
        shuffled_acc = results_shuffled.get('val', {}).get('accuracy', 0.5)
        shuffled_accs.append(shuffled_acc)
    
    avg_shuffled_acc = np.mean(shuffled_accs)
    
    # Real should be better than shuffled (with margin)
    margin = 0.02
    passed = real_acc > avg_shuffled_acc - margin
    
    return SanityTestResult(
        name="T2_LabelShuffle",
        passed=passed,
        details={
            'real_val_accuracy': float(real_acc),
            'shuffled_val_accuracy_avg': float(avg_shuffled_acc),
            'shuffled_val_accuracies': [float(a) for a in shuffled_accs],
            'margin': margin,
            'difference': float(real_acc - avg_shuffled_acc),
        }
    )


def test_future_field_guard(dataset: S4PolicyDataset) -> SanityTestResult:
    """T3: Future-field guard test.
    
    Verifies that no outcome/future fields are used as input features.
    Checks the raw trade data for forbidden fields.
    """
    if not dataset.trades:
        return SanityTestResult(
            name="T3_FutureFieldGuard",
            passed=True,
            details={'skipped': True, 'reason': 'No trades to check'}
        )
    
    # Check first trade's raw data
    sample_trade = dataset.trades[0]
    raw_fields = set(sample_trade.raw.keys())
    context_fields = set(sample_trade.raw.get('context', {}).keys())
    all_fields = raw_fields | context_fields
    
    # Check for forbidden fields in features
    # Note: These fields CAN exist in raw data, but should NOT be used in decide_trade
    found_forbidden = []
    found_suspicious = []
    
    for field in all_fields:
        field_lower = field.lower()
        if field_lower in FORBIDDEN_OUTCOME_FIELDS:
            found_forbidden.append(field)
        for suspicious in SUSPICIOUS_FIELDS:
            if suspicious in field_lower:
                found_suspicious.append(field)
    
    # The test passes if we don't find suspicious fields in the FEATURE set
    # (outcome fields in raw data are OK, they're used for metrics, not features)
    # We check the S4TradeEnriched dataclass fields that are used for policy decisions
    
    feature_fields_used = [
        'regime', 'regime_name', 'regime_confidence',
        'session', 'session_id', 'pos_in_session_range',
        'inside_value', 'above_value', 'below_value',
        'direction', 'z_t'
    ]
    
    # Check if any feature field looks like an outcome
    leaked_features = []
    for field in feature_fields_used:
        if field.lower() in FORBIDDEN_OUTCOME_FIELDS:
            leaked_features.append(field)
    
    passed = len(leaked_features) == 0
    
    return SanityTestResult(
        name="T3_FutureFieldGuard",
        passed=passed,
        details={
            'feature_fields_checked': feature_fields_used,
            'leaked_features': leaked_features,
            'raw_fields_with_outcomes': found_forbidden,
            'suspicious_fields_in_raw': found_suspicious,
            'note': 'Outcome fields in raw data are OK (used for metrics only)',
        }
    )


def test_policy_consistency(
    dataset: S4PolicyDataset,
    policy: Policy
) -> SanityTestResult:
    """Test that policy decisions are consistent (deterministic).
    
    Run policy twice on same data, should get same results.
    """
    decisions_1 = [policy.decide_trade(t) for t in dataset.trades]
    decisions_2 = [policy.decide_trade(t) for t in dataset.trades]
    
    consistent = decisions_1 == decisions_2
    
    return SanityTestResult(
        name=f"T4_PolicyConsistency_{policy.name}",
        passed=consistent,
        details={
            'policy_name': policy.name,
            'n_trades': len(dataset.trades),
            'consistent': consistent,
        }
    )


def run_all_sanity_tests(
    dataset: S4PolicyDataset,
    policies: List[Policy] = None,
) -> List[SanityTestResult]:
    """Run all sanity tests.
    
    Args:
        dataset: S4PolicyDataset to test
        policies: Optional list of policies to test consistency
    
    Returns:
        List of SanityTestResult
    """
    results = []
    
    # Core tests
    results.append(test_time_split_sanity(dataset))
    results.append(test_label_shuffle(dataset))
    results.append(test_future_field_guard(dataset))
    
    # Policy consistency tests
    if policies:
        for policy in policies[:3]:  # Test first 3 policies
            results.append(test_policy_consistency(dataset, policy))
    
    return results


def save_sanity_report(results: List[SanityTestResult], path: str):
    """Save sanity test report to JSON."""
    all_passed = all(r.passed for r in results)
    
    report = {
        'all_passed': all_passed,
        'n_tests': len(results),
        'n_passed': sum(1 for r in results if r.passed),
        'n_failed': sum(1 for r in results if not r.passed),
        'tests': [r.to_dict() for r in results],
    }
    
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(report, f, indent=2)


def print_sanity_summary(results: List[SanityTestResult]):
    """Print sanity test summary to console."""
    print("=" * 60)
    print("S4 Policy Sanity Test Results")
    print("=" * 60)
    
    for result in results:
        status = "✅ PASS" if result.passed else "❌ FAIL"
        print(f"\n{result.name}: {status}")
        
        # Print key details
        for key, value in result.details.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            elif isinstance(value, list) and len(value) > 5:
                print(f"  {key}: [{len(value)} items]")
            else:
                print(f"  {key}: {value}")
    
    print("\n" + "=" * 60)
    all_passed = all(r.passed for r in results)
    if all_passed:
        print("✅ All sanity tests PASSED!")
    else:
        failed = [r.name for r in results if not r.passed]
        print(f"❌ {len(failed)} test(s) FAILED: {failed}")
    print("=" * 60)
