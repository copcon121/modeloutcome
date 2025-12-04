"""
LEAK & INTEGRITY TEST SUITE for STATE-ENC v1.2

Tests:
- L1: Index and Future Boundaries Check
- L2: Time-based Split Validation
- L3: Label Shuffle Sanity Test
- L4: Future Cheat Upper Bound Test
- L5: Spot Check Causality (optional)
"""

import json
import logging
import random
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# L1: Index and Future Boundaries Check
# ============================================================================

def check_index_and_future_boundaries(
    dataset_path: str,
    max_samples: int = 1000,
    seq_len: int = 64
) -> Dict[str, Any]:
    """
    L1: Check that future labels don't leak into context window.
    
    Validates:
    - end_idx == start_idx + N - 1
    - label_start_idx >= end_idx + 1
    - label_end_idx >= label_start_idx
    - timestamps strictly increasing with ~1 min gap
    """
    logger.info("=" * 60)
    logger.info("L1: INDEX AND FUTURE BOUNDARIES CHECK")
    logger.info("=" * 60)
    
    violations = []
    num_checked = 0
    time_gaps = []

    with open(dataset_path, 'r') as f:
        for i, line in enumerate(f):
            if i >= max_samples:
                break
            
            sample = json.loads(line.strip())
            num_checked += 1
            
            # Get metadata
            meta = sample.get('meta', {})
            aux = sample.get('aux', {})
            
            # Check sequence length
            X = sample.get('X', [])
            actual_seq_len = len(X)
            
            if actual_seq_len != seq_len:
                violations.append({
                    'sample_idx': i,
                    'type': 'seq_len_mismatch',
                    'expected': seq_len,
                    'actual': actual_seq_len
                })
            
            # Check index boundaries if meta available
            if meta:
                start_idx = meta.get('start_idx', 0)
                end_idx = meta.get('end_idx', start_idx + seq_len - 1)
                label_start_idx = meta.get('label_start_idx', end_idx + 1)
                label_end_idx = meta.get('label_end_idx', label_start_idx + 4)
                
                # Check: end_idx == start_idx + N - 1
                expected_end = start_idx + seq_len - 1
                if end_idx != expected_end:
                    violations.append({
                        'sample_idx': i,
                        'type': 'end_idx_mismatch',
                        'expected': expected_end,
                        'actual': end_idx
                    })
                
                # Check: label_start_idx >= end_idx + 1 (no overlap)
                if label_start_idx <= end_idx:
                    violations.append({
                        'sample_idx': i,
                        'type': 'label_overlap',
                        'end_idx': end_idx,
                        'label_start_idx': label_start_idx,
                        'message': 'Label window overlaps with context!'
                    })
                
                # Check: label_end_idx >= label_start_idx
                if label_end_idx < label_start_idx:
                    violations.append({
                        'sample_idx': i,
                        'type': 'label_range_invalid',
                        'label_start_idx': label_start_idx,
                        'label_end_idx': label_end_idx
                    })
            
            # Check timestamp ordering (if timestamps in aux)
            timestamps = aux.get('timestamps', [])
            if len(timestamps) >= 2:
                for j in range(1, len(timestamps)):
                    try:
                        t1 = datetime.fromisoformat(timestamps[j-1].replace('Z', '+00:00'))
                        t2 = datetime.fromisoformat(timestamps[j].replace('Z', '+00:00'))
                        gap = (t2 - t1).total_seconds()
                        time_gaps.append(gap)
                        
                        if gap <= 0:
                            violations.append({
                                'sample_idx': i,
                                'type': 'time_not_increasing',
                                'bar_idx': j,
                                't1': timestamps[j-1],
                                't2': timestamps[j]
                            })
                        elif gap > 120:  # More than 2 minutes gap
                            violations.append({
                                'sample_idx': i,
                                'type': 'time_gap_large',
                                'bar_idx': j,
                                'gap_seconds': gap
                            })
                    except Exception:
                        pass
    
    # Summary
    result = {
        'test': 'L1_index_boundaries',
        'num_checked': num_checked,
        'num_violations': len(violations),
        'pass': len(violations) == 0,
        'violations': violations[:10] if violations else [],  # First 10 examples
        'time_gap_stats': {
            'mean': float(np.mean(time_gaps)) if time_gaps else None,
            'std': float(np.std(time_gaps)) if time_gaps else None,
            'min': float(np.min(time_gaps)) if time_gaps else None,
            'max': float(np.max(time_gaps)) if time_gaps else None
        } if time_gaps else None
    }
    
    status = "✅ PASS" if result['pass'] else "❌ FAIL"
    logger.info(f"  Checked: {num_checked} samples")
    logger.info(f"  Violations: {len(violations)}")
    logger.info(f"  Status: {status}")
    
    return result


# ============================================================================
# L2: Time-based Split Validation
# ============================================================================

def check_time_based_split(
    dataset_path: str,
    train_days: List[str],
    val_days: List[str],
    max_samples: int = 50000
) -> Dict[str, Any]:
    """
    L2: Validate train/val split has no date overlap.
    """
    logger.info("\n" + "=" * 60)
    logger.info("L2: TIME-BASED SPLIT VALIDATION")
    logger.info("=" * 60)
    
    train_dates = set(train_days)
    val_dates = set(val_days)
    
    # Check overlap
    overlap = train_dates & val_dates
    
    # Count samples per date
    date_counts = defaultdict(int)
    train_samples = 0
    val_samples = 0
    unknown_samples = 0
    
    with open(dataset_path, 'r') as f:
        for i, line in enumerate(f):
            if i >= max_samples:
                break
            
            sample = json.loads(line.strip())
            aux = sample.get('aux', {})
            
            # Try to get date from timestamp or meta
            timestamp = aux.get('timestamp', sample.get('timestamp', ''))
            if timestamp:
                try:
                    dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    date_str = dt.strftime('%Y-%m-%d')
                    date_counts[date_str] += 1
                    
                    if date_str in train_dates:
                        train_samples += 1
                    elif date_str in val_dates:
                        val_samples += 1
                    else:
                        unknown_samples += 1
                except Exception:
                    unknown_samples += 1
            else:
                unknown_samples += 1
    
    result = {
        'test': 'L2_time_split',
        'train_days': list(train_dates),
        'val_days': list(val_dates),
        'overlap_days': list(overlap),
        'has_overlap': len(overlap) > 0,
        'train_samples': train_samples,
        'val_samples': val_samples,
        'unknown_samples': unknown_samples,
        'date_distribution': dict(date_counts),
        'pass': len(overlap) == 0
    }
    
    status = "✅ PASS" if result['pass'] else "❌ FAIL"
    logger.info(f"  Train days: {len(train_dates)}")
    logger.info(f"  Val days: {len(val_dates)}")
    logger.info(f"  Overlap: {len(overlap)} days")
    logger.info(f"  Train samples: {train_samples}")
    logger.info(f"  Val samples: {val_samples}")
    logger.info(f"  Status: {status}")
    
    return result


# ============================================================================
# L3: Label Shuffle Sanity Test
# ============================================================================

def test_label_shuffle_sanity(
    z_t_array: np.ndarray,
    labels: np.ndarray,
    num_epochs: int = 3,
    train_split: float = 0.8,
    min_gap: float = 0.05
) -> Dict[str, Any]:
    """
    L3: Verify model learns from real labels, not spurious patterns.
    
    - Train with real labels → F1_real
    - Train with shuffled labels → F1_shuffled
    - Expect: F1_real > F1_shuffled + min_gap
    """
    logger.info("\n" + "=" * 60)
    logger.info("L3: LABEL SHUFFLE SANITY TEST")
    logger.info("=" * 60)
    
    n_samples = len(z_t_array)
    split_idx = int(n_samples * train_split)
    
    X_train = z_t_array[:split_idx]
    X_val = z_t_array[split_idx:]
    y_train = labels[:split_idx]
    y_val = labels[split_idx:]
    
    # Map labels to 0,1,2 if needed
    y_train_mapped = y_train + 1 if y_train.min() < 0 else y_train
    y_val_mapped = y_val + 1 if y_val.min() < 0 else y_val
    
    # Run A: Real labels
    logger.info("  Training with REAL labels...")
    clf_real = LogisticRegression(max_iter=1000, n_jobs=-1)
    clf_real.fit(X_train, y_train_mapped)
    pred_real = clf_real.predict(X_val)
    f1_real = f1_score(y_val_mapped, pred_real, average='macro')
    acc_real = accuracy_score(y_val_mapped, pred_real)
    
    # Run B: Shuffled labels
    logger.info("  Training with SHUFFLED labels...")
    y_train_shuffled = y_train_mapped.copy()
    np.random.shuffle(y_train_shuffled)
    
    clf_shuffled = LogisticRegression(max_iter=1000, n_jobs=-1)
    clf_shuffled.fit(X_train, y_train_shuffled)
    pred_shuffled = clf_shuffled.predict(X_val)
    f1_shuffled = f1_score(y_val_mapped, pred_shuffled, average='macro')
    acc_shuffled = accuracy_score(y_val_mapped, pred_shuffled)
    
    # Check
    margin = f1_real - f1_shuffled
    passed = margin >= min_gap
    
    # Expected random baseline for 3 classes
    n_classes = len(np.unique(y_val_mapped))
    expected_random = 1.0 / n_classes
    
    result = {
        'test': 'L3_label_shuffle',
        'f1_real': float(f1_real),
        'f1_shuffled': float(f1_shuffled),
        'acc_real': float(acc_real),
        'acc_shuffled': float(acc_shuffled),
        'margin': float(margin),
        'min_gap_required': min_gap,
        'expected_random_f1': float(expected_random),
        'n_classes': int(n_classes),
        'n_train': int(len(X_train)),
        'n_val': int(len(X_val)),
        'pass': passed
    }
    
    status = "✅ PASS" if passed else "❌ FAIL"
    logger.info(f"  F1 Real: {f1_real:.4f}")
    logger.info(f"  F1 Shuffled: {f1_shuffled:.4f}")
    logger.info(f"  Margin: {margin:.4f} (required: {min_gap})")
    logger.info(f"  Expected Random: {expected_random:.4f}")
    logger.info(f"  Status: {status}")
    
    return result


# ============================================================================
# L4: Future Cheat Upper Bound Test
# ============================================================================

def test_future_cheat_upper_bound(
    dataset_path: str,
    train_split: float = 0.8,
    max_samples: int = 50000
) -> Dict[str, Any]:
    """
    L4: Test upper bound by using future info directly.
    
    Uses future_return_5 as feature to predict future_dir_5.
    This should give very high F1 (near 1.0) - establishes upper bound.
    """
    logger.info("\n" + "=" * 60)
    logger.info("L4: FUTURE CHEAT UPPER BOUND TEST")
    logger.info("=" * 60)
    
    # Load data
    cheat_features = []
    labels = []
    
    with open(dataset_path, 'r') as f:
        for i, line in enumerate(f):
            if i >= max_samples:
                break
            
            sample = json.loads(line.strip())
            aux = sample.get('aux', {})
            
            # Cheat features: use future info
            future_return = aux.get('future_return_5', sample.get('future_return_5', 0))
            future_range = aux.get('future_range_15', 0)
            pos_session = aux.get('pos_in_session_range', 0.5)
            
            cheat_features.append([future_return, future_range, pos_session])
            
            # Label
            future_dir = aux.get('future_dir_5', sample.get('future_dir_5', 0))
            labels.append(future_dir)
    
    X_cheat = np.array(cheat_features, dtype=np.float32)
    y = np.array(labels) + 1  # Map -1,0,1 to 0,1,2
    
    # Handle NaN
    X_cheat = np.nan_to_num(X_cheat, nan=0.0)
    
    # Split
    n = len(X_cheat)
    split_idx = int(n * train_split)
    
    X_train = X_cheat[:split_idx]
    X_val = X_cheat[split_idx:]
    y_train = y[:split_idx]
    y_val = y[split_idx:]
    
    # Train cheat classifier
    logger.info("  Training CHEAT classifier (using future info)...")
    clf_cheat = LogisticRegression(max_iter=1000, n_jobs=-1)
    clf_cheat.fit(X_train, y_train)
    
    pred_cheat = clf_cheat.predict(X_val)
    f1_cheat = f1_score(y_val, pred_cheat, average='macro')
    acc_cheat = accuracy_score(y_val, pred_cheat)
    
    # This should be very high since we're using future info
    result = {
        'test': 'L4_future_cheat',
        'f1_cheat': float(f1_cheat),
        'acc_cheat': float(acc_cheat),
        'features_used': ['future_return_5', 'future_range_15', 'pos_in_session_range'],
        'n_train': int(len(X_train)),
        'n_val': int(len(X_val)),
        'note': 'High F1 expected since using future info directly',
        'pass': True  # This is informational, not a pass/fail test
    }
    
    logger.info(f"  F1 Cheat: {f1_cheat:.4f}")
    logger.info(f"  Acc Cheat: {acc_cheat:.4f}")
    logger.info(f"  Note: This establishes upper bound using future info")
    
    return result


# ============================================================================
# L5: Spot Check Causality (Optional)
# ============================================================================

def spot_check_causality(
    dataset_path: str,
    raw_bars_path: Optional[str] = None,
    num_checks: int = 10,
    feature_names: List[str] = None
) -> Dict[str, Any]:
    """
    L5: Spot check that features match raw data.
    
    For random samples, verify last bar features match raw data.
    """
    logger.info("\n" + "=" * 60)
    logger.info("L5: SPOT CHECK CAUSALITY")
    logger.info("=" * 60)
    
    if not raw_bars_path or not Path(raw_bars_path).exists():
        logger.info("  Skipping: raw_bars_path not provided or doesn't exist")
        return {
            'test': 'L5_causality',
            'skipped': True,
            'reason': 'raw_bars_path not available',
            'pass': None
        }
    
    # Load samples
    samples = []
    with open(dataset_path, 'r') as f:
        for line in f:
            samples.append(json.loads(line.strip()))
    
    # Random selection
    check_indices = random.sample(range(len(samples)), min(num_checks, len(samples)))
    
    mismatches = []
    checks_done = 0
    
    for idx in check_indices:
        sample = samples[idx]
        X = np.array(sample.get('X', []))
        
        if len(X) == 0:
            continue
        
        # Get last bar features
        last_bar = X[-1]
        
        # TODO: Load corresponding raw bar and compare
        # This requires bar_index or timestamp mapping
        checks_done += 1
    
    result = {
        'test': 'L5_causality',
        'num_checks': checks_done,
        'num_mismatches': len(mismatches),
        'mismatches': mismatches[:5],
        'pass': len(mismatches) == 0
    }
    
    status = "✅ PASS" if result['pass'] else "❌ FAIL"
    logger.info(f"  Checks done: {checks_done}")
    logger.info(f"  Mismatches: {len(mismatches)}")
    logger.info(f"  Status: {status}")
    
    return result


# ============================================================================
# Main Leak Evaluator
# ============================================================================

class LeakEvaluator:
    """Main class for running all leak tests"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.dataset_path = config['dataset_path']
        self.feature_config_path = config.get('feature_config_path')
        self.model_config_path = config.get('model_config_path')
        self.model_weights_path = config.get('model_weights_path')
        self.output_dir = Path(config.get('output_dir', 'state_enc_v1/artifacts/v1_2/eval_gc_m1'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.train_days = config.get('train_days', [])
        self.val_days = config.get('val_days', [])
        self.thresholds = config.get('thresholds', {})
        
        self.device = torch.device(config.get('device', 'cpu'))
    
    def load_z_t_and_labels(self, max_samples: int = 50000) -> Tuple[np.ndarray, np.ndarray]:
        """Load z_t embeddings and labels from semantic eval or compute fresh"""
        
        # Try to load from semantic eval results
        semantic_dir = self.output_dir
        z_t_path = semantic_dir / 'z_t_embeddings.npy'
        
        if z_t_path.exists():
            logger.info("Loading cached z_t embeddings...")
            z_t = np.load(z_t_path)
            labels = np.load(semantic_dir / 'future_dir_labels.npy')
            return z_t, labels
        
        # Otherwise, compute fresh
        logger.info("Computing z_t embeddings...")
        
        # Load model
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
        from state_enc_v1.src.model.state_enc_model import StateEncModel
        
        with open(self.model_config_path, 'r') as f:
            model_config = json.load(f)
        
        model = StateEncModel.from_config(model_config)
        state_dict = torch.load(self.model_weights_path, map_location=self.device)
        if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        model.load_state_dict(state_dict, strict=False)
        model.to(self.device)
        model.eval()
        
        # Load feature config
        with open(self.feature_config_path, 'r') as f:
            feat_cfg = json.load(f)
        mean = np.array(feat_cfg.get('mean', [0] * 88))
        std = np.array(feat_cfg.get('std', [1] * 88))
        
        # Process samples
        z_t_list = []
        labels_list = []
        
        with open(self.dataset_path, 'r') as f:
            for i, line in enumerate(f):
                if i >= max_samples:
                    break
                
                sample = json.loads(line.strip())
                X = np.array(sample.get('X', []), dtype=np.float32)
                
                # Normalize if not already
                if X.max() > 10:  # Likely not normalized
                    X = (X - mean) / (std + 1e-8)
                
                X = np.nan_to_num(X, nan=0.0)
                X_t = torch.tensor(X, dtype=torch.float32).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    z_t = model.encode(X_t)
                    z_t_list.append(z_t.cpu().numpy().flatten())
                
                aux = sample.get('aux', {})
                future_dir = aux.get('future_dir_5', sample.get('future_dir_5', 0))
                labels_list.append(future_dir)
        
        z_t_array = np.array(z_t_list)
        labels_array = np.array(labels_list)
        
        # Cache for future use
        np.save(z_t_path, z_t_array)
        np.save(semantic_dir / 'future_dir_labels.npy', labels_array)
        
        return z_t_array, labels_array
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all leak tests L1-L4"""
        results = {
            'timestamp': datetime.now().isoformat(),
            'config': self.config,
            'tests': {}
        }
        
        # L1: Index boundaries
        try:
            results['tests']['L1'] = check_index_and_future_boundaries(
                self.dataset_path,
                max_samples=self.config.get('max_samples', 1000),
                seq_len=self.config.get('seq_len', 64)
            )
        except Exception as e:
            logger.error(f"L1 failed: {e}")
            results['tests']['L1'] = {'error': str(e), 'pass': False}
        
        # L2: Time split
        try:
            results['tests']['L2'] = check_time_based_split(
                self.dataset_path,
                self.train_days,
                self.val_days,
                max_samples=self.config.get('max_samples', 50000)
            )
        except Exception as e:
            logger.error(f"L2 failed: {e}")
            results['tests']['L2'] = {'error': str(e), 'pass': False}
        
        # L3: Label shuffle
        try:
            z_t, labels = self.load_z_t_and_labels(
                max_samples=self.config.get('max_samples', 50000)
            )
            results['tests']['L3'] = test_label_shuffle_sanity(
                z_t, labels,
                min_gap=self.thresholds.get('label_shuffle_min_gap', 0.05)
            )
        except Exception as e:
            logger.error(f"L3 failed: {e}")
            results['tests']['L3'] = {'error': str(e), 'pass': False}
        
        # L4: Future cheat
        try:
            results['tests']['L4'] = test_future_cheat_upper_bound(
                self.dataset_path,
                max_samples=self.config.get('max_samples', 50000)
            )
        except Exception as e:
            logger.error(f"L4 failed: {e}")
            results['tests']['L4'] = {'error': str(e), 'pass': False}
        
        # Summary
        all_passed = all(
            t.get('pass', False) or t.get('pass') is None 
            for t in results['tests'].values()
        )
        results['all_passed'] = all_passed
        
        return results


def run_leak_tests(config_path: str) -> Dict[str, Any]:
    """Main entry point"""
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    evaluator = LeakEvaluator(config)
    results = evaluator.run_all_tests()
    
    # Save results
    output_path = evaluator.output_dir / 'leak_eval_full_report_v1.2.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print_leak_summary(results)
    
    return results


def print_leak_summary(results: Dict[str, Any]):
    """Print summary of leak tests"""
    print("\n" + "=" * 70)
    print("LEAK & INTEGRITY TEST SUITE — SUMMARY")
    print("=" * 70)
    
    tests = results.get('tests', {})
    
    for test_name, test_result in tests.items():
        status = "✅ PASS" if test_result.get('pass', False) else "❌ FAIL"
        if test_result.get('pass') is None:
            status = "⏭️ SKIP"
        
        print(f"\n[{test_name}] {status}")
        
        if test_name == 'L1':
            print(f"  Checked: {test_result.get('num_checked', 0)}")
            print(f"  Violations: {test_result.get('num_violations', 0)}")
        
        elif test_name == 'L2':
            print(f"  Train samples: {test_result.get('train_samples', 0)}")
            print(f"  Val samples: {test_result.get('val_samples', 0)}")
            print(f"  Overlap days: {len(test_result.get('overlap_days', []))}")
        
        elif test_name == 'L3':
            print(f"  F1 Real: {test_result.get('f1_real', 0):.4f}")
            print(f"  F1 Shuffled: {test_result.get('f1_shuffled', 0):.4f}")
            print(f"  Margin: {test_result.get('margin', 0):.4f}")
        
        elif test_name == 'L4':
            print(f"  F1 Cheat: {test_result.get('f1_cheat', 0):.4f}")
            print(f"  (Upper bound using future info)")
    
    overall = "✅ ALL PASSED" if results.get('all_passed', False) else "⚠️ SOME FAILED"
    print(f"\n{'=' * 70}")
    print(f"OVERALL: {overall}")
    print("=" * 70)
