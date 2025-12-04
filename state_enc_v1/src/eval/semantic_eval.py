"""
Semantic Evaluation for STATE-ENC v1.2

Tests:
- S1: Linear Probe Future_Dir_5
- S2: Regime Probe (regime_hint)
- S3: Cluster & Market Behavior Map
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, classification_report
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Dataset for Semantic Evaluation
# ============================================================================

class SemanticEvalDataset(Dataset):
    """Dataset for semantic evaluation from encoder_dataset_real.jsonl"""
    
    def __init__(self, data_path: str, feature_config_path: str, 
                 max_samples: Optional[int] = None):
        self.samples = []
        self.feature_names = []
        self.feature_dim = 0
        
        # Load feature config
        with open(feature_config_path, 'r') as f:
            feat_cfg = json.load(f)
        self.feature_names = feat_cfg.get('feature_names', [])
        self.feature_dim = feat_cfg.get('feature_dim', len(self.feature_names))
        self.mean = np.array(feat_cfg.get('mean', [0] * self.feature_dim))
        self.std = np.array(feat_cfg.get('std', [1] * self.feature_dim))
        
        # Load samples
        logger.info(f"Loading data from {data_path}")
        with open(data_path, 'r') as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                sample = json.loads(line.strip())
                self.samples.append(sample)
        
        logger.info(f"Loaded {len(self.samples)} samples")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        
        # Get features
        X = np.array(sample.get('X', sample.get('features', [])), dtype=np.float32)
        
        # Normalize
        if len(X.shape) == 2:
            X = (X - self.mean) / (self.std + 1e-8)
        
        # Get labels from aux or direct fields
        aux = sample.get('aux', {})
        future_dir_5 = aux.get('future_dir_5', sample.get('future_dir_5', 0))
        future_return_5 = aux.get('future_return_5', sample.get('future_return_5', 0.0))
        regime_hint = aux.get('asm_regime_hint', sample.get('regime_hint', 0))
        
        # Additional aux fields
        future_range_15 = aux.get('future_range_15', 0.0)
        pos_in_session = aux.get('pos_in_session_range', 0.5)
        
        return {
            'X': torch.tensor(X, dtype=torch.float32),
            'future_dir_5': int(future_dir_5),
            'future_return_5': float(future_return_5),
            'regime_hint': int(regime_hint),
            'future_range_15': float(future_range_15),
            'pos_in_session': float(pos_in_session),
            'idx': idx
        }
    
    def get_raw_features(self, idx: int, feature_subset: List[str]) -> np.ndarray:
        """Get raw features for baseline comparison"""
        sample = self.samples[idx]
        X = np.array(sample.get('X', sample.get('features', [])), dtype=np.float32)
        
        # Get last bar features
        if len(X.shape) == 2:
            last_bar = X[-1]
        else:
            last_bar = X
        
        # Extract subset
        subset_values = []
        for fname in feature_subset:
            if fname in self.feature_names:
                fidx = self.feature_names.index(fname)
                if fidx < len(last_bar):
                    subset_values.append(last_bar[fidx])
                else:
                    subset_values.append(0.0)
            else:
                subset_values.append(0.0)
        
        return np.array(subset_values, dtype=np.float32)


# ============================================================================
# Simple MLP Classifier for Probing
# ============================================================================

class MLPClassifier(nn.Module):
    """Simple MLP for probing tasks"""
    
    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ============================================================================
# Semantic Evaluator Main Class
# ============================================================================

class SemanticEvaluator:
    """Main evaluator for semantic tests S1, S2, S3"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device(config.get('device', 'cpu'))
        
        # Paths
        self.model_path = config['model_path']
        self.model_config_path = config['model_config_path']
        self.feature_config_path = config['feature_config_path']
        self.dataset_path = config['dataset_path']
        self.output_dir = Path(config.get('output_dir', 'state_enc_v1/artifacts/v1_2/eval'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Parameters
        self.batch_size = config.get('batch_size', 64)
        self.max_samples = config.get('max_samples', 50000)
        self.train_split = config.get('train_split', 0.8)
        self.kmeans_k = config.get('kmeans_k', 8)
        
        # Baseline features for S1
        self.baseline_features = config.get('baseline_features', [
            'c', 'hl_range', 'body', 'delta', 'volume',
            'premium_zone', 'inside_value', 'dist_to_vah', 'dist_to_val',
            'ext_trend_dir', 'int_trend_dir', 'asm_regime_hint'
        ])
        
        # Load model
        self.model = self._load_model()
        self.model.eval()
        
        # Load dataset
        self.dataset = SemanticEvalDataset(
            self.dataset_path, 
            self.feature_config_path,
            self.max_samples
        )

    def _load_model(self):
        """Load STATE-ENC model"""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
        
        from state_enc_v1.src.model.state_enc_model import StateEncModel
        
        with open(self.model_config_path, 'r') as f:
            model_config = json.load(f)
        
        model = StateEncModel.from_config(model_config)
        
        # Load weights
        state_dict = torch.load(self.model_path, map_location=self.device)
        if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        model.load_state_dict(state_dict, strict=False)
        model.to(self.device)
        
        logger.info(f"Loaded model from {self.model_path}")
        return model
    
    @torch.no_grad()
    def encode_all(self) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Encode all samples and return z_t embeddings + labels"""
        loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)
        
        all_z_t = []
        all_labels = defaultdict(list)
        all_raw_features = []
        
        logger.info("Encoding all samples...")
        for batch in loader:
            X = batch['X'].to(self.device)
            
            # Handle NaN
            X = torch.nan_to_num(X, nan=0.0)
            
            # Encode
            z_t = self.model.encode(X)
            all_z_t.append(z_t.cpu().numpy())
            
            # Collect labels
            all_labels['future_dir_5'].extend(batch['future_dir_5'].numpy())
            all_labels['future_return_5'].extend(batch['future_return_5'].numpy())
            all_labels['regime_hint'].extend(batch['regime_hint'].numpy())
            all_labels['future_range_15'].extend(batch['future_range_15'].numpy())
            all_labels['pos_in_session'].extend(batch['pos_in_session'].numpy())
        
        # Get raw features for baseline
        logger.info("Extracting baseline features...")
        for i in range(len(self.dataset)):
            raw = self.dataset.get_raw_features(i, self.baseline_features)
            all_raw_features.append(raw)
        
        z_t_array = np.vstack(all_z_t)
        raw_array = np.array(all_raw_features)
        labels = {k: np.array(v) for k, v in all_labels.items()}
        
        logger.info(f"Encoded {len(z_t_array)} samples, z_t shape: {z_t_array.shape}")
        
        return z_t_array, raw_array, labels
    
    def time_split(self, n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """Time-based train/val split (80% train, 20% val)"""
        split_idx = int(n_samples * self.train_split)
        train_idx = np.arange(split_idx)
        val_idx = np.arange(split_idx, n_samples)
        return train_idx, val_idx


# ============================================================================
# TEST S1: Linear Probe Future_Dir_5
# ============================================================================

def run_future_dir_probe(evaluator: SemanticEvaluator) -> Dict[str, Any]:
    """
    S1: Compare baseline raw features vs z_t for predicting future_dir_5
    """
    logger.info("=" * 60)
    logger.info("TEST S1: LINEAR PROBE FUTURE_DIR_5")
    logger.info("=" * 60)
    
    # Encode all
    z_t, raw_features, labels = evaluator.encode_all()
    future_dir = labels['future_dir_5']
    
    # Map -1,0,1 to 0,1,2 for classification
    future_dir_mapped = future_dir + 1  # -1->0, 0->1, 1->2
    
    # Time split
    train_idx, val_idx = evaluator.time_split(len(z_t))
    
    results = {}
    
    # --- Baseline Raw ---
    logger.info("\nTraining Baseline-Raw classifier...")
    X_train_raw = raw_features[train_idx]
    X_val_raw = raw_features[val_idx]
    y_train = future_dir_mapped[train_idx]
    y_val = future_dir_mapped[val_idx]
    
    # Handle NaN
    X_train_raw = np.nan_to_num(X_train_raw, nan=0.0)
    X_val_raw = np.nan_to_num(X_val_raw, nan=0.0)
    
    clf_raw = LogisticRegression(max_iter=1000, multi_class='multinomial', n_jobs=-1)
    clf_raw.fit(X_train_raw, y_train)
    
    pred_raw = clf_raw.predict(X_val_raw)
    acc_raw = accuracy_score(y_val, pred_raw)
    f1_raw = f1_score(y_val, pred_raw, average='macro')
    
    results['baseline_raw'] = {
        'accuracy': float(acc_raw),
        'macro_f1': float(f1_raw),
        'features_used': evaluator.baseline_features,
        'n_train': int(len(train_idx)),
        'n_val': int(len(val_idx))
    }
    
    logger.info(f"  Baseline-Raw: Acc={acc_raw:.4f}, F1={f1_raw:.4f}")

    # --- Probe z_t ---
    logger.info("\nTraining Probe-z_t classifier...")
    X_train_z = z_t[train_idx]
    X_val_z = z_t[val_idx]
    
    clf_z = LogisticRegression(max_iter=1000, multi_class='multinomial', n_jobs=-1)
    clf_z.fit(X_train_z, y_train)
    
    pred_z = clf_z.predict(X_val_z)
    acc_z = accuracy_score(y_val, pred_z)
    f1_z = f1_score(y_val, pred_z, average='macro')
    
    results['probe_z_t'] = {
        'accuracy': float(acc_z),
        'macro_f1': float(f1_z),
        'z_t_dim': int(z_t.shape[1]),
        'n_train': int(len(train_idx)),
        'n_val': int(len(val_idx))
    }
    
    logger.info(f"  Probe-z_t: Acc={acc_z:.4f}, F1={f1_z:.4f}")
    
    # --- Summary ---
    improvement_acc = (acc_z - acc_raw) / (acc_raw + 1e-8) * 100
    improvement_f1 = (f1_z - f1_raw) / (f1_raw + 1e-8) * 100
    
    results['comparison'] = {
        'accuracy_improvement_pct': float(improvement_acc),
        'f1_improvement_pct': float(improvement_f1),
        'z_t_better': bool(f1_z > f1_raw)
    }
    
    logger.info(f"\n  Improvement: Acc={improvement_acc:+.2f}%, F1={improvement_f1:+.2f}%")
    
    # Save
    output_path = evaluator.output_dir / 'semantic_probe_future_dir.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"  Saved to {output_path}")
    
    return results


# ============================================================================
# TEST S2: Regime Probe
# ============================================================================

def run_regime_probe(evaluator: SemanticEvaluator) -> Dict[str, Any]:
    """
    S2: Probe z_t for predicting regime_hint
    """
    logger.info("\n" + "=" * 60)
    logger.info("TEST S2: REGIME PROBE (REGIME_HINT)")
    logger.info("=" * 60)
    
    # Encode all
    z_t, _, labels = evaluator.encode_all()
    regime_hint = labels['regime_hint']
    
    # Filter out UNKNOWN (0)
    valid_mask = regime_hint != 0
    z_t_valid = z_t[valid_mask]
    regime_valid = regime_hint[valid_mask]
    
    logger.info(f"  Valid samples (regime != 0): {len(z_t_valid)}/{len(z_t)}")
    
    if len(z_t_valid) < 100:
        logger.warning("  Not enough valid samples for regime probe!")
        return {'error': 'insufficient_samples', 'n_valid': len(z_t_valid)}
    
    # Time split on valid samples
    n_valid = len(z_t_valid)
    split_idx = int(n_valid * evaluator.train_split)
    
    X_train = z_t_valid[:split_idx]
    X_val = z_t_valid[split_idx:]
    y_train = regime_valid[:split_idx]
    y_val = regime_valid[split_idx:]
    
    # Train MLP classifier
    logger.info("\nTraining regime classifier...")
    
    num_classes = int(regime_valid.max()) + 1
    clf = LogisticRegression(max_iter=1000, multi_class='multinomial', n_jobs=-1)
    clf.fit(X_train, y_train)
    
    pred = clf.predict(X_val)
    acc = accuracy_score(y_val, pred)
    f1 = f1_score(y_val, pred, average='macro')
    cm = confusion_matrix(y_val, pred)
    
    results = {
        'accuracy': float(acc),
        'macro_f1': float(f1),
        'n_train': int(len(X_train)),
        'n_val': int(len(X_val)),
        'num_classes': int(num_classes),
        'confusion_matrix': cm.tolist(),
        'class_distribution': {
            str(k): int(v) for k, v in zip(*np.unique(y_val, return_counts=True))
        }
    }
    
    logger.info(f"  Regime Probe: Acc={acc:.4f}, F1={f1:.4f}")
    logger.info(f"  Classes: {num_classes}, Distribution: {results['class_distribution']}")
    
    # Save
    output_path = evaluator.output_dir / 'semantic_probe_regime.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"  Saved to {output_path}")
    
    return results


# ============================================================================
# TEST S3: Cluster & Market Behavior Map
# ============================================================================

def run_cluster_analysis(evaluator: SemanticEvaluator) -> Dict[str, Any]:
    """
    S3: KMeans clustering on z_t and analyze market behavior per cluster
    """
    logger.info("\n" + "=" * 60)
    logger.info("TEST S3: CLUSTER & MARKET BEHAVIOR MAP")
    logger.info("=" * 60)
    
    # Encode all
    z_t, _, labels = evaluator.encode_all()
    
    # Run KMeans
    k = evaluator.kmeans_k
    logger.info(f"\nRunning KMeans with K={k}...")
    
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(z_t)
    
    # Analyze each cluster
    clusters = {}
    
    for cluster_id in range(k):
        mask = cluster_labels == cluster_id
        n_samples = int(mask.sum())
        
        cluster_info = {
            'num_samples': n_samples,
            'pct_of_total': float(n_samples / len(z_t) * 100)
        }
        
        # Mean future return
        future_return = labels['future_return_5'][mask]
        cluster_info['mean_future_return_5'] = float(np.mean(future_return))
        cluster_info['std_future_return_5'] = float(np.std(future_return))
        
        # Mean future range (if available)
        future_range = labels['future_range_15'][mask]
        if np.any(future_range != 0):
            cluster_info['mean_future_range_15'] = float(np.mean(future_range))
        
        # Position in session
        pos_session = labels['pos_in_session'][mask]
        cluster_info['mean_pos_in_session'] = float(np.mean(pos_session))

        # Regime distribution
        regime = labels['regime_hint'][mask]
        regime_counts = {}
        for r in np.unique(regime):
            count = int((regime == r).sum())
            regime_counts[str(int(r))] = {
                'count': count,
                'pct': float(count / n_samples * 100)
            }
        cluster_info['regime_distribution'] = regime_counts
        
        # Future direction distribution
        future_dir = labels['future_dir_5'][mask]
        dir_counts = {}
        for d in [-1, 0, 1]:
            count = int((future_dir == d).sum())
            dir_counts[str(d)] = {
                'count': count,
                'pct': float(count / n_samples * 100) if n_samples > 0 else 0
            }
        cluster_info['future_dir_distribution'] = dir_counts
        
        # Dominant characteristics
        dominant_regime = max(regime_counts.items(), key=lambda x: x[1]['pct'])[0]
        dominant_dir = max(dir_counts.items(), key=lambda x: x[1]['pct'])[0]
        
        cluster_info['dominant_regime'] = int(dominant_regime)
        cluster_info['dominant_future_dir'] = int(dominant_dir)
        
        clusters[f'cluster_{cluster_id}'] = cluster_info
    
    # Generate cluster descriptions
    descriptions = []
    regime_names = {0: 'UNKNOWN', 1: 'CHOP', 2: 'BULL', 3: 'BEAR', 4: 'DRIVE', 5: 'REVERSAL'}
    dir_names = {-1: 'DOWN', 0: 'FLAT', 1: 'UP'}
    
    for cid, info in clusters.items():
        regime_name = regime_names.get(info['dominant_regime'], f"R{info['dominant_regime']}")
        dir_name = dir_names.get(info['dominant_future_dir'], 'FLAT')
        pos = info['mean_pos_in_session']
        ret = info['mean_future_return_5']
        
        pos_desc = 'near_high' if pos > 0.7 else ('near_low' if pos < 0.3 else 'mid_range')
        ret_desc = 'positive_return' if ret > 0.001 else ('negative_return' if ret < -0.001 else 'neutral')
        
        desc = f"{cid}: {regime_name}, {dir_name}, {pos_desc}, {ret_desc} ({info['num_samples']} samples)"
        descriptions.append(desc)
        info['description'] = desc
    
    results = {
        'kmeans_k': k,
        'total_samples': len(z_t),
        'clusters': clusters,
        'cluster_descriptions': descriptions,
        'inertia': float(kmeans.inertia_)
    }
    
    logger.info(f"\nCluster Analysis Results:")
    for desc in descriptions:
        logger.info(f"  {desc}")
    
    # Save
    output_path = evaluator.output_dir / 'semantic_clusters.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"\n  Saved to {output_path}")
    
    return results


# ============================================================================
# Main Runner
# ============================================================================

def run_all_semantic_tests(config_path: str) -> Dict[str, Any]:
    """Run all semantic tests S1, S2, S3"""
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    evaluator = SemanticEvaluator(config)
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'config': config
    }
    
    # S1: Future Dir Probe
    try:
        results['S1_future_dir_probe'] = run_future_dir_probe(evaluator)
    except Exception as e:
        logger.error(f"S1 failed: {e}")
        results['S1_future_dir_probe'] = {'error': str(e)}
    
    # S2: Regime Probe
    try:
        results['S2_regime_probe'] = run_regime_probe(evaluator)
    except Exception as e:
        logger.error(f"S2 failed: {e}")
        results['S2_regime_probe'] = {'error': str(e)}
    
    # S3: Cluster Analysis
    try:
        results['S3_cluster_analysis'] = run_cluster_analysis(evaluator)
    except Exception as e:
        logger.error(f"S3 failed: {e}")
        results['S3_cluster_analysis'] = {'error': str(e)}
    
    # Save full report
    output_path = evaluator.output_dir / 'semantic_eval_full_report.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print_summary(results)
    
    return results


def print_summary(results: Dict[str, Any]):
    """Print summary table"""
    print("\n" + "=" * 70)
    print("SEMANTIC EVALUATION SUMMARY — STATE-ENC v1.2")
    print("=" * 70)
    
    # S1 Summary
    s1 = results.get('S1_future_dir_probe', {})
    if 'error' not in s1:
        print("\n[S1] FUTURE_DIR_5 PROBE:")
        print("-" * 50)
        baseline = s1.get('baseline_raw', {})
        probe = s1.get('probe_z_t', {})
        comp = s1.get('comparison', {})
        
        print(f"  {'Method':<20} {'Accuracy':<12} {'Macro F1':<12}")
        print(f"  {'-'*44}")
        print(f"  {'Baseline-Raw':<20} {baseline.get('accuracy', 0):<12.4f} {baseline.get('macro_f1', 0):<12.4f}")
        print(f"  {'Probe-z_t':<20} {probe.get('accuracy', 0):<12.4f} {probe.get('macro_f1', 0):<12.4f}")
        print(f"\n  z_t improvement: {comp.get('f1_improvement_pct', 0):+.2f}% F1")
    
    # S2 Summary
    s2 = results.get('S2_regime_probe', {})
    if 'error' not in s2:
        print("\n[S2] REGIME PROBE:")
        print("-" * 50)
        print(f"  Accuracy: {s2.get('accuracy', 0):.4f}")
        print(f"  Macro F1: {s2.get('macro_f1', 0):.4f}")
        print(f"  Classes: {s2.get('num_classes', 0)}")
    
    # S3 Summary
    s3 = results.get('S3_cluster_analysis', {})
    if 'error' not in s3:
        print("\n[S3] CLUSTER ANALYSIS:")
        print("-" * 50)
        print(f"  K={s3.get('kmeans_k', 0)}, Total samples={s3.get('total_samples', 0)}")
        print("\n  Cluster Descriptions:")
        for desc in s3.get('cluster_descriptions', [])[:5]:
            print(f"    {desc}")
    
    print("\n" + "=" * 70)
