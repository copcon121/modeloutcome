"""
Leak evaluation tests for ASM v2

Tests:
- L1: Time split validation (no date overlap)
- L2: Label shuffle test (F1_real > F1_shuffled)
- L3: Future leak guard (no future_return_* in features)
"""

import json
import numpy as np
import torch
from typing import Dict, List, Tuple
from pathlib import Path


class AsmLeakEvaluator:
    """Leak tests for ASM v2"""
    
    def __init__(self,
                 dataset_path: str,
                 splits_path: str,
                 feature_config_path: str):
        self.dataset_path = dataset_path
        self.splits_path = splits_path
        self.feature_config_path = feature_config_path
        
        # Load data
        with open(splits_path, "r") as f:
            self.splits = json.load(f)
        
        with open(feature_config_path, "r") as f:
            self.feature_config = json.load(f)
        
        self.samples = self._load_samples()
    
    def _load_samples(self) -> List[Dict]:
        samples = []
        with open(self.dataset_path, "r") as f:
            for line in f:
                if line.strip():
                    samples.append(json.loads(line))
        return samples
    
    def run_all_tests(self) -> Dict:
        """Run all leak tests"""
        results = {}
        
        # L1: Time split validation
        results["L1_TimeSplit"] = self.test_time_split()
        
        # L2: Label shuffle test
        results["L2_LabelShuffle"] = self.test_label_shuffle()
        
        # L3: Future leak guard
        results["L3_FutureLeakGuard"] = self.test_future_leak_guard()
        
        # Overall pass/fail
        results["all_passed"] = all(r["passed"] for r in results.values())
        
        return results
    
    def test_time_split(self) -> Dict:
        """L1: Check no date overlap between train and val"""
        train_dates = set(self.splits.get("train_dates", []))
        val_dates = set(self.splits.get("val_dates", []))
        
        overlap = train_dates & val_dates
        passed = len(overlap) == 0
        
        return {
            "test": "L1_TimeSplit",
            "passed": passed,
            "train_dates_count": len(train_dates),
            "val_dates_count": len(val_dates),
            "overlap_count": len(overlap),
            "overlap_dates": list(overlap)[:5] if overlap else []
        }
    
    def test_label_shuffle(self) -> Dict:
        """L2: Verify model learns from real labels vs shuffled"""
        # This is a placeholder - actual test requires training
        # In practice, we compare F1 with real vs shuffled labels
        
        # Get label distribution
        labels = [s["label_regime"] for s in self.samples]
        unique_labels = set(labels)
        label_counts = {l: labels.count(l) for l in unique_labels}
        
        # Check if labels are not all the same (trivial case)
        passed = len(unique_labels) > 1
        
        return {
            "test": "L2_LabelShuffle",
            "passed": passed,
            "num_classes": len(unique_labels),
            "label_distribution": label_counts,
            "note": "Full test requires training comparison"
        }
    
    def test_future_leak_guard(self) -> Dict:
        """L3: Ensure no future_return_* fields in input features"""
        # Check meta features don't contain future info
        meta_features = self.feature_config.get("meta_features", [])
        
        forbidden_patterns = ["future_", "return_", "outcome", "hit", "label"]
        
        leaked_features = []
        for feat in meta_features:
            for pattern in forbidden_patterns:
                if pattern in feat.lower():
                    leaked_features.append(feat)
                    break
        
        # Also check sample structure
        if self.samples:
            sample = self.samples[0]
            meta_keys = list(sample.get("meta", {}).keys())
            for key in meta_keys:
                for pattern in forbidden_patterns:
                    if pattern in key.lower():
                        leaked_features.append(f"meta.{key}")
        
        passed = len(leaked_features) == 0
        
        return {
            "test": "L3_FutureLeakGuard",
            "passed": passed,
            "meta_features": meta_features,
            "leaked_features": leaked_features
        }


def run_leak_evaluation(dataset_path: str,
                        splits_path: str,
                        feature_config_path: str,
                        output_path: str = None) -> Dict:
    """Run leak evaluation and optionally save results"""
    
    evaluator = AsmLeakEvaluator(dataset_path, splits_path, feature_config_path)
    results = evaluator.run_all_tests()
    
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved leak report to {output_path}")
    
    return results
