"""
Semantic evaluation for ASM v2

Analyzes model predictions across sessions, regimes, etc.
"""

import json
import numpy as np
import torch
from typing import Dict, List, Optional
from pathlib import Path
from collections import defaultdict


class AsmSemanticEvaluator:
    """Semantic evaluation for ASM v2"""
    
    def __init__(self,
                 model,
                 dataset_path: str,
                 feature_config_path: str,
                 device: str = "cuda"):
        self.model = model
        self.device = device
        self.dataset_path = dataset_path
        
        with open(feature_config_path, "r") as f:
            self.feature_config = json.load(f)
        
        self.idx_to_regime = self.feature_config.get("idx_to_regime", {})
        self.regime_names = self.feature_config.get("regime_names", {})
        
        self.samples = self._load_samples()
    
    def _load_samples(self) -> List[Dict]:
        samples = []
        with open(self.dataset_path, "r") as f:
            for line in f:
                if line.strip():
                    samples.append(json.loads(line))
        return samples
    
    def evaluate(self) -> Dict:
        """Run semantic evaluation"""
        self.model.eval()
        
        results = {
            "per_session": defaultdict(lambda: {"correct": 0, "total": 0}),
            "per_regime": defaultdict(lambda: {"correct": 0, "total": 0}),
            "confusion_by_session": defaultdict(lambda: defaultdict(int))
        }
        
        session_map = {0: "ASIA", 1: "LDN", 2: "NY"}
        
        with torch.no_grad():
            for sample in self.samples:
                # Prepare input
                z_t = torch.tensor(sample["z_t"], dtype=torch.float32)
                meta = sample["meta"]
                meta_vec = torch.tensor([
                    meta.get("session_id", 0),
                    meta.get("pos_in_session_range", 0.5),
                    meta.get("inside_value", 0),
                    meta.get("above_value", 0),
                    meta.get("below_value", 0),
                    meta.get("minute_of_day_norm", 0.0)
                ], dtype=torch.float32)
                
                x = torch.cat([z_t, meta_vec]).unsqueeze(0).to(self.device)
                
                # Predict
                outputs = self.model(x)
                pred = outputs["logits"].argmax(dim=-1).item()
                
                # Ground truth
                label = sample["label_regime"]
                
                # Session
                session_id = int(meta.get("session_id", 0))
                session_name = session_map.get(session_id, "UNKNOWN")
                
                # Update stats
                results["per_session"][session_name]["total"] += 1
                if pred == label:
                    results["per_session"][session_name]["correct"] += 1
                
                regime_name = self.regime_names.get(str(sample.get("regime_hint_raw", label)), f"regime_{label}")
                results["per_regime"][regime_name]["total"] += 1
                if pred == label:
                    results["per_regime"][regime_name]["correct"] += 1
                
                # Confusion by session
                results["confusion_by_session"][session_name][f"{label}->{pred}"] += 1
        
        # Compute accuracies
        summary = {
            "per_session_accuracy": {},
            "per_regime_accuracy": {},
            "total_samples": len(self.samples)
        }
        
        for session, stats in results["per_session"].items():
            if stats["total"] > 0:
                summary["per_session_accuracy"][session] = {
                    "accuracy": stats["correct"] / stats["total"],
                    "total": stats["total"]
                }
        
        for regime, stats in results["per_regime"].items():
            if stats["total"] > 0:
                summary["per_regime_accuracy"][regime] = {
                    "accuracy": stats["correct"] / stats["total"],
                    "total": stats["total"]
                }
        
        return summary
