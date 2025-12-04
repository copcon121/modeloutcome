"""Evaluation metrics for ASM v2"""

import torch
import numpy as np
from typing import Dict, List, Tuple
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


def compute_accuracy(preds: np.ndarray, targets: np.ndarray) -> float:
    """Compute accuracy"""
    return accuracy_score(targets, preds)


def compute_macro_f1(preds: np.ndarray, targets: np.ndarray) -> float:
    """Compute macro F1 score"""
    return f1_score(targets, preds, average="macro", zero_division=0)


def compute_per_class_f1(preds: np.ndarray, targets: np.ndarray, num_classes: int) -> Dict[int, float]:
    """Compute per-class F1 scores"""
    f1s = f1_score(targets, preds, average=None, zero_division=0)
    return {i: float(f1s[i]) if i < len(f1s) else 0.0 for i in range(num_classes)}


def compute_confusion_matrix(preds: np.ndarray, targets: np.ndarray, num_classes: int) -> np.ndarray:
    """Compute confusion matrix"""
    return confusion_matrix(targets, preds, labels=list(range(num_classes)))


class AsmMetrics:
    """Metrics tracker for ASM v2"""
    
    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.reset()
    
    def reset(self):
        self.all_preds = []
        self.all_targets = []
        self.total_loss = 0.0
        self.num_batches = 0
    
    def update(self, preds: torch.Tensor, targets: torch.Tensor, loss: float = 0.0):
        """Update metrics with batch results"""
        self.all_preds.extend(preds.cpu().numpy().tolist())
        self.all_targets.extend(targets.cpu().numpy().tolist())
        self.total_loss += loss
        self.num_batches += 1
    
    def compute(self) -> Dict[str, float]:
        """Compute all metrics"""
        preds = np.array(self.all_preds)
        targets = np.array(self.all_targets)
        
        acc = compute_accuracy(preds, targets)
        macro_f1 = compute_macro_f1(preds, targets)
        per_class_f1 = compute_per_class_f1(preds, targets, self.num_classes)
        cm = compute_confusion_matrix(preds, targets, self.num_classes)
        
        avg_loss = self.total_loss / max(self.num_batches, 1)
        
        return {
            "accuracy": acc,
            "macro_f1": macro_f1,
            "per_class_f1": per_class_f1,
            "confusion_matrix": cm.tolist(),
            "avg_loss": avg_loss,
            "num_samples": len(preds)
        }
