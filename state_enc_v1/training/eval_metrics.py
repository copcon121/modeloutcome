"""
Evaluation metrics for STATE-ENC v1

Metrics for:
- Direction classification (accuracy, F1, confusion matrix)
- Regime classification
- Return prediction (MAE, correlation)
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import defaultdict


def compute_accuracy(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Compute classification accuracy.
    
    Args:
        preds: [B] predicted class indices
        targets: [B] target class indices
        
    Returns:
        Accuracy as float
    """
    if len(preds) == 0:
        return 0.0
    correct = (preds == targets).sum().item()
    return correct / len(preds)


def compute_accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Compute accuracy from logits.
    
    Args:
        logits: [B, C] class logits
        targets: [B] target indices
    """
    preds = logits.argmax(dim=-1)
    return compute_accuracy(preds, targets)


def compute_f1_score(preds: torch.Tensor, 
                     targets: torch.Tensor,
                     num_classes: int,
                     average: str = "macro") -> Dict[str, float]:
    """
    Compute F1 score.
    
    Args:
        preds: [B] predicted class indices
        targets: [B] target class indices
        num_classes: Number of classes
        average: "macro", "micro", or "weighted"
        
    Returns:
        Dict with precision, recall, f1
    """
    preds = preds.cpu().numpy()
    targets = targets.cpu().numpy()
    
    # Per-class metrics
    precisions = []
    recalls = []
    f1s = []
    supports = []
    
    for c in range(num_classes):
        tp = ((preds == c) & (targets == c)).sum()
        fp = ((preds == c) & (targets != c)).sum()
        fn = ((preds != c) & (targets == c)).sum()
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        support = (targets == c).sum()
        
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        supports.append(support)
    
    if average == "macro":
        return {
            "precision": np.mean(precisions),
            "recall": np.mean(recalls),
            "f1": np.mean(f1s)
        }
    elif average == "weighted":
        total = sum(supports)
        if total == 0:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        weights = [s / total for s in supports]
        return {
            "precision": sum(p * w for p, w in zip(precisions, weights)),
            "recall": sum(r * w for r, w in zip(recalls, weights)),
            "f1": sum(f * w for f, w in zip(f1s, weights))
        }
    else:  # micro
        tp_total = sum((preds == c) & (targets == c) for c in range(num_classes))
        fp_total = sum((preds == c) & (targets != c) for c in range(num_classes))
        fn_total = sum((preds != c) & (targets == c) for c in range(num_classes))
        
        precision = tp_total / (tp_total + fp_total + 1e-8)
        recall = tp_total / (tp_total + fn_total + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        
        return {"precision": precision, "recall": recall, "f1": f1}


def compute_confusion_matrix(preds: torch.Tensor,
                             targets: torch.Tensor,
                             num_classes: int) -> np.ndarray:
    """
    Compute confusion matrix.
    
    Args:
        preds: [B] predicted class indices
        targets: [B] target class indices
        num_classes: Number of classes
        
    Returns:
        Confusion matrix [num_classes, num_classes]
        cm[i, j] = count of samples with true label i predicted as j
    """
    preds = preds.cpu().numpy()
    targets = targets.cpu().numpy()
    
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(targets, preds):
        cm[t, p] += 1
    
    return cm


def compute_regression_metrics(preds: torch.Tensor,
                               targets: torch.Tensor) -> Dict[str, float]:
    """
    Compute regression metrics.
    
    Args:
        preds: [B] predicted values
        targets: [B] target values
        
    Returns:
        Dict with mae, mse, rmse, correlation
    """
    preds = preds.cpu().numpy()
    targets = targets.cpu().numpy()
    
    mae = np.mean(np.abs(preds - targets))
    mse = np.mean((preds - targets) ** 2)
    rmse = np.sqrt(mse)
    
    # Correlation
    if len(preds) > 1 and np.std(preds) > 0 and np.std(targets) > 0:
        correlation = np.corrcoef(preds, targets)[0, 1]
    else:
        correlation = 0.0
    
    return {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "correlation": correlation
    }


def compute_direction_metrics(dir_logits: torch.Tensor,
                              dir_targets: torch.Tensor,
                              return_preds: Optional[torch.Tensor] = None,
                              return_targets: Optional[torch.Tensor] = None) -> Dict[str, float]:
    """
    Compute all direction-related metrics.
    
    Args:
        dir_logits: [B, 3] direction logits
        dir_targets: [B] direction labels (0, 1, 2)
        return_preds: [B] return predictions (optional)
        return_targets: [B] return targets (optional)
        
    Returns:
        Dict with all metrics
    """
    dir_preds = dir_logits.argmax(dim=-1)
    
    metrics = {
        "dir_accuracy": compute_accuracy(dir_preds, dir_targets)
    }
    
    # F1 scores
    f1_metrics = compute_f1_score(dir_preds, dir_targets, num_classes=3, average="macro")
    metrics["dir_f1_macro"] = f1_metrics["f1"]
    metrics["dir_precision"] = f1_metrics["precision"]
    metrics["dir_recall"] = f1_metrics["recall"]
    
    # Per-class accuracy
    for c in range(3):
        mask = dir_targets == c
        if mask.sum() > 0:
            class_acc = (dir_preds[mask] == c).float().mean().item()
            class_name = ["down", "neutral", "up"][c]
            metrics[f"dir_acc_{class_name}"] = class_acc
    
    # Return metrics
    if return_preds is not None and return_targets is not None:
        ret_metrics = compute_regression_metrics(return_preds, return_targets)
        metrics["return_mae"] = ret_metrics["mae"]
        metrics["return_correlation"] = ret_metrics["correlation"]
    
    return metrics


def compute_regime_metrics(regime_logits: torch.Tensor,
                           regime_targets: torch.Tensor,
                           ignore_unknown: bool = True) -> Dict[str, float]:
    """
    Compute regime classification metrics.
    
    Args:
        regime_logits: [B, 6] regime logits
        regime_targets: [B] regime labels
        ignore_unknown: Whether to ignore class 0 (unknown)
        
    Returns:
        Dict with metrics
    """
    regime_preds = regime_logits.argmax(dim=-1)
    
    if ignore_unknown:
        # Filter out unknown class
        mask = regime_targets != 0
        regime_preds = regime_preds[mask]
        regime_targets = regime_targets[mask]
    
    if len(regime_targets) == 0:
        return {"regime_accuracy": 0.0, "regime_f1": 0.0}
    
    metrics = {
        "regime_accuracy": compute_accuracy(regime_preds, regime_targets)
    }
    
    f1_metrics = compute_f1_score(regime_preds, regime_targets, num_classes=6, average="macro")
    metrics["regime_f1"] = f1_metrics["f1"]
    
    return metrics


class MetricsAccumulator:
    """
    Accumulator for computing metrics over batches.
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all accumulators"""
        self.dir_logits_list = []
        self.dir_targets_list = []
        self.return_preds_list = []
        self.return_targets_list = []
        self.regime_logits_list = []
        self.regime_targets_list = []
        self.losses = defaultdict(list)
    
    def update(self,
               outputs: Dict[str, torch.Tensor],
               targets: Dict[str, torch.Tensor],
               losses: Optional[Dict[str, torch.Tensor]] = None):
        """
        Update accumulators with batch data.
        
        Args:
            outputs: Model outputs
            targets: Target tensors
            losses: Loss values (optional)
        """
        # Direction
        if "dir_logits" in outputs:
            self.dir_logits_list.append(outputs["dir_logits"].detach().cpu())
        if "future_dir_5" in targets:
            self.dir_targets_list.append(targets["future_dir_5"].detach().cpu())
        
        # Return
        if "return_pred" in outputs:
            self.return_preds_list.append(outputs["return_pred"].detach().cpu())
        if "future_return_5" in targets:
            self.return_targets_list.append(targets["future_return_5"].detach().cpu())
        
        # Regime
        if "regime_logits" in outputs:
            self.regime_logits_list.append(outputs["regime_logits"].detach().cpu())
        if "regime_hint" in targets:
            self.regime_targets_list.append(targets["regime_hint"].detach().cpu())
        
        # Losses
        if losses:
            for k, v in losses.items():
                if isinstance(v, torch.Tensor):
                    self.losses[k].append(v.item())
    
    def compute(self) -> Dict[str, float]:
        """
        Compute all metrics from accumulated data.
        
        Returns:
            Dict with all metrics
        """
        metrics = {}
        
        # Direction metrics
        if self.dir_logits_list and self.dir_targets_list:
            dir_logits = torch.cat(self.dir_logits_list, dim=0)
            dir_targets = torch.cat(self.dir_targets_list, dim=0)
            
            return_preds = None
            return_targets = None
            if self.return_preds_list:
                return_preds = torch.cat(self.return_preds_list, dim=0)
            if self.return_targets_list:
                return_targets = torch.cat(self.return_targets_list, dim=0)
            
            dir_metrics = compute_direction_metrics(
                dir_logits, dir_targets, return_preds, return_targets
            )
            metrics.update(dir_metrics)
        
        # Regime metrics
        if self.regime_logits_list and self.regime_targets_list:
            regime_logits = torch.cat(self.regime_logits_list, dim=0)
            regime_targets = torch.cat(self.regime_targets_list, dim=0)
            
            regime_metrics = compute_regime_metrics(regime_logits, regime_targets)
            metrics.update(regime_metrics)
        
        # Average losses
        for k, v in self.losses.items():
            if v:
                metrics[f"avg_{k}"] = np.mean(v)
        
        return metrics


if __name__ == "__main__":
    # Test metrics
    batch_size = 100
    
    # Mock data
    dir_logits = torch.randn(batch_size, 3)
    dir_targets = torch.randint(0, 3, (batch_size,))
    return_preds = torch.randn(batch_size) * 0.001
    return_targets = torch.randn(batch_size) * 0.001
    
    # Compute metrics
    metrics = compute_direction_metrics(
        dir_logits, dir_targets, return_preds, return_targets
    )
    
    print("Direction metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    
    # Test accumulator
    accumulator = MetricsAccumulator()
    
    for _ in range(10):
        outputs = {
            "dir_logits": torch.randn(32, 3),
            "return_pred": torch.randn(32) * 0.001,
            "regime_logits": torch.randn(32, 6)
        }
        targets = {
            "future_dir_5": torch.randint(0, 3, (32,)),
            "future_return_5": torch.randn(32) * 0.001,
            "regime_hint": torch.randint(0, 6, (32,))
        }
        losses = {"loss_total": torch.tensor(0.5)}
        
        accumulator.update(outputs, targets, losses)
    
    final_metrics = accumulator.compute()
    print("\nAccumulated metrics:")
    for k, v in final_metrics.items():
        print(f"  {k}: {v:.4f}")
