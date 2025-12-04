"""
Loss functions for STATE-ENC v1 training

Multi-task losses for:
- Future direction classification
- Future return regression
- Regime classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    
    def __init__(self, 
                 alpha: Optional[torch.Tensor] = None,
                 gamma: float = 2.0,
                 reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: [B, C] logits
            targets: [B] class indices
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.alpha is not None:
            alpha = self.alpha.to(inputs.device)
            alpha_t = alpha[targets]
            focal_loss = alpha_t * focal_loss
        
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


class DirectionLoss(nn.Module):
    """
    Loss for future direction prediction.
    
    Supports:
    - Cross entropy
    - Focal loss (for imbalanced classes)
    - Label smoothing
    """
    
    def __init__(self,
                 num_classes: int = 3,
                 use_focal: bool = False,
                 focal_gamma: float = 2.0,
                 label_smoothing: float = 0.0,
                 class_weights: Optional[torch.Tensor] = None):
        super().__init__()
        
        self.num_classes = num_classes
        self.use_focal = use_focal
        
        if use_focal:
            self.loss_fn = FocalLoss(alpha=class_weights, gamma=focal_gamma)
        else:
            self.loss_fn = nn.CrossEntropyLoss(
                weight=class_weights,
                label_smoothing=label_smoothing
            )
    
    def forward(self, 
                logits: torch.Tensor, 
                targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [B, num_classes] direction logits
            targets: [B] direction labels (0, 1, 2)
        """
        return self.loss_fn(logits, targets)


class ReturnLoss(nn.Module):
    """
    Loss for future return regression.
    
    Supports:
    - MSE
    - Smooth L1 (Huber)
    - Quantile loss
    """
    
    def __init__(self,
                 loss_type: str = "smooth_l1",
                 beta: float = 0.001):
        super().__init__()
        
        self.loss_type = loss_type
        self.beta = beta
        
        if loss_type == "mse":
            self.loss_fn = nn.MSELoss()
        elif loss_type == "smooth_l1":
            self.loss_fn = nn.SmoothL1Loss(beta=beta)
        else:
            self.loss_fn = nn.MSELoss()
    
    def forward(self,
                pred: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: [B] predicted returns
            target: [B] actual returns
        """
        return self.loss_fn(pred, target)


class RegimeLoss(nn.Module):
    """
    Loss for regime classification.
    
    Handles unknown regime (class 0) by optionally masking it.
    """
    
    def __init__(self,
                 num_classes: int = 6,
                 ignore_unknown: bool = True,
                 unknown_class: int = 0,
                 label_smoothing: float = 0.0):
        super().__init__()
        
        self.ignore_unknown = ignore_unknown
        self.unknown_class = unknown_class
        
        # Use ignore_index to mask unknown class
        ignore_idx = unknown_class if ignore_unknown else -100
        self.loss_fn = nn.CrossEntropyLoss(
            ignore_index=ignore_idx,
            label_smoothing=label_smoothing
        )
    
    def forward(self,
                logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [B, num_classes] regime logits
            targets: [B] regime labels
        """
        return self.loss_fn(logits, targets)


class MultiHeadLoss(nn.Module):
    """
    Combined multi-head loss for STATE-ENC training.
    
    Combines:
    - Direction loss
    - Return loss
    - Regime loss
    
    With configurable weights.
    """
    
    def __init__(self,
                 loss_weights: Optional[Dict[str, float]] = None,
                 direction_config: Optional[Dict] = None,
                 return_config: Optional[Dict] = None,
                 regime_config: Optional[Dict] = None):
        super().__init__()
        
        # Default weights
        self.loss_weights = loss_weights or {
            "future_dir": 1.0,
            "future_return": 0.1,
            "regime": 0.5
        }
        
        # Direction loss
        dir_cfg = direction_config or {}
        self.direction_loss = DirectionLoss(
            num_classes=dir_cfg.get("num_classes", 3),
            use_focal=dir_cfg.get("use_focal", False),
            focal_gamma=dir_cfg.get("focal_gamma", 2.0),
            label_smoothing=dir_cfg.get("label_smoothing", 0.0)
        )
        
        # Return loss
        ret_cfg = return_config or {}
        self.return_loss = ReturnLoss(
            loss_type=ret_cfg.get("loss_type", "smooth_l1"),
            beta=ret_cfg.get("beta", 0.001)
        )
        
        # Regime loss
        reg_cfg = regime_config or {}
        self.regime_loss = RegimeLoss(
            num_classes=reg_cfg.get("num_classes", 6),
            ignore_unknown=reg_cfg.get("ignore_unknown", True),
            label_smoothing=reg_cfg.get("label_smoothing", 0.0)
        )
    
    def forward(self,
                outputs: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute multi-head loss.
        
        Args:
            outputs: Model outputs dict
            targets: Target dict with future_dir_5, future_return_5, regime_hint
            
        Returns:
            Dict with individual losses and total loss
        """
        losses = {}
        total_loss = 0.0
        
        # Direction loss
        if "dir_logits" in outputs and "future_dir_5" in targets:
            dir_loss = self.direction_loss(
                outputs["dir_logits"],
                targets["future_dir_5"]
            )
            losses["loss_dir"] = dir_loss
            total_loss = total_loss + self.loss_weights.get("future_dir", 1.0) * dir_loss
        
        # Return loss
        if "return_pred" in outputs and "future_return_5" in targets:
            ret_loss = self.return_loss(
                outputs["return_pred"],
                targets["future_return_5"]
            )
            losses["loss_return"] = ret_loss
            total_loss = total_loss + self.loss_weights.get("future_return", 0.1) * ret_loss
        
        # Regime loss
        if "regime_logits" in outputs and "regime_hint" in targets:
            reg_loss = self.regime_loss(
                outputs["regime_logits"],
                targets["regime_hint"]
            )
            losses["loss_regime"] = reg_loss
            total_loss = total_loss + self.loss_weights.get("regime", 0.5) * reg_loss
        
        losses["loss_total"] = total_loss
        
        return losses


def compute_multihead_loss(outputs: Dict[str, torch.Tensor],
                           targets: Dict[str, torch.Tensor],
                           loss_weights: Dict[str, float]) -> Dict[str, torch.Tensor]:
    """
    Functional interface for computing multi-head loss.
    
    Args:
        outputs: Model outputs
        targets: Target tensors
        loss_weights: Weight for each loss component
        
    Returns:
        Dict with losses
    """
    loss_module = MultiHeadLoss(loss_weights=loss_weights)
    return loss_module(outputs, targets)


if __name__ == "__main__":
    # Test losses
    batch_size = 8
    num_classes = 3
    
    # Mock outputs
    outputs = {
        "dir_logits": torch.randn(batch_size, num_classes),
        "return_pred": torch.randn(batch_size),
        "regime_logits": torch.randn(batch_size, 6)
    }
    
    # Mock targets
    targets = {
        "future_dir_5": torch.randint(0, 3, (batch_size,)),
        "future_return_5": torch.randn(batch_size) * 0.001,
        "regime_hint": torch.randint(0, 6, (batch_size,))
    }
    
    # Compute loss
    loss_weights = {"future_dir": 1.0, "future_return": 0.1, "regime": 0.5}
    losses = compute_multihead_loss(outputs, targets, loss_weights)
    
    print("Losses:")
    for k, v in losses.items():
        print(f"  {k}: {v.item():.4f}")
