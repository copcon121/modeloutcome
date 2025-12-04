"""Loss functions for ASM v2"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingCrossEntropy(nn.Module):
    """Cross entropy with label smoothing"""
    
    def __init__(self, smoothing: float = 0.05, reduction: str = "mean"):
        super().__init__()
        self.smoothing = smoothing
        self.reduction = reduction
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [B, C] unnormalized logits
            targets: [B] class indices
        """
        n_classes = logits.size(-1)
        
        # Create smoothed targets
        with torch.no_grad():
            smooth_targets = torch.zeros_like(logits)
            smooth_targets.fill_(self.smoothing / (n_classes - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
        
        # Compute loss
        log_probs = F.log_softmax(logits, dim=-1)
        loss = -(smooth_targets * log_probs).sum(dim=-1)
        
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class AsmLoss(nn.Module):
    """Combined loss for ASM v2"""
    
    def __init__(self, label_smoothing: float = 0.05):
        super().__init__()
        self.ce_loss = LabelSmoothingCrossEntropy(smoothing=label_smoothing)
    
    def forward(self, outputs: dict, targets: torch.Tensor) -> dict:
        """
        Args:
            outputs: dict with "logits" key
            targets: [B] class indices
        """
        logits = outputs["logits"]
        ce = self.ce_loss(logits, targets)
        
        return {
            "loss": ce,
            "ce_loss": ce
        }
