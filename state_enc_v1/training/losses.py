"""
Loss functions for STATE-ENC v1.2

Combined loss:
- dir_loss
- future_return_loss * 0.1
- regime_loss * 0.5
- shock_loss * 0.5
- reconstruct_loss * 0.2
- order_loss * 0.3
- latent_anchor_loss * 0.3
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class FocalLoss(nn.Module):
    """Focal Loss"""
    def __init__(self, gamma: float = 1.5, weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


class MultiHeadLossV12(nn.Module):
    """
    Combined multi-head loss for STATE-ENC v1.2
    """
    
    def __init__(self, loss_weights: Optional[Dict[str, float]] = None):
        super().__init__()
        
        self.loss_weights = loss_weights or {
            "future_dir": 1.0,
            "future_return": 0.1,
            "regime": 0.5,
            "shock": 0.5,
            "reconstruct": 0.2,
            "order": 0.3,
            "anchor": 0.3
        }
        
        self.focal_loss = FocalLoss(gamma=1.5)
        self.return_loss = nn.SmoothL1Loss(beta=0.001)
        self.regime_loss = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)
        self.shock_loss = nn.CrossEntropyLoss()
        self.order_loss = nn.CrossEntropyLoss()
        self.reconstruct_loss = nn.MSELoss()
    
    def forward(self, outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute all losses"""
        losses = {}
        device = outputs["z_t"].device
        total_loss = torch.tensor(0.0, device=device)
        
        # Direction loss (Focal)
        if "dir_logits" in outputs and "future_dir_5" in targets:
            dir_loss = self.focal_loss(outputs["dir_logits"], targets["future_dir_5"])
            losses["loss_dir"] = dir_loss
            total_loss = total_loss + self.loss_weights.get("future_dir", 1.0) * dir_loss
        
        # Return loss (SmoothL1)
        if "return_pred" in outputs and "future_return_5" in targets:
            ret_loss = self.return_loss(outputs["return_pred"], targets["future_return_5"])
            losses["loss_return"] = ret_loss
            total_loss = total_loss + self.loss_weights.get("future_return", 0.1) * ret_loss
        
        # Regime loss
        if "regime_logits" in outputs and "regime_hint" in targets:
            reg_loss = self.regime_loss(outputs["regime_logits"], targets["regime_hint"])
            losses["loss_regime"] = reg_loss
            total_loss = total_loss + self.loss_weights.get("regime", 0.5) * reg_loss
        
        # Shock loss (NEW) - only if labels exist
        if "shock_logits" in outputs and "shock_labels" in outputs:
            try:
                shock_loss = self.shock_loss(outputs["shock_logits"], outputs["shock_labels"])
                if not torch.isnan(shock_loss):
                    losses["loss_shock"] = shock_loss
                    total_loss = total_loss + self.loss_weights.get("shock", 0.5) * shock_loss
            except Exception:
                pass
        
        # Reconstruct loss (NEW)
        if "reconstructed_features" in outputs and "original_x" in outputs:
            try:
                recon = outputs["reconstructed_features"]
                original = outputs["original_x"]
                recon_loss = F.mse_loss(recon, original)
                if not torch.isnan(recon_loss):
                    losses["loss_reconstruct"] = recon_loss
                    total_loss = total_loss + self.loss_weights.get("reconstruct", 0.2) * recon_loss
            except Exception:
                pass
        
        # Order loss (NEW)
        if "order_logits" in outputs and "order_labels" in outputs:
            try:
                order_loss = self.order_loss(outputs["order_logits"], outputs["order_labels"])
                if not torch.isnan(order_loss):
                    losses["loss_order"] = order_loss
                    total_loss = total_loss + self.loss_weights.get("order", 0.3) * order_loss
            except Exception:
                pass
        
        # Anchor loss (NEW)
        if "anchor_distance" in outputs:
            try:
                anchor_loss = outputs["anchor_distance"].mean()
                if not torch.isnan(anchor_loss):
                    losses["loss_anchor"] = anchor_loss
                    total_loss = total_loss + self.loss_weights.get("anchor", 0.3) * anchor_loss
            except Exception:
                pass
        
        losses["loss_total"] = total_loss
        return losses


# Backward compatibility
MultiHeadLoss = MultiHeadLossV12


def compute_multihead_loss(outputs: Dict[str, torch.Tensor],
                           targets: Dict[str, torch.Tensor],
                           loss_weights: Dict[str, float]) -> Dict[str, torch.Tensor]:
    """Functional interface"""
    loss_module = MultiHeadLossV12(loss_weights=loss_weights)
    return loss_module(outputs, targets)
