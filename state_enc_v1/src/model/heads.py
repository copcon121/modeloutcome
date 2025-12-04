"""
Prediction heads for STATE-ENC v1

Multiple heads for different tasks:
- SelfSupervisedHead: Predict future direction and return
- RegimeHead: Predict market regime
- MetaS4Head: Generic output for meta-learning (placeholder)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class SelfSupervisedHead(nn.Module):
    """
    Self-supervised prediction head.
    
    Predicts:
    - future_dir_5: Direction classification (3 classes: down, neutral, up)
    - future_return_5: Return regression (optional)
    """
    
    def __init__(self,
                 input_dim: int,
                 num_dir_classes: int = 3,
                 predict_return: bool = True,
                 hidden_dim: Optional[int] = None,
                 dropout: float = 0.1):
        """
        Args:
            input_dim: Input embedding dimension (d_model)
            num_dir_classes: Number of direction classes (default 3: -1, 0, +1)
            predict_return: Whether to predict return value
            hidden_dim: Hidden layer dimension (default: input_dim)
            dropout: Dropout rate
        """
        super().__init__()
        
        self.predict_return = predict_return
        hidden_dim = hidden_dim or input_dim
        
        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Direction classification head
        self.dir_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_dir_classes)
        )
        
        # Return regression head
        if predict_return:
            self.return_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1)
            )
    
    def forward(self, z_t: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            z_t: State embedding [B, d_model]
            
        Returns:
            Dict with:
                - dir_logits: [B, num_classes]
                - return_pred: [B] (if predict_return)
        """
        h = self.shared(z_t)
        
        outputs = {
            "dir_logits": self.dir_head(h)
        }
        
        if self.predict_return:
            outputs["return_pred"] = self.return_head(h).squeeze(-1)
        
        return outputs


class RegimeHead(nn.Module):
    """
    Market regime classification head.
    
    Predicts regime from ASM v1:
    - 0: unknown
    - 1: range
    - 2: trend_up
    - 3: trend_down
    - 4: opening_drive_up
    - 5: opening_drive_down
    """
    
    def __init__(self,
                 input_dim: int,
                 num_classes: int = 6,
                 hidden_dim: Optional[int] = None,
                 dropout: float = 0.1):
        """
        Args:
            input_dim: Input embedding dimension
            num_classes: Number of regime classes
            hidden_dim: Hidden layer dimension
            dropout: Dropout rate
        """
        super().__init__()
        
        hidden_dim = hidden_dim or input_dim
        
        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(self, z_t: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            z_t: State embedding [B, d_model]
            
        Returns:
            Dict with:
                - regime_logits: [B, num_classes]
        """
        return {
            "regime_logits": self.head(z_t)
        }


class MetaS4Head(nn.Module):
    """
    Meta S4 head for future extension.
    
    Outputs generic vector that can be used for:
    - Trade filtering
    - Risk adjustment
    - Strategy selection
    
    Currently a placeholder that outputs a configurable dimension vector.
    """
    
    def __init__(self,
                 input_dim: int,
                 output_dim: int = 4,
                 hidden_dim: Optional[int] = None,
                 dropout: float = 0.1):
        """
        Args:
            input_dim: Input embedding dimension
            output_dim: Output vector dimension
            hidden_dim: Hidden layer dimension
            dropout: Dropout rate
        """
        super().__init__()
        
        hidden_dim = hidden_dim or input_dim
        
        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, z_t: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            z_t: State embedding [B, d_model]
            
        Returns:
            Dict with:
                - meta_output: [B, output_dim]
        """
        return {
            "meta_output": self.head(z_t)
        }


class ContrastiveHead(nn.Module):
    """
    Contrastive learning head for self-supervised pre-training.
    
    Projects embeddings to a lower-dimensional space for contrastive loss.
    """
    
    def __init__(self,
                 input_dim: int,
                 proj_dim: int = 64,
                 hidden_dim: Optional[int] = None):
        super().__init__()
        
        hidden_dim = hidden_dim or input_dim
        
        self.projector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, proj_dim)
        )
    
    def forward(self, z_t: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            z_t: State embedding [B, d_model]
            
        Returns:
            Dict with:
                - proj: [B, proj_dim] - L2 normalized projection
        """
        proj = self.projector(z_t)
        proj = F.normalize(proj, dim=-1)
        return {"proj": proj}


class VolatilityHead(nn.Module):
    """
    Volatility prediction head.
    
    Predicts future volatility/range for risk management.
    """
    
    def __init__(self,
                 input_dim: int,
                 hidden_dim: Optional[int] = None,
                 dropout: float = 0.1):
        super().__init__()
        
        hidden_dim = hidden_dim or input_dim
        
        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Softplus()  # Ensure positive output
        )
    
    def forward(self, z_t: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            z_t: State embedding [B, d_model]
            
        Returns:
            Dict with:
                - volatility_pred: [B] - predicted volatility
        """
        return {
            "volatility_pred": self.head(z_t).squeeze(-1)
        }


def create_head(head_type: str, config: dict) -> nn.Module:
    """
    Factory function to create head from config.
    
    Args:
        head_type: Type of head ("self_supervised", "regime", "meta_s4", etc.)
        config: Head configuration dict
        
    Returns:
        Head module
    """
    input_dim = config["input_dim"]
    dropout = config.get("dropout", 0.1)
    
    if head_type == "self_supervised":
        return SelfSupervisedHead(
            input_dim=input_dim,
            num_dir_classes=config.get("num_dir_classes", 3),
            predict_return=config.get("predict_return", True),
            dropout=dropout
        )
    elif head_type == "regime":
        return RegimeHead(
            input_dim=input_dim,
            num_classes=config.get("num_classes", 6),
            dropout=dropout
        )
    elif head_type == "meta_s4":
        return MetaS4Head(
            input_dim=input_dim,
            output_dim=config.get("output_dim", 4),
            dropout=dropout
        )
    elif head_type == "contrastive":
        return ContrastiveHead(
            input_dim=input_dim,
            proj_dim=config.get("proj_dim", 64)
        )
    elif head_type == "volatility":
        return VolatilityHead(
            input_dim=input_dim,
            dropout=dropout
        )
    else:
        raise ValueError(f"Unknown head type: {head_type}")


if __name__ == "__main__":
    # Test heads
    d_model = 128
    batch_size = 4
    
    z_t = torch.randn(batch_size, d_model)
    
    # Test SelfSupervisedHead
    ss_head = SelfSupervisedHead(d_model)
    ss_out = ss_head(z_t)
    print(f"SelfSupervised - dir_logits: {ss_out['dir_logits'].shape}, return_pred: {ss_out['return_pred'].shape}")
    
    # Test RegimeHead
    regime_head = RegimeHead(d_model)
    regime_out = regime_head(z_t)
    print(f"Regime - regime_logits: {regime_out['regime_logits'].shape}")
    
    # Test MetaS4Head
    meta_head = MetaS4Head(d_model, output_dim=4)
    meta_out = meta_head(z_t)
    print(f"MetaS4 - meta_output: {meta_out['meta_output'].shape}")
