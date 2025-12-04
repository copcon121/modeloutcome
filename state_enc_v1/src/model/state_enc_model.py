"""
StateEncModel v1.2

Fixes for test failures:
- ShockDetectionHead for shock sensitivity
- FeatureReconstructHead for missing feature robustness
- OrderDetectionHead for bar order sensitivity
- LatentAnchorHead for drift control
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Any

from .encoder_backbone import StateEncoderBackbone
from .heads import (
    SelfSupervisedHead, RegimeHead, MetaS4Head,
    ShockDetectionHead, FeatureReconstructHead, 
    OrderDetectionHead, LatentAnchorHead,
    DataAugmentationV12
)


class StateEncModel(nn.Module):
    """
    STATE-ENC v1.2 Model
    
    New heads:
    - ShockDetectionHead: detect market shocks
    - FeatureReconstructHead: reconstruct masked features
    - OrderDetectionHead: detect shuffled sequences
    - LatentAnchorHead: anchor loss for drift control
    """
    
    def __init__(self,
                 input_dim: int,
                 d_model: int = 64,
                 num_heads: int = 4,
                 num_layers: int = 4,
                 dim_feedforward: int = 256,
                 dropout: float = 0.075,
                 sequence_length: int = 128,
                 pooling: str = "last",
                 heads_config: Optional[Dict] = None):
        super().__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        self.sequence_length = sequence_length
        
        # Encoder backbone v1.2
        self.encoder = StateEncoderBackbone(
            input_dim=input_dim,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_len=sequence_length,
            pooling=pooling
        )
        
        # Default heads config
        if heads_config is None:
            heads_config = {
                "self_supervised": {"enabled": True},
                "regime": {"enabled": True, "num_classes": 6},
                "shock": {"enabled": True},
                "reconstruct": {"enabled": True},
                "order": {"enabled": True},
                "anchor": {"enabled": True},
                "meta_s4": {"enabled": False}
            }
        
        # Initialize heads
        self.heads = nn.ModuleDict()
        
        # Self-supervised head
        if heads_config.get("self_supervised", {}).get("enabled", True):
            self.heads["self_supervised"] = SelfSupervisedHead(
                input_dim=d_model,
                feature_dim=input_dim,
                dropout=dropout
            )
        
        # Regime head
        if heads_config.get("regime", {}).get("enabled", True):
            self.heads["regime"] = RegimeHead(
                input_dim=d_model,
                num_classes=heads_config.get("regime", {}).get("num_classes", 6),
                dropout=dropout
            )
        
        # Shock detection head (NEW)
        if heads_config.get("shock", {}).get("enabled", True):
            self.heads["shock"] = ShockDetectionHead(d_model, dropout)
        
        # Feature reconstruct head (NEW)
        if heads_config.get("reconstruct", {}).get("enabled", True):
            self.heads["reconstruct"] = FeatureReconstructHead(d_model, input_dim, dropout=dropout)
        
        # Order detection head (NEW)
        if heads_config.get("order", {}).get("enabled", True):
            self.heads["order"] = OrderDetectionHead(d_model, dropout)
        
        # Latent anchor head (NEW)
        if heads_config.get("anchor", {}).get("enabled", True):
            self.heads["anchor"] = LatentAnchorHead(d_model)
        
        # Meta S4 head
        if heads_config.get("meta_s4", {}).get("enabled", False):
            self.heads["meta_s4"] = MetaS4Head(d_model, dropout=dropout)
        
        # Data augmentation v1.2
        self.augmentation = DataAugmentationV12()
    
    def set_epoch(self, epoch: int):
        """Set current epoch for scheduled augmentation"""
        self.augmentation.set_epoch(epoch)
    
    def forward(self,
                x: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                return_all_heads: bool = True,
                augment: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: [B, N, D] input features
            mask: [B, N] padding mask
            return_all_heads: compute all head outputs
            augment: apply data augmentation
        """
        outputs = {}
        
        # Data augmentation
        if augment and self.training:
            aug_result = self.augmentation(x, training=True)
            x_input = aug_result["x"]
            outputs["original_x"] = aug_result["original_x"]
            outputs["shock_labels"] = aug_result["shock_labels"]
            outputs["order_labels"] = aug_result["order_labels"]
            outputs["feature_mask"] = aug_result["feature_mask"]
        else:
            x_input = x
        
        # Encode
        z_seq, z_t = self.encoder(x_input, mask)
        
        outputs["z_seq"] = z_seq
        outputs["z_t"] = z_t
        
        if return_all_heads:
            # Self-supervised head
            if "self_supervised" in self.heads:
                ss_out = self.heads["self_supervised"](z_t)
                outputs.update(ss_out)
            
            # Regime head
            if "regime" in self.heads:
                reg_out = self.heads["regime"](z_t)
                outputs.update(reg_out)
            
            # Shock detection head
            if "shock" in self.heads:
                shock_out = self.heads["shock"](z_t)
                outputs.update(shock_out)
            
            # Feature reconstruct head
            if "reconstruct" in self.heads:
                recon_out = self.heads["reconstruct"](z_seq)
                outputs.update(recon_out)
            
            # Order detection head
            if "order" in self.heads:
                order_out = self.heads["order"](z_t)
                outputs.update(order_out)
            
            # Latent anchor head
            if "anchor" in self.heads:
                anchor_out = self.heads["anchor"](z_t)
                outputs.update(anchor_out)
            
            # Meta S4 head
            if "meta_s4" in self.heads:
                meta_out = self.heads["meta_s4"](z_t)
                outputs.update(meta_out)
        
        return outputs
    
    def encode(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Get only z_t embedding"""
        _, z_t = self.encoder(x, mask)
        return z_t
    
    def get_embedding_dim(self) -> int:
        return self.d_model
    
    def get_head_names(self) -> list:
        return list(self.heads.keys())
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "StateEncModel":
        return cls(
            input_dim=config["input_dim"],
            d_model=config.get("d_model", 64),
            num_heads=config.get("num_heads", 4),
            num_layers=config.get("num_layers", 4),
            dim_feedforward=config.get("dim_feedforward", 256),
            dropout=config.get("dropout", 0.075),
            sequence_length=config.get("sequence_length", 128),
            pooling=config.get("pooling", "last"),
            heads_config=config.get("heads", None)
        )
    
    def get_config(self) -> Dict[str, Any]:
        return {
            "input_dim": self.input_dim,
            "d_model": self.d_model,
            "sequence_length": self.sequence_length,
            "heads": list(self.heads.keys())
        }


def load_state_enc_model(checkpoint_path: str,
                         config_path: Optional[str] = None,
                         device: str = "cpu") -> StateEncModel:
    """Load trained model"""
    import json
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if "config" in checkpoint:
        config = checkpoint["config"]
    elif config_path:
        with open(config_path, "r") as f:
            config = json.load(f)
    else:
        raise ValueError("Config not found")
    
    model = StateEncModel.from_config(config)
    
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    return model
