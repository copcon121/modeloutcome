"""
Main StateEncModel for STATE-ENC v1

Combines encoder backbone with multiple prediction heads.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Any

from .encoder_backbone import StateEncoderBackbone, create_encoder
from .heads import SelfSupervisedHead, RegimeHead, MetaS4Head, create_head


class StateEncModel(nn.Module):
    """
    Complete State Encoder Model.
    
    Architecture:
    - Encoder backbone (Transformer or Lightweight)
    - Multiple prediction heads:
        - SelfSupervisedHead: future direction & return
        - RegimeHead: market regime classification
        - MetaS4Head: generic output for meta-learning
    
    Usage:
        model = StateEncModel(input_dim=95, d_model=128, ...)
        outputs = model(x)  # x: [B, N, D]
        z_t = outputs["z_t"]  # Market state embedding
        dir_logits = outputs["dir_logits"]  # Direction prediction
    """
    
    def __init__(self,
                 input_dim: int,
                 d_model: int = 128,
                 num_heads: int = 8,
                 num_layers: int = 4,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1,
                 sequence_length: int = 128,
                 pooling: str = "last",
                 positional_encoding: str = "sinusoidal",
                 heads_config: Optional[Dict[str, Dict]] = None):
        """
        Args:
            input_dim: Input feature dimension D
            d_model: Model/embedding dimension
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            dim_feedforward: FFN hidden dimension
            dropout: Dropout rate
            sequence_length: Maximum sequence length N
            pooling: Pooling method for z_t
            positional_encoding: Type of positional encoding
            heads_config: Configuration for prediction heads
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        self.sequence_length = sequence_length
        
        # Encoder backbone
        self.encoder = StateEncoderBackbone(
            input_dim=input_dim,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_len=sequence_length,
            pooling=pooling,
            positional_encoding=positional_encoding
        )
        
        # Default heads config
        if heads_config is None:
            heads_config = {
                "self_supervised": {"enabled": True, "num_dir_classes": 3, "predict_return": True},
                "regime": {"enabled": True, "num_classes": 6},
                "meta_s4": {"enabled": True, "output_dim": 4}
            }
        
        # Initialize heads
        self.heads = nn.ModuleDict()
        
        # Self-supervised head
        ss_config = heads_config.get("self_supervised", {})
        if ss_config.get("enabled", True):
            self.heads["self_supervised"] = SelfSupervisedHead(
                input_dim=d_model,
                num_dir_classes=ss_config.get("num_dir_classes", 3),
                predict_return=ss_config.get("predict_return", True),
                dropout=dropout
            )
        
        # Regime head
        regime_config = heads_config.get("regime", {})
        if regime_config.get("enabled", True):
            self.heads["regime"] = RegimeHead(
                input_dim=d_model,
                num_classes=regime_config.get("num_classes", 6),
                dropout=dropout
            )
        
        # Meta S4 head
        meta_config = heads_config.get("meta_s4", {})
        if meta_config.get("enabled", True):
            self.heads["meta_s4"] = MetaS4Head(
                input_dim=d_model,
                output_dim=meta_config.get("output_dim", 4),
                dropout=dropout
            )
    
    def forward(self, 
                x: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                return_all_heads: bool = True) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input tensor [B, N, D]
            mask: Optional attention mask [B, N]
            return_all_heads: Whether to compute all head outputs
            
        Returns:
            Dict containing:
                - z_seq: [B, N, d_model] sequence embeddings
                - z_t: [B, d_model] final state embedding
                - dir_logits: [B, 3] direction logits (if self_supervised enabled)
                - return_pred: [B] return prediction (if enabled)
                - regime_logits: [B, 6] regime logits (if regime enabled)
                - meta_output: [B, 4] meta output (if meta_s4 enabled)
        """
        # Encode sequence
        z_seq, z_t = self.encoder(x, mask)
        
        outputs = {
            "z_seq": z_seq,
            "z_t": z_t
        }
        
        if return_all_heads:
            # Self-supervised head
            if "self_supervised" in self.heads:
                ss_out = self.heads["self_supervised"](z_t)
                outputs["dir_logits"] = ss_out["dir_logits"]
                if "return_pred" in ss_out:
                    outputs["return_pred"] = ss_out["return_pred"]
            
            # Regime head
            if "regime" in self.heads:
                regime_out = self.heads["regime"](z_t)
                outputs["regime_logits"] = regime_out["regime_logits"]
            
            # Meta S4 head
            if "meta_s4" in self.heads:
                meta_out = self.heads["meta_s4"](z_t)
                outputs["meta_output"] = meta_out["meta_output"]
        
        return outputs
    
    def encode(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Get only the state embedding z_t (for inference).
        
        Args:
            x: Input tensor [B, N, D]
            mask: Optional attention mask
            
        Returns:
            z_t: [B, d_model] state embedding
        """
        _, z_t = self.encoder(x, mask)
        return z_t
    
    def get_embedding_dim(self) -> int:
        """Return embedding dimension"""
        return self.d_model
    
    def freeze_encoder(self):
        """Freeze encoder weights (for fine-tuning heads only)"""
        for param in self.encoder.parameters():
            param.requires_grad = False
    
    def unfreeze_encoder(self):
        """Unfreeze encoder weights"""
        for param in self.encoder.parameters():
            param.requires_grad = True
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "StateEncModel":
        """
        Create model from config dict.
        
        Args:
            config: Model configuration
            
        Returns:
            StateEncModel instance
        """
        return cls(
            input_dim=config["input_dim"],
            d_model=config.get("d_model", 128),
            num_heads=config.get("num_heads", 8),
            num_layers=config.get("num_layers", 4),
            dim_feedforward=config.get("dim_feedforward", 512),
            dropout=config.get("dropout", 0.1),
            sequence_length=config.get("sequence_length", 128),
            pooling=config.get("pooling", "last"),
            positional_encoding=config.get("positional_encoding", "sinusoidal"),
            heads_config=config.get("heads", None)
        )
    
    def get_config(self) -> Dict[str, Any]:
        """Get model configuration for saving"""
        return {
            "input_dim": self.input_dim,
            "d_model": self.d_model,
            "sequence_length": self.sequence_length,
            "num_heads": self.encoder.transformer.layers[0].self_attn.num_heads,
            "num_layers": len(self.encoder.transformer.layers),
            "pooling": self.encoder.pooling
        }


def load_state_enc_model(checkpoint_path: str, 
                         config_path: Optional[str] = None,
                         device: str = "cpu") -> StateEncModel:
    """
    Load trained StateEncModel from checkpoint.
    
    Args:
        checkpoint_path: Path to .pt checkpoint file
        config_path: Path to model_config.json (optional if saved in checkpoint)
        device: Device to load model on
        
    Returns:
        Loaded StateEncModel
    """
    import json
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get config
    if "config" in checkpoint:
        config = checkpoint["config"]
    elif config_path:
        with open(config_path, "r") as f:
            config = json.load(f)
    else:
        raise ValueError("Config not found in checkpoint and config_path not provided")
    
    # Create model
    model = StateEncModel.from_config(config)
    
    # Load weights
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    return model


if __name__ == "__main__":
    # Test model
    model = StateEncModel(
        input_dim=95,
        d_model=128,
        num_heads=8,
        num_layers=4,
        sequence_length=128
    )
    
    # Test forward
    x = torch.randn(4, 128, 95)  # [B, N, D]
    outputs = model(x)
    
    print("Model outputs:")
    for k, v in outputs.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {v.shape}")
    
    # Test encode only
    z_t = model.encode(x)
    print(f"\nEncode only - z_t: {z_t.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
