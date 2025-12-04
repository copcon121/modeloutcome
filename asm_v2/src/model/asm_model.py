"""
ASM v2 Model - Regime Classifier

Uses z_t embedding from STATE-ENC + meta features to classify market regime.
Architecture: GRN (Gated Residual Network) based classifier.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Any
import json


class GatedLinearUnit(nn.Module):
    """Gated Linear Unit"""
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.gate = nn.Linear(input_dim, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x) * torch.sigmoid(self.gate(x))


class GatedResidualNetwork(nn.Module):
    """Gated Residual Network block"""
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.glu = GatedLinearUnit(hidden_dim, output_dim)
        self.layer_norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Skip connection projection if dims differ
        self.skip_proj = nn.Linear(input_dim, output_dim) if input_dim != output_dim else None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Skip connection
        skip = self.skip_proj(x) if self.skip_proj else x
        
        # GRN path
        h = F.elu(self.fc1(x))
        h = self.dropout(h)
        h = F.elu(self.fc2(h))
        h = self.dropout(h)
        h = self.glu(h)
        
        # Residual + LayerNorm
        return self.layer_norm(skip + h)


class AsmModel(nn.Module):
    """
    ASM v2 Model - Regime Classifier
    
    Input: z_t (64-dim) + meta features (6-dim) = 70-dim
    Output: regime class logits
    """
    
    def __init__(self,
                 z_dim: int = 64,
                 meta_dim: int = 6,
                 hidden_dim: int = 128,
                 num_layers: int = 2,
                 dropout: float = 0.1,
                 num_classes: int = 5,
                 use_grn: bool = True):
        super().__init__()
        
        self.z_dim = z_dim
        self.meta_dim = meta_dim
        self.input_dim = z_dim + meta_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.use_grn = use_grn
        
        if use_grn:
            # GRN-based architecture
            layers = []
            in_dim = self.input_dim
            for i in range(num_layers):
                out_dim = hidden_dim if i < num_layers - 1 else hidden_dim
                layers.append(GatedResidualNetwork(in_dim, hidden_dim, out_dim, dropout))
                in_dim = out_dim
            self.backbone = nn.Sequential(*layers)
        else:
            # Simple MLP
            layers = []
            in_dim = self.input_dim
            for i in range(num_layers):
                layers.append(nn.Linear(in_dim, hidden_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
                in_dim = hidden_dim
            self.backbone = nn.Sequential(*layers)
        
        # Classification head
        self.classifier = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: [B, input_dim] concatenated z_t + meta features
            
        Returns:
            Dict with logits and probs
        """
        h = self.backbone(x)
        logits = self.classifier(h)
        probs = F.softmax(logits, dim=-1)
        
        return {
            "logits": logits,
            "probs": probs,
            "features": h
        }
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Get predicted class indices"""
        with torch.no_grad():
            out = self.forward(x)
            return out["logits"].argmax(dim=-1)
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "AsmModel":
        return cls(
            z_dim=config.get("z_dim", 64),
            meta_dim=config.get("meta_dim", 6),
            hidden_dim=config.get("hidden_dim", 128),
            num_layers=config.get("num_layers", 2),
            dropout=config.get("dropout", 0.1),
            num_classes=config.get("num_classes", 5),
            use_grn=config.get("use_grn", True)
        )
    
    def get_config(self) -> Dict[str, Any]:
        return {
            "z_dim": self.z_dim,
            "meta_dim": self.meta_dim,
            "hidden_dim": self.hidden_dim,
            "num_classes": self.num_classes,
            "use_grn": self.use_grn
        }


def load_asm_model(checkpoint_path: str,
                   config_path: Optional[str] = None,
                   device: str = "cpu") -> AsmModel:
    """Load trained ASM model"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if "config" in checkpoint:
        config = checkpoint["config"]
    elif config_path:
        with open(config_path, "r") as f:
            config = json.load(f)
    else:
        raise ValueError("Config not found")
    
    model = AsmModel.from_config(config)
    
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    return model
