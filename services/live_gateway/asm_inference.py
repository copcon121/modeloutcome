"""
ASM Inference for Live Gateway
Loads and runs ASM-GRU64-v1.0-C3 model
"""

import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).parent.parent.parent


# ==============================================================================
# MODEL DEFINITION (must match training)
# ==============================================================================


class ASMGRUClassifier(nn.Module):
    """GRU-based classifier for Auction State Model."""

    def __init__(
        self,
        input_dim: int = 100,
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_classes: int = 3,
        dropout: float = 0.2,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )

        gru_output_dim = hidden_dim * self.num_directions
        self.classifier = nn.Sequential(
            nn.Linear(gru_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        _, h_n = self.gru(x)
        if self.bidirectional:
            h_forward = h_n[-2, :, :]
            h_backward = h_n[-1, :, :]
            h_final = torch.cat([h_forward, h_backward], dim=1)
        else:
            h_final = h_n[-1, :, :]
        return self.classifier(h_final)


# ==============================================================================
# ASM MODEL LOADER
# ==============================================================================


class ASMModelLoader:
    """
    Load and run inference with ASM-GRU64-v1.0-C3 model.
    
    Model outputs 3 classes: UP (0), DOWN (1), NEUTRAL (2)
    """
    
    # Default paths
    DEFAULT_MODEL_PATH = ROOT / "output/asm_models_v1/ASM-GRU64-v1.0-C3.pt"
    
    # Model config (from training)
    INPUT_DIM = 100
    SEQ_LEN = 60
    HIDDEN_DIM = 64
    NUM_LAYERS = 2
    NUM_CLASSES = 3
    DROPOUT = 0.2
    
    # Label mapping
    LABEL_NAMES = {0: "UP", 1: "DOWN", 2: "NEUTRAL"}
    
    def __init__(self, model_path: Path = None, device: str = "cpu"):
        """
        Initialize ASM model loader.
        
        Args:
            model_path: Path to model .pt file
            device: Device to run inference on ('cpu' or 'cuda')
        """
        self.model_path = model_path or self.DEFAULT_MODEL_PATH
        self.device = torch.device(device)
        self.model: Optional[nn.Module] = None
        self.loaded = False
    
    def load(self) -> bool:
        """Load model from checkpoint. Returns True if successful."""
        try:
            if not self.model_path.exists():
                print(f"WARNING: ASM model not found at {self.model_path}")
                return False
            
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
            
            # Get config from checkpoint if available
            config = checkpoint.get("config", {})
            input_dim = config.get("input_dim", self.INPUT_DIM)
            hidden_dim = config.get("hidden_dim", self.HIDDEN_DIM)
            num_layers = config.get("num_layers", self.NUM_LAYERS)
            dropout = config.get("dropout", self.DROPOUT)
            
            # Create model
            self.model = ASMGRUClassifier(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                num_classes=self.NUM_CLASSES,
                dropout=dropout,
            )
            
            # Load weights
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.to(self.device)
            self.model.eval()
            self.loaded = True
            
            print(f"ASM Model loaded: {self.model_path}")
            print(f"  Device: {self.device}")
            print(f"  Input: ({self.SEQ_LEN}, {input_dim})")
            
            return True
            
        except Exception as e:
            print(f"ERROR loading ASM model: {e}")
            return False
    
    def predict_proba(self, context: np.ndarray) -> Dict[str, float]:
        """
        Predict probabilities for a context sequence.
        
        Args:
            context: Context sequence, shape (60, 100) or (1, 60, 100)
            
        Returns:
            Dict with p_up, p_down, p_neutral, p_shift
        """
        if not self.loaded or self.model is None:
            return {
                "p_up": 0.33,
                "p_down": 0.33,
                "p_neutral": 0.34,
                "p_shift": 0.66,
            }
        
        # Convert to tensor if needed
        if isinstance(context, np.ndarray):
            X_seq = torch.from_numpy(context).float()
        else:
            X_seq = context
        
        # Add batch dimension if needed
        if X_seq.dim() == 2:
            X_seq = X_seq.unsqueeze(0)  # (1, 60, 100)
        
        # Move to device
        X_seq = X_seq.to(self.device)
        
        # Inference
        with torch.no_grad():
            logits = self.model(X_seq)
            probs = torch.softmax(logits, dim=1)
        
        # Extract probabilities
        probs = probs.cpu().numpy()[0]
        
        return {
            "p_up": float(probs[0]),
            "p_down": float(probs[1]),
            "p_neutral": float(probs[2]),
            "p_shift": float(probs[0] + probs[1]),  # P(UP) + P(DOWN)
        }


# Global model instance
asm_model = ASMModelLoader()
