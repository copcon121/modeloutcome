#!/usr/bin/env python3
"""
ASM Inference v1.0
==================
Load and run inference with ASM-GRU64-v1.0-C3 model.

Usage:
    from scripts.asm_inference_v1 import ASMModelV1Loader
    
    loader = ASMModelV1Loader()
    probs = loader.predict_proba(X_seq)  # X_seq: (60, 100) or (1, 60, 100)
"""

from pathlib import Path
from typing import Dict, Union

import numpy as np
import torch
import torch.nn as nn


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


class ASMModelV1Loader:
    """
    Load and run inference with ASM-GRU64-v1.0-C3 model.
    
    Model outputs 3 classes: UP (0), DOWN (1), NEUTRAL (2)
    """
    
    # Default paths
    DEFAULT_MODEL_PATH = Path("output/asm_models_v1/ASM-GRU64-v1.0-C3.pt")
    DEFAULT_METRICS_PATH = Path("output/asm_models_v1/ASM-GRU64-v1.0-C3_metrics.json")
    
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
        
        # Load model
        self.model = self._load_model()
        self.model.eval()
        
        print(f"ASM Model loaded: {self.model_path}")
        print(f"  Device: {self.device}")
        print(f"  Input: ({self.SEQ_LEN}, {self.INPUT_DIM})")
    
    def _load_model(self) -> nn.Module:
        """Load model from checkpoint."""
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        
        # Get config from checkpoint if available
        config = checkpoint.get("config", {})
        input_dim = config.get("input_dim", self.INPUT_DIM)
        hidden_dim = config.get("hidden_dim", self.HIDDEN_DIM)
        num_layers = config.get("num_layers", self.NUM_LAYERS)
        dropout = config.get("dropout", self.DROPOUT)
        
        # Create model
        model = ASMGRUClassifier(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_classes=self.NUM_CLASSES,
            dropout=dropout,
        )
        
        # Load weights
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self.device)
        
        return model
    
    def predict_proba(self, X_seq: Union[np.ndarray, torch.Tensor]) -> Dict[str, float]:
        """
        Predict probabilities for a single context sequence.
        
        Args:
            X_seq: Context sequence, shape (60, 100) or (1, 60, 100)
            
        Returns:
            Dict with p_up, p_down, p_neutral, p_shift
        """
        # Convert to tensor if needed
        if isinstance(X_seq, np.ndarray):
            X_seq = torch.from_numpy(X_seq).float()
        
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
    
    def predict_batch(self, X_batch: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Predict probabilities for a batch of sequences.
        
        Args:
            X_batch: Batch of sequences, shape (N, 60, 100)
            
        Returns:
            Array of shape (N, 3) with probabilities [p_up, p_down, p_neutral]
        """
        if isinstance(X_batch, np.ndarray):
            X_batch = torch.from_numpy(X_batch).float()
        
        X_batch = X_batch.to(self.device)
        
        with torch.no_grad():
            logits = self.model(X_batch)
            probs = torch.softmax(logits, dim=1)
        
        return probs.cpu().numpy()


# ==============================================================================
# TEST
# ==============================================================================


if __name__ == "__main__":
    # Test loading
    loader = ASMModelV1Loader()
    
    # Test with random input
    X_test = np.random.randn(60, 100).astype(np.float32)
    probs = loader.predict_proba(X_test)
    
    print(f"\nTest prediction:")
    print(f"  p_up:      {probs['p_up']:.4f}")
    print(f"  p_down:    {probs['p_down']:.4f}")
    print(f"  p_neutral: {probs['p_neutral']:.4f}")
    print(f"  p_shift:   {probs['p_shift']:.4f}")
