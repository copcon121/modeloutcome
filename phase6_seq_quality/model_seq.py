"""
Sequence Quality Model - GRU Architecture

Lightweight GRU-based model that preserves temporal structure.
"""

import torch
import torch.nn as nn


class QualitySeqGRU(nn.Module):
    """
    GRU-based quality classifier
    
    Architecture:
        Input: [B, 60, D] sequence + [B, 1] side
        -> Concat side as extra feature channel
        -> GRU layers
        -> MLP head
        -> Output: [B, 1] logit
    """
    
    def __init__(self, input_dim, hidden_dim=128, num_layers=1, dropout=0.1):
        """
        Args:
            input_dim: Feature dimension D (66)
            hidden_dim: GRU hidden size
            num_layers: Number of GRU layers
            dropout: Dropout rate
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Side embedding (simple: just expand and concat)
        # We'll concatenate side as an extra feature channel
        self.gru_input_dim = input_dim + 1  # D + side
        
        # GRU layers
        self.gru = nn.GRU(
            input_size=self.gru_input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        # MLP head
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )
        
        # Count params
        total_params = sum(p.numel() for p in self.parameters())
        print(f"[QualitySeqGRU] Initialized with {total_params:,} parameters")
        print(f"  Input dim: {input_dim} + 1 (side) = {self.gru_input_dim}")
        print(f"  Hidden dim: {hidden_dim}")
        print(f"  Num layers: {num_layers}")
    
    def forward(self, X_seq, side):
        """
        Forward pass
        
        Args:
            X_seq: [B, T, D] sequence (T=60, D=66)
            side: [B, 1] or [B] with +1/-1
        
        Returns:
            logits: [B, 1]
        """
        B, T, D = X_seq.shape
        
        # Handle side shape
        if side.dim() == 1:
            side = side.unsqueeze(1)  # [B] -> [B, 1]
        
        # Expand side across time: [B, 1] -> [B, T, 1]
        side_expanded = side.unsqueeze(1).expand(B, T, 1)
        
        # Concatenate side with features
        X_with_side = torch.cat([X_seq, side_expanded], dim=-1)  # [B, T, D+1]
        
        # GRU forward
        # output: [B, T, hidden_dim]
        # h_n: [num_layers, B, hidden_dim]
        output, h_n = self.gru(X_with_side)
        
        # Use last hidden state from last layer
        last_hidden = h_n[-1]  # [B, hidden_dim]
        
        # MLP head
        logits = self.head(last_hidden)  # [B, 1]
        
        return logits


if __name__ == "__main__":
    # Test model
    print("="*60)
    print("TESTING SEQUENCE MODEL")
    print("="*60)
    
    # Create model
    model = QualitySeqGRU(input_dim=66, hidden_dim=128, num_layers=1, dropout=0.1)
    
    # Dummy input
    B = 4
    T = 60
    D = 66
    
    X_dummy = torch.randn(B, T, D)
    side_dummy = torch.FloatTensor([1, -1, 1, -1]).unsqueeze(1)
    
    # Forward
    logits = model(X_dummy, side_dummy)
    
    print(f"\nInput shapes:")
    print(f"  X_seq: {X_dummy.shape}")
    print(f"  side: {side_dummy.shape}")
    print(f"\nOutput:")
    print(f"  logits: {logits.shape}")
    print(f"  values: {logits.squeeze()}")
    
    # Test with sigmoid
    probs = torch.sigmoid(logits)
    print(f"  probs: {probs.squeeze()}")
    
    print(f"\nModel test complete!")
