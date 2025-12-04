"""
Encoder backbone for STATE-ENC v1

Transformer-based encoder that converts sequence of bar features to embeddings.
"""

import math
import torch
import torch.nn as nn
from typing import Optional, Tuple


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for transformer.
    """
    
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, d_model]
        Returns:
            [B, N, d_model] with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class LearnedPositionalEncoding(nn.Module):
    """
    Learned positional encoding.
    """
    
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class StateEncoderBackbone(nn.Module):
    """
    Transformer encoder backbone for market state encoding.
    
    Architecture:
    1. Linear projection: D -> d_model
    2. Positional encoding
    3. N layers of TransformerEncoder
    4. Pooling to get final embedding z_t
    
    Outputs:
    - z_seq: [B, N, d_model] - full sequence embeddings
    - z_t: [B, d_model] - final state embedding (pooled)
    """
    
    def __init__(self,
                 input_dim: int,
                 d_model: int = 128,
                 num_heads: int = 8,
                 num_layers: int = 4,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1,
                 max_len: int = 512,
                 pooling: str = "last",
                 positional_encoding: str = "sinusoidal"):
        """
        Args:
            input_dim: Input feature dimension D
            d_model: Model dimension
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            dim_feedforward: FFN hidden dimension
            dropout: Dropout rate
            max_len: Maximum sequence length
            pooling: Pooling method ("last", "mean", "cls")
            positional_encoding: Type of positional encoding ("sinusoidal", "learned")
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        self.pooling = pooling
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # CLS token for cls pooling
        if pooling == "cls":
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        
        # Positional encoding
        if positional_encoding == "learned":
            self.pos_encoder = LearnedPositionalEncoding(d_model, max_len, dropout)
        else:
            self.pos_encoder = PositionalEncoding(d_model, max_len, dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-LN for better training stability
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model)
        )
        
        # Output projection (optional refinement)
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model)
        )
    
    def forward(self, 
                x: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input tensor [B, N, D]
            mask: Optional attention mask [B, N]
            
        Returns:
            z_seq: Sequence embeddings [B, N, d_model]
            z_t: Final state embedding [B, d_model]
        """
        B, N, D = x.shape
        
        # Project input
        x = self.input_proj(x)  # [B, N, d_model]
        
        # Add CLS token if using cls pooling
        if self.pooling == "cls":
            cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, d_model]
            x = torch.cat([cls_tokens, x], dim=1)  # [B, N+1, d_model]
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer encoding
        z_seq = self.transformer(x, src_key_padding_mask=mask)
        
        # Pooling to get z_t
        if self.pooling == "cls":
            z_t = z_seq[:, 0, :]  # CLS token
            z_seq = z_seq[:, 1:, :]  # Remove CLS from sequence output
        elif self.pooling == "mean":
            if mask is not None:
                # Masked mean pooling
                mask_expanded = (~mask).unsqueeze(-1).float()
                z_t = (z_seq * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
            else:
                z_t = z_seq.mean(dim=1)
        else:  # "last"
            z_t = z_seq[:, -1, :]  # Last token
        
        # Output projection
        z_t = self.output_proj(z_t)
        
        return z_seq, z_t
    
    def get_embedding_dim(self) -> int:
        """Return embedding dimension"""
        return self.d_model


class LightweightEncoder(nn.Module):
    """
    Lightweight encoder alternative using 1D convolutions + LSTM.
    
    Faster than transformer for inference, suitable for real-time trading.
    """
    
    def __init__(self,
                 input_dim: int,
                 d_model: int = 128,
                 num_conv_layers: int = 3,
                 lstm_layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        
        # Conv layers for local pattern extraction
        conv_layers = []
        in_channels = input_dim
        for i in range(num_conv_layers):
            out_channels = d_model if i == num_conv_layers - 1 else d_model // 2
            conv_layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm1d(out_channels),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
            in_channels = out_channels
        
        self.conv = nn.Sequential(*conv_layers)
        
        # LSTM for sequential modeling
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0,
            bidirectional=False
        )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model)
        )
    
    def forward(self, 
                x: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, N, D]
        Returns:
            z_seq: [B, N, d_model]
            z_t: [B, d_model]
        """
        # Conv expects [B, C, N]
        x = x.transpose(1, 2)  # [B, D, N]
        x = self.conv(x)  # [B, d_model, N]
        x = x.transpose(1, 2)  # [B, N, d_model]
        
        # LSTM
        z_seq, (h_n, c_n) = self.lstm(x)  # z_seq: [B, N, d_model]
        
        # Last hidden state as z_t
        z_t = h_n[-1]  # [B, d_model]
        z_t = self.output_proj(z_t)
        
        return z_seq, z_t
    
    def get_embedding_dim(self) -> int:
        return self.d_model


def create_encoder(config: dict) -> nn.Module:
    """
    Factory function to create encoder from config.
    
    Args:
        config: Model config dict
        
    Returns:
        Encoder module
    """
    encoder_type = config.get("encoder_type", "transformer")
    
    if encoder_type == "lightweight":
        return LightweightEncoder(
            input_dim=config["input_dim"],
            d_model=config.get("d_model", 128),
            num_conv_layers=config.get("num_conv_layers", 3),
            lstm_layers=config.get("lstm_layers", 2),
            dropout=config.get("dropout", 0.1)
        )
    else:
        return StateEncoderBackbone(
            input_dim=config["input_dim"],
            d_model=config.get("d_model", 128),
            num_heads=config.get("num_heads", 8),
            num_layers=config.get("num_layers", 4),
            dim_feedforward=config.get("dim_feedforward", 512),
            dropout=config.get("dropout", 0.1),
            max_len=config.get("sequence_length", 512),
            pooling=config.get("pooling", "last"),
            positional_encoding=config.get("positional_encoding", "sinusoidal")
        )


if __name__ == "__main__":
    # Test encoder
    encoder = StateEncoderBackbone(
        input_dim=95,
        d_model=128,
        num_heads=8,
        num_layers=4
    )
    
    x = torch.randn(4, 128, 95)  # [B, N, D]
    z_seq, z_t = encoder(x)
    
    print(f"Input shape: {x.shape}")
    print(f"z_seq shape: {z_seq.shape}")
    print(f"z_t shape: {z_t.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in encoder.parameters())
    print(f"Total parameters: {total_params:,}")
