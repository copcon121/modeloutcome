"""
Encoder backbone for STATE-ENC v1.2

Fixes:
- RoPE (Rotary Position Embedding) for bar order sensitivity
- 1D Causal Conv before backbone
- GRN (Gated Residual Network) instead of GEGLU
- LatentNorm for drift control
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding with learnable scale"""
    
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.075):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.scale = nn.Parameter(torch.ones(1) * 2.0)  # Increased scale for order sensitivity
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.scale * self.pe[:, :x.size(1), :]
        return self.dropout(x)


class ALiBiPositionalBias(nn.Module):
    """ALiBi (Attention with Linear Biases) for better order sensitivity"""
    
    def __init__(self, num_heads: int, max_len: int = 512):
        super().__init__()
        self.num_heads = num_heads
        
        # Compute slopes for each head
        slopes = torch.tensor([2 ** (-8 * i / num_heads) for i in range(1, num_heads + 1)])
        self.register_buffer('slopes', slopes)
        
        # Pre-compute position differences
        positions = torch.arange(max_len)
        pos_diff = positions.unsqueeze(0) - positions.unsqueeze(1)  # [max_len, max_len]
        self.register_buffer('pos_diff', pos_diff)
    
    def forward(self, seq_len: int) -> torch.Tensor:
        """Returns bias of shape [num_heads, seq_len, seq_len]"""
        pos_diff = self.pos_diff[:seq_len, :seq_len]  # [seq_len, seq_len]
        bias = self.slopes.unsqueeze(1).unsqueeze(2) * pos_diff.unsqueeze(0)  # [num_heads, seq_len, seq_len]
        return bias


class GatedResidualNetwork(nn.Module):
    """
    Gated Residual Network (GRN) - better than GEGLU for robustness
    
    GRN(x) = LayerNorm(x + GLU(Dense(ELU(Dense(x)))))
    """
    
    def __init__(self, d_model: int, hidden_dim: int, dropout: float = 0.075):
        super().__init__()
        
        self.fc1 = nn.Linear(d_model, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, d_model * 2)  # For GLU
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        
        # ELU activation
        h = F.elu(self.fc1(x))
        h = self.dropout(h)
        
        # GLU gate
        h = self.fc2(h)
        h, gate = h.chunk(2, dim=-1)
        h = h * torch.sigmoid(gate)
        h = self.dropout(h)
        
        # Residual + LayerNorm
        return self.norm(residual + h)


class CausalConv1d(nn.Module):
    """1D Causal Convolution for order sensitivity - Enhanced v1.2"""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 5):
        super().__init__()
        self.padding = kernel_size - 1
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=0)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=0)  # Additional conv
        self.norm = nn.LayerNorm(out_channels)
        self.norm2 = nn.LayerNorm(out_channels)
        
        # Position-aware gate
        self.pos_gate = nn.Sequential(
            nn.Linear(out_channels, out_channels),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, N, D]"""
        residual = x
        
        # First causal conv
        x = x.transpose(1, 2)  # [B, D, N]
        x = F.pad(x, (self.padding, 0))
        x = self.conv(x)
        x = x.transpose(1, 2)  # [B, N, D]
        x = F.gelu(x)
        x = self.norm(x)
        
        # Second causal conv for stronger order encoding
        x_t = x.transpose(1, 2)
        x_t = F.pad(x_t, (2, 0))  # kernel=3, padding=2
        x_t = self.conv2(x_t)
        x_t = x_t.transpose(1, 2)
        x_t = F.gelu(x_t)
        x_t = self.norm2(x_t)
        
        # Position-aware gating
        gate = self.pos_gate(x_t)
        x = x * gate + residual * (1 - gate)
        
        return x


class TransformerLayerV12(nn.Module):
    """Transformer layer v1.2 with GRN and RoPE"""
    
    def __init__(self, d_model: int, num_heads: int, dim_feedforward: int, dropout: float = 0.075):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.grn = GatedResidualNetwork(d_model, dim_feedforward, dropout)
        self.dropout = nn.Dropout(dropout)
        # Removed RoPE - using sinusoidal PE at backbone level
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Pre-norm attention
        residual = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x, key_padding_mask=mask)
        x = residual + self.dropout(x)
        
        # GRN instead of FFN
        x = self.grn(x)
        
        return x


class LatentNorm(nn.Module):
    """Latent normalization layer for drift control - Light version v1.2"""
    
    def __init__(self, d_model: int):
        super().__init__()
        # Light normalization - just scale, no centering
        self.scale = nn.Parameter(torch.ones(d_model))
        self.eps = 1e-6
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # Light normalization - preserve relative differences
        z_norm = z / (z.norm(dim=-1, keepdim=True) + self.eps)
        
        # Scale back to reasonable magnitude
        z_out = z_norm * self.scale * z.norm(dim=-1, keepdim=True).mean()
        
        return z_out


class StateEncoderBackbone(nn.Module):
    """
    STATE-ENC v1.2 Backbone
    
    Architecture:
    1. Input projection
    2. Causal Conv1d (kernel=5) for order sensitivity
    3. RoPE position encoding
    4. N x TransformerLayer with GRN
    5. LatentNorm for drift control
    6. Pooling -> z_t
    """
    
    def __init__(self,
                 input_dim: int,
                 d_model: int = 64,
                 num_heads: int = 4,
                 num_layers: int = 4,
                 dim_feedforward: int = 256,
                 dropout: float = 0.075,
                 max_len: int = 256,
                 pooling: str = "last"):
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
        
        # Causal Conv1d for order sensitivity
        self.causal_conv = CausalConv1d(d_model, d_model, kernel_size=5)
        
        # Positional encoding
        self.pos_enc = PositionalEncoding(d_model, max_len, dropout)
        
        # Transformer layers with GRN
        self.layers = nn.ModuleList([
            TransformerLayerV12(d_model, num_heads, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        
        # Final norm
        self.final_norm = nn.LayerNorm(d_model)
        
        # LatentNorm for drift control
        self.latent_norm = LatentNorm(d_model)
        
        # Initialize weights
        self._init_weights()
        
        # Output projection with sensitivity amplification
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model)
        )
        
        # Sensitivity gate - amplifies differences
        self.sensitivity_gate = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )
        
        # Learnable temperature
        self.temperature = nn.Parameter(torch.ones(1) * 0.5)  # Lower temp = more sensitive
    
    def _init_weights(self):
        """Initialize weights properly"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, N, D]
            mask: [B, N]
        Returns:
            z_seq: [B, N, d_model]
            z_t: [B, d_model]
        """
        # Input projection
        x = self.input_proj(x)
        
        # Causal conv for order sensitivity
        x = self.causal_conv(x)
        
        # Positional encoding
        x = self.pos_enc(x)
        
        # Transformer layers
        for layer in self.layers:
            x = layer(x, mask)
        
        # Final norm
        z_seq = self.final_norm(x)
        
        # Pooling
        if self.pooling == "mean":
            if mask is not None:
                mask_expanded = (~mask).unsqueeze(-1).float()
                z_t = (z_seq * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
            else:
                z_t = z_seq.mean(dim=1)
        else:  # "last"
            z_t = z_seq[:, -1, :]
        
        # LatentNorm for drift control
        z_t = self.latent_norm(z_t)
        
        # Output projection
        z_proj = self.output_proj(z_t)
        
        # Sensitivity gate - amplifies important features
        gate = self.sensitivity_gate(z_t)
        z_t = z_proj * (1 + gate)  # Amplify by gate
        
        # Temperature scaling
        z_t = z_t / self.temperature.clamp(min=0.1)
        
        return z_seq, z_t
    
    def get_embedding_dim(self) -> int:
        return self.d_model


def create_encoder(config: dict) -> StateEncoderBackbone:
    return StateEncoderBackbone(
        input_dim=config["input_dim"],
        d_model=config.get("d_model", 64),
        num_heads=config.get("num_heads", 4),
        num_layers=config.get("num_layers", 4),
        dim_feedforward=config.get("dim_feedforward", 256),
        dropout=config.get("dropout", 0.075),
        max_len=config.get("sequence_length", 256),
        pooling=config.get("pooling", "last")
    )
