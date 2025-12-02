"""
MLP Model for Quality Classification

Tabular MLP baseline that flattens 60x66 context window.
"""

import torch
import torch.nn as nn


class QualityMLP(nn.Module):
    """
    Tabular MLP for binary quality classification (KEEP vs DROP)
    
    Input: Flattened [60*66] context + side encoding = 3961 features
    Output: Single logit (for BCEWithLogitsLoss)
    """
    
    def __init__(self, input_dim=3961, hidden_dims=[256, 128], dropout=0.2):
        super().__init__()
        
        self.input_dim = input_dim
        
        layers = []
        prev_dim = input_dim
        
        # Hidden layers
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        #Output layer (single logit)
        layers.append(nn.Linear(prev_dim, 1))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Args:
            x: [batch, input_dim] flattened features
        
        Returns:
            logits: [batch, 1] unnormalized scores
        """
        return self.model(x).squeeze(-1)  # [batch]


def prepare_features(X, side):
    """
    Prepare features for MLP input
    
    Args:
        X: [batch, 60, 66] context windows
        side: [batch] side encoding (+1 long, -1 short)
    
    Returns:
        x_flat: [batch, 3961] flattened features
    """
    batch_size = X.shape[0]
    
    # Flatten context
    x_flat = X.view(batch_size, -1)  # [batch, 60*66=3960]
    
    # Append side as extra feature
    side_feat = side.float().view(batch_size, 1)  # [batch, 1]
    
    # Concatenate
    x_input = torch.cat([x_flat, side_feat], dim=1)  # [batch, 3961]
    
    return x_input
