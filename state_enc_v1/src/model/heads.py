"""
Prediction heads for STATE-ENC v1.2

New heads:
- ShockDetectionHead: detect market shocks
- FeatureReconstructHead: reconstruct masked features
- OrderDetectionHead: detect shuffled sequences
- LatentAnchorHead: anchor loss for drift control
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
import random


class FocalLoss(nn.Module):
    """Focal Loss with gamma=1.5"""
    def __init__(self, gamma: float = 1.5, weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


class ShockDetectionHead(nn.Module):
    """
    Detect market shocks (binary classification) - Enhanced v1.2
    Shock = |return_1| > threshold OR true_range spike
    Uses attention to focus on shock-relevant features
    """
    
    def __init__(self, input_dim: int, dropout: float = 0.075):
        super().__init__()
        
        # Shock-sensitive feature extractor
        self.shock_encoder = nn.Sequential(
            nn.Linear(input_dim, input_dim * 4),
            nn.LayerNorm(input_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim * 4, input_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # Attention for shock detection
        self.shock_attn = nn.MultiheadAttention(input_dim * 2, num_heads=4, dropout=dropout, batch_first=True)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(input_dim * 2, input_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim, 2)  # Binary: shock / no-shock
        )
        
        # Shock magnitude predictor (auxiliary)
        self.magnitude_head = nn.Sequential(
            nn.Linear(input_dim * 2, input_dim),
            nn.GELU(),
            nn.Linear(input_dim, 1)
        )
    
    def forward(self, z_t: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Encode shock features
        h = self.shock_encoder(z_t)
        
        # Self-attention for shock patterns
        h_unsq = h.unsqueeze(1)  # [B, 1, D*2]
        h_attn, _ = self.shock_attn(h_unsq, h_unsq, h_unsq)
        h = h + h_attn.squeeze(1)  # Residual
        
        # Classify
        logits = self.classifier(h)
        magnitude = self.magnitude_head(h).squeeze(-1)
        
        return {
            "shock_logits": logits,
            "shock_magnitude": magnitude
        }


class FeatureReconstructHead(nn.Module):
    """
    Reconstruct masked features for robustness
    """
    
    def __init__(self, d_model: int, feature_dim: int, num_reconstruct: int = 10, dropout: float = 0.075):
        super().__init__()
        self.num_reconstruct = num_reconstruct
        
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.LayerNorm(d_model * 2),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, feature_dim)
        )
    
    def forward(self, z_seq: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Reconstruct features from sequence embeddings"""
        return {"reconstructed_features": self.decoder(z_seq)}


class OrderDetectionHead(nn.Module):
    """
    Detect if sequence has been shuffled (binary classification) - Enhanced v1.2
    Uses position-aware encoding to detect order violations
    """
    
    def __init__(self, input_dim: int, dropout: float = 0.075):
        super().__init__()
        
        # Order-sensitive encoder
        self.order_encoder = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.LayerNorm(input_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim * 2, input_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # Position consistency checker
        self.pos_checker = nn.Sequential(
            nn.Linear(input_dim * 2, input_dim),
            nn.GELU(),
            nn.Linear(input_dim, input_dim // 2),
            nn.GELU(),
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(input_dim // 2, input_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim // 4, 2)  # Binary: shuffled / not-shuffled
        )
        
        # Order score (continuous)
        self.order_score = nn.Sequential(
            nn.Linear(input_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, z_t: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.order_encoder(z_t)
        h = self.pos_checker(h)
        
        logits = self.classifier(h)
        score = self.order_score(h).squeeze(-1)
        
        return {
            "order_logits": logits,
            "order_score": score
        }


class LatentAnchorHead(nn.Module):
    """
    Compute anchor loss for drift control - Enhanced v1.2
    Same-session, low-vol bars should have similar embeddings
    Uses multiple anchors and contrastive learning
    """
    
    def __init__(self, d_model: int, num_anchors: int = 4):
        super().__init__()
        self.d_model = d_model
        self.num_anchors = num_anchors
        
        # Multiple anchors for different regimes
        self.register_buffer('anchors', torch.zeros(num_anchors, d_model))
        self.register_buffer('anchor_counts', torch.zeros(num_anchors))
        self.momentum = 0.95
        
        # Anchor assignment network
        self.anchor_assigner = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, num_anchors)
        )
        
        # Drift predictor
        self.drift_predictor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1)
        )
    
    def update_anchors(self, z_t: torch.Tensor, assignments: torch.Tensor):
        """Update anchors with soft assignment"""
        with torch.no_grad():
            soft_assign = F.softmax(assignments, dim=-1)  # [B, num_anchors]
            
            for i in range(self.num_anchors):
                weights = soft_assign[:, i]  # [B]
                if weights.sum() > 0.1:
                    weighted_mean = (z_t * weights.unsqueeze(-1)).sum(dim=0) / (weights.sum() + 1e-8)
                    
                    if self.anchor_counts[i] == 0:
                        self.anchors[i] = weighted_mean
                    else:
                        self.anchors[i] = self.momentum * self.anchors[i] + (1 - self.momentum) * weighted_mean
                    self.anchor_counts[i] += weights.sum()
    
    def forward(self, z_t: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Assign to anchors
        assignments = self.anchor_assigner(z_t)  # [B, num_anchors]
        
        if self.training:
            self.update_anchors(z_t, assignments)
        
        # Compute distance to nearest anchor
        soft_assign = F.softmax(assignments, dim=-1)  # [B, num_anchors]
        
        # Distance to each anchor
        z_expanded = z_t.unsqueeze(1)  # [B, 1, D]
        anchors_expanded = self.anchors.unsqueeze(0)  # [1, num_anchors, D]
        
        distances = ((z_expanded - anchors_expanded) ** 2).mean(dim=-1)  # [B, num_anchors]
        
        # Weighted distance (soft assignment)
        anchor_dist = (distances * soft_assign).sum(dim=-1)  # [B]
        
        # Drift prediction
        drift_pred = self.drift_predictor(z_t).squeeze(-1)
        
        return {
            "anchor_distance": anchor_dist,
            "anchor_assignments": assignments,
            "drift_prediction": drift_pred,
            "anchors": self.anchors
        }


class SelfSupervisedHead(nn.Module):
    """Self-supervised head v1.2"""
    
    def __init__(self, input_dim: int, feature_dim: int, num_dir_classes: int = 3,
                 predict_return: bool = True, dropout: float = 0.075):
        super().__init__()
        self.predict_return = predict_return
        
        hidden_dim = input_dim * 2
        
        self.dir_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_dir_classes)
        )
        
        if predict_return:
            self.return_head = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1)
            )
    
    def forward(self, z_t: torch.Tensor) -> Dict[str, torch.Tensor]:
        outputs = {"dir_logits": self.dir_head(z_t)}
        if self.predict_return:
            outputs["return_pred"] = self.return_head(z_t).squeeze(-1)
        return outputs


class RegimeHead(nn.Module):
    """Regime classification head v1.2"""
    
    def __init__(self, input_dim: int, num_classes: int = 6, dropout: float = 0.075):
        super().__init__()
        hidden_dim = input_dim * 2
        
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
        return {"regime_logits": self.head(z_t)}


class MetaS4Head(nn.Module):
    """Meta S4 head"""
    
    def __init__(self, input_dim: int, output_dim: int = 4, dropout: float = 0.075):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.LayerNorm(input_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim, output_dim)
        )
    
    def forward(self, z_t: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {"meta_output": self.head(z_t)}


class DataAugmentationV12:
    """
    Data augmentation v1.2 with scheduled feature dropout and shock injection
    Enhanced for better shock and order sensitivity
    """
    
    def __init__(self, epoch: int = 1):
        self.epoch = epoch
        self.ohlc_indices = [0, 1, 2, 3]
    
    def set_epoch(self, epoch: int):
        self.epoch = epoch
    
    def get_dropout_rate(self) -> float:
        """Scheduled dropout rate"""
        if self.epoch <= 2:
            return 0.05
        elif self.epoch <= 4:
            return 0.10
        else:
            return 0.20
    
    def inject_shock(self, x: torch.Tensor, shock_prob: float = 0.35) -> Tuple[torch.Tensor, torch.Tensor]:
        """Inject synthetic shocks - Enhanced for better detection"""
        B, N, D = x.shape
        shocked = x.clone()
        shock_labels = torch.zeros(B, dtype=torch.long, device=x.device)
        
        for b in range(B):
            if random.random() < shock_prob:
                # Inject shock in last 5-10 bars
                shock_len = random.randint(3, 8)
                shock_start = N - shock_len
                
                # Multiple shock types combined
                shock_magnitude = random.uniform(2.0, 4.0)
                
                # Price shock
                for idx in self.ohlc_indices:
                    if idx < D:
                        direction = random.choice([1, -1])
                        shocked[b, shock_start:, idx] += direction * shocked[b, shock_start:, idx].abs() * (shock_magnitude - 1)
                
                # Volume spike
                if 9 < D:
                    shocked[b, shock_start:, 9] *= shock_magnitude * 1.5
                
                # Delta spike
                if 15 < D:
                    shocked[b, shock_start:, 15] *= shock_magnitude
                
                # Add noise to make it more realistic
                noise = torch.randn_like(shocked[b, shock_start:]) * 0.1
                shocked[b, shock_start:] += noise
                
                shock_labels[b] = 1
        
        return shocked, shock_labels
    
    def shuffle_sequence(self, x: torch.Tensor, shuffle_prob: float = 0.4) -> Tuple[torch.Tensor, torch.Tensor]:
        """Shuffle some sequences for order detection - Enhanced"""
        B, N, D = x.shape
        shuffled = x.clone()
        order_labels = torch.zeros(B, dtype=torch.long, device=x.device)
        
        for b in range(B):
            if random.random() < shuffle_prob:
                # More aggressive shuffling
                num_swaps = random.randint(3, 6)
                
                for _ in range(num_swaps):
                    # Swap non-adjacent positions for more disruption
                    i = random.randint(0, N // 2 - 1)
                    j = random.randint(N // 2, N - 1)
                    shuffled[b, i], shuffled[b, j] = shuffled[b, j].clone(), shuffled[b, i].clone()
                
                order_labels[b] = 1
        
        return shuffled, order_labels
    
    def mask_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Mask features with scheduled dropout"""
        B, N, D = x.shape
        dropout_rate = self.get_dropout_rate()
        
        mask = torch.rand(B, N, D, device=x.device) < dropout_rate
        masked = x.clone()
        masked[mask] = 0.0
        
        return masked, mask
    
    def add_temporal_noise(self, x: torch.Tensor, noise_scale: float = 0.05) -> torch.Tensor:
        """Add temporal-aware noise"""
        B, N, D = x.shape
        
        # Noise increases towards the end of sequence
        time_weights = torch.linspace(0.5, 1.5, N, device=x.device).unsqueeze(0).unsqueeze(-1)
        noise = torch.randn_like(x) * noise_scale * time_weights
        
        return x + noise
    
    def __call__(self, x: torch.Tensor, training: bool = True) -> Dict[str, torch.Tensor]:
        """Apply all augmentations"""
        if not training:
            return {
                "x": x,
                "shock_labels": torch.zeros(x.shape[0], dtype=torch.long, device=x.device),
                "order_labels": torch.zeros(x.shape[0], dtype=torch.long, device=x.device),
                "feature_mask": torch.zeros_like(x, dtype=torch.bool),
                "original_x": x
            }
        
        original_x = x.clone()
        
        # Apply augmentations in sequence
        x_shocked, shock_labels = self.inject_shock(x.clone())
        x_shuffled, order_labels = self.shuffle_sequence(x_shocked)
        x_masked, feature_mask = self.mask_features(x_shuffled)
        x_noisy = self.add_temporal_noise(x_masked)
        
        return {
            "x": x_noisy,
            "original_x": original_x,
            "shock_labels": shock_labels,
            "order_labels": order_labels,
            "feature_mask": feature_mask
        }
