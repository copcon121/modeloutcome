"""
Multi-Model Quality Inference System

Supports both tabular (legacy) and sequence (primary) quality models.
Provides unified routing interface for production deployment.
"""

import sys
import json
from pathlib import Path
import numpy as np
import torch

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from phase4_quality_tabular.model import QualityMLP, prepare_features as prepare_tabular
from phase6_seq_quality.model_seq import QualitySeqGRU


class TabularQualityService:
    """
    Legacy tabular quality model service
    
    Uses MLP on flattened [60×66] features
    """
    
    def __init__(self, model_path=None, normalizer_path=None, device='cpu'):
        self.device = torch.device(device)
        
        # Default paths
        if model_path is None or normalizer_path is None:
            root = Path(__file__).parent.parent
            model_path = model_path or root / "output/phase4_quality/model_tabular_quality_v1_best.pt"
            normalizer_path = normalizer_path or root / "output/phase4_quality/normalizer_stats.pt"
        
        # Load model
        self.model = QualityMLP(input_dim=3961)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        # Load normalizer
        normalizer = torch.load(normalizer_path, map_location=self.device)
        self.norm_mean = normalizer['mean'].to(self.device)
        self.norm_std = normalizer['std'].to(self.device)
        
        print(f"[TabularQualityService] Loaded (LEGACY)")
    
    def predict_p_keep(self, X_seq, side):
        """
        Predict keep probability
        
        Args:
            X_seq: [60, 66] or [1, 60, 66]
            side: scalar or [1]
        
        Returns:
            float: p_keep
        """
        # Ensure batch dimension
        if isinstance(X_seq, np.ndarray):
            X_seq = torch.FloatTensor(X_seq)
        
        if X_seq.dim() == 2:
            X_seq = X_seq.unsqueeze(0)
        
        # Handle side
        if isinstance(side, (int, float, np.integer)):
            side = torch.LongTensor([side])
        elif isinstance(side, np.ndarray):
            side = torch.LongTensor(side)
        
        # Flatten and normalize (tabular approach)
        X_flat = prepare_tabular(X_seq, side)
        X_norm = (X_flat - self.norm_mean) / self.norm_std
        X_norm = X_norm.to(self.device)
        
        # Predict
        with torch.no_grad():
            logit = self.model(X_norm)
            p_keep = torch.sigmoid(logit).item()
        
        return p_keep


class SeqQualityService:
    """
    Primary sequence quality model service
    
    Uses GRU on [60×66] time-series
    """
    
    def __init__(self, model_path=None, normalizer_path=None, device='cpu'):
        self.device = torch.device(device)
        
        # Default paths
        if model_path is None or normalizer_path is None:
            root = Path(__file__).parent.parent
            model_path = model_path or root / "output/phase6_seq_quality/model_seq_quality_v1_best.pt"
            normalizer_path = normalizer_path or root / "output/phase6_seq_quality/normalizer_stats_seq.pt"
        
        # Load model
        self.model = QualitySeqGRU(input_dim=66, hidden_dim=128, num_layers=1, dropout=0.1)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        # Load normalizer
        normalizer = torch.load(normalizer_path, map_location=self.device)
        self.norm_mean = normalizer['mean'].to(self.device)  # [D]
        self.norm_std = normalizer['std'].to(self.device)  # [D]
        
        print(f"[SeqQualityService] Loaded (PRIMARY)")
    
    def predict_p_keep(self, X_seq, side):
        """
        Predict keep probability
        
        Args:
            X_seq: [60, 66] or [1, 60, 66]
            side: scalar or [1]
        
        Returns:
            float: p_keep
        """
        # Convert to tensor
        if isinstance(X_seq, np.ndarray):
            X_seq = torch.FloatTensor(X_seq)
        
        # Ensure batch dimension
        if X_seq.dim() == 2:
            X_seq = X_seq.unsqueeze(0)  # [1, 60, 66]
        
        # Normalize (per-feature across time)
        X_norm = (X_seq - self.norm_mean) / (self.norm_std + 1e-8)
        X_norm = X_norm.to(self.device)
        
        # Handle side
        if isinstance(side, (int, float, np.integer)):
            side_tensor = torch.FloatTensor([[float(side)]]).to(self.device)
        elif isinstance(side, np.ndarray):
            side_tensor = torch.FloatTensor([[float(side)]]).to(self.device)
        else:
            side_tensor = side.float().to(self.device)
            if side_tensor.dim() == 0:
                side_tensor = side_tensor.unsqueeze(0).unsqueeze(0)
            elif side_tensor.dim() == 1:
                side_tensor = side_tensor.unsqueeze(1)
        
        # Predict
        with torch.no_grad():
            logit = self.model(X_norm, side_tensor)
            p_keep = torch.sigmoid(logit).item()
        
        return p_keep


class QualityModelRouter:
    """
    Routes  requests to appropriate quality model (tabular or sequence)
    
    Reads config from quality_model_modes.json
    """
    
    def __init__(self, modes_config_path=None, device='cpu'):
        """
        Args:
            modes_config_path: Path to quality_model_modes.json
            device: 'cpu' or 'cuda'
        """
        self.device = device
        
        # Default config path
        if modes_config_path is None:
            root = Path(__file__).parent.parent
            modes_config_path = root / "output/phase5_quality/quality_model_modes.json"
        
        # Load config
        with open(modes_config_path, 'r') as f:
            self.config = json.load(f)
        
        self.global_default = self.config['global_default_model']
        
        # Initialize services (lazy loading could be added)
        print(f"[QualityModelRouter] Initializing services...")
        self.tabular_service = TabularQualityService(device=device)
        self.seq_service = SeqQualityService(device=device)
        
        print(f"[QualityModelRouter] Ready")
        print(f"  Global default: {self.global_default}")
        print(f"  Available models: {list(self.config['models'].keys())}")
    
    def predict(self, X_seq, side, model_type=None, mode=None, threshold=None):
        """
        Predict with model routing
        
        Args:
            X_seq: [60, D] sequence
            side: +1 or -1
            model_type: 'tabular_v1' or 'seq_v1' or None (uses global default)
            mode: Mode name or None (uses model default)
            threshold: Custom threshold or None (uses config)
        
        Returns:
            dict: {
                'model_type': str,
                'mode': str or None,
                'threshold': float,
                'p_keep': float,
                'keep': bool
            }
        """
        # Determine model type
        if model_type is None:
            model_type = self.global_default
        
        # Validate model type
        if model_type not in self.config['models']:
            raise ValueError(f"Invalid model_type: {model_type}. Available: {list(self.config['models'].keys())}")
        
        model_config = self.config['models'][model_type]
        
        # Determine mode
        if mode is None:
            mode = model_config['default_mode']
        
        # Validate mode
        if mode not in model_config['modes']:
            raise ValueError(f"Invalid mode '{mode}' for model '{model_type}'. Available: {list(model_config['modes'].keys())}")
        
        mode_config = model_config['modes'][mode]
        
        # Determine threshold
        if threshold is None:
            threshold = mode_config['threshold']
        
        # Route to appropriate service
        if model_type == 'tabular_v1':
            p_keep = self.tabular_service.predict_p_keep(X_seq, side)
        elif model_type == 'seq_v1':
            p_keep = self.seq_service.predict_p_keep(X_seq, side)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
        
        # Decision
        keep = p_keep >= threshold
        
        return {
            'model_type': model_type,
            'mode': mode,
            'threshold': threshold,
            'p_keep': p_keep,
            'keep': keep,
            'side': int(side) if isinstance(side, (np.integer, torch.Tensor)) else side
        }


def create_router(device='cpu'):
    """Factory function to create router with default paths"""
    return QualityModelRouter(device=device)


if __name__ == "__main__":
    # Test router
    print("="*60)
    print("TESTING MULTI-MODEL ROUTER")
    print("="*60)
    
    router = create_router()
    
    # Dummy data
    X_dummy = np.random.randn(60, 66).astype(np.float32)
    side_dummy = 1
    
    # Test 1: Default (should use seq_v1)
    print(f"\n1. DEFAULT (no args):")
    result = router.predict(X_dummy, side_dummy)
    print(f"   Model: {result['model_type']}, Mode: {result['mode']}, p_keep: {result['p_keep']:.4f}, keep: {result['keep']}")
    
    # Test 2: Tabular balanced
    print(f"\n2. TABULAR BALANCED:")
    result = router.predict(X_dummy, side_dummy, model_type='tabular_v1', mode='balanced')
    print(f"   Model: {result['model_type']}, Mode: {result['mode']}, p_keep: {result['p_keep']:.4f}, keep: {result['keep']}")
    
    # Test 3: Seq conservative
    print(f"\n3. SEQ CONSERVATIVE:")
    result = router.predict(X_dummy, side_dummy, model_type='seq_v1', mode='seq_conservative')
    print(f"   Model: {result['model_type']}, Mode: {result['mode']}, Threshold: {result['threshold']}, p_keep: {result['p_keep']:.4f}, keep: {result['keep']}")
    
    # Test 4: Custom threshold
    print(f"\n4. CUSTOM THRESHOLD:")
    result = router.predict(X_dummy, side_dummy, threshold=0.9)
    print(f"   Model: {result['model_type']}, Threshold: {result['threshold']}, p_keep: {result['p_keep']:.4f}, keep: {result['keep']}")
    
    print(f"\n{'='*60}")
    print("ROUTER TEST COMPLETE!")
    print("="*60)
