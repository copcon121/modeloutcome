"""
ASM v2 Dataset Builder

Builds ASM dataset from encoder_dataset_gc_m1_v1.2.jsonl using STATE-ENC v1.2.
"""

import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from tqdm import tqdm
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from state_enc_v1.src.model.state_enc_model import load_state_enc_model


class AsmDatasetBuilder:
    """Build ASM dataset from encoder dataset + STATE-ENC embeddings"""
    
    def __init__(self,
                 encoder_dataset_path: str,
                 state_enc_model_path: str,
                 state_enc_model_config_path: str,
                 meta_features: List[str],
                 regime_mapping: Dict[str, str],
                 device: str = "cuda"):
        self.encoder_dataset_path = encoder_dataset_path
        self.meta_features = meta_features
        self.regime_mapping = regime_mapping
        self.device = device if torch.cuda.is_available() else "cpu"
        
        # Load STATE-ENC model (frozen)
        print(f"Loading STATE-ENC v1.2 from {state_enc_model_path}...")
        self.state_enc = load_state_enc_model(
            state_enc_model_path,
            state_enc_model_config_path,
            device=self.device
        )
        self.state_enc.eval()
        self.z_dim = self.state_enc.get_embedding_dim()
        print(f"STATE-ENC loaded, z_dim={self.z_dim}")
    
    def build(self, output_path: str, splits_output_path: str,
              feature_config_output_path: str, train_ratio: float = 0.8) -> Dict[str, Any]:
        """Build ASM dataset and splits"""
        
        # Load encoder dataset
        samples = self._load_encoder_dataset()
        print(f"Loaded {len(samples)} samples from encoder dataset")
        
        # Filter samples with valid regime hint
        valid_samples = []
        regime_counts = {}
        
        for sample in samples:
            regime_hint = sample.get("regime_hint", 0)
            if regime_hint > 0:
                valid_samples.append(sample)
                regime_counts[regime_hint] = regime_counts.get(regime_hint, 0) + 1
        
        print(f"Valid samples with regime_hint > 0: {len(valid_samples)}")
        print(f"Regime distribution: {regime_counts}")
        
        # Build regime class mapping
        unique_regimes = sorted(regime_counts.keys())
        regime_to_idx = {r: i for i, r in enumerate(unique_regimes)}
        idx_to_regime = {i: r for r, i in regime_to_idx.items()}
        num_classes = len(unique_regimes)
        print(f"Number of regime classes: {num_classes}")
        
        # Process samples and compute embeddings
        asm_samples = []
        dates = []
        
        print("Computing embeddings...")
        batch_size = 32
        
        for i in tqdm(range(0, len(valid_samples), batch_size)):
            batch = valid_samples[i:i+batch_size]
            
            # Stack X tensors
            X_batch = torch.tensor(
                np.array([s["X"] for s in batch]),
                dtype=torch.float32,
                device=self.device
            )
            
            # Compute embeddings
            with torch.no_grad():
                z_batch = self.state_enc.encode(X_batch)  # [B, z_dim]
            
            z_batch = z_batch.cpu().numpy()
            
            for j, sample in enumerate(batch):
                z_t = z_batch[j].tolist()
                
                # Extract meta features from aux
                aux = sample.get("aux", {})
                meta = self._extract_meta_features(aux, sample)
                
                # Get regime label
                regime_hint = sample["regime_hint"]
                label_regime = regime_to_idx[regime_hint]
                
                # Extract date for splitting
                # Date is embedded in the sample - we'll use aux or infer
                date = aux.get("date", f"unknown_{i+j}")
                dates.append(date)
                
                asm_sample = {
                    "z_t": z_t,
                    "meta": meta,
                    "label_regime": label_regime,
                    "regime_hint_raw": regime_hint
                }
                asm_samples.append(asm_sample)
        
        print(f"Built {len(asm_samples)} ASM samples")
        
        # Create time-based split
        unique_dates = sorted(set(dates))
        n_train_dates = int(len(unique_dates) * train_ratio)
        train_dates = set(unique_dates[:n_train_dates])
        val_dates = set(unique_dates[n_train_dates:])
        
        train_indices = [i for i, d in enumerate(dates) if d in train_dates]
        val_indices = [i for i, d in enumerate(dates) if d in val_dates]
        
        print(f"Train dates: {len(train_dates)}, Val dates: {len(val_dates)}")
        print(f"Train samples: {len(train_indices)}, Val samples: {len(val_indices)}")
        
        # Save ASM dataset
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            for sample in asm_samples:
                f.write(json.dumps(sample) + "\n")
        print(f"Saved ASM dataset to {output_path}")
        
        # Save splits
        splits = {
            "train_indices": train_indices,
            "val_indices": val_indices,
            "train_dates": list(train_dates),
            "val_dates": list(val_dates)
        }
        with open(splits_output_path, "w") as f:
            json.dump(splits, f, indent=2)
        print(f"Saved splits to {splits_output_path}")
        
        # Save feature config
        feature_config = {
            "z_dim": self.z_dim,
            "meta_dim": len(self.meta_features),
            "meta_features": self.meta_features,
            "num_classes": num_classes,
            "regime_to_idx": {str(k): v for k, v in regime_to_idx.items()},
            "idx_to_regime": {str(k): v for k, v in idx_to_regime.items()},
            "regime_names": {str(k): self.regime_mapping.get(str(k), f"regime_{k}") 
                           for k in unique_regimes}
        }
        Path(feature_config_output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(feature_config_output_path, "w") as f:
            json.dump(feature_config, f, indent=2)
        print(f"Saved feature config to {feature_config_output_path}")
        
        return {
            "num_samples": len(asm_samples),
            "num_classes": num_classes,
            "train_samples": len(train_indices),
            "val_samples": len(val_indices),
            "regime_counts": regime_counts
        }
    
    def _load_encoder_dataset(self) -> List[Dict]:
        """Load encoder dataset from JSONL"""
        samples = []
        with open(self.encoder_dataset_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))
        return samples
    
    def _extract_meta_features(self, aux: Dict, sample: Dict) -> Dict[str, float]:
        """Extract meta features from aux dict"""
        meta = {}
        
        # Session ID (from sample if available)
        session_map = {"ASIA": 0, "LDN": 1, "NY": 2}
        session = sample.get("session", "ASIA")
        meta["session_id"] = session_map.get(session, 0)
        
        # Position in session range
        meta["pos_in_session_range"] = aux.get("pos_in_session_range", 0.5)
        
        # Value area features (from last bar in X if available)
        # These are typically at indices 72, 73, 74 in feature vector
        X = sample.get("X", [])
        if len(X) > 0:
            last_bar = X[-1]
            # inside_value, above_value, below_value at indices 72, 73, 74
            meta["inside_value"] = 1 if len(last_bar) > 72 and last_bar[72] > 0 else 0
            meta["above_value"] = 1 if len(last_bar) > 73 and last_bar[73] > 0 else 0
            meta["below_value"] = 1 if len(last_bar) > 74 and last_bar[74] > 0 else 0
            # minute_of_day_norm at index 83
            meta["minute_of_day_norm"] = last_bar[83] if len(last_bar) > 83 else 0.0
        else:
            meta["inside_value"] = 0
            meta["above_value"] = 0
            meta["below_value"] = 0
            meta["minute_of_day_norm"] = 0.0
        
        return meta
