"""
S4_LDN Context Enricher

Enriches S4_LDN trade events with:
- z_t embedding from STATE-ENC v1.2
- regime_pred from ASM v2
- meta features
"""

import json
import glob
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
from tqdm import tqdm
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from state_enc_v1.src.model.state_enc_model import load_state_enc_model
from asm_v2.src.model.asm_model import load_asm_model


class S4ContextEnricher:
    """Enriches S4_LDN trades with STATE-ENC embeddings and ASM regime predictions"""
    
    def __init__(self,
                 state_enc_model_path: str,
                 state_enc_config_path: str,
                 state_enc_feature_config_path: str,
                 asm_model_path: str,
                 asm_config_path: str,
                 asm_feature_config_path: str,
                 device: str = "cuda"):
        
        self.device = device if torch.cuda.is_available() else "cpu"
        
        # Load STATE-ENC v1.2
        print(f"Loading STATE-ENC v1.2...")
        self.state_enc = load_state_enc_model(
            state_enc_model_path,
            state_enc_config_path,
            device=self.device
        )
        self.state_enc.eval()
        
        # Load feature config for STATE-ENC
        with open(state_enc_feature_config_path, "r") as f:
            self.state_enc_feature_config = json.load(f)
        
        # Load ASM v2
        print(f"Loading ASM v2...")
        self.asm_model = load_asm_model(
            asm_model_path,
            asm_config_path,
            device=self.device
        )
        self.asm_model.eval()
        
        # Load ASM feature config
        with open(asm_feature_config_path, "r") as f:
            self.asm_feature_config = json.load(f)
        
        self.regime_names = self.asm_feature_config.get("regime_names", {})
        self.idx_to_regime = self.asm_feature_config.get("idx_to_regime", {})
        
        # Sequence length from STATE-ENC config
        with open(state_enc_config_path, "r") as f:
            model_config = json.load(f)
        self.seq_len = model_config.get("sequence_length", 64)
        self.feature_dim = model_config.get("input_dim", 88)
        
        print(f"Initialized: seq_len={self.seq_len}, feature_dim={self.feature_dim}")
    
    def load_bar_data(self, bars_glob: str) -> Dict[str, List[Dict]]:
        """Load bar data from JSONL files, indexed by date"""
        bars_by_date = {}
        
        files = sorted(glob.glob(bars_glob))
        print(f"Loading bar data from {len(files)} files...")
        
        for filepath in files:
            with open(filepath, "r") as f:
                for line in f:
                    if line.strip():
                        raw_bar = json.loads(line)
                        # Handle smc_export format: timestamp field, nested bar data
                        time_str = raw_bar.get("timestamp", raw_bar.get("time", ""))
                        if time_str:
                            date = time_str[:10]  # YYYY-MM-DD
                            if date not in bars_by_date:
                                bars_by_date[date] = []
                            
                            # Flatten bar data for feature extraction
                            bar = {"time": time_str, "timestamp": time_str}
                            bar.update(raw_bar.get("bar", {}))
                            bar.update(raw_bar.get("tick_features", {}))
                            bar["bar_index"] = raw_bar.get("bar_index", 0)
                            bar["session"] = raw_bar.get("session", "ASIA")
                            
                            bars_by_date[date].append(bar)
        
        # Sort bars by time within each date
        for date in bars_by_date:
            bars_by_date[date].sort(key=lambda x: x.get("timestamp", x.get("time", "")))
        
        total_bars = sum(len(v) for v in bars_by_date.values())
        print(f"Loaded {total_bars} bars across {len(bars_by_date)} dates")
        
        return bars_by_date
    
    def load_s4_trades(self, s4_file: str) -> List[Dict]:
        """Load S4_LDN trade events"""
        trades = []
        with open(s4_file, "r") as f:
            for line in f:
                if line.strip():
                    trades.append(json.loads(line))
        print(f"Loaded {len(trades)} S4 trades")
        return trades
    
    def enrich_trades(self,
                      bars_by_date: Dict[str, List[Dict]],
                      trades: List[Dict],
                      output_path: str) -> Dict[str, Any]:
        """Enrich trades with context"""
        
        enriched_trades = []
        stats = {
            "total_trades": len(trades),
            "enriched_trades": 0,
            "skipped_no_bars": 0,
            "regime_distribution": {}
        }
        
        print("Enriching trades...")
        for trade in tqdm(trades):
            # Extract trade time
            trade_time = trade.get("time", "")
            if not trade_time:
                stats["skipped_no_bars"] += 1
                continue
            
            trade_date = trade_time[:10]
            
            # Get bars for this date
            bars = bars_by_date.get(trade_date, [])
            if len(bars) < self.seq_len:
                stats["skipped_no_bars"] += 1
                continue
            
            # Find bar index for trade time
            bar_idx = self._find_bar_index(bars, trade_time)
            if bar_idx < self.seq_len:
                stats["skipped_no_bars"] += 1
                continue
            
            # Get context bars (N bars before and including entry)
            context_bars = bars[bar_idx - self.seq_len + 1:bar_idx + 1]
            
            # Build feature tensor
            X = self._build_feature_tensor(context_bars)
            if X is None:
                stats["skipped_no_bars"] += 1
                continue
            
            # Compute embedding
            with torch.no_grad():
                X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(0).to(self.device)
                z_t = self.state_enc.encode(X_tensor).squeeze(0).cpu().numpy()
            
            # Extract meta features from entry bar
            entry_bar = context_bars[-1]
            meta = self._extract_meta_features(entry_bar)
            
            # Compute regime prediction
            with torch.no_grad():
                meta_vec = torch.tensor([
                    meta["session_id"],
                    meta["pos_in_session_range"],
                    meta["inside_value"],
                    meta["above_value"],
                    meta["below_value"],
                    meta["minute_of_day_norm"]
                ], dtype=torch.float32)
                
                x = torch.cat([torch.tensor(z_t), meta_vec]).unsqueeze(0).to(self.device)
                asm_out = self.asm_model(x)
                regime_pred = asm_out["logits"].argmax(dim=-1).item()
            
            # Get regime name
            regime_hint_raw = self.idx_to_regime.get(str(regime_pred), regime_pred)
            regime_name = self.regime_names.get(str(regime_hint_raw), f"regime_{regime_pred}")
            
            # Build enriched trade
            enriched_trade = trade.copy()
            enriched_trade["context"] = {
                "z_t": z_t.tolist(),
                "asm_regime_pred": regime_pred,
                "asm_regime_name": regime_name,
                "session_id": meta["session_id"],
                "pos_in_session_range": meta["pos_in_session_range"],
                "inside_value": meta["inside_value"],
                "above_value": meta["above_value"],
                "below_value": meta["below_value"]
            }
            
            enriched_trades.append(enriched_trade)
            stats["enriched_trades"] += 1
            
            # Track regime distribution
            if regime_name not in stats["regime_distribution"]:
                stats["regime_distribution"][regime_name] = {"total": 0, "win": 0, "loss": 0}
            stats["regime_distribution"][regime_name]["total"] += 1
            
            outcome = trade.get("label", "flat")
            if outcome == "win":
                stats["regime_distribution"][regime_name]["win"] += 1
            elif outcome == "loss":
                stats["regime_distribution"][regime_name]["loss"] += 1
        
        # Save enriched trades
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            for trade in enriched_trades:
                f.write(json.dumps(trade) + "\n")
        
        print(f"Saved {len(enriched_trades)} enriched trades to {output_path}")
        
        return stats
    
    def _find_bar_index(self, bars: List[Dict], trade_time: str) -> int:
        """Find bar index matching trade time"""
        # Normalize trade_time format
        trade_time_norm = trade_time.replace("Z", "").replace(".000", "")
        
        for i, bar in enumerate(bars):
            bar_time = bar.get("timestamp", bar.get("time", ""))
            bar_time_norm = bar_time.replace("Z", "").split(".")[0]
            
            if bar_time_norm >= trade_time_norm[:19]:  # Compare YYYY-MM-DDTHH:MM:SS
                return i
        return len(bars) - 1
    
    def _build_feature_tensor(self, bars: List[Dict]) -> Optional[np.ndarray]:
        """Build feature tensor from bars"""
        # This is a simplified version - in practice, use the same
        # normalization as the encoder dataset builder
        
        feature_names = self.state_enc_feature_config.get("feature_names", [])
        
        X = np.zeros((len(bars), self.feature_dim), dtype=np.float32)
        
        for i, bar in enumerate(bars):
            for j, feat_name in enumerate(feature_names):
                if j >= self.feature_dim:
                    break
                X[i, j] = bar.get(feat_name, 0.0)
        
        return X
    
    def _extract_meta_features(self, bar: Dict) -> Dict[str, float]:
        """Extract meta features from bar"""
        session_map = {"ASIA": 0, "LDN": 1, "NY": 2}
        session = bar.get("session", "ASIA")
        
        return {
            "session_id": session_map.get(session, 0),
            "pos_in_session_range": bar.get("pos_in_session_range", 0.5),
            "inside_value": 1 if bar.get("inside_value", 0) > 0 else 0,
            "above_value": 1 if bar.get("above_value", 0) > 0 else 0,
            "below_value": 1 if bar.get("below_value", 0) > 0 else 0,
            "minute_of_day_norm": bar.get("minute_of_day_norm", 0.0)
        }
