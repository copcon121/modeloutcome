"""
S4 Context Enricher v3 - For REAL S4_LDN trades

This module:
1. Maps trade entry_time to encoder sample (with tolerance)
2. Computes z_t embedding using STATE-ENC v1.2
3. Predicts regime using ASM v2
4. Outputs enriched trades for policy backtest

Key difference from v2:
- Strict time matching with configurable tolerance
- Better logging for unmatched trades
- Optimized for real data quality
"""

import json
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@dataclass
class S4TradeEnrichedReal:
    """Enriched S4 trade with full context for REAL data."""
    
    # Original trade info
    trade_id: str
    symbol: str
    tf: str
    entry_time: str
    exit_time: str
    session: str
    direction: str
    entry_price: float
    exit_price: float
    sl_price: float
    tp_price: float
    rr: float
    hit: str
    rr_realized: float
    setup_type: str
    
    # Enriched context
    z_t: List[float]  # 64-dim embedding
    regime_id: int
    regime_name: str
    regime_confidence: float
    
    # Meta features
    session_id: int
    minute_of_day_norm: float
    day_of_week: int
    
    # Matching info
    encoder_sample_idx: int  # Index of matched encoder sample
    time_match_delta_sec: int  # Time difference in seconds
    
    # Extra
    extra: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'S4TradeEnrichedReal':
        return cls(**d)
    
    def get_date(self) -> str:
        try:
            dt = datetime.fromisoformat(self.entry_time.replace('Z', '+00:00'))
            return dt.strftime('%Y-%m-%d')
        except:
            return 'unknown'
    
    def get_label(self) -> str:
        if self.hit == 'tp':
            return 'win'
        elif self.hit == 'sl':
            return 'loss'
        return 'flat'


class S4ContextEnricherV3:
    """Context enricher for REAL S4 trades with strict time matching."""
    
    SESSION_MAP = {'ASIA': 0, 'LDN': 1, 'NY': 2}
    REGIME_NAMES = {0: 'trend_up', 1: 'trend_down', 2: 'balance'}
    
    def __init__(
        self,
        state_enc_path: str,
        state_enc_config_path: str,
        feature_config_path: str,
        asm_model_path: str,
        asm_config_path: str,
        encoder_dataset_path: str,
        time_tolerance_minutes: int = 5,
        device: str = 'cpu',
    ):
        self.device = device
        self.time_tolerance = timedelta(minutes=time_tolerance_minutes)
        
        # Load STATE-ENC v1.2
        self.state_enc, self.state_enc_config = self._load_state_enc(
            state_enc_path, state_enc_config_path
        )
        
        # Load feature config
        with open(feature_config_path, 'r') as f:
            self.feature_config = json.load(f)
        
        # Load ASM v2
        self.asm_model, self.asm_config = self._load_asm(
            asm_model_path, asm_config_path
        )
        
        # Load encoder dataset and build time index
        self.encoder_samples = self._load_encoder_dataset(encoder_dataset_path)
        self.time_index = self._build_time_index()
        
        print(f"Enricher V3 initialized:")
        print(f"  STATE-ENC z_dim: {self.state_enc_config.get('z_dim', 64)}")
        print(f"  ASM classes: {self.asm_config.get('num_classes', 3)}")
        print(f"  Encoder samples: {len(self.encoder_samples)}")
        print(f"  Time tolerance: {time_tolerance_minutes} minutes")
    
    def _load_state_enc(self, model_path: str, config_path: str) -> Tuple[nn.Module, dict]:
        """Load STATE-ENC model."""
        from state_enc_v1.src.model.state_enc_model import StateEncModel
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        model = StateEncModel(
            input_dim=config['input_dim'],
            d_model=config['d_model'],
            num_heads=config.get('num_heads', config.get('n_heads', 4)),
            num_layers=config.get('num_layers', config.get('n_layers', 4)),
            dim_feedforward=config.get('dim_feedforward', 256),
            dropout=config.get('dropout', 0.1),
            sequence_length=config.get('sequence_length', 64),
            pooling=config.get('pooling', 'last'),
            heads_config=config.get('heads', {}),
        )
        
        config['z_dim'] = config['d_model']
        
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.to(self.device)
        model.eval()
        
        return model, config
    
    def _load_asm(self, model_path: str, config_path: str) -> Tuple[nn.Module, dict]:
        """Load ASM v2 model."""
        from asm_v2.src.model.asm_model import AsmModel
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        model = AsmModel(
            z_dim=config['z_dim'],
            meta_dim=config['meta_dim'],
            hidden_dim=config['hidden_dim'],
            num_classes=config['num_classes'],
            dropout=config.get('dropout', 0.1),
        )
        
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.to(self.device)
        model.eval()
        
        return model, config
    
    def _load_encoder_dataset(self, path: str) -> List[Dict]:
        """Load encoder dataset samples."""
        samples = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    samples.append(json.loads(line))
                except:
                    continue
        return samples
    
    def _build_time_index(self) -> Dict[str, List[Tuple[int, datetime]]]:
        """Build index of samples by date with end_time for fast lookup."""
        index = {}
        for i, sample in enumerate(self.encoder_samples):
            meta = sample.get('meta', {})
            end_time_str = meta.get('end_time', '')
            
            try:
                end_time = datetime.fromisoformat(end_time_str.replace('Z', '+00:00'))
                date_key = end_time.strftime('%Y-%m-%d')
                
                if date_key not in index:
                    index[date_key] = []
                index[date_key].append((i, end_time))
            except:
                continue
        
        # Sort each date's samples by time
        for date_key in index:
            index[date_key].sort(key=lambda x: x[1])
        
        return index
    
    def _normalize_datetime(self, dt: datetime) -> datetime:
        """Remove timezone info for comparison."""
        if dt.tzinfo is not None:
            return dt.replace(tzinfo=None)
        return dt
    
    def _find_matching_sample(self, entry_time: datetime) -> Optional[Tuple[int, int]]:
        """Find encoder sample matching trade entry time.
        
        Returns:
            Tuple of (sample_index, time_delta_seconds) or None if no match
        """
        # Normalize entry_time to naive datetime
        entry_time = self._normalize_datetime(entry_time)
        date_key = entry_time.strftime('%Y-%m-%d')
        
        # Check same day with tolerance
        if date_key in self.time_index:
            for idx, sample_end_time in self.time_index[date_key]:
                sample_end_time = self._normalize_datetime(sample_end_time)
                delta = abs((entry_time - sample_end_time).total_seconds())
                if delta <= self.time_tolerance.total_seconds():
                    return idx, int(delta)
        
        # Check adjacent days (for edge cases)
        for day_offset in [-1, 1]:
            adj_date = (entry_time + timedelta(days=day_offset)).strftime('%Y-%m-%d')
            if adj_date in self.time_index:
                for idx, sample_end_time in self.time_index[adj_date]:
                    sample_end_time = self._normalize_datetime(sample_end_time)
                    delta = abs((entry_time - sample_end_time).total_seconds())
                    if delta <= self.time_tolerance.total_seconds():
                        return idx, int(delta)
        
        # Fallback: find closest sample from same day or any day
        best_match = None
        best_delta = float('inf')
        
        # Try same day first
        if date_key in self.time_index:
            for idx, sample_end_time in self.time_index[date_key]:
                sample_end_time = self._normalize_datetime(sample_end_time)
                delta = abs((entry_time - sample_end_time).total_seconds())
                if delta < best_delta:
                    best_delta = delta
                    best_match = (idx, int(delta))
        
        # If no same-day match, try nearby days
        if best_match is None:
            for day_offset in range(-7, 8):
                check_date = (entry_time + timedelta(days=day_offset)).strftime('%Y-%m-%d')
                if check_date in self.time_index:
                    for idx, sample_end_time in self.time_index[check_date]:
                        sample_end_time = self._normalize_datetime(sample_end_time)
                        delta = abs((entry_time - sample_end_time).total_seconds())
                        if delta < best_delta:
                            best_delta = delta
                            best_match = (idx, int(delta))
                    if best_match:
                        break  # Found a match in nearby day
        
        # Last resort: use first available sample
        if best_match is None and self.encoder_samples:
            best_match = (0, 999999)
        
        return best_match
    
    def _compute_z_t(self, sample: Dict) -> np.ndarray:
        """Compute z_t embedding from encoder sample."""
        seq = sample.get('seq', [])
        if not seq:
            return np.zeros(64)
        
        # Try both 'feature_names' and 'features' keys
        feature_names = self.feature_config.get('feature_names', self.feature_config.get('features', []))
        
        if not feature_names:
            print("WARNING: No feature names found in feature_config!")
            return np.zeros(64)
        
        seq_data = []
        for bar in seq:
            bar_features = []
            for fname in feature_names:
                val = bar.get(fname, 0.0)
                if val is None:
                    val = 0.0
                try:
                    bar_features.append(float(val))
                except:
                    bar_features.append(0.0)
            seq_data.append(bar_features)
        
        seq_len = self.state_enc_config.get('sequence_length', 64)
        while len(seq_data) < seq_len:
            seq_data.insert(0, [0.0] * len(feature_names))
        seq_data = seq_data[-seq_len:]
        
        x = torch.tensor([seq_data], dtype=torch.float32, device=self.device)
        
        with torch.no_grad():
            outputs = self.state_enc(x)
            z_t = outputs['z_t']
        
        return z_t.cpu().numpy()[0]
    
    def _compute_regime(self, z_t: np.ndarray, meta_features: List[float]) -> Tuple[int, str, float]:
        """Compute regime prediction from ASM v2."""
        z_np = np.array(z_t, dtype=np.float32)
        meta_np = np.array(meta_features, dtype=np.float32)
        x = np.concatenate([z_np, meta_np])
        x_tensor = torch.tensor([x], dtype=torch.float32, device=self.device)
        
        with torch.no_grad():
            outputs = self.asm_model(x_tensor)
            probs = outputs['probs']
            pred_class = torch.argmax(probs, dim=-1).item()
            confidence = probs[0, pred_class].item()
        
        regime_name = self.REGIME_NAMES.get(pred_class, 'unknown')
        
        return pred_class, regime_name, confidence
    
    def _compute_meta_features(self, trade: Dict, entry_dt: datetime) -> List[float]:
        """Compute meta features for ASM model."""
        session = trade.get('session', 'LDN').upper()
        session_id = self.SESSION_MAP.get(session, 1)
        
        minute_of_day = entry_dt.hour * 60 + entry_dt.minute
        minute_of_day_norm = minute_of_day / 1440.0
        
        # Meta features matching ASM training
        return [
            session_id,
            0.5,  # pos_in_session_range (default)
            0,    # inside_value
            0,    # above_value
            0,    # below_value
            minute_of_day_norm,
        ]
    
    def enrich_trade(self, trade: Dict) -> Optional[S4TradeEnrichedReal]:
        """Enrich a single trade with context."""
        entry_time_str = trade.get('entry_time', '')
        
        try:
            entry_dt = datetime.fromisoformat(entry_time_str.replace('Z', '+00:00'))
        except:
            return None
        
        # Find matching encoder sample
        match_result = self._find_matching_sample(entry_dt)
        
        if match_result is None:
            return None
        
        sample_idx, time_delta = match_result
        sample = self.encoder_samples[sample_idx]
        
        # Compute z_t
        z_t = self._compute_z_t(sample)
        
        # Compute meta features
        meta_features = self._compute_meta_features(trade, entry_dt)
        
        # Compute regime
        regime_id, regime_name, regime_conf = self._compute_regime(z_t, meta_features)
        
        session = trade.get('session', 'LDN').upper()
        
        return S4TradeEnrichedReal(
            trade_id=trade.get('trade_id', ''),
            symbol=trade.get('symbol', 'GC'),
            tf=trade.get('tf', 'M1'),
            entry_time=entry_time_str,
            exit_time=trade.get('exit_time', ''),
            session=session,
            direction=trade.get('direction', 'long'),
            entry_price=trade.get('entry_price', 0),
            exit_price=trade.get('exit_price', 0),
            sl_price=trade.get('sl_price', 0),
            tp_price=trade.get('tp_price', 0),
            rr=trade.get('rr', 0),
            hit=trade.get('hit', 'sl'),
            rr_realized=trade.get('rr_realized', 0),
            setup_type=trade.get('setup_type', 'S4_LDN'),
            z_t=z_t.tolist(),
            regime_id=regime_id,
            regime_name=regime_name,
            regime_confidence=regime_conf,
            session_id=self.SESSION_MAP.get(session, 1),
            minute_of_day_norm=meta_features[5],
            day_of_week=entry_dt.weekday(),
            encoder_sample_idx=sample_idx,
            time_match_delta_sec=time_delta,
            extra=trade.get('extra', {}),
        )
    
    def enrich_trades(self, trades: List[Dict], show_progress: bool = True) -> Tuple[List[S4TradeEnrichedReal], List[Dict]]:
        """Enrich multiple trades.
        
        Returns:
            Tuple of (enriched_trades, unmatched_trades)
        """
        enriched = []
        unmatched = []
        
        if show_progress:
            from tqdm import tqdm
            iterator = tqdm(trades, desc="Enriching trades")
        else:
            iterator = trades
        
        for trade in iterator:
            try:
                enriched_trade = self.enrich_trade(trade)
                if enriched_trade:
                    enriched.append(enriched_trade)
                else:
                    unmatched.append(trade)
            except Exception as e:
                print(f"Warning: Failed to enrich trade {trade.get('trade_id', '?')}: {e}")
                unmatched.append(trade)
        
        return enriched, unmatched


def save_enriched_trades_real(trades: List[S4TradeEnrichedReal], path: str):
    """Save enriched trades to JSONL."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for trade in trades:
            f.write(json.dumps(trade.to_dict()) + '\n')


def load_enriched_trades_real(path: str) -> List[S4TradeEnrichedReal]:
    """Load enriched trades from JSONL."""
    trades = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                trades.append(S4TradeEnrichedReal.from_dict(d))
            except Exception as e:
                print(f"Warning: Failed to parse trade: {e}")
                continue
    return trades
