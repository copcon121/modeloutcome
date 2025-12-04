"""
S4 Context Enricher v2 - Full enrichment pipeline for S4 trades

This module:
1. Loads standardized S4 trades
2. For each trade, builds context sequence from encoder dataset
3. Computes z_t embedding using STATE-ENC v1.2
4. Predicts regime using ASM v2
5. Adds meta features
6. Outputs enriched trades for policy backtest
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

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@dataclass
class S4TradeEnrichedV2:
    """Enriched S4 trade with full context."""
    
    # Original trade info
    trade_id: str
    symbol: str
    tf: str
    time: str
    date: str
    day_of_week: int
    session: str
    bar_index: int
    direction: str
    entry: float
    sl: float
    tp: float
    outcome_rr: float
    hit: str
    label: str
    setup_type: str
    signal_type: str
    
    # Enriched context
    z_t: List[float]  # 64-dim embedding
    regime_id: int
    regime_name: str
    regime_confidence: float
    regime_onehot: List[float]  # [3] for 3 classes
    
    # Meta features
    session_id: int  # 0=ASIA, 1=LDN, 2=NY
    pos_in_session_range: float
    minute_of_day_norm: float
    day_of_week_norm: float
    inside_value: int
    above_value: int
    below_value: int
    
    # Optional extra
    meta: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'S4TradeEnrichedV2':
        return cls(**d)


class S4ContextEnricherV2:
    """Full context enricher for S4 trades."""
    
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
        device: str = 'cpu',
    ):
        self.device = device
        
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
        
        # Load encoder dataset for context lookup
        self.encoder_samples = self._load_encoder_dataset(encoder_dataset_path)
        
        # Build date index for fast lookup
        self.date_index = self._build_date_index()
        
        print(f"Enricher initialized:")
        print(f"  STATE-ENC z_dim: {self.state_enc_config.get('z_dim', 64)}")
        print(f"  ASM classes: {self.asm_config.get('num_classes', 3)}")
        print(f"  Encoder samples: {len(self.encoder_samples)}")
    
    def _load_state_enc(self, model_path: str, config_path: str) -> Tuple[nn.Module, dict]:
        """Load STATE-ENC model."""
        from state_enc_v1.src.model.state_enc_model import StateEncModel
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Map config keys (handle different naming conventions)
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
        
        # Add z_dim to config for later use
        config['z_dim'] = config['d_model']  # z_t dim = d_model
        
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
        from asm_v2.src.model.asm_model import AsmModel as ASMModel
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        model = ASMModel(
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
    
    def _build_date_index(self) -> Dict[str, List[int]]:
        """Build index of samples by date for fast lookup."""
        index = {}
        for i, sample in enumerate(self.encoder_samples):
            meta = sample.get('meta', {})
            date = meta.get('date', 'unknown')
            if date not in index:
                index[date] = []
            index[date].append(i)
        return index
    
    def _find_context_sample(self, trade_date: str, trade_session: str) -> Optional[Dict]:
        """Find best matching encoder sample for trade context."""
        # Try exact date match
        if trade_date in self.date_index:
            indices = self.date_index[trade_date]
            # Prefer same session
            for idx in indices:
                sample = self.encoder_samples[idx]
                if sample.get('meta', {}).get('session', '').upper() == trade_session:
                    return sample
            # Fallback to any sample from same date
            if indices:
                return self.encoder_samples[indices[0]]
        
        # Try nearby dates
        try:
            dt = datetime.fromisoformat(trade_date)
            for offset in [1, -1, 2, -2, 3, -3]:
                nearby_date = (dt + timedelta(days=offset)).strftime('%Y-%m-%d')
                if nearby_date in self.date_index:
                    return self.encoder_samples[self.date_index[nearby_date][0]]
        except:
            pass
        
        # Fallback to random sample
        if self.encoder_samples:
            return self.encoder_samples[0]
        
        return None
    
    def _compute_z_t(self, sample: Dict) -> np.ndarray:
        """Compute z_t embedding from encoder sample."""
        seq = sample.get('seq', [])
        if not seq:
            return np.zeros(64)
        
        # Extract features according to feature config
        feature_names = self.feature_config.get('features', [])
        
        # Build tensor
        seq_data = []
        for bar in seq:
            bar_features = []
            for fname in feature_names:
                val = bar.get(fname, 0.0)
                if val is None:
                    val = 0.0
                bar_features.append(float(val))
            seq_data.append(bar_features)
        
        # Pad/truncate to expected length
        seq_len = self.state_enc_config.get('seq_len', 64)
        while len(seq_data) < seq_len:
            seq_data.insert(0, [0.0] * len(feature_names))
        seq_data = seq_data[-seq_len:]
        
        # Convert to tensor
        x = torch.tensor([seq_data], dtype=torch.float32, device=self.device)
        
        # Forward pass
        with torch.no_grad():
            outputs = self.state_enc(x)
            z_t = outputs['z_t']  # [1, z_dim]
        
        return z_t.cpu().numpy()[0]
    
    def _compute_regime(self, z_t: np.ndarray, meta_features: List[float]) -> Tuple[int, str, float, List[float]]:
        """Compute regime prediction from ASM v2."""
        # Prepare input - ASM model expects concatenated [z_t, meta]
        z_np = np.array(z_t, dtype=np.float32)
        meta_np = np.array(meta_features, dtype=np.float32)
        x = np.concatenate([z_np, meta_np])
        x_tensor = torch.tensor([x], dtype=torch.float32, device=self.device)
        
        # Forward pass
        with torch.no_grad():
            outputs = self.asm_model(x_tensor)  # Returns dict with logits, probs
            probs = outputs['probs']  # [1, num_classes]
            pred_class = torch.argmax(probs, dim=-1).item()
            confidence = probs[0, pred_class].item()
            onehot = probs[0].cpu().numpy().tolist()
        
        regime_name = self.REGIME_NAMES.get(pred_class, 'unknown')
        
        return pred_class, regime_name, confidence, onehot
    
    def _compute_meta_features(self, trade: Dict) -> Tuple[List[float], Dict[str, Any]]:
        """Compute meta features for trade."""
        session = trade.get('session', 'LDN').upper()
        session_id = self.SESSION_MAP.get(session, 1)
        
        # Parse time
        time_str = trade.get('time', '')
        try:
            dt = datetime.fromisoformat(time_str.replace('Z', '+00:00'))
            minute_of_day = dt.hour * 60 + dt.minute
            minute_of_day_norm = minute_of_day / 1440.0
            day_of_week = dt.weekday()
            day_of_week_norm = day_of_week / 6.0
        except:
            minute_of_day_norm = 0.5
            day_of_week_norm = 0.5
        
        # Position in session (simplified)
        pos_in_session = 0.5  # Default middle
        
        # Value area flags (simplified - would need actual VA data)
        inside_value = 0
        above_value = 0
        below_value = 0
        
        # Meta features for ASM model (must match training)
        meta_features = [
            session_id,
            pos_in_session,
            inside_value,
            above_value,
            below_value,
            minute_of_day_norm,
        ]
        
        meta_dict = {
            'session_id': session_id,
            'pos_in_session_range': pos_in_session,
            'minute_of_day_norm': minute_of_day_norm,
            'day_of_week_norm': day_of_week_norm,
            'inside_value': inside_value,
            'above_value': above_value,
            'below_value': below_value,
        }
        
        return meta_features, meta_dict
    
    def enrich_trade(self, trade: Dict) -> S4TradeEnrichedV2:
        """Enrich a single trade with context."""
        # Find context sample
        trade_date = trade.get('date', '')
        trade_session = trade.get('session', 'LDN').upper()
        context_sample = self._find_context_sample(trade_date, trade_session)
        
        # Compute z_t
        if context_sample:
            z_t = self._compute_z_t(context_sample)
        else:
            z_t = np.zeros(64)
        
        # Compute meta features
        meta_features, meta_dict = self._compute_meta_features(trade)
        
        # Compute regime
        regime_id, regime_name, regime_conf, regime_onehot = self._compute_regime(z_t, meta_features)
        
        # Build enriched trade
        return S4TradeEnrichedV2(
            trade_id=trade.get('trade_id', ''),
            symbol=trade.get('symbol', 'GC'),
            tf=trade.get('tf', 'M1'),
            time=trade.get('time', ''),
            date=trade.get('date', ''),
            day_of_week=trade.get('day_of_week', 0),
            session=trade_session,
            bar_index=trade.get('bar_index', 0),
            direction=trade.get('direction', 'long'),
            entry=trade.get('entry', 0),
            sl=trade.get('sl', 0),
            tp=trade.get('tp', 0),
            outcome_rr=trade.get('outcome_rr', 0),
            hit=trade.get('hit', 'sl'),
            label=trade.get('label', 'loss'),
            setup_type=trade.get('setup_type', 'unknown'),
            signal_type=trade.get('signal_type', ''),
            z_t=z_t.tolist(),
            regime_id=regime_id,
            regime_name=regime_name,
            regime_confidence=regime_conf,
            regime_onehot=regime_onehot,
            session_id=meta_dict['session_id'],
            pos_in_session_range=meta_dict['pos_in_session_range'],
            minute_of_day_norm=meta_dict['minute_of_day_norm'],
            day_of_week_norm=meta_dict['day_of_week_norm'],
            inside_value=meta_dict['inside_value'],
            above_value=meta_dict['above_value'],
            below_value=meta_dict['below_value'],
            meta=trade.get('meta', {}),
        )
    
    def enrich_trades(self, trades: List[Dict], show_progress: bool = True) -> List[S4TradeEnrichedV2]:
        """Enrich multiple trades."""
        enriched = []
        
        if show_progress:
            from tqdm import tqdm
            iterator = tqdm(trades, desc="Enriching trades")
        else:
            iterator = trades
        
        for trade in iterator:
            try:
                enriched_trade = self.enrich_trade(trade)
                enriched.append(enriched_trade)
            except Exception as e:
                print(f"Warning: Failed to enrich trade {trade.get('trade_id', '?')}: {e}")
                continue
        
        return enriched


def save_enriched_trades(trades: List[S4TradeEnrichedV2], path: str):
    """Save enriched trades to JSONL."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for trade in trades:
            f.write(json.dumps(trade.to_dict()) + '\n')


def load_enriched_trades_v2(path: str) -> List[S4TradeEnrichedV2]:
    """Load enriched trades from JSONL."""
    trades = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                trades.append(S4TradeEnrichedV2.from_dict(d))
            except Exception as e:
                print(f"Warning: Failed to parse trade: {e}")
                continue
    return trades
