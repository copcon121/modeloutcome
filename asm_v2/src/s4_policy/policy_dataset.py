"""
S4 Policy Dataset - Load enriched trades for policy evaluation

Schema assumptions for enriched trades (from s4_context_enricher.py):
REQUIRED fields:
- symbol: str (e.g., "GC 12-25")
- tf: str (e.g., "M1")
- time: str ISO format (e.g., "2025-09-01T02:30:00.000Z")
- signal: str ("long" or "short")
- entry: float (entry price)
- sl: float (stop loss price)
- tp: float (take profit price)
- outcome_rr: float (realized R multiple)
- label: str ("win", "loss", "flat")
- context.asm_regime_pred: int (regime class index)
- context.asm_regime_name: str (regime name)
- context.z_t: list[float] length 64

OPTIONAL fields:
- bar_index: int
- session: str (from trade or context)
- context.session_id: int
- context.pos_in_session_range: float
- context.inside_value: int
- context.above_value: int
- context.below_value: int
- context.regime_confidence: float (if available)
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
import numpy as np


@dataclass
class S4TradeEnriched:
    """Enriched S4 trade with context from STATE-ENC and ASM v2."""
    
    # Core trade info (REQUIRED)
    symbol: str
    tf: str
    time: datetime
    direction: str  # "long" or "short"
    entry_price: float
    sl_price: float
    tp_price: float
    rr_outcome: float  # realized R
    label: str  # "win", "loss", "flat"/"be"
    
    # ASM v2 regime (REQUIRED)
    regime: int  # class index
    regime_name: str  # human readable
    
    # STATE-ENC embedding (REQUIRED)
    z_t: List[float]  # 64-dim
    
    # Optional meta
    regime_confidence: float = 0.0  # prob of predicted class
    session: str = "unknown"
    session_id: int = 0
    pos_in_session_range: float = 0.5
    inside_value: int = 0
    above_value: int = 0
    below_value: int = 0
    bar_index: int = 0
    
    # Raw data for debugging
    raw: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'S4TradeEnriched':
        """Parse from enriched trade JSON dict."""
        context = d.get('context', {})
        
        # Parse time
        time_str = d.get('time', '')
        try:
            time = datetime.fromisoformat(time_str.replace('Z', '+00:00'))
        except:
            time = datetime.now()
        
        # Direction from signal
        signal = d.get('signal', 'long')
        direction = signal if signal in ['long', 'short'] else 'long'
        
        # Session - try multiple sources
        session = d.get('session', context.get('session', 'unknown'))
        
        return cls(
            symbol=d.get('symbol', 'unknown'),
            tf=d.get('tf', 'M1'),
            time=time,
            direction=direction,
            entry_price=float(d.get('entry', 0)),
            sl_price=float(d.get('sl', 0)),
            tp_price=float(d.get('tp', 0)),
            rr_outcome=float(d.get('outcome_rr', 0)),
            label=d.get('label', 'loss'),
            regime=int(context.get('asm_regime_pred', 0)),
            regime_name=context.get('asm_regime_name', 'unknown'),
            z_t=context.get('z_t', [0.0] * 64),
            regime_confidence=float(context.get('regime_confidence', 0.0)),
            session=session,
            session_id=int(context.get('session_id', 0)),
            pos_in_session_range=float(context.get('pos_in_session_range', 0.5)),
            inside_value=int(context.get('inside_value', 0)),
            above_value=int(context.get('above_value', 0)),
            below_value=int(context.get('below_value', 0)),
            bar_index=int(d.get('bar_index', 0)),
            raw=d
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert back to dict."""
        return {
            'symbol': self.symbol,
            'tf': self.tf,
            'time': self.time.isoformat(),
            'direction': self.direction,
            'entry_price': self.entry_price,
            'sl_price': self.sl_price,
            'tp_price': self.tp_price,
            'rr_outcome': self.rr_outcome,
            'label': self.label,
            'regime': self.regime,
            'regime_name': self.regime_name,
            'regime_confidence': self.regime_confidence,
            'session': self.session,
            'session_id': self.session_id,
            'pos_in_session_range': self.pos_in_session_range,
            'z_t': self.z_t,
        }
    
    def get_date(self) -> str:
        """Get date string for time-based splitting."""
        return self.time.strftime('%Y-%m-%d')


def load_enriched_trades(path: str) -> List[S4TradeEnriched]:
    """Load enriched trades from JSONL file."""
    trades = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                trade = S4TradeEnriched.from_dict(d)
                trades.append(trade)
            except Exception as e:
                print(f"Warning: Failed to parse trade: {e}")
                continue
    
    # Sort by time
    trades.sort(key=lambda t: t.time)
    return trades


class S4PolicyDataset:
    """Dataset for S4 policy evaluation."""
    
    def __init__(self, trades: List[S4TradeEnriched]):
        self.trades = trades
    
    @classmethod
    def from_file(cls, path: str) -> 'S4PolicyDataset':
        """Load from JSONL file."""
        trades = load_enriched_trades(path)
        return cls(trades)
    
    def __len__(self) -> int:
        return len(self.trades)
    
    def __getitem__(self, idx: int) -> S4TradeEnriched:
        return self.trades[idx]
    
    def get_dates(self) -> List[str]:
        """Get unique dates sorted."""
        dates = sorted(set(t.get_date() for t in self.trades))
        return dates
    
    def time_split(self, train_ratio: float = 0.8) -> Tuple['S4PolicyDataset', 'S4PolicyDataset']:
        """Split by time (dates), not shuffle."""
        dates = self.get_dates()
        n_train = int(len(dates) * train_ratio)
        train_dates = set(dates[:n_train])
        val_dates = set(dates[n_train:])
        
        train_trades = [t for t in self.trades if t.get_date() in train_dates]
        val_trades = [t for t in self.trades if t.get_date() in val_dates]
        
        return S4PolicyDataset(train_trades), S4PolicyDataset(val_trades)
    
    def to_numpy(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Convert to numpy arrays for ML model.
        
        Returns:
            z_t: (N, 64) embeddings
            meta: (N, M) meta features
            labels: (N,) binary labels (1=win, 0=loss)
        """
        z_t_list = []
        meta_list = []
        labels = []
        
        for t in self.trades:
            if t.label in ['win', 'loss']:  # Skip flat/be for binary classification
                z_t_list.append(t.z_t)
                meta_list.append([
                    t.regime,
                    t.session_id,
                    t.pos_in_session_range,
                    t.inside_value,
                    t.above_value,
                    t.below_value,
                    1 if t.direction == 'long' else 0,
                ])
                labels.append(1 if t.label == 'win' else 0)
        
        return (
            np.array(z_t_list, dtype=np.float32),
            np.array(meta_list, dtype=np.float32),
            np.array(labels, dtype=np.int64)
        )
    
    def get_regime_distribution(self) -> Dict[str, int]:
        """Get regime distribution."""
        dist = {}
        for t in self.trades:
            name = t.regime_name
            dist[name] = dist.get(name, 0) + 1
        return dist
    
    def get_label_distribution(self) -> Dict[str, int]:
        """Get label distribution."""
        dist = {}
        for t in self.trades:
            dist[t.label] = dist.get(t.label, 0) + 1
        return dist
    
    def filter_by_label(self, labels: List[str]) -> 'S4PolicyDataset':
        """Filter trades by label."""
        filtered = [t for t in self.trades if t.label in labels]
        return S4PolicyDataset(filtered)
