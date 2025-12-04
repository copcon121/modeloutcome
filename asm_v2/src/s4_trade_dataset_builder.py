"""
S4 Trade Dataset Builder - Build standardized trade dataset from raw S4 trades

This module:
1. Reads raw S4 trade files (from Ninja export or manual)
2. Standardizes schema
3. Adds derived fields (date, day_of_week, bar_index_in_session, etc.)
4. Outputs clean JSONL for enrichment pipeline
"""

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import random


@dataclass
class S4TradeRaw:
    """Standardized S4 trade record."""
    
    # Core identifiers
    trade_id: str
    symbol: str
    tf: str
    
    # Timing
    time: str  # ISO format
    date: str  # YYYY-MM-DD
    day_of_week: int  # 0=Mon, 6=Sun
    session: str  # ASIA, LDN, NY
    bar_index: int
    
    # Trade details
    direction: str  # long, short
    entry: float
    sl: float
    tp: float
    
    # Outcome
    outcome_rr: float
    hit: str  # tp, sl, be, timeout
    label: str  # win, loss, flat
    
    # Setup info
    setup_type: str  # fvg_retest, ob_retest, etc.
    signal_type: str  # s4_ldn_long, s4_ldn_short
    
    # Optional meta
    meta: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'S4TradeRaw':
        """Parse from raw dict with flexible field names."""
        # Parse time
        time_str = d.get('time', d.get('entry_time', ''))
        try:
            dt = datetime.fromisoformat(time_str.replace('Z', '+00:00'))
            date_str = dt.strftime('%Y-%m-%d')
            day_of_week = dt.weekday()
        except:
            dt = datetime.now()
            date_str = dt.strftime('%Y-%m-%d')
            day_of_week = dt.weekday()
        
        # Direction normalization
        direction = d.get('direction', d.get('signal', 'long')).lower()
        if direction not in ['long', 'short']:
            direction = 'long' if 'long' in direction.lower() else 'short'
        
        # Hit/outcome normalization
        hit = d.get('hit', d.get('outcome', 'sl')).lower()
        if hit not in ['tp', 'sl', 'be', 'timeout']:
            hit = 'sl'
        
        # Label from hit
        label = d.get('label', '')
        if not label:
            if hit == 'tp':
                label = 'win'
            elif hit == 'sl':
                label = 'loss'
            else:
                label = 'flat'
        
        # Session normalization
        session = d.get('session', 'LDN').upper()
        if session in ['ASIA', 'ASIAN']:
            session = 'ASIA'
        elif session in ['LDN', 'LONDON']:
            session = 'LDN'
        elif session in ['NY', 'NEW_YORK', 'NEWYORK']:
            session = 'NY'
        
        return cls(
            trade_id=d.get('trade_id', d.get('id', f"trade_{hash(time_str) % 100000}")),
            symbol=d.get('symbol', 'GC'),
            tf=d.get('tf', d.get('timeframe', 'M1')),
            time=time_str,
            date=date_str,
            day_of_week=day_of_week,
            session=session,
            bar_index=int(d.get('bar_index', 0)),
            direction=direction,
            entry=float(d.get('entry', d.get('entry_price', 0))),
            sl=float(d.get('sl', d.get('stop_loss', 0))),
            tp=float(d.get('tp', d.get('take_profit', 0))),
            outcome_rr=float(d.get('outcome_rr', d.get('rr', 0))),
            hit=hit,
            label=label,
            setup_type=d.get('setup_type', d.get('setup', 'unknown')),
            signal_type=d.get('signal_type', f"s4_ldn_{direction}"),
            meta=d.get('meta', {}),
        )


def load_raw_trades(path: str) -> List[S4TradeRaw]:
    """Load raw trades from JSONL file."""
    trades = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                trade = S4TradeRaw.from_dict(d)
                trades.append(trade)
            except Exception as e:
                print(f"Warning: Failed to parse trade: {e}")
                continue
    
    # Sort by time
    trades.sort(key=lambda t: t.time)
    return trades


def save_trades(trades: List[S4TradeRaw], path: str):
    """Save trades to JSONL file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for trade in trades:
            f.write(json.dumps(trade.to_dict()) + '\n')


def generate_synthetic_trades(
    n_trades: int = 200,
    start_date: str = "2025-09-01",
    end_date: str = "2025-11-30",
    symbol: str = "GC 12-25",
    base_price: float = 2600.0,
    win_rate: float = 0.55,
    avg_win_rr: float = 2.5,
    avg_loss_rr: float = -1.0,
) -> List[S4TradeRaw]:
    """Generate synthetic S4 trades for testing.
    
    Creates realistic distribution of trades across dates, sessions, directions.
    """
    import random
    from datetime import timedelta
    
    random.seed(42)
    
    start = datetime.fromisoformat(start_date)
    end = datetime.fromisoformat(end_date)
    total_days = (end - start).days
    
    sessions = ['ASIA', 'LDN', 'NY']
    session_weights = [0.2, 0.5, 0.3]  # LDN most common
    session_hours = {'ASIA': (2, 8), 'LDN': (8, 14), 'NY': (14, 20)}
    
    directions = ['long', 'short']
    setup_types = ['fvg_retest', 'ob_retest', 'liquidity_sweep', 'breaker_block']
    
    trades = []
    price = base_price
    
    for i in range(n_trades):
        # Random date
        day_offset = random.randint(0, total_days)
        trade_date = start + timedelta(days=day_offset)
        
        # Skip weekends
        while trade_date.weekday() >= 5:
            day_offset = random.randint(0, total_days)
            trade_date = start + timedelta(days=day_offset)
        
        # Random session
        session = random.choices(sessions, weights=session_weights)[0]
        hour_range = session_hours[session]
        hour = random.randint(hour_range[0], hour_range[1] - 1)
        minute = random.randint(0, 59)
        
        trade_time = trade_date.replace(hour=hour, minute=minute, second=0)
        
        # Direction
        direction = random.choice(directions)
        
        # Price movement
        price += random.uniform(-20, 20)
        entry = round(price, 1)
        
        # SL/TP
        risk = random.uniform(3, 8)  # Risk in points
        if direction == 'long':
            sl = round(entry - risk, 1)
            tp = round(entry + risk * avg_win_rr, 1)
        else:
            sl = round(entry + risk, 1)
            tp = round(entry - risk * avg_win_rr, 1)
        
        # Outcome
        is_win = random.random() < win_rate
        if is_win:
            hit = 'tp'
            outcome_rr = round(random.uniform(avg_win_rr * 0.8, avg_win_rr * 1.2), 2)
            label = 'win'
        else:
            # Some losses, some BE
            if random.random() < 0.1:
                hit = 'be'
                outcome_rr = 0.0
                label = 'flat'
            else:
                hit = 'sl'
                outcome_rr = round(random.uniform(avg_loss_rr * 1.2, avg_loss_rr * 0.8), 2)
                label = 'loss'
        
        trade = S4TradeRaw(
            trade_id=f"s4_gc_{i:05d}",
            symbol=symbol,
            tf='M1',
            time=trade_time.isoformat() + 'Z',
            date=trade_time.strftime('%Y-%m-%d'),
            day_of_week=trade_time.weekday(),
            session=session,
            bar_index=random.randint(50, 300),
            direction=direction,
            entry=entry,
            sl=sl,
            tp=tp,
            outcome_rr=outcome_rr,
            hit=hit,
            label=label,
            setup_type=random.choice(setup_types),
            signal_type=f"s4_ldn_{direction}",
            meta={},
        )
        trades.append(trade)
    
    # Sort by time
    trades.sort(key=lambda t: t.time)
    return trades


def build_trade_dataset(
    input_path: Optional[str] = None,
    output_path: str = "asm_v2/artifacts/gc_m1/s4_ldn_trades_gc_m1_v1.jsonl",
    generate_if_missing: bool = True,
    n_synthetic: int = 200,
) -> List[S4TradeRaw]:
    """Build standardized trade dataset.
    
    Args:
        input_path: Path to raw trades JSONL (optional)
        output_path: Path to save standardized trades
        generate_if_missing: If True, generate synthetic trades if input missing
        n_synthetic: Number of synthetic trades to generate
    
    Returns:
        List of standardized trades
    """
    trades = []
    
    if input_path and Path(input_path).exists():
        print(f"Loading raw trades from: {input_path}")
        trades = load_raw_trades(input_path)
        print(f"Loaded {len(trades)} trades")
    elif generate_if_missing:
        print(f"Generating {n_synthetic} synthetic trades for testing...")
        trades = generate_synthetic_trades(n_trades=n_synthetic)
        print(f"Generated {len(trades)} synthetic trades")
    else:
        raise FileNotFoundError(f"Raw trades file not found: {input_path}")
    
    # Save standardized trades
    save_trades(trades, output_path)
    print(f"Saved {len(trades)} trades to: {output_path}")
    
    return trades


def get_trade_stats(trades: List[S4TradeRaw]) -> Dict[str, Any]:
    """Get summary statistics for trades."""
    if not trades:
        return {}
    
    dates = sorted(set(t.date for t in trades))
    sessions = {}
    directions = {}
    labels = {}
    
    for t in trades:
        sessions[t.session] = sessions.get(t.session, 0) + 1
        directions[t.direction] = directions.get(t.direction, 0) + 1
        labels[t.label] = labels.get(t.label, 0) + 1
    
    rr_outcomes = [t.outcome_rr for t in trades]
    
    return {
        'n_trades': len(trades),
        'date_range': f"{dates[0]} to {dates[-1]}",
        'n_days': len(dates),
        'sessions': sessions,
        'directions': directions,
        'labels': labels,
        'win_rate': labels.get('win', 0) / max(1, labels.get('win', 0) + labels.get('loss', 0)),
        'avg_rr': sum(rr_outcomes) / len(rr_outcomes) if rr_outcomes else 0,
        'total_r': sum(rr_outcomes),
    }
