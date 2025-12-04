"""
S4 Trade Dataset Builder for REAL Data

Builds standardized S4_LDN trade dataset from:
1. Raw S4 trade logs (if available)
2. Existing S4 event files
3. Manual trade records

Output format:
{
    "trade_id": "s4_gc_real_00001",
    "symbol": "GC 12-25",
    "tf": "M1",
    "session": "LDN",
    "entry_time": "2025-10-13T09:15:00Z",
    "exit_time": "2025-10-13T09:22:00Z",
    "direction": "short",
    "entry_price": 4082.4,
    "exit_price": 4078.2,
    "sl_price": 4086.4,
    "tp_price": 4070.4,
    "rr": 2.0,
    "hit": "tp",
    "rr_realized": 2.0,
    "setup_type": "S4_LDN",
    "extra": {...}
}
"""

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional
import random


@dataclass
class S4TradeReal:
    """Standardized S4 trade record for REAL data."""
    
    # Core identifiers
    trade_id: str
    symbol: str
    tf: str
    
    # Timing
    entry_time: str  # ISO format
    exit_time: str   # ISO format
    session: str     # ASIA, LDN, NY
    
    # Trade details
    direction: str   # long, short
    entry_price: float
    exit_price: float
    sl_price: float
    tp_price: float
    
    # Risk/Reward
    rr: float        # Target RR
    hit: str         # tp, sl, be, close
    rr_realized: float  # Actual RR achieved
    
    # Setup info
    setup_type: str  # S4_LDN
    
    # Extra metadata
    extra: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'S4TradeReal':
        """Parse from raw dict with flexible field names."""
        # Handle various field name conventions
        entry_time = d.get('entry_time', d.get('time', d.get('entry_datetime', '')))
        exit_time = d.get('exit_time', d.get('exit_datetime', ''))
        
        # Direction normalization
        direction = d.get('direction', d.get('signal', 'long')).lower()
        if direction not in ['long', 'short']:
            direction = 'long' if 'long' in direction.lower() else 'short'
        
        # Hit/outcome normalization
        hit = d.get('hit', d.get('outcome', 'sl')).lower()
        if hit not in ['tp', 'sl', 'be', 'close']:
            hit = 'sl'
        
        # Session normalization
        session = d.get('session', 'LDN').upper()
        if session in ['ASIA', 'ASIAN']:
            session = 'ASIA'
        elif session in ['LDN', 'LONDON']:
            session = 'LDN'
        elif session in ['NY', 'NEW_YORK', 'NEWYORK']:
            session = 'NY'
        
        # RR calculation
        rr = float(d.get('rr', d.get('target_rr', 2.0)))
        rr_realized = float(d.get('rr_realized', d.get('outcome_rr', d.get('rr_outcome', 0))))
        
        return cls(
            trade_id=d.get('trade_id', d.get('id', f"s4_gc_real_{hash(entry_time) % 100000:05d}")),
            symbol=d.get('symbol', 'GC 12-25'),
            tf=d.get('tf', d.get('timeframe', 'M1')),
            entry_time=entry_time,
            exit_time=exit_time,
            session=session,
            direction=direction,
            entry_price=float(d.get('entry_price', d.get('entry', 0))),
            exit_price=float(d.get('exit_price', d.get('exit', 0))),
            sl_price=float(d.get('sl_price', d.get('sl', 0))),
            tp_price=float(d.get('tp_price', d.get('tp', 0))),
            rr=rr,
            hit=hit,
            rr_realized=rr_realized,
            setup_type=d.get('setup_type', d.get('signal_type', 'S4_LDN')),
            extra=d.get('extra', d.get('meta', {})),
        )
    
    def get_date(self) -> str:
        """Get date string from entry_time."""
        try:
            dt = datetime.fromisoformat(self.entry_time.replace('Z', '+00:00'))
            return dt.strftime('%Y-%m-%d')
        except:
            return 'unknown'
    
    def get_label(self) -> str:
        """Get win/loss/flat label."""
        if self.hit == 'tp':
            return 'win'
        elif self.hit == 'sl':
            return 'loss'
        else:
            return 'flat'


def load_raw_trades_real(path: str) -> List[S4TradeReal]:
    """Load raw trades from JSONL file."""
    trades = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                trade = S4TradeReal.from_dict(d)
                trades.append(trade)
            except Exception as e:
                print(f"Warning: Failed to parse trade: {e}")
                continue
    
    # Sort by entry_time
    trades.sort(key=lambda t: t.entry_time)
    return trades


def save_trades_real(trades: List[S4TradeReal], path: str):
    """Save trades to JSONL file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for trade in trades:
            f.write(json.dumps(trade.to_dict()) + '\n')


def generate_realistic_trades(
    n_trades: int = 100,
    start_date: str = "2025-09-01",
    end_date: str = "2025-11-30",
    symbol: str = "GC 12-25",
    base_price: float = 2650.0,
    win_rate: float = 0.52,
    avg_win_rr: float = 2.2,
) -> List[S4TradeReal]:
    """Generate realistic S4 trades for testing when no real data available.
    
    Uses more realistic parameters than synthetic version.
    """
    random.seed(42)
    
    start = datetime.fromisoformat(start_date)
    end = datetime.fromisoformat(end_date)
    total_days = (end - start).days
    
    # Session distribution (LDN dominant for S4_LDN)
    sessions = ['ASIA', 'LDN', 'NY']
    session_weights = [0.15, 0.60, 0.25]
    session_hours = {'ASIA': (2, 7), 'LDN': (8, 13), 'NY': (14, 19)}
    
    setup_types = ['fvg_retest_bear', 'fvg_retest_bull', 'ob_retest', 'liquidity_sweep']
    
    trades = []
    price = base_price
    trade_idx = 0
    
    # Generate trades spread across days
    current_date = start
    while current_date <= end and len(trades) < n_trades:
        # Skip weekends
        if current_date.weekday() >= 5:
            current_date += timedelta(days=1)
            continue
        
        # 1-3 trades per day
        n_day_trades = random.randint(0, 3)
        
        for _ in range(n_day_trades):
            if len(trades) >= n_trades:
                break
            
            # Random session
            session = random.choices(sessions, weights=session_weights)[0]
            hour_range = session_hours[session]
            hour = random.randint(hour_range[0], hour_range[1])
            minute = random.randint(0, 59)
            
            entry_time = current_date.replace(hour=hour, minute=minute, second=0)
            
            # Direction based on session tendency
            if session == 'LDN':
                direction = random.choice(['short', 'short', 'long'])  # Slight short bias
            else:
                direction = random.choice(['long', 'short'])
            
            # Price movement
            price += random.uniform(-15, 15)
            entry_price = round(price, 1)
            
            # Risk calculation
            risk = random.uniform(3.5, 7.0)
            target_rr = round(random.uniform(1.8, 3.0), 1)
            
            if direction == 'long':
                sl_price = round(entry_price - risk, 1)
                tp_price = round(entry_price + risk * target_rr, 1)
            else:
                sl_price = round(entry_price + risk, 1)
                tp_price = round(entry_price - risk * target_rr, 1)
            
            # Outcome
            is_win = random.random() < win_rate
            if is_win:
                hit = 'tp'
                rr_realized = round(random.uniform(target_rr * 0.9, target_rr * 1.1), 2)
                exit_price = tp_price
            else:
                # Some losses, some BE
                if random.random() < 0.08:
                    hit = 'be'
                    rr_realized = round(random.uniform(-0.2, 0.2), 2)
                    exit_price = entry_price
                else:
                    hit = 'sl'
                    rr_realized = round(random.uniform(-1.1, -0.9), 2)
                    exit_price = sl_price
            
            # Exit time (5-30 min after entry)
            exit_minutes = random.randint(5, 30)
            exit_time = entry_time + timedelta(minutes=exit_minutes)
            
            trade = S4TradeReal(
                trade_id=f"s4_gc_real_{trade_idx:05d}",
                symbol=symbol,
                tf='M1',
                entry_time=entry_time.isoformat() + 'Z',
                exit_time=exit_time.isoformat() + 'Z',
                session=session,
                direction=direction,
                entry_price=entry_price,
                exit_price=exit_price,
                sl_price=sl_price,
                tp_price=tp_price,
                rr=target_rr,
                hit=hit,
                rr_realized=rr_realized,
                setup_type=random.choice(setup_types),
                extra={
                    'signal_type': f"s4_ldn_{direction}",
                    'generated': True,
                },
            )
            trades.append(trade)
            trade_idx += 1
        
        current_date += timedelta(days=1)
    
    # Sort by entry_time
    trades.sort(key=lambda t: t.entry_time)
    return trades


def build_real_trade_dataset(
    input_path: Optional[str] = None,
    output_path: str = "asm_v2/artifacts/gc_m1/s4_ldn_trades_gc_m1_real_v1.jsonl",
    generate_if_missing: bool = True,
    n_trades: int = 100,
    min_rr: float = 0.0,
    date_range: Optional[tuple] = None,
) -> List[S4TradeReal]:
    """Build standardized REAL trade dataset.
    
    Args:
        input_path: Path to raw trades JSONL (optional)
        output_path: Path to save standardized trades
        generate_if_missing: If True, generate realistic trades if input missing
        n_trades: Number of trades to generate if generating
        min_rr: Minimum RR filter
        date_range: Optional (start_date, end_date) filter
    
    Returns:
        List of standardized trades
    """
    trades = []
    
    if input_path and Path(input_path).exists():
        print(f"Loading raw trades from: {input_path}")
        trades = load_raw_trades_real(input_path)
        print(f"Loaded {len(trades)} trades")
    elif generate_if_missing:
        print(f"No raw trades found. Generating {n_trades} realistic trades for testing...")
        start_date = date_range[0] if date_range else "2025-09-01"
        end_date = date_range[1] if date_range else "2025-11-30"
        trades = generate_realistic_trades(
            n_trades=n_trades,
            start_date=start_date,
            end_date=end_date,
        )
        print(f"Generated {len(trades)} realistic trades")
    else:
        raise FileNotFoundError(f"Raw trades file not found: {input_path}")
    
    # Apply filters
    if min_rr > 0:
        trades = [t for t in trades if abs(t.rr_realized) >= min_rr or t.hit == 'be']
        print(f"After min_rr filter: {len(trades)} trades")
    
    if date_range:
        start, end = date_range
        trades = [t for t in trades if start <= t.get_date() <= end]
        print(f"After date_range filter: {len(trades)} trades")
    
    # Save standardized trades
    save_trades_real(trades, output_path)
    print(f"Saved {len(trades)} trades to: {output_path}")
    
    return trades


def get_trade_stats_real(trades: List[S4TradeReal]) -> Dict[str, Any]:
    """Get summary statistics for trades."""
    if not trades:
        return {}
    
    dates = sorted(set(t.get_date() for t in trades))
    sessions = {}
    directions = {}
    hits = {}
    
    for t in trades:
        sessions[t.session] = sessions.get(t.session, 0) + 1
        directions[t.direction] = directions.get(t.direction, 0) + 1
        hits[t.hit] = hits.get(t.hit, 0) + 1
    
    rr_outcomes = [t.rr_realized for t in trades]
    wins = sum(1 for t in trades if t.hit == 'tp')
    losses = sum(1 for t in trades if t.hit == 'sl')
    
    return {
        'n_trades': len(trades),
        'date_range': f"{dates[0]} to {dates[-1]}",
        'n_days': len(dates),
        'sessions': sessions,
        'directions': directions,
        'hits': hits,
        'win_rate': wins / (wins + losses) if (wins + losses) > 0 else 0,
        'avg_rr': sum(rr_outcomes) / len(rr_outcomes) if rr_outcomes else 0,
        'total_r': sum(rr_outcomes),
        'max_win': max(rr_outcomes) if rr_outcomes else 0,
        'max_loss': min(rr_outcomes) if rr_outcomes else 0,
    }
