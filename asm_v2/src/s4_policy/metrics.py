"""
S4 Policy Metrics - Compute backtest metrics from trade results

Metrics computed:
- Win rate
- Average R
- Expectancy (mean R)
- Std R
- Max drawdown (in R)
- Profit factor
- Sharpe-like ratio
"""

from typing import List, Tuple, Dict, Any
import numpy as np
from .policy_dataset import S4TradeEnriched


def compute_win_rate(labels: List[str]) -> float:
    """Compute win rate from labels."""
    if not labels:
        return 0.0
    wins = sum(1 for l in labels if l == 'win')
    total = sum(1 for l in labels if l in ['win', 'loss'])
    return wins / total if total > 0 else 0.0


def compute_expectancy(rr_outcomes: List[float]) -> float:
    """Compute expectancy (mean R)."""
    if not rr_outcomes:
        return 0.0
    return np.mean(rr_outcomes)


def compute_std_rr(rr_outcomes: List[float]) -> float:
    """Compute standard deviation of R."""
    if len(rr_outcomes) < 2:
        return 0.0
    return np.std(rr_outcomes, ddof=1)


def compute_cumulative_r(rr_outcomes: List[float]) -> np.ndarray:
    """Compute cumulative R series."""
    if not rr_outcomes:
        return np.array([0.0])
    return np.cumsum(rr_outcomes)


def compute_drawdown(rr_outcomes: List[float]) -> Tuple[float, np.ndarray]:
    """Compute max drawdown in R and drawdown series.
    
    Returns:
        max_dd: maximum drawdown (positive number)
        dd_series: drawdown at each point
    """
    if not rr_outcomes:
        return 0.0, np.array([0.0])
    
    cum_r = compute_cumulative_r(rr_outcomes)
    running_max = np.maximum.accumulate(cum_r)
    dd_series = running_max - cum_r
    max_dd = np.max(dd_series)
    
    return float(max_dd), dd_series


def compute_profit_factor(rr_outcomes: List[float]) -> float:
    """Compute profit factor (gross profit / gross loss)."""
    if not rr_outcomes:
        return 0.0
    
    gross_profit = sum(r for r in rr_outcomes if r > 0)
    gross_loss = abs(sum(r for r in rr_outcomes if r < 0))
    
    if gross_loss == 0:
        return float('inf') if gross_profit > 0 else 0.0
    
    return gross_profit / gross_loss


def compute_sharpe_like(rr_outcomes: List[float]) -> float:
    """Compute Sharpe-like ratio (mean / std)."""
    if len(rr_outcomes) < 2:
        return 0.0
    
    mean_r = np.mean(rr_outcomes)
    std_r = np.std(rr_outcomes, ddof=1)
    
    if std_r == 0:
        return float('inf') if mean_r > 0 else 0.0
    
    return mean_r / std_r


def compute_metrics(trades: List[S4TradeEnriched]) -> Dict[str, Any]:
    """Compute all metrics from list of trades.
    
    Returns dict with:
        n_trades, win_count, loss_count, be_count,
        win_rate, avg_rr, expectancy, std_rr,
        max_drawdown_r, profit_factor, sharpe_like,
        cum_r_final
    """
    if not trades:
        return {
            'n_trades': 0,
            'win_count': 0,
            'loss_count': 0,
            'be_count': 0,
            'win_rate': 0.0,
            'avg_rr': 0.0,
            'expectancy': 0.0,
            'std_rr': 0.0,
            'max_drawdown_r': 0.0,
            'profit_factor': 0.0,
            'sharpe_like': 0.0,
            'cum_r_final': 0.0,
        }
    
    labels = [t.label for t in trades]
    rr_outcomes = [t.rr_outcome for t in trades]
    
    win_count = sum(1 for l in labels if l == 'win')
    loss_count = sum(1 for l in labels if l == 'loss')
    be_count = sum(1 for l in labels if l in ['flat', 'be'])
    
    max_dd, _ = compute_drawdown(rr_outcomes)
    cum_r = compute_cumulative_r(rr_outcomes)
    
    return {
        'n_trades': len(trades),
        'win_count': win_count,
        'loss_count': loss_count,
        'be_count': be_count,
        'win_rate': compute_win_rate(labels),
        'avg_rr': float(np.mean(rr_outcomes)) if rr_outcomes else 0.0,
        'expectancy': compute_expectancy(rr_outcomes),
        'std_rr': compute_std_rr(rr_outcomes),
        'max_drawdown_r': max_dd,
        'profit_factor': compute_profit_factor(rr_outcomes),
        'sharpe_like': compute_sharpe_like(rr_outcomes),
        'cum_r_final': float(cum_r[-1]) if len(cum_r) > 0 else 0.0,
    }


def compute_metrics_by_regime(trades: List[S4TradeEnriched]) -> Dict[str, Dict[str, Any]]:
    """Compute metrics grouped by regime."""
    by_regime = {}
    for t in trades:
        regime = t.regime_name
        if regime not in by_regime:
            by_regime[regime] = []
        by_regime[regime].append(t)
    
    return {regime: compute_metrics(regime_trades) 
            for regime, regime_trades in by_regime.items()}


def compute_metrics_by_session(trades: List[S4TradeEnriched]) -> Dict[str, Dict[str, Any]]:
    """Compute metrics grouped by session."""
    by_session = {}
    for t in trades:
        session = t.session
        if session not in by_session:
            by_session[session] = []
        by_session[session].append(t)
    
    return {session: compute_metrics(session_trades) 
            for session, session_trades in by_session.items()}
