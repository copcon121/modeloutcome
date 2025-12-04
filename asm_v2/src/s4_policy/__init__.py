"""
S4_LDN Policy & Backtest Module v1

This module provides:
- Policy dataset loading from enriched trades
- Rule-based and ML-based trade policies
- Backtest engine for policy evaluation
- Leak/sanity tests for policy layer
"""

from .policy_dataset import S4TradeEnriched, S4PolicyDataset, load_enriched_trades
from .policy_rules import Policy, PolicyConfig, create_policy
from .backtester import Backtester, PolicyResult
from .metrics import compute_metrics, compute_drawdown

__all__ = [
    'S4TradeEnriched',
    'S4PolicyDataset', 
    'load_enriched_trades',
    'Policy',
    'PolicyConfig',
    'create_policy',
    'Backtester',
    'PolicyResult',
    'compute_metrics',
    'compute_drawdown',
]
