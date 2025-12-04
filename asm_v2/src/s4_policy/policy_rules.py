"""
S4 Policy Rules - Rule-based trade/skip policies

Each policy decides whether to KEEP or SKIP a trade based on:
- Regime from ASM v2
- Meta features (session, pos_in_session, etc.)
- z_t embedding (for ML-based policies)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Callable
from .policy_dataset import S4TradeEnriched


@dataclass
class PolicyConfig:
    """Configuration for a policy."""
    name: str
    policy_type: str  # "baseline", "regime", "session", "confidence", "combo", "ml"
    params: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'PolicyConfig':
        return cls(
            name=d.get('name', 'unknown'),
            policy_type=d.get('policy_type', 'baseline'),
            params=d.get('params', {})
        )


class Policy(ABC):
    """Base class for trade policies."""
    
    def __init__(self, config: PolicyConfig):
        self.config = config
        self.name = config.name
    
    @abstractmethod
    def decide_trade(self, trade: S4TradeEnriched) -> bool:
        """Decide whether to keep (True) or skip (False) a trade."""
        pass
    
    def __repr__(self) -> str:
        return f"Policy({self.name})"


class BaselineAllPolicy(Policy):
    """P0: Keep all trades (baseline for comparison)."""
    
    def decide_trade(self, trade: S4TradeEnriched) -> bool:
        return True


class RegimeOnlyPolicy(Policy):
    """Filter trades by specific regime(s).
    
    Params:
        regimes: list of regime names to KEEP (e.g., ["trend_down"])
    """
    
    def __init__(self, config: PolicyConfig):
        super().__init__(config)
        self.allowed_regimes = config.params.get('regimes', [])
    
    def decide_trade(self, trade: S4TradeEnriched) -> bool:
        return trade.regime_name in self.allowed_regimes


class RegimeExcludePolicy(Policy):
    """Exclude trades with specific regime(s).
    
    Params:
        exclude_regimes: list of regime names to SKIP
    """
    
    def __init__(self, config: PolicyConfig):
        super().__init__(config)
        self.exclude_regimes = config.params.get('exclude_regimes', [])
    
    def decide_trade(self, trade: S4TradeEnriched) -> bool:
        return trade.regime_name not in self.exclude_regimes


class SessionFilterPolicy(Policy):
    """Filter trades by session.
    
    Params:
        sessions: list of session names to KEEP (e.g., ["LDN"])
    """
    
    def __init__(self, config: PolicyConfig):
        super().__init__(config)
        self.allowed_sessions = config.params.get('sessions', [])
    
    def decide_trade(self, trade: S4TradeEnriched) -> bool:
        return trade.session in self.allowed_sessions


class RegimeConfidencePolicy(Policy):
    """Filter by regime confidence threshold.
    
    Params:
        min_confidence: float threshold (e.g., 0.6)
    """
    
    def __init__(self, config: PolicyConfig):
        super().__init__(config)
        self.min_confidence = config.params.get('min_confidence', 0.5)
    
    def decide_trade(self, trade: S4TradeEnriched) -> bool:
        return trade.regime_confidence >= self.min_confidence


class ComboPolicy(Policy):
    """Combine multiple conditions with AND logic.
    
    Params:
        regimes: list of allowed regimes (optional)
        sessions: list of allowed sessions (optional)
        min_confidence: float (optional)
        pos_range: [min, max] for pos_in_session_range (optional)
        directions: list of allowed directions (optional)
    """
    
    def __init__(self, config: PolicyConfig):
        super().__init__(config)
        self.regimes = config.params.get('regimes', None)
        self.sessions = config.params.get('sessions', None)
        self.min_confidence = config.params.get('min_confidence', None)
        self.pos_range = config.params.get('pos_range', None)
        self.directions = config.params.get('directions', None)
    
    def decide_trade(self, trade: S4TradeEnriched) -> bool:
        # Check regime
        if self.regimes is not None:
            if trade.regime_name not in self.regimes:
                return False
        
        # Check session
        if self.sessions is not None:
            if trade.session not in self.sessions:
                return False
        
        # Check confidence
        if self.min_confidence is not None:
            if trade.regime_confidence < self.min_confidence:
                return False
        
        # Check position in session range
        if self.pos_range is not None:
            min_pos, max_pos = self.pos_range
            if not (min_pos <= trade.pos_in_session_range <= max_pos):
                return False
        
        # Check direction
        if self.directions is not None:
            if trade.direction not in self.directions:
                return False
        
        return True


class DirectionRegimePolicy(Policy):
    """Match direction with regime (trend alignment).
    
    Logic:
    - Long trades: prefer trend_up
    - Short trades: prefer trend_down
    
    Params:
        strict: if True, only allow aligned trades
        allow_balance: if True, also allow balance regime
    """
    
    def __init__(self, config: PolicyConfig):
        super().__init__(config)
        self.strict = config.params.get('strict', True)
        self.allow_balance = config.params.get('allow_balance', False)
    
    def decide_trade(self, trade: S4TradeEnriched) -> bool:
        regime = trade.regime_name
        direction = trade.direction
        
        # Check alignment
        aligned = (
            (direction == 'long' and regime == 'trend_up') or
            (direction == 'short' and regime == 'trend_down')
        )
        
        if aligned:
            return True
        
        if self.allow_balance and regime == 'balance':
            return True
        
        return not self.strict


class MLPredictionPolicy(Policy):
    """Policy based on ML model prediction.
    
    Params:
        threshold: p(win) threshold to keep trade
        model: reference to trained model (set externally)
    """
    
    def __init__(self, config: PolicyConfig):
        super().__init__(config)
        self.threshold = config.params.get('threshold', 0.5)
        self.model = None  # Set externally after training
    
    def set_model(self, model):
        """Set the trained model."""
        self.model = model
    
    def decide_trade(self, trade: S4TradeEnriched) -> bool:
        if self.model is None:
            return True  # No model, keep all
        
        # Get prediction
        p_win = self.model.predict_proba_single(trade)
        return p_win >= self.threshold


# Policy factory
POLICY_REGISTRY = {
    'baseline': BaselineAllPolicy,
    'regime_only': RegimeOnlyPolicy,
    'regime_exclude': RegimeExcludePolicy,
    'session': SessionFilterPolicy,
    'confidence': RegimeConfidencePolicy,
    'combo': ComboPolicy,
    'direction_regime': DirectionRegimePolicy,
    'ml': MLPredictionPolicy,
}


def create_policy(config: PolicyConfig) -> Policy:
    """Create policy from config."""
    policy_cls = POLICY_REGISTRY.get(config.policy_type)
    if policy_cls is None:
        raise ValueError(f"Unknown policy type: {config.policy_type}")
    return policy_cls(config)


def create_policies_from_config(policy_configs: List[Dict[str, Any]]) -> List[Policy]:
    """Create multiple policies from config list."""
    policies = []
    for cfg_dict in policy_configs:
        config = PolicyConfig.from_dict(cfg_dict)
        policy = create_policy(config)
        policies.append(policy)
    return policies


# Pre-defined policy configs for convenience
DEFAULT_POLICIES = [
    {
        "name": "P0_baseline_all",
        "policy_type": "baseline",
        "params": {}
    },
    {
        "name": "P1_regime_trend_down",
        "policy_type": "regime_only",
        "params": {"regimes": ["trend_down"]}
    },
    {
        "name": "P2_regime_no_balance",
        "policy_type": "regime_exclude",
        "params": {"exclude_regimes": ["balance"]}
    },
    {
        "name": "P3_session_ldn",
        "policy_type": "session",
        "params": {"sessions": ["LDN"]}
    },
    {
        "name": "P4_direction_aligned",
        "policy_type": "direction_regime",
        "params": {"strict": True, "allow_balance": False}
    },
    {
        "name": "P5_direction_aligned_balance",
        "policy_type": "direction_regime",
        "params": {"strict": True, "allow_balance": True}
    },
    {
        "name": "P6_combo_trend_ldn",
        "policy_type": "combo",
        "params": {
            "regimes": ["trend_down", "trend_up"],
            "sessions": ["LDN"]
        }
    },
    {
        "name": "P7_combo_trend_down_mid_session",
        "policy_type": "combo",
        "params": {
            "regimes": ["trend_down"],
            "pos_range": [0.2, 0.8]
        }
    },
]
