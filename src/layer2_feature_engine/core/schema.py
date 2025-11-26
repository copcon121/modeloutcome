"""
Data schemas for the Feature Engine
Defines core data structures used throughout Layer 2
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime


@dataclass
class RawBar:
    """
    Raw OHLCV bar received from NinjaTrader (Layer 1)
    """
    ts: datetime  # Timestamp
    open: float
    high: float
    low: float
    close: float
    volume: float

    # Optional orderflow data (from Rithmic)
    delta: Optional[float] = 0.0
    buy_volume: Optional[float] = 0.0
    sell_volume: Optional[float] = 0.0

    # Optional Level 2 depth data
    l2_bid_depth: Optional[List[float]] = field(default_factory=list)
    l2_ask_depth: Optional[List[float]] = field(default_factory=list)

    def __post_init__(self):
        """Convert string timestamp to datetime if needed"""
        if isinstance(self.ts, str):
            self.ts = datetime.fromisoformat(self.ts.replace('Z', '+00:00'))

    @property
    def range_size(self) -> float:
        """Bar range (high - low)"""
        return self.high - self.low

    @property
    def body_size(self) -> float:
        """Bar body size (abs(close - open))"""
        return abs(self.close - self.open)

    @property
    def is_bullish(self) -> bool:
        """True if close > open"""
        return self.close > self.open


@dataclass
class FeatureBar:
    """
    Fully engineered feature set for a single bar
    Combines OHLCV, SMC, Volume Profile, L2, and time features
    """
    ts: datetime
    features: Dict[str, float]  # Dictionary of all computed features

    def to_list(self, feature_names: List[str]) -> List[float]:
        """Convert features dict to ordered list based on feature_names"""
        return [self.features.get(name, 0.0) for name in feature_names]

    def __len__(self) -> int:
        """Number of features"""
        return len(self.features)


@dataclass
class Record:
    """
    Complete training/inference record
    Includes context window + label + metadata
    """
    context: List[FeatureBar]  # List of FeatureBar (e.g., 60 bars)
    label: Optional[int] = None  # 0=long, 1=short, 2=skip (None for inference)

    # Metadata (for analysis and debugging)
    max_up_R: Optional[float] = None
    max_down_R: Optional[float] = None
    entry_price: Optional[float] = None
    atr: Optional[float] = None

    @property
    def context_len(self) -> int:
        """Number of bars in context"""
        return len(self.context)

    @property
    def feature_dim(self) -> int:
        """Number of features per bar"""
        return len(self.context[0]) if self.context else 0

    def to_feature_matrix(self, feature_names: List[str]) -> List[List[float]]:
        """
        Convert context to 2D matrix [context_len, feature_dim]
        """
        return [bar.to_list(feature_names) for bar in self.context]


@dataclass
class VolumeProfileState:
    """
    Volume Profile state for a session or window
    """
    vah: float  # Value Area High
    val: float  # Value Area Low
    poc: float  # Point of Control
    hvn_levels: List[float]  # High Volume Nodes
    lvn_levels: List[float]  # Low Volume Nodes

    def distance_to_vah(self, price: float) -> float:
        """Distance from price to VAH (normalized)"""
        return (price - self.vah) / self.vah if self.vah != 0 else 0.0

    def distance_to_val(self, price: float) -> float:
        """Distance from price to VAL (normalized)"""
        return (price - self.val) / self.val if self.val != 0 else 0.0

    def distance_to_poc(self, price: float) -> float:
        """Distance from price to POC (normalized)"""
        return (price - self.poc) / self.poc if self.poc != 0 else 0.0


@dataclass
class SMCStructure:
    """
    Smart Money Concepts structure state
    Tracks swings, BOS, CHoCH, and zones
    """
    swing_highs: List[int]  # Indices of swing high bars
    swing_lows: List[int]   # Indices of swing low bars

    # Structure breaks
    bos_up_indices: List[int]
    bos_down_indices: List[int]
    choch_up_indices: List[int]
    choch_down_indices: List[int]

    # Zones (Order Blocks, Fair Value Gaps)
    active_obs_up: List[Dict]  # [{price, strength, bar_index}, ...]
    active_obs_down: List[Dict]
    active_fvgs_up: List[Dict]
    active_fvgs_down: List[Dict]


@dataclass
class L2DepthState:
    """
    Level 2 market depth aggregated state
    """
    bid_pressure: float  # Sum of bid volumes
    ask_pressure: float  # Sum of ask volumes
    depth_imbalance: float  # (bid - ask) / (bid + ask)

    # Top N levels
    bid_volumes: List[float]
    ask_volumes: List[float]
    bid_prices: List[float]
    ask_prices: List[float]
