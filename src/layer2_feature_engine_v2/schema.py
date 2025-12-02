"""
Phase 2 Feature Engine V2 - Data Schemas
All data classes for raw data, states, and features
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict


# ===== RAW DATA FROM EXPORTER =====

@dataclass
class RawBar:
    """
    Raw bar data from NinjaTrader exporter (JSONL format)
    Maps to actual export format: o/h/l/c (not open/high/low/close)
    """
    # Metadata
    symbol: str
    timeframe: str
    timestamp: datetime
    bar_index: int
    
    # OHLCV (shorthand names from exporter)
    o: float    # open
    h: float    # high
    l: float    # low
    c: float    # close
    volume: float
    
    # Orderflow (from Volumdelta indicator)
    delta: float              # DeltasClose[1] from Volumdelta
    buy_volume: float         # (volume + delta) / 2
    sell_volume: float        # volume - buy_volume
    
    # Market depth (stub values = close)
    best_bid: float
    best_ask: float
    
    # Tick features (raw from exporter)
    tick_speed: float         # Total ticks in bar (NOT per second)
    aggr_buy_speed: float     # Buy volume (NOT per second)
    aggr_sell_speed: float    # Sell volume (NOT per second)
    price_speed: float        # Intrabar range (H - L)
    
    # VWAP (from exporter)
    vwap_daily: float = 0.0   # Daily VWAP from NinjaTrader
    
    @property
    def is_bullish(self) -> bool:
        """True if close > open"""
        return self.c > self.o
    
    @property
    def body_size(self) -> float:
        """Absolute body size"""
        return abs(self.c - self.o)
    
    @property
    def range_size(self) -> float:
        """High - Low range"""
        return self.h - self.l
    
    @property
    def upper_wick(self) -> float:
        """Upper wick size"""
        return self.h - max(self.o, self.c)
    
    @property
    def lower_wick(self) -> float:
        """Lower wick size"""
        return min(self.o, self.c) - self.l


# ===== SWING STATES =====

@dataclass
class InternalSwingState:
    """
    Internal swing state (wave 5) - window-based detection
    """
    # Current leg direction: 1=BULLISH_LEG (from low), 0=BEARISH_LEG (from high), -1=undefined
    # Matches Pine Script LuxAlgo: BULLISH_LEG=1, BEARISH_LEG=0
    last_leg: int = -1
    
    # Last confirmed swings
    swing_high_price: Optional[float] = None
    swing_high_bar_index: int = -1
    swing_high_crossed: bool = False
    
    swing_low_price: Optional[float] = None
    swing_low_bar_index: int = -1
    swing_low_crossed: bool = False
    
    # Previous swings (for pattern detection)
    prev_swing_high_price: Optional[float] = None
    prev_swing_low_price: Optional[float] = None
    
    # Trend bias: +1 bullish, -1 bearish, 0 neutral
    trend_bias: int = 0


@dataclass
class ExternalSwingState:
    """
    External swing state (wave 50) - larger window
    """
    # Current leg direction
    last_leg: int = -1
    
    # Last confirmed swings
    swing_high_price: Optional[float] = None
    swing_high_bar_index: int = -1
    swing_high_crossed: bool = False
    
    swing_low_price: Optional[float] = None
    swing_low_bar_index: int = -1
    swing_low_crossed: bool = False
    
    # Previous swings
    prev_swing_high_price: Optional[float] = None
    prev_swing_low_price: Optional[float] = None
    
    # Trend bias
    trend_bias: int = 0


# ===== STRUCTURE STATE =====

@dataclass
class SMCState:
    """
    Complete SMC state (internal + external + zones + sweeps)
    """
    # ===== INTERNAL LAYER (wave 5) =====
    int_trend_dir: int = 0              # +1 up, -1 down, 0 neutral
    int_bos_up: bool = False
    int_bos_down: bool = False
    int_choch_up: bool = False
    int_choch_down: bool = False
    int_swing_high_price: Optional[float] = None
    int_swing_low_price: Optional[float] = None
    int_swing_high_bars_ago: int = 999
    int_swing_low_bars_ago: int = 999
    
    # ===== EXTERNAL LAYER (wave 50) =====
    ext_trend_dir: int = 0              # +1 up, -1 down, 0 neutral
    ext_bos_up: bool = False
    ext_bos_down: bool = False
    ext_choch_up: bool = False
    ext_choch_down: bool = False
    ext_swing_high_price: Optional[float] = None
    ext_swing_low_price: Optional[float] = None
    ext_swing_high_bars_ago: int = 999
    ext_swing_low_bars_ago: int = 999
    
    # ===== LIQUIDITY SWEEPS =====
    swept_prev_int_high: bool = False
    swept_prev_int_low: bool = False
    swept_prev_ext_high: bool = False
    swept_prev_ext_low: bool = False
    
    # ===== ZONES (FVG, OB) =====
    in_bull_fvg: bool = False
    in_bear_fvg: bool = False
    near_bull_fvg: bool = False
    near_bear_fvg: bool = False
    in_bull_ob: bool = False
    in_bear_ob: bool = False
    near_bull_ob: bool = False
    near_bear_ob: bool = False
    dist_to_nearest_fvg: float = 999.0
    dist_to_nearest_ob: float = 999.0


# ===== ZONES DATA =====

@dataclass
class FVGZone:
    """Fair Value Gap zone"""
    is_bullish: bool
    bar_index: int          # Middle bar of 3-bar pattern
    top: float              # Upper bound
    bottom: float           # Lower bound
    gap_size: float
    mitigated: bool = False
    invalidated: bool = False


@dataclass
class OBZone:
    """Order Block zone"""
    is_bullish: bool
    bar_index: int          # Source bar (last opposite candle)
    top: float              # OB top
    bottom: float           # OB bottom
    mitigated: bool = False
    invalidated: bool = False
    hit_count: int = 0      # Times price touched zone


@dataclass
class ZonesState:
    """Active zones state"""
    active_fvgs: List[FVGZone] = field(default_factory=list)
    active_obs: List[OBZone] = field(default_factory=list)
    
    # Current bar zone status
    in_bull_fvg: bool = False
    in_bear_fvg: bool = False
    near_bull_fvg: bool = False
    near_bear_fvg: bool = False
    in_bull_ob: bool = False
    in_bear_ob: bool = False
    near_bull_ob: bool = False
    near_bear_ob: bool = False
    dist_to_nearest_fvg: float = 999.0
    dist_to_nearest_ob: float = 999.0


# ===== VOLUME PROFILE STATE =====

@dataclass
class VolumeProfileState:
    """Volume Profile state"""
    poc_price: Optional[float] = None      # Point of Control
    val_price: Optional[float] = None      # Value Area Low
    vah_price: Optional[float] = None      # Value Area High
    
    # Current bar position relative to VP
    in_value_area: bool = False
    above_value_area: bool = False
    below_value_area: bool = False
    
    # Distances (normalized)
    dist_to_poc: float = 0.0
    dist_to_vah: float = 0.0
    dist_to_val: float = 0.0


# ===== FEATURE BAR (FINAL OUTPUT) =====

@dataclass
class FeatureBar:
    """
    Final feature vector for one bar
    Combines: Price + Orderflow + SMC (int/ext) + Zones + VP
    Total: ~60-80 features
    """
    
    # ===== 1. PRICE / OHLCV FEATURES (8) =====
    close: float
    high_low_range: float               # H - L
    body: float                          # abs(C - O)
    upper_wick: float
    lower_wick: float
    close_return: float                  # (C_t - C_{t-1}) / C_{t-1}
    volume: float
    volume_change: float                 # (V_t - V_{t-1}) / V_{t-1}
    
    # ===== 2. ORDERFLOW FEATURES (8) =====
    delta: float
    delta_over_volume: float             # delta / volume
    buy_volume: float
    sell_volume: float
    buy_sell_ratio: float                # buy / sell
    tick_speed: float
    aggr_buy_speed: float
    aggr_sell_speed: float
    price_speed: float
    
    # ===== 3. INTERNAL SMC FEATURES (12) =====
    int_trend_dir: int
    int_bos_up: bool
    int_bos_down: bool
    int_choch_up: bool
    int_choch_down: bool
    int_swing_high_distance: float       # (C - swing_high) / tick_size
    int_swing_low_distance: float        # (swing_low - C) / tick_size
    bars_since_int_swing_high: int
    bars_since_int_swing_low: int
    swept_prev_int_high: bool
    swept_prev_int_low: bool
    int_bias_bullish: bool               # int_trend_dir > 0
    
    # ===== 4. EXTERNAL SMC FEATURES (12) =====
    ext_trend_dir: int
    ext_bos_up: bool
    ext_bos_down: bool
    ext_choch_up: bool
    ext_choch_down: bool
    ext_swing_high_distance: float
    ext_swing_low_distance: float
    bars_since_ext_swing_high: int
    bars_since_ext_swing_low: int
    swept_prev_ext_high: bool
    swept_prev_ext_low: bool
    ext_bias_bullish: bool
    
    # ===== 5. ZONES FEATURES (10) =====
    in_bull_fvg: bool
    in_bear_fvg: bool
    near_bull_fvg: bool
    near_bear_fvg: bool
    # Internal OB zones
    int_in_bull_ob: bool
    int_in_bear_ob: bool
    int_near_bull_ob: bool
    int_near_bear_ob: bool
    
    # External OB zones
    ext_in_bull_ob: bool
    ext_in_bear_ob: bool
    ext_near_bull_ob: bool
    ext_near_bear_ob: bool
    
    dist_to_nearest_fvg: float
    dist_to_nearest_ob: float
    nearest_fvg_size: float
    
    # ===== 6. VOLUME PROFILE FEATURES (9) =====
    vp_poc_price: float
    vp_val_price: float
    vp_vah_price: float
    vp_in_value_area: bool
    vp_above_value_area: bool
    vp_below_value_area: bool
    vp_dist_to_poc: float
    vp_dist_to_vah: float
    vp_dist_to_val: float
    
    # ===== 7. WAVE STRENGTH FEATURES (v2) =====
    impulse_strength: float          # 0-100
    pullback_strength: float         # 0-100
    cum_delta_5: float
    cum_delta_10: float
    cum_delta_20: float

    # ===== 8. VWAP FEATURES (2) =====
    vwap_daily: float                # Daily VWAP from NinjaTrader
    dist_to_vwap: float             # (close - vwap) / tick_size
    
    # ===== 9. MACRO TREND CONTEXT (14 Features) =====
    # M5 Context (7 features) - Immediate context for M1 trading
    m5_trend_up: float              # 1.0 if Bullish, 0.0 otherwise
    m5_trend_down: float            # 1.0 if Bearish, 0.0 otherwise
    m5_premium: float               # 1.0 if > midpoint
    m5_discount: float              # 1.0 if < midpoint
    dist_to_m5_swing_high: float    # Normalized by ATR
    dist_to_m5_swing_low: float
    near_m5_fvg: float              # 1.0 if near/inside
    
    # H1 Context (7 features) - Medium-term trend
    h1_trend_up: float
    h1_trend_down: float
    h1_premium: float
    h1_discount: float
    dist_to_h1_swing_high: float
    dist_to_h1_swing_low: float
    near_h1_fvg: float
    
    # ===== 10. ENHANCED MACRO CONTEXT (10 NEW Features) =====
    # Derived from existing: trend_dir = trend_up - trend_down, pd_zone = premium - discount
    # NEW features only (5 per TF):
    
    # M5 Enhanced (5 new features)
    m5_swing_phase: float           # 0=range, 1=impulse, 2=pullback
    m5_price_pos_in_range: float    # 0-1, position in rolling range
    m5_bos_up_count_recent: float   # Count BOS up in recent X bars
    m5_bos_down_count_recent: float # Count BOS down in recent X bars
    m5_ob_imbalance: float          # n_buy_OB - n_sell_OB, clipped [-3,3]
    
    # H1 Enhanced (5 new features)
    h1_swing_phase: float
    h1_price_pos_in_range: float
    h1_bos_up_count_recent: float
    h1_bos_down_count_recent: float
    h1_ob_imbalance: float
    
    # TOTAL: ~100 features (90 existing + 10 new)
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for easier manipulation"""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, bool):
                result[key] = 1.0 if value else 0.0
            elif isinstance(value, (int, float)):
                result[key] = float(value)
            elif value is None:
                result[key] = 0.0
        return result
    
    def to_array(self) -> List[float]:
        """Convert to ordered array (for numpy)"""
        return list(self.to_dict().values())
    
    @classmethod
    def feature_names(cls) -> List[str]:
        """Get ordered list of feature names"""
        dummy = cls(
            close=0, high_low_range=0, body=0, upper_wick=0, lower_wick=0,
            close_return=0, volume=0, volume_change=0,
            delta=0, delta_over_volume=0, buy_volume=0, sell_volume=0,
            buy_sell_ratio=0, tick_speed=0, aggr_buy_speed=0, aggr_sell_speed=0,
            price_speed=0,
            int_trend_dir=0, int_bos_up=False, int_bos_down=False,
            int_choch_up=False, int_choch_down=False,
            int_swing_high_distance=0, int_swing_low_distance=0,
            bars_since_int_swing_high=0, bars_since_int_swing_low=0,
            swept_prev_int_high=False, swept_prev_int_low=False,
            int_bias_bullish=False,
            ext_trend_dir=0, ext_bos_up=False, ext_bos_down=False,
            ext_choch_up=False, ext_choch_down=False,
            ext_swing_high_distance=0, ext_swing_low_distance=0,
            bars_since_ext_swing_high=0, bars_since_ext_swing_low=0,
            swept_prev_ext_high=False, swept_prev_ext_low=False,
            ext_bias_bullish=False,
            in_bull_fvg=False, in_bear_fvg=False,
            near_bull_fvg=False, near_bear_fvg=False,
            int_in_bull_ob=False, int_in_bear_ob=False,
            int_near_bull_ob=False, int_near_bear_ob=False,
            ext_in_bull_ob=False, ext_in_bear_ob=False,
            ext_near_bull_ob=False, ext_near_bear_ob=False,
            dist_to_nearest_fvg=0, dist_to_nearest_ob=0,
            nearest_fvg_size=0.0,
            vp_poc_price=0, vp_val_price=0, vp_vah_price=0,
            vp_in_value_area=False, vp_above_value_area=False,
            vp_below_value_area=False,
            vp_dist_to_poc=0, vp_dist_to_vah=0, vp_dist_to_val=0,
            impulse_strength=0.0, pullback_strength=0.0,
            cum_delta_5=0.0, cum_delta_10=0.0, cum_delta_20=0.0,
            vwap_daily=0, dist_to_vwap=0,
            m5_trend_up=0, m5_trend_down=0, m5_premium=0, m5_discount=0,
            dist_to_m5_swing_high=0, dist_to_m5_swing_low=0, near_m5_fvg=0,
            h1_trend_up=0, h1_trend_down=0, h1_premium=0, h1_discount=0,
            dist_to_h1_swing_high=0, dist_to_h1_swing_low=0, near_h1_fvg=0,
            # Enhanced macro features (10 new)
            m5_swing_phase=0, m5_price_pos_in_range=0,
            m5_bos_up_count_recent=0, m5_bos_down_count_recent=0, m5_ob_imbalance=0,
            h1_swing_phase=0, h1_price_pos_in_range=0,
            h1_bos_up_count_recent=0, h1_bos_down_count_recent=0, h1_ob_imbalance=0
        )
        return list(dummy.to_dict().keys())
