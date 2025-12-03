"""
S4 Rule Engine for Live Gateway
Implements S4_HighVol_FVG_London rules
"""

from typing import Dict, Optional, Tuple
from dataclasses import dataclass

import sys
from pathlib import Path
ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.layer2_feature_engine_v2.schema import FeatureBar


# ==============================================================================
# S4 CONFIGURATION
# ==============================================================================

S4_CONFIG = {
    "session": "London",
    "session_start_hour": 8,
    "session_end_hour": 14,
    "rr_target": 2.0,
}


# ==============================================================================
# S4 RULE ENGINE
# ==============================================================================


@dataclass
class S4Setup:
    """S4 setup result"""
    is_valid: bool
    side: int  # +1 long, -1 short, 0 none
    session: str
    high_vol: bool
    in_fvg: bool
    ext_trend: int
    entry_price: float = 0.0
    sl_price: float = 0.0
    tp_price: float = 0.0


def detect_session(hour: int) -> str:
    """Detect trading session from hour (UTC)"""
    if 0 <= hour < 8:
        return "Asia"
    elif 8 <= hour < 14:
        return "London"
    else:
        return "NY"


def check_s4_setup(
    feature_bar: FeatureBar,
    feature_dict: Dict,
    is_high_vol: bool,
    hour: int,
    session_filter: str = "London"
) -> S4Setup:
    """
    Check if current bar qualifies as S4 setup.
    
    S4 Rule:
    1. London session (08:00 - 14:00 UTC)
    2. High volatility regime
    3. In FVG zone + trend continuation
    
    Returns:
        S4Setup with setup details
    """
    session = detect_session(hour)
    
    # Default result
    result = S4Setup(
        is_valid=False,
        side=0,
        session=session,
        high_vol=is_high_vol,
        in_fvg=False,
        ext_trend=int(feature_dict.get("ext_trend_dir", 0))
    )
    
    # Session filter
    if session_filter and session != session_filter:
        return result
    
    # High vol regime
    if not is_high_vol:
        return result
    
    # FVG + Trend conditions
    in_bull_fvg = feature_dict.get("in_bull_fvg", 0) == 1
    in_bear_fvg = feature_dict.get("in_bear_fvg", 0) == 1
    ext_trend = int(feature_dict.get("ext_trend_dir", 0))
    
    result.in_fvg = in_bull_fvg or in_bear_fvg
    result.ext_trend = ext_trend
    
    # Long setup: in bull FVG + uptrend
    if in_bull_fvg and ext_trend > 0:
        result.is_valid = True
        result.side = 1
        result.entry_price, result.sl_price, result.tp_price = calculate_sl_tp(
            feature_dict, side=1, rr=S4_CONFIG["rr_target"]
        )
        return result
    
    # Short setup: in bear FVG + downtrend
    if in_bear_fvg and ext_trend < 0:
        result.is_valid = True
        result.side = -1
        result.entry_price, result.sl_price, result.tp_price = calculate_sl_tp(
            feature_dict, side=-1, rr=S4_CONFIG["rr_target"]
        )
        return result
    
    return result


def calculate_sl_tp(feature_dict: Dict, side: int, rr: float = 2.0) -> Tuple[float, float, float]:
    """
    Calculate entry, SL and TP for a trade.
    
    Returns:
        (entry, sl, tp)
    """
    close = feature_dict.get("close", 0)
    high_low_range = feature_dict.get("high_low_range", 1)
    low = close - high_low_range / 2  # Approximate
    high = close + high_low_range / 2
    
    atr_approx = high_low_range * 2
    
    if side == 1:  # Long
        entry = close
        sl = low - atr_approx * 0.5
        risk = entry - sl
        tp = entry + risk * rr
    else:  # Short
        entry = close
        sl = high + atr_approx * 0.5
        risk = sl - entry
        tp = entry - risk * rr
    
    return entry, sl, tp
