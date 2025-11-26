"""
Level 2 Market Depth Features
Computes bid/ask pressure, depth imbalance, and aggression indicators
"""

from typing import List, Dict
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from layer2_feature_engine.core.schema import RawBar, L2DepthState


def compute_l2_state(bars: List[RawBar], top_n: int = 10) -> L2DepthState:
    """
    Compute aggregated Level 2 depth state from bars

    NOTE: This is a placeholder implementation.
    Real L2 data requires integration with Rithmic API or similar market depth provider.

    Args:
        bars: List of RawBar objects
        top_n: Number of top levels to aggregate

    Returns:
        L2DepthState object
    """
    if not bars:
        return _default_l2_state()

    # In real implementation, you would aggregate L2 depth from bars
    # For now, we use placeholder values based on volume

    recent_bars = bars[-min(10, len(bars)):]

    # Estimate bid/ask pressure from volume (placeholder logic)
    total_bid_pressure = 0.0
    total_ask_pressure = 0.0

    for bar in recent_bars:
        # If bar is bullish, assume more ask pressure (buyers)
        # If bar is bearish, assume more bid pressure (sellers)
        if bar.is_bullish:
            total_ask_pressure += bar.volume
            total_bid_pressure += bar.volume * 0.4  # Assume 60/40 split
        else:
            total_bid_pressure += bar.volume
            total_ask_pressure += bar.volume * 0.4

    # Compute imbalance
    total_pressure = total_bid_pressure + total_ask_pressure
    if total_pressure > 0:
        imbalance = (total_bid_pressure - total_ask_pressure) / total_pressure
    else:
        imbalance = 0.0

    # Placeholder for actual L2 depth levels
    bid_volumes = [0.0] * top_n
    ask_volumes = [0.0] * top_n
    bid_prices = [0.0] * top_n
    ask_prices = [0.0] * top_n

    # TODO: Real implementation would extract from bar.l2_bid_depth and bar.l2_ask_depth

    return L2DepthState(
        bid_pressure=total_bid_pressure,
        ask_pressure=total_ask_pressure,
        depth_imbalance=imbalance,
        bid_volumes=bid_volumes,
        ask_volumes=ask_volumes,
        bid_prices=bid_prices,
        ask_prices=ask_prices
    )


def extract_bar_l2_features(i: int, l2_state: L2DepthState, bar: RawBar) -> Dict[str, float]:
    """
    Extract Level 2 features for a specific bar

    Args:
        i: Bar index
        l2_state: L2DepthState object
        bar: RawBar object

    Returns:
        Dict of features
    """
    features = {
        # Bid/Ask pressure (normalized)
        'l2_bid_pressure': _normalize_pressure(l2_state.bid_pressure),
        'l2_ask_pressure': _normalize_pressure(l2_state.ask_pressure),

        # Depth imbalance (-1 to +1)
        'l2_depth_imbalance': l2_state.depth_imbalance,

        # Aggression indicators (placeholder)
        # TODO: Real implementation would compute aggression from trade direction
        'l2_aggression_buy': 0.0,
        'l2_aggression_sell': 0.0,

        # Top level ratios (placeholder)
        'l2_top_bid_ask_ratio': _compute_top_ratio(l2_state),
    }

    return features


def _normalize_pressure(pressure: float) -> float:
    """
    Normalize pressure to [0, 1] range using log scale

    Args:
        pressure: Raw pressure value

    Returns:
        Normalized pressure
    """
    import numpy as np
    return np.tanh(pressure / 10000.0)  # Assuming typical volume ~10k


def _compute_top_ratio(l2_state: L2DepthState) -> float:
    """
    Compute ratio of top bid to top ask volume

    Args:
        l2_state: L2DepthState object

    Returns:
        Ratio (0-1 scale)
    """
    if not l2_state.bid_volumes or not l2_state.ask_volumes:
        return 0.5  # Neutral

    top_bid = l2_state.bid_volumes[0]
    top_ask = l2_state.ask_volumes[0]

    if top_bid + top_ask == 0:
        return 0.5

    return top_bid / (top_bid + top_ask)


def _default_l2_state() -> L2DepthState:
    """Return default L2DepthState when not enough data"""
    return L2DepthState(
        bid_pressure=0.0,
        ask_pressure=0.0,
        depth_imbalance=0.0,
        bid_volumes=[],
        ask_volumes=[],
        bid_prices=[],
        ask_prices=[]
    )
