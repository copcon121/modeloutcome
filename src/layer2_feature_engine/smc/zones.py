"""
SMC Zones Detection (Order Blocks, Fair Value Gaps)
Identifies institutional supply/demand zones
"""

from typing import List, Dict, Tuple
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from layer2_feature_engine.core.schema import RawBar, SMCStructure


def detect_order_blocks(bars: List[RawBar]) -> Tuple[List[Dict], List[Dict]]:
    """
    Detect bullish and bearish order blocks

    Order Block: Last opposite-colored bar before a strong move
    - Bullish OB: Last bearish bar before bullish move
    - Bearish OB: Last bullish bar before bearish move

    Args:
        bars: List of RawBar objects

    Returns:
        Tuple of (bullish_obs, bearish_obs)
        Each OB is a dict: {'bar_index': int, 'price': float, 'strength': float}
    """
    bullish_obs = []
    bearish_obs = []

    if len(bars) < 5:
        return bullish_obs, bearish_obs

    for i in range(2, len(bars) - 2):
        # Look for strong bullish move (next 2 bars are bullish)
        if bars[i+1].is_bullish and bars[i+2].is_bullish:
            # Check if current bar is bearish (last down move)
            if not bars[i].is_bullish:
                strength = (bars[i+1].close - bars[i].close) / bars[i].close
                bullish_obs.append({
                    'bar_index': i,
                    'price': bars[i].low,  # Support level
                    'strength': strength
                })

        # Look for strong bearish move
        if not bars[i+1].is_bullish and not bars[i+2].is_bullish:
            # Check if current bar is bullish (last up move)
            if bars[i].is_bullish:
                strength = (bars[i].close - bars[i+1].close) / bars[i].close
                bearish_obs.append({
                    'bar_index': i,
                    'price': bars[i].high,  # Resistance level
                    'strength': strength
                })

    # Keep only recent OBs (last 20)
    return bullish_obs[-20:], bearish_obs[-20:]


def detect_fair_value_gaps(bars: List[RawBar]) -> Tuple[List[Dict], List[Dict]]:
    """
    Detect Fair Value Gaps (FVG)

    FVG: A 3-bar pattern where there's a gap between bar 1 and bar 3
    - Bullish FVG: Gap up (bar3.low > bar1.high)
    - Bearish FVG: Gap down (bar3.high < bar1.low)

    Args:
        bars: List of RawBar objects

    Returns:
        Tuple of (bullish_fvgs, bearish_fvgs)
        Each FVG is a dict: {'bar_index': int, 'gap_start': float, 'gap_end': float, 'size': float}
    """
    bullish_fvgs = []
    bearish_fvgs = []

    if len(bars) < 3:
        return bullish_fvgs, bearish_fvgs

    for i in range(len(bars) - 2):
        bar1 = bars[i]
        bar2 = bars[i+1]
        bar3 = bars[i+2]

        # Bullish FVG
        if bar3.low > bar1.high:
            gap_size = bar3.low - bar1.high
            bullish_fvgs.append({
                'bar_index': i+1,  # Middle bar
                'gap_start': bar1.high,
                'gap_end': bar3.low,
                'size': gap_size
            })

        # Bearish FVG
        if bar3.high < bar1.low:
            gap_size = bar1.low - bar3.high
            bearish_fvgs.append({
                'bar_index': i+1,
                'gap_start': bar3.high,
                'gap_end': bar1.low,
                'size': gap_size
            })

    # Keep only recent FVGs (last 10)
    return bullish_fvgs[-10:], bearish_fvgs[-10:]


def extract_zone_features(i: int, bars: List[RawBar], smc_structure: SMCStructure) -> Dict[str, float]:
    """
    Extract zone-related features for a specific bar

    Args:
        i: Bar index
        bars: List of RawBar objects
        smc_structure: SMCStructure object

    Returns:
        Dict of features
    """
    bar = bars[i]
    features = {}

    # Distance to nearest Order Blocks
    dist_to_ob_up = _distance_to_nearest_zone(bar.close, smc_structure.active_obs_up, 'price')
    dist_to_ob_down = _distance_to_nearest_zone(bar.close, smc_structure.active_obs_down, 'price')

    features['dist_to_ob_up'] = dist_to_ob_up
    features['dist_to_ob_down'] = dist_to_ob_down

    # Distance to nearest Fair Value Gaps
    dist_to_fvg_up = _distance_to_nearest_fvg(bar.close, smc_structure.active_fvgs_up)
    dist_to_fvg_down = _distance_to_nearest_fvg(bar.close, smc_structure.active_fvgs_down)

    features['dist_to_fvg_up'] = dist_to_fvg_up
    features['dist_to_fvg_down'] = dist_to_fvg_down

    # Inside FVG flags
    features['inside_fvg_up'] = 1.0 if _is_inside_fvg(bar.close, smc_structure.active_fvgs_up) else 0.0
    features['inside_fvg_down'] = 1.0 if _is_inside_fvg(bar.close, smc_structure.active_fvgs_down) else 0.0

    return features


def _distance_to_nearest_zone(price: float, zones: List[Dict], price_key: str) -> float:
    """
    Calculate distance to nearest zone (normalized)

    Args:
        price: Current price
        zones: List of zone dicts
        price_key: Key for price in zone dict

    Returns:
        Normalized distance (0-1)
    """
    if not zones:
        return 1.0  # Max distance

    distances = [abs(price - zone[price_key]) / price for zone in zones]
    min_dist = min(distances)
    return min(min_dist * 10, 1.0)  # Scale and cap at 1.0


def _distance_to_nearest_fvg(price: float, fvgs: List[Dict]) -> float:
    """Calculate distance to nearest FVG"""
    if not fvgs:
        return 1.0

    distances = []
    for fvg in fvgs:
        # Distance to middle of gap
        gap_mid = (fvg['gap_start'] + fvg['gap_end']) / 2
        dist = abs(price - gap_mid) / price
        distances.append(dist)

    min_dist = min(distances)
    return min(min_dist * 10, 1.0)


def _is_inside_fvg(price: float, fvgs: List[Dict]) -> bool:
    """Check if price is inside any FVG"""
    for fvg in fvgs:
        if fvg['gap_start'] <= price <= fvg['gap_end']:
            return True
    return False
