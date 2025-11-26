"""
Market Structure Detection (BOS, CHoCH, Sweeps)
Break of Structure, Change of Character, and Liquidity Sweeps
"""

from typing import List, Dict
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from layer2_feature_engine.core.schema import RawBar, SMCStructure


def compute_structure_flags(
    bars: List[RawBar],
    swing_highs: List[int],
    swing_lows: List[int]
) -> Dict[str, List[int]]:
    """
    Compute structure break flags (BOS, CHoCH)

    BOS (Break of Structure): Price breaks above recent swing high (bullish)
                              or below recent swing low (bearish)

    CHoCH (Change of Character): Price breaks counter-trend swing
                                 (signals potential reversal)

    Args:
        bars: List of RawBar objects
        swing_highs: List of swing high indices
        swing_lows: List of swing low indices

    Returns:
        Dict with keys: 'bos_up', 'bos_down', 'choch_up', 'choch_down'
        Each value is a list of indices where the event occurred
    """
    bos_up = []
    bos_down = []
    choch_up = []
    choch_down = []

    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return {
            'bos_up': bos_up,
            'bos_down': bos_down,
            'choch_up': choch_up,
            'choch_down': choch_down
        }

    # Track current trend (1 = bullish, -1 = bearish, 0 = neutral)
    trend = 0

    for i in range(1, len(bars)):
        # Check for BOS up (break above recent swing high)
        recent_high_idx = [s for s in swing_highs if s < i]
        if recent_high_idx:
            recent_high = bars[recent_high_idx[-1]].high
            if bars[i].close > recent_high:
                if trend <= 0:
                    choch_up.append(i)
                    trend = 1
                else:
                    bos_up.append(i)

        # Check for BOS down (break below recent swing low)
        recent_low_idx = [s for s in swing_lows if s < i]
        if recent_low_idx:
            recent_low = bars[recent_low_idx[-1]].low
            if bars[i].close < recent_low:
                if trend >= 0:
                    choch_down.append(i)
                    trend = -1
                else:
                    bos_down.append(i)

    return {
        'bos_up': bos_up,
        'bos_down': bos_down,
        'choch_up': choch_up,
        'choch_down': choch_down
    }


def extract_bar_structure_features(i: int, smc_structure: SMCStructure) -> Dict[str, float]:
    """
    Extract structure features for a specific bar

    Args:
        i: Bar index
        smc_structure: SMCStructure object

    Returns:
        Dict of features
    """
    features = {
        # Is this bar a swing point?
        'is_swing_high': 1.0 if i in smc_structure.swing_highs else 0.0,
        'is_swing_low': 1.0 if i in smc_structure.swing_lows else 0.0,

        # Structure breaks at this bar
        'bos_up': 1.0 if i in smc_structure.bos_up_indices else 0.0,
        'bos_down': 1.0 if i in smc_structure.bos_down_indices else 0.0,
        'choch_up': 1.0 if i in smc_structure.choch_up_indices else 0.0,
        'choch_down': 1.0 if i in smc_structure.choch_down_indices else 0.0,

        # Bars since last swing
        'bars_since_swing_high': _bars_since_last(i, smc_structure.swing_highs),
        'bars_since_swing_low': _bars_since_last(i, smc_structure.swing_lows),

        # Bars since last structure break
        'bars_since_bos_up': _bars_since_last(i, smc_structure.bos_up_indices),
        'bars_since_bos_down': _bars_since_last(i, smc_structure.bos_down_indices),
    }

    return features


def _bars_since_last(current_idx: int, event_indices: List[int]) -> float:
    """
    Calculate bars since last event

    Args:
        current_idx: Current bar index
        event_indices: List of indices where event occurred

    Returns:
        Number of bars since last event (normalized, max 100)
    """
    valid_events = [e for e in event_indices if e < current_idx]
    if not valid_events:
        return 1.0  # Normalized max

    bars_since = current_idx - valid_events[-1]
    return min(bars_since / 100.0, 1.0)  # Normalize to [0, 1]
