"""
Swing High/Low Detection
Identifies pivot points in price action
"""

from typing import List, Tuple
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from layer2_feature_engine.core.schema import RawBar


def detect_swings(bars: List[RawBar], lookback: int = 2) -> Tuple[List[int], List[int]]:
    """
    Detect swing highs and swing lows

    A swing high is a bar where the high is higher than the highs of
    `lookback` bars before and after it.

    A swing low is a bar where the low is lower than the lows of
    `lookback` bars before and after it.

    Args:
        bars: List of RawBar objects
        lookback: Number of bars to check on each side

    Returns:
        Tuple of (swing_high_indices, swing_low_indices)
    """
    if len(bars) < 2 * lookback + 1:
        return [], []

    swing_highs = []
    swing_lows = []

    for i in range(lookback, len(bars) - lookback):
        # Check for swing high
        is_swing_high = True
        for j in range(i - lookback, i):
            if bars[j].high >= bars[i].high:
                is_swing_high = False
                break
        if is_swing_high:
            for j in range(i + 1, i + lookback + 1):
                if bars[j].high >= bars[i].high:
                    is_swing_high = False
                    break

        if is_swing_high:
            swing_highs.append(i)

        # Check for swing low
        is_swing_low = True
        for j in range(i - lookback, i):
            if bars[j].low <= bars[i].low:
                is_swing_low = False
                break
        if is_swing_low:
            for j in range(i + 1, i + lookback + 1):
                if bars[j].low <= bars[i].low:
                    is_swing_low = False
                    break

        if is_swing_low:
            swing_lows.append(i)

    return swing_highs, swing_lows


def get_nearest_swing_high(index: int, swing_highs: List[int]) -> int:
    """
    Get the nearest swing high before the given index

    Args:
        index: Current bar index
        swing_highs: List of swing high indices

    Returns:
        Index of nearest swing high, or -1 if none found
    """
    valid_swings = [s for s in swing_highs if s < index]
    return valid_swings[-1] if valid_swings else -1


def get_nearest_swing_low(index: int, swing_lows: List[int]) -> int:
    """
    Get the nearest swing low before the given index

    Args:
        index: Current bar index
        swing_lows: List of swing low indices

    Returns:
        Index of nearest swing low, or -1 if none found
    """
    valid_swings = [s for s in swing_lows if s < index]
    return valid_swings[-1] if valid_swings else -1
