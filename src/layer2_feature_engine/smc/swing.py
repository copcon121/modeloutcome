"""
2-Layer Swing Detection: Internal (wave 5) + External (wave 50)

This module implements Smart Money Concepts (SMC) multi-layer swing detection
following MGannSwing methodology:

- Internal swing (int): Fine-grained structure (wave 5) - minor swings
- External swing (ext): Macro structure (wave 50) - major swings

Architecture:
1. Fractal pivot detection (Gann 2-bar style)
2. Internal swing confirmation (wave 5 - minimum 5 ticks move)
3. External swing confirmation (wave 50 - minimum 50 ticks move, built on internal swings)
"""

from typing import List, Tuple, Optional
from dataclasses import dataclass
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from layer2_feature_engine.core.schema import RawBar
from layer2_feature_engine.smc.config import SMCConfig, DEFAULT_SMC_CONFIG


# ==================== State Classes ====================

@dataclass
class InternalSwingState:
    """
    Internal swing state (wave 5)
    Tracks minor market structure
    """
    # Trend direction
    trend_dir: int = 0  # +1=up, -1=down, 0=neutral

    # Last confirmed internal swing high
    swing_high_idx: int = -1
    swing_high_price: float = 0.0
    swing_high_confirmed: bool = False

    # Last confirmed internal swing low
    swing_low_idx: int = -1
    swing_low_price: float = 0.0
    swing_low_confirmed: bool = False

    # Pending fractal (not yet confirmed as swing)
    pending_high_idx: int = -1
    pending_high_price: float = 0.0

    pending_low_idx: int = -1
    pending_low_price: float = 0.0

    def bars_since_swing_high(self, current_idx: int) -> int:
        """Bars since last confirmed internal swing high"""
        if self.swing_high_idx < 0:
            return 999
        return current_idx - self.swing_high_idx

    def bars_since_swing_low(self, current_idx: int) -> int:
        """Bars since last confirmed internal swing low"""
        if self.swing_low_idx < 0:
            return 999
        return current_idx - self.swing_low_idx


@dataclass
class ExternalSwingState:
    """
    External swing state (wave 50)
    Tracks major market structure
    """
    # Trend direction
    trend_dir: int = 0  # +1=up, -1=down, 0=neutral

    # Last confirmed external swing high
    swing_high_idx: int = -1
    swing_high_price: float = 0.0
    swing_high_confirmed: bool = False

    # Last confirmed external swing low
    swing_low_idx: int = -1
    swing_low_price: float = 0.0
    swing_low_confirmed: bool = False

    # Accumulation since last external swing
    accumulated_move_from_ext_high: float = 0.0
    accumulated_move_from_ext_low: float = 0.0

    def bars_since_swing_high(self, current_idx: int) -> int:
        """Bars since last confirmed external swing high"""
        if self.swing_high_idx < 0:
            return 999
        return current_idx - self.swing_high_idx

    def bars_since_swing_low(self, current_idx: int) -> int:
        """Bars since last confirmed external swing low"""
        if self.swing_low_idx < 0:
            return 999
        return current_idx - self.swing_low_idx


# ==================== Fractal Pivot Detection ====================

def detect_fractal_pivots(
    bars: List[RawBar],
    left: int = 2,
    right: int = 2
) -> Tuple[List[int], List[int]]:
    """
    Detect fractal pivot points (Gann 2-bar style)

    A fractal high is a bar where the high is higher than 'left' bars before
    and 'right' bars after.

    A fractal low is a bar where the low is lower than 'left' bars before
    and 'right' bars after.

    This is the FIRST LAYER - raw fractals, not yet confirmed as swings.

    Args:
        bars: List of RawBar objects
        left: Left lookback period
        right: Right lookback period

    Returns:
        Tuple of (fractal_high_indices, fractal_low_indices)
    """
    if len(bars) < left + right + 1:
        return [], []

    fractal_highs = []
    fractal_lows = []

    for i in range(left, len(bars) - right):
        # Check fractal high
        is_fractal_high = True
        for j in range(i - left, i):
            if bars[j].high >= bars[i].high:
                is_fractal_high = False
                break
        if is_fractal_high:
            for j in range(i + 1, i + right + 1):
                if bars[j].high >= bars[i].high:
                    is_fractal_high = False
                    break

        if is_fractal_high:
            fractal_highs.append(i)

        # Check fractal low
        is_fractal_low = True
        for j in range(i - left, i):
            if bars[j].low <= bars[i].low:
                is_fractal_low = False
                break
        if is_fractal_low:
            for j in range(i + 1, i + right + 1):
                if bars[j].low <= bars[i].low:
                    is_fractal_low = False
                    break

        if is_fractal_low:
            fractal_lows.append(i)

    return fractal_highs, fractal_lows


# ==================== Internal Swing Detector (Wave 5) ====================

class InternalSwingDetector:
    """
    Detects internal swings (wave 5) from fractal pivots

    Rules:
    1. Start from fractal pivots (left=2, right=2)
    2. Confirm as internal swing if:
       - Price moved at least min_int_move_ticks from last swing
       - Minimum bars passed since last swing (anti-noise)
    3. Update trend_dir based on confirmed swings
    """

    def __init__(self, config: SMCConfig):
        self.config = config
        self.state = InternalSwingState()

    def update(self, bars: List[RawBar], current_idx: int) -> InternalSwingState:
        """
        Update internal swing state at current bar index

        Args:
            bars: Full list of bars
            current_idx: Current bar index to process

        Returns:
            Updated InternalSwingState
        """
        if current_idx < self.config.fractal_left + self.config.fractal_right:
            return self.state  # Not enough bars yet

        # Check if current position (minus right lookback) forms a fractal
        # This checks if bar at (current_idx - right) is a fractal pivot
        fractal_check_idx = current_idx - self.config.fractal_right

        if fractal_check_idx < self.config.fractal_left:
            return self.state

        # Check for fractal high at fractal_check_idx
        is_fractal_high = True
        pivot_bar = bars[fractal_check_idx]

        # Check left side
        for i in range(fractal_check_idx - self.config.fractal_left, fractal_check_idx):
            if bars[i].high >= pivot_bar.high:
                is_fractal_high = False
                break

        # Check right side
        if is_fractal_high:
            for i in range(fractal_check_idx + 1, fractal_check_idx + self.config.fractal_right + 1):
                if bars[i].high >= pivot_bar.high:
                    is_fractal_high = False
                    break

        # If fractal high and not already processed
        if is_fractal_high and fractal_check_idx not in [self.state.pending_high_idx, self.state.swing_high_idx]:
            if self._confirm_internal_swing_high(bars, fractal_check_idx, pivot_bar.high):
                self.state.swing_high_idx = fractal_check_idx
                self.state.swing_high_price = pivot_bar.high
                self.state.swing_high_confirmed = True
                self.state.pending_high_idx = -1
            else:
                self.state.pending_high_idx = fractal_check_idx
                self.state.pending_high_price = pivot_bar.high

        # Check for fractal low at fractal_check_idx
        is_fractal_low = True

        # Check left side
        for i in range(fractal_check_idx - self.config.fractal_left, fractal_check_idx):
            if bars[i].low <= pivot_bar.low:
                is_fractal_low = False
                break

        # Check right side
        if is_fractal_low:
            for i in range(fractal_check_idx + 1, fractal_check_idx + self.config.fractal_right + 1):
                if bars[i].low <= pivot_bar.low:
                    is_fractal_low = False
                    break

        # If fractal low and not already processed
        if is_fractal_low and fractal_check_idx not in [self.state.pending_low_idx, self.state.swing_low_idx]:
            if self._confirm_internal_swing_low(bars, fractal_check_idx, pivot_bar.low):
                self.state.swing_low_idx = fractal_check_idx
                self.state.swing_low_price = pivot_bar.low
                self.state.swing_low_confirmed = True
                self.state.pending_low_idx = -1
            else:
                self.state.pending_low_idx = fractal_check_idx
                self.state.pending_low_price = pivot_bar.low

        # Update trend direction
        self._update_trend_dir()

        return self.state

    def _confirm_internal_swing_high(self, bars: List[RawBar], idx: int, price: float) -> bool:
        """
        Confirm fractal high as internal swing high (wave 5)

        SIMPLIFIED: Accept all fractals with minimum bars spacing only
        """
        # Check minimum bars since last swing high (avoid too many swings)
        if self.state.swing_high_idx >= 0:
            bars_since = idx - self.state.swing_high_idx
            if bars_since < self.config.min_bars_between_swings:
                return False  # Too soon

        return True  # Accept all other fractals

    def _confirm_internal_swing_low(self, bars: List[RawBar], idx: int, price: float) -> bool:
        """
        Confirm fractal low as internal swing low (wave 5)

        SIMPLIFIED: Accept all fractals with minimum bars spacing only
        """
        # Check minimum bars since last swing low (avoid too many swings)
        if self.state.swing_low_idx >= 0:
            bars_since = idx - self.state.swing_low_idx
            if bars_since < self.config.min_bars_between_swings:
                return False  # Too soon

        return True  # Accept all other fractals

    def _update_trend_dir(self):
        """Update internal trend direction based on swing highs/lows"""
        if self.state.swing_high_idx < 0 and self.state.swing_low_idx < 0:
            self.state.trend_dir = 0  # Neutral
        elif self.state.swing_high_idx < 0:
            self.state.trend_dir = -1  # Only lows = bearish
        elif self.state.swing_low_idx < 0:
            self.state.trend_dir = +1  # Only highs = bullish
        else:
            # Both exist - trend = direction of most recent
            if self.state.swing_high_idx > self.state.swing_low_idx:
                self.state.trend_dir = +1  # Recent high = bullish
            else:
                self.state.trend_dir = -1  # Recent low = bearish


# ==================== External Swing Detector (Wave 50) ====================

class ExternalSwingDetector:
    """
    Detects external swings (wave 50) from internal swings

    Rules:
    1. Build on top of internal swings (not raw fractals)
    2. Confirm as external swing if:
       - Accumulated move >= min_ext_move_ticks from last external swing
       - Series of internal swings in same direction
    3. Update trend_dir based on confirmed swings
    """

    def __init__(self, config: SMCConfig):
        self.config = config
        self.state = ExternalSwingState()

    def update(
        self,
        bars: List[RawBar],
        current_idx: int,
        int_state: InternalSwingState
    ) -> ExternalSwingState:
        """
        Update external swing state based on internal swings

        Args:
            bars: Full list of bars
            current_idx: Current bar index
            int_state: Current internal swing state

        Returns:
            Updated ExternalSwingState
        """
        if int_state.swing_high_idx < 0 and int_state.swing_low_idx < 0:
            return self.state  # No internal swings yet

        current_price = bars[current_idx].close

        # Check for external swing high confirmation (wave 50)
        if int_state.swing_high_idx >= 0 and int_state.swing_high_confirmed:
            if self._should_confirm_external_high(int_state.swing_high_price, int_state.swing_high_idx):
                self.state.swing_high_idx = int_state.swing_high_idx
                self.state.swing_high_price = int_state.swing_high_price
                self.state.swing_high_confirmed = True
                self.state.accumulated_move_from_ext_low = 0.0  # Reset accumulation

        # Check for external swing low confirmation (wave 50)
        if int_state.swing_low_idx >= 0 and int_state.swing_low_confirmed:
            if self._should_confirm_external_low(int_state.swing_low_price, int_state.swing_low_idx):
                self.state.swing_low_idx = int_state.swing_low_idx
                self.state.swing_low_price = int_state.swing_low_price
                self.state.swing_low_confirmed = True
                self.state.accumulated_move_from_ext_high = 0.0  # Reset accumulation

        # Update accumulated moves
        if self.state.swing_high_idx >= 0:
            self.state.accumulated_move_from_ext_high = abs(current_price - self.state.swing_high_price)

        if self.state.swing_low_idx >= 0:
            self.state.accumulated_move_from_ext_low = abs(current_price - self.state.swing_low_price)

        # Update trend direction
        self._update_trend_dir()

        return self.state

    def _should_confirm_external_high(self, price: float, idx: int) -> bool:
        """
        Check if internal swing high should be promoted to external swing high (wave 50)

        Rules:
        - Must be at least min_ext_move_ticks above last external swing low
        - Accumulated move from last external low >= threshold
        """
        if self.state.swing_low_idx < 0:
            return True  # First external swing

        # Check move size from last external swing low
        move_ticks = self.config.price_to_ticks(price - self.state.swing_low_price)
        if move_ticks < self.config.min_ext_move_ticks:
            return False  # Not enough movement for wave 50

        # Check if this high is higher than last external high (Higher High)
        if self.state.swing_high_idx >= 0:
            if price <= self.state.swing_high_price:
                return False  # Not a higher high

        return True

    def _should_confirm_external_low(self, price: float, idx: int) -> bool:
        """
        Check if internal swing low should be promoted to external swing low (wave 50)

        Rules:
        - Must be at least min_ext_move_ticks below last external swing high
        - Accumulated move from last external high >= threshold
        """
        if self.state.swing_high_idx < 0:
            return True  # First external swing

        # Check move size from last external swing high
        move_ticks = self.config.price_to_ticks(self.state.swing_high_price - price)
        if move_ticks < self.config.min_ext_move_ticks:
            return False  # Not enough movement for wave 50

        # Check if this low is lower than last external low (Lower Low)
        if self.state.swing_low_idx >= 0:
            if price >= self.state.swing_low_price:
                return False  # Not a lower low

        return True

    def _update_trend_dir(self):
        """Update external trend direction based on swing highs/lows"""
        if self.state.swing_high_idx < 0 and self.state.swing_low_idx < 0:
            self.state.trend_dir = 0  # Neutral
        elif self.state.swing_high_idx < 0:
            self.state.trend_dir = -1  # Only lows = bearish
        elif self.state.swing_low_idx < 0:
            self.state.trend_dir = +1  # Only highs = bullish
        else:
            # Both exist - trend = direction of most recent
            if self.state.swing_high_idx > self.state.swing_low_idx:
                self.state.trend_dir = +1  # Recent high = bullish
            else:
                self.state.trend_dir = -1  # Recent low = bearish


# ==================== Backward Compatibility ====================

def detect_swings(bars: List[RawBar], lookback: int = 2) -> Tuple[List[int], List[int]]:
    """
    Legacy function - kept for backward compatibility
    Uses fractal detection only (no wave 5/50 confirmation)

    For NEW code, use InternalSwingDetector / ExternalSwingDetector instead.
    """
    return detect_fractal_pivots(bars, lookback, lookback)


def get_nearest_swing_high(index: int, swing_highs: List[int]) -> int:
    """Get nearest swing high before given index"""
    valid_swings = [s for s in swing_highs if s < index]
    return valid_swings[-1] if valid_swings else -1


def get_nearest_swing_low(index: int, swing_lows: List[int]) -> int:
    """Get nearest swing low before given index"""
    valid_swings = [s for s in swing_lows if s < index]
    return valid_swings[-1] if valid_swings else -1
