"""
Phase 2 Feature Engine V2 - Swing Detection
Window-based swing detection matching NinjaTrader logic
Uses rolling MAX/MIN approach (NOT Gann fractals!)
"""

from collections import deque
from typing import Optional, List
import logging

from ..schema import RawBar, InternalSwingState, ExternalSwingState

from ..config import SMCConfig


logger = logging.getLogger(__name__)


# Leg constants (matches NinjaTrader)
BULLISH_LEG = 0   # Swing from low (expecting rally)
BEARISH_LEG = 1   # Swing from high (expecting pullback)


class InternalSwingDetector:
    """
    Internal swing detector using rolling window MAX/MIN approach
    Matches NinjaTrader IntWindow=5 logic (wave 5)
    
    Algorithm (from NinjaTrader UpdateStructure_MTF):
    1. Maintain rolling window of highs/lows
    2. New BEARISH leg when: current high > MAX(previous window highs)
    3. New BULLISH leg when: current low < MIN(previous window lows)
    4. Update swing high/low when leg changes
    """
    
    def __init__(self, config: SMCConfig, tick_size: float):
        self.config = config
        self.tick_size = tick_size
        self.window = config.swing_int_window
        
        # Rolling windows - need extra space for delayed checking
        # Check at idx=window, comparing with window bars → need window + idx + buffer
        max_size = max(100, self.window * 3)  # Generous buffer
        self.highs = deque(maxlen=max_size)
        self.lows = deque(maxlen=max_size)
        self.bar_indices = deque(maxlen=max_size)
        
        # Current state
        self.state = InternalSwingState()
        self.current_bar_index = 0
        
    def update(self, bar: RawBar) -> InternalSwingState:
        """
        Update internal swing state with new bar
        
        CRITICAL: Matches NinjaTrader UpdateStructure_MTF exactly
        
        NinjaTrader logic:
        - int idx = win;  // e.g., idx = 5
        - Check: highs[idx] > MAX(highs, win)[0]
        - Where highs[idx] = bar 'idx' bars ago
        - And MAX(highs, win)[0] = max of last 'win' bars from [0]
        
        In NinjaTrader: [0] = current/newest, [idx] = idx bars ago
        In Python deque: [-1] = newest, [-idx-1] = idx bars ago
        
        Returns:
            Updated InternalSwingState
        """
        # Add to rolling window
        self.highs.append(bar.h)
        self.lows.append(bar.l)
        self.bar_indices.append(bar.bar_index)
        self.current_bar_index = bar.bar_index
        
        # Need enough bars: currentBar <= idx + 2 → need at least idx + 3 bars
        if len(self.highs) <= self.window + 2:
            return self.state
        
        # idx = win (position to check, bars ago from current)
        idx = self.window  # e.g., 5 for internal
        
        # In Python: check bar at position -(idx+1) from end
        # Example: if idx=5, and we have 10 bars [0..9], we check bar at index -6 = bar[4]
        # This is "5 bars ago" from current bar[9]
        check_high = self.highs[-(idx + 1)]
        check_low = self.lows[-(idx + 1)]
        check_bar_index = self.bar_indices[-(idx + 1)]
        
        # MAX/MIN of last 'win' bars (from current [0] backwards)
        # This is bars [0] to [win-1] in NinjaTrader = last 'win' bars in Python
        window_highs = list(self.highs)[-self.window:]  # Last 'window' bars (newest)
        window_lows = list(self.lows)[-self.window:]
        
        max_window_high = max(window_highs)
        min_window_low = min(window_lows)
        
        # Check if bar at idx breaks out of window
        # NinjaTrader: newLegHigh = highs[idx] > maxSeries[0]
        new_leg_high = check_high > max_window_high
        new_leg_low = check_low < min_window_low
        
        # Determine leg
        leg_now = self.state.last_leg
        if new_leg_high:
            leg_now = BEARISH_LEG
        elif new_leg_low:
            leg_now = BULLISH_LEG
        
        # First leg initialization
        if self.state.last_leg == -1 and (new_leg_high or new_leg_low):
            self.state.last_leg = leg_now
            if leg_now == BULLISH_LEG:
                self.state.swing_low_price = check_low
                self.state.swing_low_bar_index = check_bar_index
                logger.debug(f"[IntSwing] First leg: BULLISH at bar {check_bar_index}, low={check_low}")
            else:
                self.state.swing_high_price = check_high
                self.state.swing_high_bar_index = check_bar_index
                logger.debug(f"[IntSwing] First leg: BEARISH at bar {check_bar_index}, high={check_high}")
            return self.state
        
        # Check for leg change
        start_of_new_leg = (self.state.last_leg != -1) and (leg_now != self.state.last_leg)
        
        if start_of_new_leg:
            if leg_now == BULLISH_LEG:
                self.state.prev_swing_low_price = self.state.swing_low_price
                self.state.swing_low_price = check_low
                self.state.swing_low_bar_index = check_bar_index
                self.state.swing_low_crossed = False
                self.state.last_leg = BULLISH_LEG
                self.state.trend_bias = -1
                
                logger.debug(f"[IntSwing] New BULLISH leg at bar {check_bar_index}, low={check_low}")
                
            else:  # BEARISH_LEG
                self.state.prev_swing_high_price = self.state.swing_high_price
                self.state.swing_high_price = check_high
                self.state.swing_high_bar_index = check_bar_index
                self.state.swing_high_crossed = False
                self.state.last_leg = BEARISH_LEG
                self.state.trend_bias = 1
                
                logger.debug(f"[IntSwing] New BEARISH leg at bar {check_bar_index}, high={check_high}")
        
        return self.state
    
    def get_state(self) -> InternalSwingState:
        """Get current state"""
        return self.state


class ExternalSwingDetector:
    """
    External swing detector using larger rolling window
    Matches NinjaTrader WinExt=50 logic (wave 50)
    
    Same algorithm as Internal but with larger window
    """
    
    def __init__(self, config: SMCConfig, tick_size: float):
        self.config = config
        self.tick_size = tick_size
        self.window = config.swing_ext_window
        
        # Rolling windows - need extra space
        max_size = max(200, self.window * 3)
        self.highs = deque(maxlen=max_size)
        self.lows = deque(maxlen=max_size)
        self.bar_indices = deque(maxlen=max_size)
        
        # Current state
        self.state = ExternalSwingState()
        self.current_bar_index = 0
        
    def update(self, bar: RawBar) -> ExternalSwingState:
        """Update external swing - same logic as internal but larger window"""
        self.highs.append(bar.h)
        self.lows.append(bar.l)
        self.bar_indices.append(bar.bar_index)
        self.current_bar_index = bar.bar_index
        
        if len(self.highs) <= self.window + 2:
            return self.state
        
        idx = self.window
        check_high = self.highs[-(idx + 1)]
        check_low = self.lows[-(idx + 1)]
        check_bar_index = self.bar_indices[-(idx + 1)]
        
        window_highs = list(self.highs)[-self.window:]
        window_lows = list(self.lows)[-self.window:]
        
        max_window_high = max(window_highs)
        min_window_low = min(window_lows)
        
        new_leg_high = check_high > max_window_high
        new_leg_low = check_low < min_window_low
        
        leg_now = self.state.last_leg
        if new_leg_high:
            leg_now = BEARISH_LEG
        elif new_leg_low:
            leg_now = BULLISH_LEG
        
        if self.state.last_leg == -1 and (new_leg_high or new_leg_low):
            self.state.last_leg = leg_now
            if leg_now == BULLISH_LEG:
                self.state.swing_low_price = check_low
                self.state.swing_low_bar_index = check_bar_index
                logger.debug(f"[ExtSwing] First leg: BULLISH at bar {check_bar_index}, low={check_low}")
            else:
                self.state.swing_high_price = check_high
                self.state.swing_high_bar_index = check_bar_index
                logger.debug(f"[ExtSwing] First leg: BEARISH at bar {check_bar_index}, high={check_high}")
            return self.state
        
        start_of_new_leg = (self.state.last_leg != -1) and (leg_now != self.state.last_leg)
        
        if start_of_new_leg:
            if leg_now == BULLISH_LEG:
                self.state.prev_swing_low_price = self.state.swing_low_price
                self.state.swing_low_price = check_low
                self.state.swing_low_bar_index = check_bar_index
                self.state.swing_low_crossed = False
                self.state.last_leg = BULLISH_LEG
                self.state.trend_bias = -1
                logger.debug(f"[ExtSwing] New BULLISH leg at bar {check_bar_index}, low={check_low}")
            else:
                self.state.prev_swing_high_price = self.state.swing_high_price
                self.state.swing_high_price = check_high
                self.state.swing_high_bar_index = check_bar_index
                self.state.swing_high_crossed = False
                self.state.last_leg = BEARISH_LEG
                self.state.trend_bias = 1
                logger.debug(f"[ExtSwing] New BEARISH leg at bar {check_bar_index}, high={check_high}")
        
        return self.state
    
    def get_state(self) -> ExternalSwingState:
        """Get current state"""
        return self.state
