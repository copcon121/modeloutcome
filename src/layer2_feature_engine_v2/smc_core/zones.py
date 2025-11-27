"""
Phase 2 Feature Engine V2 - Zone Detection (PD, FVG, OB)
Matches NinjaTrader SMC indicator zone logic
"""

from typing import Optional, List
from dataclasses import dataclass
import logging

from ..schema import RawBar, InternalSwingState, ExternalSwingState
from ..config import SMCConfig

logger = logging.getLogger(__name__)


@dataclass
class PDZoneState:
    """Premium/Discount zone state"""
    trail_up: float = 0.0      # Trailing swing high
    trail_down: float = 0.0    # Trailing swing low
    equilibrium: float = 0.0   # Mid-point (50%)
    in_premium: bool = False   # Price > EQ
    in_discount: bool = False  # Price < EQ
    price_position_pct: float = 50.0  # Position in range (0-100)


class PDZoneTracker:
    """
    Premium/Discount Zone Tracker
    
    Matches NinjaTrader UpdateTrailingHighLow logic:
    - Trail Up = Highest swing high
    - Trail Down = Lowest swing low
    - Equilibrium = (Trail Up + Trail Down) / 2
    - Premium = price above EQ
    - Discount = price below EQ
    """
    
    def __init__(self, config: SMCConfig):
        self.config = config
        self.state = PDZoneState()
    
    def update(self, bar: RawBar, swing_state: ExternalSwingState) -> PDZoneState:
        """
        Update PD zones based on swing levels
        
        Uses EXTERNAL swings (window=50) for trailing high/low
        
        Args:
            bar: Current bar
            swing_state: External swing state (uses external swings for PD)
        
        Returns:
            Updated PDZoneState
        """
        # Update trailing levels from swing state
        # NinjaTrader uses swing high/low as trailing levels
        if swing_state.swing_high_price and swing_state.swing_high_price != 0:
            # Initialize or update trail up
            if self.state.trail_up == 0:
                self.state.trail_up = swing_state.swing_high_price
            else:
                # Update to highest swing high
                self.state.trail_up = max(self.state.trail_up, swing_state.swing_high_price)
        
        if swing_state.swing_low_price and swing_state.swing_low_price != 0:
            # Initialize or update trail down
            if self.state.trail_down == 0:
                self.state.trail_down = swing_state.swing_low_price
            else:
                # Update to lowest swing low
                self.state.trail_down = min(self.state.trail_down, swing_state.swing_low_price)
        
        # Calculate equilibrium (only if both levels set)
        if self.state.trail_up != 0 and self.state.trail_down != 0:
            self.state.equilibrium = (self.state.trail_up + self.state.trail_down) / 2
            
            # Determine if in premium or discount
            current_price = bar.c
            
            if current_price > self.state.equilibrium:
                self.state.in_premium = True
                self.state.in_discount = False
            elif current_price < self.state.equilibrium:
                self.state.in_premium = False
                self.state.in_discount = True
            else:
                self.state.in_premium = False
                self.state.in_discount = False
            
            # Calculate price position percentage (0 = trail_down, 100 = trail_up)
            range_size = self.state.trail_up - self.state.trail_down
            if range_size > 0:
                position = current_price - self.state.trail_down
                self.state.price_position_pct = (position / range_size) * 100
                # Clamp to 0-100
                self.state.price_position_pct = max(0, min(100, self.state.price_position_pct))
            else:
                self.state.price_position_pct = 50.0
        
        return self.state
    
    def get_state(self) -> PDZoneState:
        """Get current state"""
        return self.state


# FVG and OB detection will be added here later
@dataclass
class FVGZone:
    """Fair Value Gap zone"""
    top: float
    bottom: float
    bar_index: int
    is_bullish: bool
    filled: bool = False
    fill_percentage: float = 0.0


@dataclass
class OBZone:
    """Order Block zone"""
    top: float
    bottom: float
    bar_index: int
    is_bullish: bool
    broken: bool = False
