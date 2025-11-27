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
    
    CRITICAL: Only updates when swings are CONFIRMED (swing_bar_index changes)
    This ensures stable PD zones for ML training!
    """
    
    def __init__(self, config: SMCConfig):
        self.config = config
        self.state = PDZoneState()
        
        # Track last confirmed swing indices to detect NEW swings
        self.last_swing_high_index = -1
        self.last_swing_low_index = -1
    
    def update(self, bar: RawBar, swing_state: ExternalSwingState) -> PDZoneState:
        """
        Update PD zones based on current swing range
        
        Uses EXTERNAL swings (window=50) for premium/discount calculation
        
        CRITICAL Logic:
        - After swing low and swing high are BOTH confirmed
        - Calculate EQ = (swing_high + swing_low) / 2
        - Premium = price > EQ (upper half of range)
        - Discount = price < EQ (lower half of range)
        
        This divides the CURRENT confirmed swing range in half!
        
        Args:
            bar: Current bar
            swing_state: External swing state (uses external swings for PD)
        
        Returns:
            Updated PDZoneState
        """
        # Update swing levels when confirmed (swing_bar_index changes)
        if (swing_state.swing_high_bar_index != -1 and 
            swing_state.swing_high_bar_index != self.last_swing_high_index and
            swing_state.swing_high_price and swing_state.swing_high_price != 0):
            
            self.state.trail_up = swing_state.swing_high_price
            self.last_swing_high_index = swing_state.swing_high_bar_index
            logger.debug(f"[PDZone] Updated swing HIGH to {self.state.trail_up} at bar {bar.bar_index}")
        
        if (swing_state.swing_low_bar_index != -1 and
            swing_state.swing_low_bar_index != self.last_swing_low_index and
            swing_state.swing_low_price and swing_state.swing_low_price != 0):
            
            self.state.trail_down = swing_state.swing_low_price
            self.last_swing_low_index = swing_state.swing_low_bar_index
            logger.debug(f"[PDZone] Updated swing LOW to {self.state.trail_down} at bar {bar.bar_index}")
        
        # Calculate PD zones from CURRENT swing range
        # Need BOTH swing high and swing low confirmed
        if self.state.trail_up != 0 and self.state.trail_down != 0:
            # Equilibrium = midpoint of current swing range
            self.state.equilibrium = (self.state.trail_up + self.state.trail_down) / 2
            
            current_price = bar.c
            
            # Premium = upper half of range (above EQ)
            # Discount = lower half of range (below EQ)
            if current_price > self.state.equilibrium:
                self.state.in_premium = True
                self.state.in_discount = False
            elif current_price < self.state.equilibrium:
                self.state.in_premium = False
                self.state.in_discount = True
            else:
                self.state.in_premium = False
                self.state.in_discount = False
            
            # Calculate position in range (0% = swing_low, 100% = swing_high)
            range_size = self.state.trail_up - self.state.trail_down
            if range_size > 0:
                position = current_price - self.state.trail_down
                self.state.price_position_pct = (position / range_size) * 100
                self.state.price_position_pct = max(0, min(100, self.state.price_position_pct))
            else:
                self.state.price_position_pct = 50.0
        
        return self.state
    
    def get_state(self) -> PDZoneState:
        """Get current state"""
        return self.state


# FVG and OB Zone Detection
@dataclass
class FVGZone:
    """Fair Value Gap zone"""
    top: float
    bottom: float
    bar_index: int
    is_bullish: bool
    filled: bool = False
    fill_percentage: float = 0.0
    midpoint: float = 0.0
    
    def __post_init__(self):
        self.midpoint = (self.top + self.bottom) / 2


@dataclass
class OBZone:
    """Order Block zone"""
    top: float
    bottom: float
    bar_index: int
    is_bullish: bool
    broken: bool = False
    source_type: str = ""  # "BOS_UP", "BOS_DOWN", "CHOCH_UP", "CHOCH_DOWN"


class FVGDetector:
    """
    Fair Value Gap Detector
    
    Detects 3-bar FVG patterns:
    - Bullish FVG: bar[2].low > bar[0].high (gap up)
    - Bearish FVG: bar[2].high < bar[0].low (gap down)
    
    Tracks fill percentage as price moves through gap
    """
    
    def __init__(self, config: SMCConfig, tick_size: float):
        self.config = config
        self.tick_size = tick_size
        self.min_gap = config.fvg_min_gap_ticks * tick_size
        
        # Store last 3 bars for pattern detection
        self.bar_buffer: List[RawBar] = []
        
        # Active FVG zones
        self.active_fvgs: List[FVGZone] = []
        
    def update(self, bar: RawBar) -> List[FVGZone]:
        """
        Update FVG detection with new bar
        
        Returns list of newly detected FVGs (if any)
        """
        # Add to buffer
        self.bar_buffer.append(bar)
        if len(self.bar_buffer) > 3:
            self.bar_buffer.pop(0)
        
        new_fvgs = []
        
        # Need 3 bars for pattern
        if len(self.bar_buffer) == 3:
            bar0, bar1, bar2 = self.bar_buffer
            
            # Check Bullish FVG: bar2.low > bar0.high
            if bar2.l > bar0.h:
                gap_size = bar2.l - bar0.h
                if gap_size >= self.min_gap:
                    fvg = FVGZone(
                        top=bar2.l,
                        bottom=bar0.h,
                        bar_index=bar2.bar_index,
                        is_bullish=True
                    )
                    new_fvgs.append(fvg)
                    self.active_fvgs.append(fvg)
                    logger.debug(f"[FVG] Bullish FVG detected at bar {bar2.bar_index}, gap={gap_size:.1f}")
            
            # Check Bearish FVG: bar2.high < bar0.low
            elif bar2.h < bar0.l:
                gap_size = bar0.l - bar2.h
                if gap_size >= self.min_gap:
                    fvg = FVGZone(
                        top=bar0.l,
                        bottom=bar2.h,
                        bar_index=bar2.bar_index,
                        is_bullish=False
                    )
                    new_fvgs.append(fvg)
                    self.active_fvgs.append(fvg)
                    logger.debug(f"[FVG] Bearish FVG detected at bar {bar2.bar_index}, gap={gap_size:.1f}")
        
        # Update fill status for all active FVGs
        self._update_fills(bar)
        
        return new_fvgs
    
    def _update_fills(self, bar: RawBar):
        """Update fill percentage for active FVGs"""
        for fvg in self.active_fvgs:
            if fvg.filled:
                continue
            
            # Check if price has entered FVG
            if bar.h >= fvg.bottom and bar.l <= fvg.top:
                # Calculate fill percentage
                gap_size = fvg.top - fvg.bottom
                
                if fvg.is_bullish:
                    # Bullish FVG fills from bottom up
                    if bar.l <= fvg.bottom:
                        filled_amount = min(bar.h, fvg.top) - fvg.bottom
                    else:
                        filled_amount = min(bar.h, fvg.top) - bar.l
                else:
                    # Bearish FVG fills from top down
                    if bar.h >= fvg.top:
                        filled_amount = fvg.top - max(bar.l, fvg.bottom)
                    else:
                        filled_amount = bar.h - max(bar.l, fvg.bottom)
                
                fvg.fill_percentage = (filled_amount / gap_size) * 100 if gap_size > 0 else 0
                
                # Mark as filled if 100%
                if fvg.fill_percentage >= 100:
                    fvg.filled = True
                    logger.debug(f"[FVG] FVG at bar {fvg.bar_index} fully filled at bar {bar.bar_index}")
    
    def get_active_fvgs(self) -> List[FVGZone]:
        """Get all active (non-filled) FVGs"""
        return [fvg for fvg in self.active_fvgs if not fvg.filled]
    
    def get_all_fvgs(self) -> List[FVGZone]:
        """
        Get ALL FVGs (including filled ones)
        
        IMPORTANT for ML: Keep full history of zones for model context!
        Filled FVGs still provide valuable information about market structure.
        """
        return self.active_fvgs


class OBDetector:
    """
    Order Block Detector
    
    Detects OB zones when BOS/CHoCH occurs:
    - Source candle = last candle before break
    - Zone = candle body or full range (configurable)
    - Tracks if OB is broken/mitigated
    """
    
    def __init__(self, config: SMCConfig, tick_size: float):
        self.config = config
        self.tick_size = tick_size
        self.use_full_candle = config.ob_use_full_candle
        
        # Store last bar for OB detection
        self.last_bar: Optional[RawBar] = None
        
        # Active OB zones
        self.active_obs: List[OBZone] = []
    
    def update(self, bar: RawBar, structure_state: dict) -> List[OBZone]:
        """
        Update OB detection based on structure breaks
        
        Args:
            bar: Current bar
            structure_state: Dict with bos_up, bos_down, choch_up, choch_down flags
        
        Returns:
            List of newly created OBs
        """
        new_obs = []
        
        # Check for BOS/CHoCH and create OB from last bar
        if self.last_bar:
            # Bullish OB from BOS/CHoCH UP
            if structure_state.get('bos_up') or structure_state.get('choch_up'):
                if self.use_full_candle:
                    top = self.last_bar.h
                    bottom = self.last_bar.l
                else:
                    top = max(self.last_bar.o, self.last_bar.c)
                    bottom = min(self.last_bar.o, self.last_bar.c)
                
                source_type = 'BOS_UP' if structure_state.get('bos_up') else 'CHOCH_UP'
                ob = OBZone(
                    top=top,
                    bottom=bottom,
                    bar_index=self.last_bar.bar_index,
                    is_bullish=True,
                    source_type=source_type
                )
                new_obs.append(ob)
                self.active_obs.append(ob)
                logger.debug(f"[OB] Bullish OB created from {source_type} at bar {self.last_bar.bar_index}")
            
            # Bearish OB from BOS/CHoCH DOWN
            elif structure_state.get('bos_down') or structure_state.get('choch_down'):
                if self.use_full_candle:
                    top = self.last_bar.h
                    bottom = self.last_bar.l
                else:
                    top = max(self.last_bar.o, self.last_bar.c)
                    bottom = min(self.last_bar.o, self.last_bar.c)
                
                source_type = 'BOS_DOWN' if structure_state.get('bos_down') else 'CHOCH_DOWN'
                ob = OBZone(
                    top=top,
                    bottom=bottom,
                    bar_index=self.last_bar.bar_index,
                    is_bullish=False,
                    source_type=source_type
                )
                new_obs.append(ob)
                self.active_obs.append(ob)
                logger.debug(f"[OB] Bearish OB created from {source_type} at bar {self.last_bar.bar_index}")
        
        # Update broken status
        self._update_broken(bar)
        
        # Store current bar for next iteration
        self.last_bar = bar
        
        return new_obs
    
    def _update_broken(self, bar: RawBar):
        """Check if OBs are broken/mitigated"""
        for ob in self.active_obs:
            if ob.broken:
                continue
            
            # Bullish OB broken if price goes below bottom
            if ob.is_bullish and bar.l < ob.bottom:
                ob.broken = True
                logger.debug(f"[OB] Bullish OB at bar {ob.bar_index} broken at bar {bar.bar_index}")
            
            # Bearish OB broken if price goes above top
            elif not ob.is_bullish and bar.h > ob.top:
                ob.broken = True
                logger.debug(f"[OB] Bearish OB at bar {ob.bar_index} broken at bar {bar.bar_index}")
    
    def get_active_obs(self) -> List[OBZone]:
        """Get all active (non-broken) OBs"""
        return [ob for ob in self.active_obs if not ob.broken]
    
    def get_all_obs(self) -> List[OBZone]:
        """
        Get ALL OBs (including broken ones)
        
        IMPORTANT for ML: Keep full history of zones for model context!
        Broken OBs still provide valuable information about market structure.
        """
        return self.active_obs
