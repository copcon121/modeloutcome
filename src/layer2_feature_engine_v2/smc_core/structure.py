"""
Phase 2 Feature Engine V2 - Structure Detection (BOS/CHoCH)
Matches NinjaTrader SMC indicator logic
"""

from typing import Optional
import logging

from ..schema import RawBar, InternalSwingState, ExternalSwingState
from ..config import SMCConfig

logger = logging.getLogger(__name__)


class StructureDetector:
    """
    Detects Break of Structure (BOS) and Change of Character (CHoCH)
    
    Algorithm (from NinjaTrader CheckBosChoCh):
    
    BOS (Break of Structure) = Price breaks previous swing in SAME direction as bias
    - Bullish bias: close > swing_high + buffer → BOS UP
    - Bearish bias: close < swing_low - buffer → BOS DOWN
    
    CHoCH (Change of Character) = Price breaks swing in OPPOSITE direction
    - Bullish bias: close < swing_low - buffer → CHoCH DOWN
    - Bearish bias: close > swing_high + buffer → CHoCH UP
    
    Uses ConfirmOnClose flag to determine if we check close or current price
    """
    
    def __init__(self, config: SMCConfig, tick_size: float):
        self.config = config
        self.tick_size = tick_size
        self.buffer = config.bos_close_buffer_ticks * tick_size
        self.confirm_on_close = config.confirm_on_close
        
        # Structure state for Internal
        self.int_structure_dir = 0  # 1=bullish, -1=bearish, 0=undefined
        self.int_bos_up = False
        self.int_bos_down = False
        self.int_choch_up = False
        self.int_choch_down = False
        
        # Structure state for External
        self.ext_structure_dir = 0
        self.ext_bos_up = False
        self.ext_bos_down = False
        self.ext_choch_up = False
        self.ext_choch_down = False
    
    def update_internal(self, bar: RawBar, swing_state: InternalSwingState):
        """
        Update internal structure (BOS/CHoCH) based on swing state
        
        NinjaTrader logic (CheckBosChoCh_MTF):
        1. Get swing high/low levels
        2. Check if price breaks with buffer
        3. Determine if BOS (same direction) or CHoCH (opposite direction)
        4. Update structure direction
        """
        # Reset pulses
        self.int_bos_up = False
        self.int_bos_down = False
        self.int_choch_up = False
        self.int_choch_down = False
        
        # Need valid swing levels
        if (swing_state.swing_high_price is None or swing_state.swing_high_price == 0 or
            swing_state.swing_low_price is None or swing_state.swing_low_price == 0):
            return
        
        # Get price to check (close or current high/low)
        if self.confirm_on_close:
            check_price_high = bar.c
            check_price_low = bar.c
        else:
            check_price_high = bar.h
            check_price_low = bar.l
        
        # Get swing levels
        swing_high = swing_state.swing_high_price
        swing_low = swing_state.swing_low_price
        
        # Check breaks with buffer
        breaks_high = check_price_high > (swing_high + self.buffer)
        breaks_low = check_price_low < (swing_low - self.buffer)
        
        # Determine BOS vs CHoCH based on current bias
        # CRITICAL: Only fire if swing NOT already crossed!
        if breaks_high and not swing_state.swing_high_crossed:
            if self.int_structure_dir >= 0:  # Bullish or undefined
                # Breaking high in bullish direction = BOS
                self.int_bos_up = True
                self.int_structure_dir = 1
                swing_state.swing_high_crossed = True  # Mark as crossed
                logger.debug(f"[IntStructure] BOS UP at bar {bar.bar_index}, price={check_price_high}, swing_high={swing_high}")
            else:  # Bearish
                # Breaking high in bearish trend = CHoCH
                self.int_choch_up = True
                self.int_structure_dir = 1
                swing_state.swing_high_crossed = True  # Mark as crossed
                logger.debug(f"[IntStructure] CHoCH UP at bar {bar.bar_index}, price={check_price_high}, swing_high={swing_high}")
        
        elif breaks_low and not swing_state.swing_low_crossed:
            if self.int_structure_dir <= 0:  # Bearish or undefined
                # Breaking low in bearish direction = BOS
                self.int_bos_down = True
                self.int_structure_dir = -1
                swing_state.swing_low_crossed = True  # Mark as crossed
                logger.debug(f"[IntStructure] BOS DOWN at bar {bar.bar_index}, price={check_price_low}, swing_low={swing_low}")
            else:  # Bullish
                # Breaking low in bullish trend = CHoCH
                self.int_choch_down = True
                self.int_structure_dir = -1
                swing_state.swing_low_crossed = True  # Mark as crossed
                logger.debug(f"[IntStructure] CHoCH DOWN at bar {bar.bar_index}, price={check_price_low}, swing_low={swing_low}")
    
    def update_external(self, bar: RawBar, swing_state: ExternalSwingState):
        """Update external structure - same logic as internal"""
        # Reset pulses
        self.ext_bos_up = False
        self.ext_bos_down = False
        self.ext_choch_up = False
        self.ext_choch_down = False
        
        if (swing_state.swing_high_price is None or swing_state.swing_high_price == 0 or
            swing_state.swing_low_price is None or swing_state.swing_low_price == 0):
            return
        
        if self.confirm_on_close:
            check_price_high = bar.c
            check_price_low = bar.c
        else:
            check_price_high = bar.h
            check_price_low = bar.l
        
        swing_high = swing_state.swing_high_price
        swing_low = swing_state.swing_low_price
        
        breaks_high = check_price_high > (swing_high + self.buffer)
        breaks_low = check_price_low < (swing_low - self.buffer)
        
        if breaks_high and not swing_state.swing_high_crossed:
            if self.ext_structure_dir >= 0:
                self.ext_bos_up = True
                self.ext_structure_dir = 1
                swing_state.swing_high_crossed = True
                logger.debug(f"[ExtStructure] BOS UP at bar {bar.bar_index}, price={check_price_high}")
            else:
                self.ext_choch_up = True
                self.ext_structure_dir = 1
                swing_state.swing_high_crossed = True
                logger.debug(f"[ExtStructure] CHoCH UP at bar {bar.bar_index}, price={check_price_high}")
        
        elif breaks_low and not swing_state.swing_low_crossed:
            if self.ext_structure_dir <= 0:
                self.ext_bos_down = True
                self.ext_structure_dir = -1
                swing_state.swing_low_crossed = True
                logger.debug(f"[ExtStructure] BOS DOWN at bar {bar.bar_index}, price={check_price_low}")
            else:
                self.ext_choch_down = True
                self.ext_structure_dir = -1
                swing_state.swing_low_crossed = True
                logger.debug(f"[ExtStructure] CHoCH DOWN at bar {bar.bar_index}, price={check_price_low}")
    
    def get_internal_state(self):
        """Get internal structure state"""
        return {
            'structure_dir': self.int_structure_dir,
            'bos_up': self.int_bos_up,
            'bos_down': self.int_bos_down,
            'choch_up': self.int_choch_up,
            'choch_down': self.int_choch_down
        }
    
    def get_external_state(self):
        """Get external structure state"""
        return {
            'structure_dir': self.ext_structure_dir,
            'bos_up': self.ext_bos_up,
            'bos_down': self.ext_bos_down,
            'choch_up': self.ext_choch_up,
            'choch_down': self.ext_choch_down
        }
