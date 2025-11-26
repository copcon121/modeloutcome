"""
SMC Engine - 2-Layer Structure Detection
Orchestrates Internal (wave 5) + External (wave 50) BOS/CHoCH detection
"""

from typing import List, Optional
from dataclasses import dataclass

from layer2_feature_engine.core.schema import RawBar
from layer2_feature_engine.smc.config import SMCConfig, DEFAULT_SMC_CONFIG
from layer2_feature_engine.smc.swing import (
    InternalSwingDetector,
    ExternalSwingDetector,
    InternalSwingState,
    ExternalSwingState
)


@dataclass
class SMCState:
    """
    Complete SMC state for both internal and external layers
    """
    # Internal layer (wave 5)
    int_trend_dir: int = 0
    int_bos_up: bool = False
    int_bos_down: bool = False
    int_choch_up: bool = False
    int_choch_down: bool = False
    int_swing_high_price: float = 0.0
    int_swing_low_price: float = 0.0
    int_swing_high_bars_ago: int = 999
    int_swing_low_bars_ago: int = 999

    # External layer (wave 50)
    ext_trend_dir: int = 0
    ext_bos_up: bool = False
    ext_bos_down: bool = False
    ext_choch_up: bool = False
    ext_choch_down: bool = False
    ext_swing_high_price: float = 0.0
    ext_swing_low_price: float = 0.0
    ext_swing_high_bars_ago: int = 999
    ext_swing_low_bars_ago: int = 999


class SMCEngine:
    """
    Main SMC Engine - processes bars and outputs 2-layer structure

    Usage:
        engine = SMCEngine(config)
        for i, bar in enumerate(bars):
            smc_state = engine.update(bars, i)
            # Use smc_state.int_*, smc_state.ext_* for features
    """

    def __init__(self, config: SMCConfig = None):
        self.config = config or DEFAULT_SMC_CONFIG

        # Swing detectors
        self.int_swing_detector = InternalSwingDetector(self.config)
        self.ext_swing_detector = ExternalSwingDetector(self.config)

        # State
        self.int_state: Optional[InternalSwingState] = None
        self.ext_state: Optional[ExternalSwingState] = None

        # Previous state for BOS/CHoCH detection
        self.prev_int_state: Optional[InternalSwingState] = None
        self.prev_ext_state: Optional[ExternalSwingState] = None

    def update(self, bars: List[RawBar], current_idx: int) -> SMCState:
        """
        Update SMC state at current bar

        Args:
            bars: Full list of bars up to current
            current_idx: Current bar index

        Returns:
            SMCState with int_* and ext_* fields
        """
        # Update internal swings (wave 5)
        self.prev_int_state = self.int_state
        self.int_state = self.int_swing_detector.update(bars, current_idx)

        # Update external swings (wave 50)
        self.prev_ext_state = self.ext_state
        self.ext_state = self.ext_swing_detector.update(bars, current_idx, self.int_state)

        # Detect BOS/CHoCH for both layers
        smc_state = SMCState()

        # Internal layer
        smc_state.int_trend_dir = self.int_state.trend_dir
        smc_state.int_swing_high_price = self.int_state.swing_high_price
        smc_state.int_swing_low_price = self.int_state.swing_low_price
        smc_state.int_swing_high_bars_ago = self.int_state.bars_since_swing_high(current_idx)
        smc_state.int_swing_low_bars_ago = self.int_state.bars_since_swing_low(current_idx)

        # Detect internal BOS/CHoCH
        int_bos, int_choch = self._detect_bos_choch_internal(bars, current_idx)
        smc_state.int_bos_up = int_bos.get('up', False)
        smc_state.int_bos_down = int_bos.get('down', False)
        smc_state.int_choch_up = int_choch.get('up', False)
        smc_state.int_choch_down = int_choch.get('down', False)

        # External layer
        smc_state.ext_trend_dir = self.ext_state.trend_dir
        smc_state.ext_swing_high_price = self.ext_state.swing_high_price
        smc_state.ext_swing_low_price = self.ext_state.swing_low_price
        smc_state.ext_swing_high_bars_ago = self.ext_state.bars_since_swing_high(current_idx)
        smc_state.ext_swing_low_bars_ago = self.ext_state.bars_since_swing_low(current_idx)

        # Detect external BOS/CHoCH
        ext_bos, ext_choch = self._detect_bos_choch_external(bars, current_idx)
        smc_state.ext_bos_up = ext_bos.get('up', False)
        smc_state.ext_bos_down = ext_bos.get('down', False)
        smc_state.ext_choch_up = ext_choch.get('up', False)
        smc_state.ext_choch_down = ext_choch.get('down', False)

        return smc_state

    def _detect_bos_choch_internal(self, bars: List[RawBar], current_idx: int):
        """
        Detect internal BOS/CHoCH (wave 5 level)

        BOS (Break of Structure): Close breaks swing in trend direction
        CHoCH (Change of Character): Close breaks swing against trend (reversal)
        """
        bos = {'up': False, 'down': False}
        choch = {'up': False, 'down': False}

        if not self.int_state or current_idx < 1:
            return bos, choch

        current_bar = bars[current_idx]
        buffer = self.config.ticks_to_price(self.config.bos_close_buffer_ticks)

        # BOS UP: Close breaks above internal swing high (bullish continuation)
        if self.int_state.swing_high_idx >= 0:
            swing_high = self.int_state.swing_high_price
            if current_bar.close > (swing_high + buffer):
                if self.int_state.trend_dir >= 0:
                    bos['up'] = True  # Continuation of uptrend
                else:
                    choch['up'] = True  # Reversal from downtrend

        # BOS DOWN: Close breaks below internal swing low (bearish continuation)
        if self.int_state.swing_low_idx >= 0:
            swing_low = self.int_state.swing_low_price
            if current_bar.close < (swing_low - buffer):
                if self.int_state.trend_dir <= 0:
                    bos['down'] = True  # Continuation of downtrend
                else:
                    choch['down'] = True  # Reversal from uptrend

        return bos, choch

    def _detect_bos_choch_external(self, bars: List[RawBar], current_idx: int):
        """
        Detect external BOS/CHoCH (wave 50 level)

        Same logic as internal, but using external swings
        """
        bos = {'up': False, 'down': False}
        choch = {'up': False, 'down': False}

        if not self.ext_state or current_idx < 1:
            return bos, choch

        current_bar = bars[current_idx]
        buffer = self.config.ticks_to_price(self.config.bos_close_buffer_ticks)

        # BOS UP: Close breaks above external swing high
        if self.ext_state.swing_high_idx >= 0:
            swing_high = self.ext_state.swing_high_price
            if current_bar.close > (swing_high + buffer):
                if self.ext_state.trend_dir >= 0:
                    bos['up'] = True  # Continuation
                else:
                    choch['up'] = True  # Reversal

        # BOS DOWN: Close breaks below external swing low
        if self.ext_state.swing_low_idx >= 0:
            swing_low = self.ext_state.swing_low_price
            if current_bar.close < (swing_low - buffer):
                if self.ext_state.trend_dir <= 0:
                    bos['down'] = True  # Continuation
                else:
                    choch['down'] = True  # Reversal

        return bos, choch
