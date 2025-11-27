"""
Phase 2 Feature Engine V2 - Configuration
SMC parameters matching NinjaTrader indicator
"""

from dataclasses import dataclass


@dataclass
class SMCConfig:
    """
    SMC Configuration matching NinjaTrader SMC_Structure indicator
    
    Key parameters:
    - swing_int_window: Internal swing window (IntWindow in Ninja) = wave 5
    - swing_ext_window: External swing window (WinExt in Ninja) = wave 50
    - Use window-based MAX/MIN approach (NOT Gann fractals)
    """
    
    # ===== SWING DETECTION =====
    # Window sizes for rolling MAX/MIN approach
    swing_int_window: int = 5      # Internal swing (wave 5)
    swing_ext_window: int = 50     # External swing (wave 50)
    
    # ===== BOS/CHoCH BUFFER =====
    # Buffer in ticks when checking close break (matches BosBufferTicks)
    bos_close_buffer_ticks: int = 1
    
    # Confirm on close (not wick) - matches ConfirmOnClose=True
    confirm_on_close: bool = True
    
    # ===== MINIMUM MOVES =====
    # Minimum move thresholds (in ticks)
    min_int_move_ticks: int = 5     # Internal swing minimum move
    min_ext_move_ticks: int = 50    # External swing minimum move
    
    # ===== FVG SETTINGS =====
    # Minimum gap size for FVG (in ticks)
    fvg_min_gap_ticks: int = 2
    
    # Auto threshold using ATR (matches FVGAutoThreshold)
    fvg_auto_threshold: bool = True
    fvg_threshold_multiplier: float = 2.0  # ATR multiplier
    
    # ===== ORDER BLOCK SETTINGS =====
    # Lookback bars for finding OB source candle (matches OBLookbackBars)
    ob_lookback_bars: int = 30
    
    # Buffer for OB zone (in ticks)
    ob_buffer_ticks: int = 1
    
    # Use full candle or body only (matches OBUseFullCandle)
    ob_use_full_candle: bool = True
    
    # ===== VOLUME PROFILE =====
    # Rolling window for VP calculation
    vp_window_bars: int = 100
    
    # Value area percentage (70% of volume around POC)
    vp_value_area_pct: float = 0.70
    
    # ===== LIQUIDITY SWEEP =====
    # Sweep confirmation: wick through + close back inside
    sweep_wick_buffer_ticks: int = 1


# ===== INSTRUMENT PROFILES =====

GC_M1_SMC_CONFIG = SMCConfig(
    # GC (Gold) specific settings
    swing_int_window=5,
    swing_ext_window=50,
    bos_close_buffer_ticks=1,
    min_int_move_ticks=5,
    min_ext_move_ticks=50,
    fvg_min_gap_ticks=2,
    ob_lookback_bars=30,
    vp_window_bars=100,
)


NQ_M1_SMC_CONFIG = SMCConfig(
    # NQ (Nasdaq) specific settings (example for future use)
    swing_int_window=5,
    swing_ext_window=50,
    bos_close_buffer_ticks=2,  # NQ may need larger buffer
    min_int_move_ticks=10,
    min_ext_move_ticks=100,
    fvg_min_gap_ticks=5,
    ob_lookback_bars=30,
    vp_window_bars=100,
)


# Default config
DEFAULT_SMC_CONFIG = GC_M1_SMC_CONFIG
