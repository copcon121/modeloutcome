"""
SMC Configuration
Defines swing detection parameters for Internal (wave 5) and External (wave 50) structure
"""

from dataclasses import dataclass


@dataclass
class SMCConfig:
    """
    Configuration for Smart Money Concepts (SMC) multi-layer swing detection

    SMC uses 2-layer swing structure:
    - Internal (int): Fine-grained swings (wave 5) - smaller timeframe structure
    - External (ext): Macro swings (wave 50) - larger timeframe structure

    This follows MGannSwing methodology where:
    - wave 5 = internal swing (minor structure)
    - wave 50 = external swing (major structure)
    """

    # ==================== Internal Swing (wave 5) ====================
    swing_int_strength: int = 5
    """Internal swing strength - minimum ticks for wave 5 confirmation"""

    # ==================== External Swing (wave 50) ====================
    swing_ext_strength: int = 50
    """External swing strength - minimum ticks for wave 50 confirmation"""

    # ==================== Fractal Pivot Detection ====================
    fractal_left: int = 2
    """Left lookback for fractal pivot detection (Gann 2-bar style)"""

    fractal_right: int = 2
    """Right lookback for fractal pivot detection"""

    # ==================== Movement Thresholds ====================
    min_int_move_ticks: int = 5
    """Minimum price move in ticks to confirm internal swing (wave 5)"""

    min_ext_move_ticks: int = 50
    """Minimum price move in ticks to confirm external swing (wave 50)"""

    tick_size: float = 0.1
    """Tick size for the instrument (GC = 0.1)"""

    # ==================== BOS/CHoCH Filters ====================
    bos_close_buffer_ticks: int = 1
    """Close must exceed swing by this many ticks to confirm BOS"""

    choch_min_swing_ratio: float = 0.5
    """
    Minimum swing ratio for CHoCH confirmation
    New swing must be at least 50% of previous swing to confirm character change
    """

    # ==================== Displacement Filters (Optional) ====================
    min_displacement_body_ratio: float = 0.0
    """Minimum body/range ratio for displacement candle (0 = disabled)"""

    min_displacement_range_ratio: float = 0.0
    """Minimum range relative to ATR for displacement (0 = disabled)"""

    # ==================== Trend Confirmation ====================
    min_bars_between_swings: int = 3
    """Minimum bars between consecutive swings of same type"""

    def ticks_to_price(self, ticks: int) -> float:
        """Convert ticks to price distance"""
        return ticks * self.tick_size

    def price_to_ticks(self, price_distance: float) -> int:
        """Convert price distance to ticks"""
        return int(abs(price_distance) / self.tick_size)


# ==================== Preset Configurations ====================

GC_M1_SMC_CONFIG = SMCConfig(
    # Gold Futures M1 settings
    swing_int_strength=5,
    swing_ext_strength=50,
    fractal_left=2,
    fractal_right=2,
    min_int_move_ticks=5,
    min_ext_move_ticks=50,
    tick_size=0.1,  # GC tick = $0.10
    bos_close_buffer_ticks=1,
    choch_min_swing_ratio=0.5,
    min_bars_between_swings=3,
)

ES_M1_SMC_CONFIG = SMCConfig(
    # E-mini S&P 500 M1 settings
    swing_int_strength=4,
    swing_ext_strength=40,
    fractal_left=2,
    fractal_right=2,
    min_int_move_ticks=4,
    min_ext_move_ticks=40,
    tick_size=0.25,  # ES tick = $0.25
    bos_close_buffer_ticks=1,
    choch_min_swing_ratio=0.5,
    min_bars_between_swings=3,
)

NQ_M1_SMC_CONFIG = SMCConfig(
    # E-mini NASDAQ M1 settings
    swing_int_strength=10,
    swing_ext_strength=100,
    fractal_left=2,
    fractal_right=2,
    min_int_move_ticks=10,
    min_ext_move_ticks=100,
    tick_size=0.25,  # NQ tick = $0.25
    bos_close_buffer_ticks=2,
    choch_min_swing_ratio=0.5,
    min_bars_between_swings=3,
)


# ==================== Default Config ====================
DEFAULT_SMC_CONFIG = GC_M1_SMC_CONFIG
