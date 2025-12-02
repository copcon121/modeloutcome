"""
Risk Configuration
Defines configurable risk parameters for SL/TP calculation
"""

import os
from dataclasses import dataclass


@dataclass
class RiskConfig:
    """
    Risk management configuration

    Attributes:
        atr_mult_sl: ATR multiplier for stop loss (e.g., 0.5 means SL = 0.5 x ATR)
        rr_target: Risk-reward ratio for take profit (e.g., 2.0 means TP = 2R)
        buffer_ticks: Additional buffer in ticks for SL (default: 2)
        tick_size: Instrument tick size (default: 0.1 for GC)
    """
    atr_mult_sl: float = 0.5
    rr_target: float = 2.0
    buffer_ticks: int = 2
    tick_size: float = 0.1
    
    def __str__(self):
        return f"RiskConfig(atr={self.atr_mult_sl}, rr={self.rr_target})"
    
    def to_dict(self):
        return {
            'atr_mult_sl': self.atr_mult_sl,
            'rr_target': self.rr_target,
            'buffer_ticks': self.buffer_ticks,
            'tick_size': self.tick_size
        }


# Default configurations (baseline used for best prior results)
BASELINE_CONFIG = RiskConfig(atr_mult_sl=0.5, rr_target=2.0)
DEFAULT_CONFIG = BASELINE_CONFIG  # Alias for backward compatibility
LEGACY_CONFIG = RiskConfig(atr_mult_sl=1.5, rr_target=3.0)  # Looser stops experiment

# Lower-DD preset: slightly wider stop, lower RR to reduce variance
LOW_DD_CONFIG = RiskConfig(atr_mult_sl=0.7, rr_target=1.8)
# Conservative preset: even wider stop and lower RR
CONSERVATIVE_CONFIG = RiskConfig(atr_mult_sl=0.8, rr_target=1.5)

PRESETS = {
    "BASELINE": BASELINE_CONFIG,
    "LOW_DD": LOW_DD_CONFIG,
    "CONSERVATIVE": CONSERVATIVE_CONFIG,
    "LEGACY": LEGACY_CONFIG,
}


def load_risk_config_from_env() -> RiskConfig:
    """
    Load risk configuration with optional env overrides.

    Environment variables:
        RISK_PRESET: one of PRESETS keys (BASELINE/LOW_DD/CONSERVATIVE/LEGACY)
        RISK_ATR_MULT_SL: float override
        RISK_RR_TARGET: float override
        RISK_BUFFER_TICKS: int override
        RISK_TICK_SIZE: float override
    """
    preset_name = os.getenv("RISK_PRESET", "BASELINE").upper()
    base_cfg = PRESETS.get(preset_name, BASELINE_CONFIG)

    # Copy to avoid mutating globals
    cfg = RiskConfig(
        atr_mult_sl=base_cfg.atr_mult_sl,
        rr_target=base_cfg.rr_target,
        buffer_ticks=base_cfg.buffer_ticks,
        tick_size=base_cfg.tick_size,
    )

    def _maybe_float(name, current):
        val = os.getenv(name)
        return current if val is None else float(val)

    def _maybe_int(name, current):
        val = os.getenv(name)
        return current if val is None else int(val)

    cfg.atr_mult_sl = _maybe_float("RISK_ATR_MULT_SL", cfg.atr_mult_sl)
    cfg.rr_target = _maybe_float("RISK_RR_TARGET", cfg.rr_target)
    cfg.buffer_ticks = _maybe_int("RISK_BUFFER_TICKS", cfg.buffer_ticks)
    cfg.tick_size = _maybe_float("RISK_TICK_SIZE", cfg.tick_size)
    return cfg
