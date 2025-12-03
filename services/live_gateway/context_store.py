"""
Context Store for Live Gateway
Manages per-symbol rolling context windows and SMC state
"""

import sys
from pathlib import Path
from collections import deque
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import numpy as np

# Add project root to path
ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.layer2_feature_engine_v2.context_manager import SMCContextManager
from src.layer2_feature_engine_v2.config import GC_M1_SMC_CONFIG
from src.layer2_feature_engine_v2.schema import RawBar, FeatureBar


# ==============================================================================
# CONFIGURATION
# ==============================================================================

ASM_SEQ_LEN = 60  # Context window for ASM model
ASM_FEATURE_DIM = 100  # ASM v1.0 uses 100 features
HIGH_VOL_LOOKBACK = 100  # Bars for high vol regime detection

# Feature columns to exclude (metadata only, NOT OHLC!)
EXCLUDE_COLS = [
    "timestamp", "bar_index", "_source_file",
]

# Features to skip for ASM v1.0 (no Weekly VA)
SKIP_PREFIXES = ["weekly_", "daily_va_"]

# FIXED FEATURE ORDER - must match training data exactly!
# From output/asm_dataset_v1/asm_dataset_v1_stats.json
ASM_FEATURE_COLS = [
    "close", "high_low_range", "body", "upper_wick", "lower_wick",
    "close_return", "volume", "volume_change", "delta", "delta_over_volume",
    "buy_volume", "sell_volume", "buy_sell_ratio", "tick_speed", "aggr_buy_speed",
    "aggr_sell_speed", "price_speed", "int_trend_dir", "int_bos_up", "int_bos_down",
    "int_choch_up", "int_choch_down", "int_swing_high_distance", "int_swing_low_distance",
    "bars_since_int_swing_high", "bars_since_int_swing_low", "swept_prev_int_high",
    "swept_prev_int_low", "int_bias_bullish", "ext_trend_dir", "ext_bos_up", "ext_bos_down",
    "ext_choch_up", "ext_choch_down", "ext_swing_high_distance", "ext_swing_low_distance",
    "bars_since_ext_swing_high", "bars_since_ext_swing_low", "swept_prev_ext_high",
    "swept_prev_ext_low", "ext_bias_bullish", "in_bull_fvg", "in_bear_fvg", "near_bull_fvg",
    "near_bear_fvg", "int_in_bull_ob", "int_in_bear_ob", "int_near_bull_ob", "int_near_bear_ob",
    "ext_in_bull_ob", "ext_in_bear_ob", "ext_near_bull_ob", "ext_near_bear_ob",
    "dist_to_nearest_fvg", "dist_to_nearest_ob", "nearest_fvg_size", "vp_poc_price",
    "vp_val_price", "vp_vah_price", "vp_in_value_area", "vp_above_value_area",
    "vp_below_value_area", "vp_dist_to_poc", "vp_dist_to_vah", "vp_dist_to_val",
    "impulse_strength", "pullback_strength", "cum_delta_5", "cum_delta_10", "cum_delta_20",
    "vwap_daily", "dist_to_vwap", "m5_trend_up", "m5_trend_down", "m5_premium", "m5_discount",
    "dist_to_m5_swing_high", "dist_to_m5_swing_low", "near_m5_fvg", "h1_trend_up",
    "h1_trend_down", "h1_premium", "h1_discount", "dist_to_h1_swing_high", "dist_to_h1_swing_low",
    "near_h1_fvg", "open", "high", "low", "m5_swing_phase", "m5_price_pos_in_range",
    "m5_bos_up_count_recent", "m5_bos_down_count_recent", "m5_ob_imbalance", "h1_swing_phase",
    "h1_price_pos_in_range", "h1_bos_up_count_recent", "h1_bos_down_count_recent",
    "h1_ob_imbalance", "global_bar_index",
]


# ==============================================================================
# CONTEXT STORE
# ==============================================================================


@dataclass
class SymbolContext:
    """Per-symbol context state"""
    symbol: str
    timeframe: str
    
    # SMC Context Manager
    smc_manager: SMCContextManager = field(default=None)
    
    # Rolling feature buffer (last N bars)
    feature_buffer: deque = field(default_factory=lambda: deque(maxlen=ASM_SEQ_LEN + HIGH_VOL_LOOKBACK))
    
    # Raw bar buffer for high vol calculation
    raw_buffer: deque = field(default_factory=lambda: deque(maxlen=HIGH_VOL_LOOKBACK))
    
    # Last feature bar (for S4 rule checks)
    last_feature_bar: Optional[FeatureBar] = None
    
    # Bar count
    bar_count: int = 0
    
    def __post_init__(self):
        if self.smc_manager is None:
            self.smc_manager = SMCContextManager(GC_M1_SMC_CONFIG, tick_size=0.1)


class ContextStore:
    """
    In-memory store for per-symbol contexts.
    Thread-safe for single-threaded FastAPI (async).
    """
    
    def __init__(self):
        self.contexts: Dict[Tuple[str, str], SymbolContext] = {}
        self._feature_cols: Optional[List[str]] = None
    
    def get_or_create(self, symbol: str, timeframe: str) -> SymbolContext:
        """Get existing context or create new one"""
        key = (symbol, timeframe)
        if key not in self.contexts:
            self.contexts[key] = SymbolContext(symbol=symbol, timeframe=timeframe)
        return self.contexts[key]
    
    def update(self, symbol: str, timeframe: str, raw_bar: RawBar) -> Tuple[FeatureBar, Dict[str, float]]:
        """
        Update context with new bar.
        
        Returns:
            (feature_bar, feature_dict) - the processed feature bar and its dict form
        """
        ctx = self.get_or_create(symbol, timeframe)
        
        # Update SMC and get feature bar
        feature_bar = ctx.smc_manager.update(raw_bar)
        feature_dict = feature_bar.to_dict()
        
        # Add OHLC and global_bar_index to match training data
        feature_dict["open"] = raw_bar.o
        feature_dict["high"] = raw_bar.h
        feature_dict["low"] = raw_bar.l
        feature_dict["global_bar_index"] = ctx.bar_count
        
        # Add to buffers
        ctx.feature_buffer.append(feature_dict)
        ctx.raw_buffer.append({
            "high_low_range": raw_bar.h - raw_bar.l,
            "volume": raw_bar.volume,
            "close": raw_bar.c,
            "low": raw_bar.l,
            "high": raw_bar.h,
        })
        
        ctx.last_feature_bar = feature_bar
        ctx.bar_count += 1
        
        return feature_bar, feature_dict
    
    def get_asm_context(self, symbol: str, timeframe: str) -> Optional[np.ndarray]:
        """
        Get ASM context window (60 x 100) for inference.
        
        Uses FIXED feature order from ASM_FEATURE_COLS to match training data exactly.
        
        Returns:
            np.ndarray of shape (60, 100) or None if not enough data
        """
        ctx = self.get_or_create(symbol, timeframe)
        
        if len(ctx.feature_buffer) < ASM_SEQ_LEN:
            return None
        
        # Build context array from last 60 bars using FIXED feature order
        recent_bars = list(ctx.feature_buffer)[-ASM_SEQ_LEN:]
        
        context = []
        for bar_dict in recent_bars:
            # Use fixed feature order from training
            row = [float(bar_dict.get(col, 0.0)) for col in ASM_FEATURE_COLS]
            context.append(row)
        
        context = np.array(context, dtype=np.float32)
        
        # Handle NaN values - replace with 0
        context = np.nan_to_num(context, nan=0.0)
        
        return context
    
    def is_high_vol_regime(self, symbol: str, timeframe: str) -> bool:
        """Check if current bar is in high volatility regime"""
        ctx = self.get_or_create(symbol, timeframe)
        
        if len(ctx.raw_buffer) < HIGH_VOL_LOOKBACK:
            return False
        
        recent = list(ctx.raw_buffer)[-HIGH_VOL_LOOKBACK:]
        ranges = [b["high_low_range"] for b in recent]
        volumes = [b["volume"] for b in recent]
        
        current = ctx.raw_buffer[-1]
        range_q66 = np.quantile(ranges, 0.66)
        avg_vol = np.mean(volumes)
        
        high_range = current["high_low_range"] > range_q66
        high_volume = current["volume"] > avg_vol * 2.0
        
        return high_range or high_volume
    
    def get_context_count(self) -> int:
        """Get number of active contexts"""
        return len(self.contexts)


# Global store instance
context_store = ContextStore()
