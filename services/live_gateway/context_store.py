"""
Context Store for Live Gateway
Manages per-symbol rolling context windows and SMC state

v1.1: Uses unified ASM_FEATURE_COLS from src/asm_feature_spec.py
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
from src.asm_feature_spec import ASM_FEATURE_COLS, ASM_SEQ_LEN, ASM_FEATURE_DIM


# ==============================================================================
# CONFIGURATION
# ==============================================================================

HIGH_VOL_LOOKBACK = 100  # Bars for high vol regime detection

# Note: ASM_SEQ_LEN, ASM_FEATURE_DIM, ASM_FEATURE_COLS imported from src.asm_feature_spec


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
