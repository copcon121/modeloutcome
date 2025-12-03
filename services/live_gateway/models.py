"""
Pydantic models for Live Gateway API
Matches SMC_Exporter_Pro_v3 JSON schema from NinjaTrader
"""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel


# ==============================================================================
# REQUEST MODELS (from NinjaTrader SMC_Exporter_Pro_v3)
# ==============================================================================


class BarData(BaseModel):
    """Bar OHLCV + orderflow data"""
    o: float
    h: float
    l: float
    c: float
    volume: float
    delta: float
    buy_volume: float
    sell_volume: float
    best_bid: float
    best_ask: float
    vwap_daily: float = 0.0


class TickFeatures(BaseModel):
    """Tick-level aggregated features"""
    tick_speed: float
    aggr_buy_speed: float
    aggr_sell_speed: float
    price_speed: float


class LiveBarEvent(BaseModel):
    """
    Complete bar event from NinjaTrader exporter.
    Matches SMC_Exporter_Pro_v3.BuildJsonForClosedBar output.
    """
    symbol: str
    timeframe: str  # "M1"
    timestamp: datetime
    bar_index: int
    bar: BarData
    tick_features: TickFeatures


# ==============================================================================
# RESPONSE MODELS
# ==============================================================================


class LiveSignalResponse(BaseModel):
    """
    Response to NinjaTrader with trade signal info.
    """
    has_signal: bool
    module: Optional[str] = None  # e.g. "S4_LDN_ASM_LowShift_0.2_v1.1"
    symbol: Optional[str] = None
    timeframe: Optional[str] = None
    side: Optional[str] = None  # "long" / "short"
    entry: Optional[float] = None
    sl: Optional[float] = None
    tp: Optional[float] = None
    rr: Optional[float] = None
    session: Optional[str] = None
    
    # ASM scores
    p_shift: Optional[float] = None
    p_up: Optional[float] = None
    p_down: Optional[float] = None
    p_neutral: Optional[float] = None
    
    # Debug info
    s4_setup: Optional[bool] = None
    high_vol: Optional[bool] = None
    in_fvg: Optional[bool] = None
    
    version: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    contexts_active: int
