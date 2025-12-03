"""
Live Gateway FastAPI Application
S4_LDN_ASM_LowShift_0.2_v1.1 Strategy Service

Endpoints:
- GET /health - Health check
- POST /live_bar - Process bar and return signal
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Add project root to path
ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.layer2_feature_engine_v2.schema import RawBar

from .models import (
    LiveBarEvent,
    LiveSignalResponse,
    HealthResponse,
)
from .context_store import context_store
from .s4_engine import check_s4_setup, detect_session, S4_CONFIG
from .asm_inference import asm_model


# ==============================================================================
# CONFIGURATION
# ==============================================================================

STRATEGY_VERSION = "S4_LDN_ASM_LowShift_0.2_v1.1"
P_SHIFT_THRESHOLD = 0.2  # ASM LowShift filter threshold

# Logging
LOG_DIR = ROOT / "logs"
LOG_FILE = LOG_DIR / "live_signals_s4_asm_v1.jsonl"


# ==============================================================================
# FASTAPI APP
# ==============================================================================

app = FastAPI(
    title="Live Gateway - S4_LDN_ASM_LowShift",
    description="Real-time signal generation for S4 HighVol FVG London + ASM LowShift filter",
    version="1.1.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==============================================================================
# STARTUP / SHUTDOWN
# ==============================================================================

@app.on_event("startup")
async def startup_event():
    """Load ASM model on startup"""
    print("=" * 60)
    print(f"Starting Live Gateway: {STRATEGY_VERSION}")
    print("=" * 60)
    
    # Create log directory
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load ASM model
    success = asm_model.load()
    if not success:
        print("WARNING: ASM model not loaded. Signals will use default probabilities.")
    
    print(f"Log file: {LOG_FILE}")
    print("Ready to receive bars!")


# ==============================================================================
# ENDPOINTS
# ==============================================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="ok",
        model_loaded=asm_model.loaded,
        contexts_active=context_store.get_context_count(),
    )


@app.post("/live_bar", response_model=LiveSignalResponse)
async def process_live_bar(event: LiveBarEvent):
    """
    Process a live bar and return trade signal.
    
    Flow:
    1. Convert to RawBar and update SMC context
    2. Check S4 setup (HighVol + FVG + London)
    3. If S4 setup, run ASM inference
    4. Apply LowShift filter (p_shift <= 0.2)
    5. Return signal response
    """
    try:
        # 1. Convert to RawBar
        raw_bar = RawBar(
            symbol=event.symbol,
            timeframe=event.timeframe,
            timestamp=event.timestamp,
            bar_index=event.bar_index,
            o=event.bar.o,
            h=event.bar.h,
            l=event.bar.l,
            c=event.bar.c,
            volume=event.bar.volume,
            delta=event.bar.delta,
            buy_volume=event.bar.buy_volume,
            sell_volume=event.bar.sell_volume,
            best_bid=event.bar.best_bid,
            best_ask=event.bar.best_ask,
            tick_speed=event.tick_features.tick_speed,
            aggr_buy_speed=event.tick_features.aggr_buy_speed,
            aggr_sell_speed=event.tick_features.aggr_sell_speed,
            price_speed=event.tick_features.price_speed,
            vwap_daily=event.bar.vwap_daily,
        )
        
        # 2. Update context and get features
        feature_bar, feature_dict = context_store.update(
            event.symbol, event.timeframe, raw_bar
        )
        
        # 3. Check high vol regime
        is_high_vol = context_store.is_high_vol_regime(event.symbol, event.timeframe)
        
        # 4. Check S4 setup
        hour = event.timestamp.hour
        s4_setup = check_s4_setup(
            feature_bar, feature_dict, is_high_vol, hour,
            session_filter=S4_CONFIG["session"]
        )
        
        # Default response (no signal)
        response = LiveSignalResponse(
            has_signal=False,
            module=STRATEGY_VERSION,
            symbol=event.symbol,
            timeframe=event.timeframe,
            session=s4_setup.session,
            s4_setup=s4_setup.is_valid,
            high_vol=is_high_vol,
            in_fvg=s4_setup.in_fvg,
            version=STRATEGY_VERSION,
        )
        
        # 5. If S4 setup, run ASM inference
        asm_probs = None
        if s4_setup.is_valid:
            # Get ASM context
            asm_context = context_store.get_asm_context(event.symbol, event.timeframe)
            
            if asm_context is not None:
                # Run ASM inference
                asm_probs = asm_model.predict_proba(asm_context)
                
                response.p_up = asm_probs["p_up"]
                response.p_down = asm_probs["p_down"]
                response.p_neutral = asm_probs["p_neutral"]
                response.p_shift = asm_probs["p_shift"]
                
                # 6. Apply LowShift filter
                if asm_probs["p_shift"] <= P_SHIFT_THRESHOLD:
                    response.has_signal = True
                    response.side = "long" if s4_setup.side == 1 else "short"
                    response.entry = s4_setup.entry_price
                    response.sl = s4_setup.sl_price
                    response.tp = s4_setup.tp_price
                    response.rr = S4_CONFIG["rr_target"]
        
        # 7. Log signal (always log when S4 setup is valid)
        if s4_setup.is_valid:
            log_entry = {
                "timestamp": event.timestamp.isoformat(),
                "symbol": event.symbol,
                "timeframe": event.timeframe,
                "bar_index": event.bar_index,
                "session": s4_setup.session,
                "s4_setup": s4_setup.is_valid,
                "side": "long" if s4_setup.side == 1 else "short",
                "entry": s4_setup.entry_price,
                "sl": s4_setup.sl_price,
                "tp": s4_setup.tp_price,
                "high_vol": is_high_vol,
                "in_fvg": s4_setup.in_fvg,
                "ext_trend": s4_setup.ext_trend,
                "p_up": asm_probs["p_up"] if asm_probs else None,
                "p_down": asm_probs["p_down"] if asm_probs else None,
                "p_neutral": asm_probs["p_neutral"] if asm_probs else None,
                "p_shift": asm_probs["p_shift"] if asm_probs else None,
                "filter_pass": response.has_signal,
                "version": STRATEGY_VERSION,
            }
            
            with open(LOG_FILE, "a") as f:
                f.write(json.dumps(log_entry) + "\n")
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
async def get_stats():
    """Get gateway statistics"""
    return {
        "version": STRATEGY_VERSION,
        "model_loaded": asm_model.loaded,
        "contexts_active": context_store.get_context_count(),
        "p_shift_threshold": P_SHIFT_THRESHOLD,
        "session_filter": S4_CONFIG["session"],
        "rr_target": S4_CONFIG["rr_target"],
    }


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
