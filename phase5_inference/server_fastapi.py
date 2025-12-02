"""
FastAPI Server for Multi-Model Quality Prediction

Supports both tabular (legacy) and sequence (primary) models.
Shadow trading logging for production readiness.

Usage:
    uvicorn phase5_inference.server_fastapi:app --reload --port 8000
"""

import sys
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Any
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import numpy as np

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from phase5_inference.inference import create_router

# Create app
app = FastAPI(
    title="Quality Model API v3 - Multi-Model",
    description="Multi-model SMC Quality Filter (Tabular + Sequence GRU)",
    version="3.0.0"
)

# Global router and shadow log
router = None
SHADOW_LOG_PATH = None

@app.on_event("startup")
async def startup_event():
    global router, SHADOW_LOG_PATH
    
    # Initialize router
    router = create_router(device='cpu')
    
    # Shadow log path
    root = Path(__file__).parent.parent
    SHADOW_LOG_PATH = root / "output/phase5_quality/shadow_trading_log.jsonl"
    SHADOW_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"[FastAPI] Multi-Model Quality Router initialized")
    print(f"[FastAPI] Global default: {router.global_default}")
    print(f"[FastAPI] Shadow log: {SHADOW_LOG_PATH}")


# Request/Response models
class PredictRequest(BaseModel):
    X: List[List[float]] = Field(..., description="Context window [60, 66]")
    side: int = Field(..., description="+1 for long, -1 for short")
    model_type: Optional[str] = Field(None, description="'tabular_v1' (legacy) or 'seq_v1' (primary)")
    mode: Optional[str] = Field(None, description="Mode name (e.g., 'balanced', 'seq_conservative')")
    threshold: Optional[float] = Field(None, description="Custom threshold (overrides mode)", ge=0.0, le=1.0)
    shadow_only: Optional[bool] = Field(False, description="If true, prediction is for shadow trading only (no real execution)")
    
    # Optional metadata for shadow logging
    meta: Optional[Dict[str, Any]] = Field(None, description="Metadata (symbol, timeframe, event_time, etc.)")


class PredictResponse(BaseModel):
    p_keep: float
    keep: bool
    model_type: str
    mode: Optional[str]
    threshold: float
    side: int
    shadow_only: bool
    timestamp: str


# Endpoints
@app.get("/")
async def root():
    return {
        "service": "Quality Model API v3",
        "version": "3.0.0",
        "models": {
            "tabular_v1": {
                "status": "legacy",
                "description": "MLP on flattened features (Phase 4)",
                "params": "1M",
                "modes": ["balanced", "conservative"]
            },
            "seq_v1": {
                "status": "primary",
                "description": "GRU on sequences (Phase 6)",
                "params": "92K",
                "modes": ["seq_balanced", "seq_conservative"],
                "recommended": True
            }
        },
        "global_default": router.global_default if router else "seq_v1",
        "endpoints": {
            "/predict_quality": "POST - Predict trade quality",
            "/health": "GET - Health check",
            "/models": "GET - Model details",
            "/docs": "GET - Swagger UI"
        }
    }


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "router_loaded": router is not None,
        "global_default_model": router.global_default if router else None,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/models")
async def get_models():
    """Get model configuration details"""
    if router is None:
        raise HTTPException(status_code=503, detail="Router not initialized")
    
    return {
        "global_default": router.global_default,
        "models": router.config['models'],
        "comparison": router.config.get('comparison_summary', {}),
        "recommendations": router.config.get('deployment_recommendation', {})
    }


@app.post("/predict_quality", response_model=PredictResponse)
async def predict_quality(request: PredictRequest):
    """
    Predict trade quality with multi-model support
    
    Defaults:
    - model_type: seq_v1 (primary)
    - mode: seq_balanced
    - threshold: from config (0.5 or 0.8)
    """
    if router is None:
        raise HTTPException(status_code=503, detail="Router not initialized")
    
    try:
        # Convert to numpy
        X_np = np.array(request.X, dtype=np.float32)
        
        # Validate shape
        if X_np.shape != (60, 66):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid X shape: expected (60, 66), got {X_np.shape}"
            )
        
        # Validate side
        if request.side not in [1, -1]:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid side: must be +1 or -1, got {request.side}"
            )
        
        # Route prediction
        result = router.predict(
            X_np,
            request.side,
            model_type=request.model_type,
            mode=request.mode,
            threshold=request.threshold
        )
        
        # Timestamp
        timestamp = datetime.now().isoformat()
        
        # Shadow logging
        log_entry = {
            "timestamp_server": timestamp,
            "model_type": result['model_type'],
            "mode": result['mode'],
            "threshold": result['threshold'],
            "shadow_only": request.shadow_only or False,
            "p_keep": result['p_keep'],
            "keep": result['keep'],
            "side": result['side'],
            "meta": request.meta or {}
        }
        
        # Append to shadow log
        with open(SHADOW_LOG_PATH, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
        
        # Build response
        return PredictResponse(
            p_keep=result['p_keep'],
            keep=result['keep'],
            model_type=result['model_type'],
            mode=result['mode'],
            threshold=result['threshold'],
            side=result['side'],
            shadow_only=request.shadow_only or False,
            timestamp=timestamp
        )
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    print("Starting Multi-Model Quality API v3...")
    print("Models: tabular_v1 (legacy), seq_v1 (primary)")
    print("Default: seq_v1 / seq_balanced")
    print("Access http://localhost:8000/docs for Swagger UI")
    uvicorn.run(app, host="0.0.0.0", port=8000)
