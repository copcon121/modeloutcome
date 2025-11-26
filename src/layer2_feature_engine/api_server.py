"""
Feature Engine API Server
Receives raw bars from NinjaTrader and forwards predictions from Model Server
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
from datetime import datetime
import httpx
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from layer2_feature_engine.core.schema import RawBar
from layer2_feature_engine.core.context_manager import ContextManager
from layer2_feature_engine.utils.logging_utils import setup_logger

logger = setup_logger('feature_engine_api', log_file='logs/feature_engine_api.log')

# Initialize FastAPI app
app = FastAPI(title="Feature Engine API", version="1.0.0")

# Global context manager
context_manager = ContextManager(context_len=60, max_history=100)

# Model server URL
MODEL_SERVER_URL = "http://localhost:5002/infer"


class BarData(BaseModel):
    """Single bar data"""
    ts: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    delta: Optional[float] = 0.0
    buy_volume: Optional[float] = 0.0
    sell_volume: Optional[float] = 0.0


class RawDataRequest(BaseModel):
    """Request from NinjaTrader"""
    symbol: str
    timeframe: str
    timestamp: str
    bars: List[BarData]


class PredictionResponse(BaseModel):
    """Response with prediction"""
    symbol: str
    timestamp: str
    prob_long: float
    prob_short: float
    prob_skip: float
    recommendation: str  # "LONG", "SHORT", or "SKIP"


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Feature Engine API Server",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "/health": "Health check",
            "/raw": "POST - Receive raw bars from NinjaTrader",
            "/stats": "GET - Context manager statistics"
        }
    }


@app.get("/health")
async def health_check():
    """Health check"""
    return {
        "status": "ok",
        "context_bars": len(context_manager.bars),
        "model_server": MODEL_SERVER_URL
    }


@app.get("/stats")
async def get_stats():
    """Get context manager statistics"""
    return {
        "total_bars": len(context_manager.bars),
        "context_len": context_manager.context_len,
        "max_history": context_manager.max_history,
        "normalizer_fitted": context_manager.normalizer.fitted if context_manager.normalizer else False
    }


@app.post("/raw", response_model=PredictionResponse)
async def receive_raw_data(request: RawDataRequest):
    """
    Receive raw bars from NinjaTrader, build features, and get prediction

    Args:
        request: RawDataRequest with bar data

    Returns:
        PredictionResponse with model predictions
    """
    try:
        logger.info(f"Received {len(request.bars)} bars for {request.symbol}")

        # Convert to RawBar objects
        raw_bars = []
        for bar_data in request.bars:
            bar = RawBar(
                ts=datetime.fromisoformat(bar_data.ts),
                open=bar_data.open,
                high=bar_data.high,
                low=bar_data.low,
                close=bar_data.close,
                volume=bar_data.volume,
                delta=bar_data.delta,
                buy_volume=bar_data.buy_volume,
                sell_volume=bar_data.sell_volume
            )
            raw_bars.append(bar)

        # Add to context manager (batch update)
        context_manager.add_bars_batch(raw_bars)

        # Build features for latest context
        feature_bars = context_manager.get_latest_context()

        if len(feature_bars) < context_manager.context_len:
            raise HTTPException(
                status_code=400,
                detail=f"Not enough bars: {len(feature_bars)} < {context_manager.context_len}"
            )

        # Extract feature names (from first bar)
        feature_names = list(feature_bars[0].features.keys())

        # Convert to feature matrix
        feature_matrix = []
        for fb in feature_bars:
            feature_matrix.append(fb.to_list(feature_names))

        logger.info(f"Built feature matrix: {len(feature_matrix)} x {len(feature_matrix[0])}")

        # Call model server
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                MODEL_SERVER_URL,
                json={"features": feature_matrix}
            )

            if response.status_code != 200:
                raise HTTPException(
                    status_code=502,
                    detail=f"Model server error: {response.status_code}"
                )

            prediction = response.json()

        # Determine recommendation
        probs = [
            prediction['prob_long'],
            prediction['prob_short'],
            prediction['prob_skip']
        ]
        max_prob = max(probs)
        max_idx = probs.index(max_prob)

        recommendations = ["LONG", "SHORT", "SKIP"]
        recommendation = recommendations[max_idx]

        result = PredictionResponse(
            symbol=request.symbol,
            timestamp=request.timestamp,
            prob_long=prediction['prob_long'],
            prob_short=prediction['prob_short'],
            prob_skip=prediction['prob_skip'],
            recommendation=recommendation
        )

        logger.info(f"Prediction: {recommendation} (prob={max_prob:.3f})")

        return result

    except httpx.RequestError as e:
        logger.error(f"Failed to connect to model server: {e}")
        raise HTTPException(
            status_code=503,
            detail=f"Model server unavailable: {e}"
        )

    except Exception as e:
        logger.error(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5001)
