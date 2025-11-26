"""
FastAPI Inference Server
Serves trained outcome model for real-time predictions
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import torch
import torch.nn.functional as F
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from layer3_model.training.train_outcome_model import SimpleTransformer, SimpleMLP
from layer2_feature_engine.utils.logging_utils import setup_logger

logger = setup_logger('inference_server', log_file='logs/inference_server.log')

# Initialize FastAPI app
app = FastAPI(title="Outcome Model Inference Server", version="1.0.0")

# Global model variable
model = None
device = None
config = None


class InferRequest(BaseModel):
    """Request schema for inference"""
    features: List[List[float]]  # [context_len, feature_dim]


class InferResponse(BaseModel):
    """Response schema for inference"""
    prob_long: float
    prob_short: float
    prob_skip: float


@app.on_event("startup")
async def load_model():
    """Load trained model at startup"""
    global model, device, config

    model_path = Path("models/outcome_transformer_v1.pt")

    if not model_path.exists():
        logger.error(f"Model file not found: {model_path}")
        logger.warning("Server starting without model - inference will fail")
        return

    logger.info(f"Loading model from {model_path}")

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint['config']

    # Instantiate model
    model_config = config['model']
    architecture = model_config['architecture']

    # Need to determine feature_dim from checkpoint or config
    # For simplicity, we'll extract it from the state dict
    state_dict = checkpoint['model_state_dict']

    if architecture == 'transformer':
        # Get feature_dim from input_proj weight shape
        feature_dim = state_dict['input_proj.weight'].shape[1]
        model = SimpleTransformer(feature_dim, model_config['transformer'])
    elif architecture == 'mlp':
        # Get input_dim from first layer
        input_dim = state_dict['network.0.weight'].shape[1]
        model = SimpleMLP(input_dim, model_config['mlp'])
    else:
        raise ValueError(f"Unknown architecture: {architecture}")

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    logger.info("Model loaded successfully!")
    logger.info(f"Architecture: {architecture}")
    logger.info(f"Validation accuracy from training: {checkpoint.get('val_acc', 'N/A')}")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "device": str(device) if device else "N/A"
    }


@app.post("/infer", response_model=InferResponse)
async def infer(request: InferRequest):
    """
    Inference endpoint

    Args:
        request: InferRequest with feature matrix

    Returns:
        InferResponse with probabilities
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Convert features to tensor
        features_tensor = torch.tensor(request.features, dtype=torch.float32)

        # Add batch dimension
        features_tensor = features_tensor.unsqueeze(0)  # [1, context_len, feature_dim]

        # Move to device
        features_tensor = features_tensor.to(device)

        # Inference
        with torch.no_grad():
            logits = model(features_tensor)  # [1, num_classes]
            probs = F.softmax(logits, dim=1)  # [1, num_classes]

        # Extract probabilities
        probs = probs.cpu().numpy()[0]

        response = InferResponse(
            prob_long=float(probs[0]),
            prob_short=float(probs[1]),
            prob_skip=float(probs[2])
        )

        logger.debug(f"Inference result: {response}")

        return response

    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Outcome Model Inference Server",
        "version": "1.0.0",
        "endpoints": {
            "/health": "Health check",
            "/infer": "POST - Run inference",
            "/docs": "API documentation"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5002)
