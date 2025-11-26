#!/bin/bash
# Run Outcome Model Inference Server (local, no Docker)

echo "Starting Outcome Model Inference Server..."

# Check if venv exists
if [ ! -d "venv" ]; then
    echo "Error: Virtual environment not found. Please create one first:"
    echo "  python -m venv venv"
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Check if model exists
if [ ! -f "models/outcome_transformer_v1.pt" ]; then
    echo "Warning: Model file not found at models/outcome_transformer_v1.pt"
    echo "Server will start but inference will fail until model is trained."
fi

# Set Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Run server
echo "Server will be available at: http://localhost:5002"
echo "API docs at: http://localhost:5002/docs"
echo ""

uvicorn src.layer3_model.inference.server:app \
    --host 0.0.0.0 \
    --port 5002 \
    --reload \
    --log-level info
