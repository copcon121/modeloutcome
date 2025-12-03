#!/usr/bin/env python3
"""
Run Live Gateway Server

Usage:
    python services/live_gateway/run_server.py
    python services/live_gateway/run_server.py --port 8001
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

import uvicorn


def main():
    parser = argparse.ArgumentParser(description="Run Live Gateway Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    args = parser.parse_args()
    
    print(f"Starting Live Gateway on {args.host}:{args.port}")
    
    # Import app directly to avoid module path issues
    from services.live_gateway.app import app
    
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
    )


if __name__ == "__main__":
    main()
