#!/usr/bin/env python3
"""
Run Live Gateway Server

Usage:
    python services/live_gateway/run_server.py
    python services/live_gateway/run_server.py --port 8001
"""

import argparse
import uvicorn


def main():
    parser = argparse.ArgumentParser(description="Run Live Gateway Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    args = parser.parse_args()
    
    print(f"Starting Live Gateway on {args.host}:{args.port}")
    
    uvicorn.run(
        "services.live_gateway.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
