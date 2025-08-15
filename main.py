#!/usr/bin/env python3
"""
Alternative entry point for the Data Analyst Agent API server.
This provides another way to start the application during deployment.
"""

import os
import uvicorn

if __name__ == "__main__":
    # Get port from environment variable (Replit/deployment platforms set this)
    port = int(os.environ.get("PORT", 5000))
    host = "0.0.0.0"
    
    print(f"Starting Data Analyst Agent API server on {host}:{port}")
    
    # Start the server using the FastAPI app from api.index
    uvicorn.run(
        "api.index:app",
        host=host,
        port=port,
        reload=False,
        log_level="info"
    )