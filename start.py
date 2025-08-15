#!/usr/bin/env python3
"""
Entry point for the Data Analyst Agent API server.
This file ensures the application starts correctly during deployment.
"""

import os
import sys
import uvicorn

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the FastAPI app
try:
    from api.index import app
    print("✅ Successfully imported FastAPI app from api.index")
except ImportError as e:
    print(f"❌ Failed to import FastAPI app: {e}")
    sys.exit(1)

if __name__ == "__main__":
    # Get port from environment variable (Replit sets this)
    port = int(os.environ.get("PORT", 5000))
    host = "0.0.0.0"
    
    print(f"🚀 Starting Data Analyst Agent API server on {host}:{port}")
    print("📊 Ready to process data analysis requests!")
    
    # Start the server
    uvicorn.run(
        "api.index:app",
        host=host,
        port=port,
        reload=False,  # Disable reload in production
        log_level="info"
    )