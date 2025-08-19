#!/usr/bin/env python3
"""
Data Analyst Agent - Alternative Entry Point
Compatible entry point for various deployment platforms.
"""

# Simple import and expose of the FastAPI app
from api.index import app

# This allows deployment platforms to find the app directly
if __name__ == "__main__":
    import uvicorn
    import os
    
    port = int(os.getenv("PORT", 5000))
    uvicorn.run(app, host="0.0.0.0", port=port)