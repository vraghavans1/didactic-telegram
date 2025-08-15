#!/usr/bin/env python3
"""
Simple entry point that directly imports and exposes the FastAPI app.
This can be used with deployment platforms that expect an 'app' variable.
"""

# Import the FastAPI application
from api.index import app

# This allows deployment platforms to find the app directly
if __name__ == "__main__":
    import uvicorn
    import os
    
    port = int(os.environ.get("PORT", 5000))
    uvicorn.run(app, host="0.0.0.0", port=port)