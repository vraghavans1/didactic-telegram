#!/usr/bin/env python3
"""
Data Analyst Agent - Primary Entry Point
Deployment-ready startup script for the Data Analyst Agent FastAPI application.
"""

import os
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Main entry point for deployment."""
    try:
        logger.info("🚀 Starting Data Analyst Agent deployment...")
        
        # Import the FastAPI app
        from api.index import app
        logger.info("✅ FastAPI application imported successfully")
        
        # Start the server with uvicorn
        import uvicorn
        
        # Get port from environment (Replit deployment sets this)
        port = int(os.getenv("PORT", 5000))
        host = "0.0.0.0"
        
        logger.info(f"🌐 Starting server on {host}:{port}")
        
        # Run the server
        uvicorn.run(
            app,
            host=host,
            port=port,
            log_level="info"
        )
        
    except Exception as e:
        logger.error(f"❌ Failed to start Data Analyst Agent: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()