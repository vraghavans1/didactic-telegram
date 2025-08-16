import logging
import traceback
import json
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Form, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import io

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import unified agent
import os
try:
    from unified_data_agent import UnifiedDataAgent
    openai_key = os.getenv('OPENAI_API_KEY')
    if openai_key:
        unified_agent = UnifiedDataAgent(openai_api_key=openai_key)
        logger.info("✅ UnifiedDataAgent loaded successfully")
    else:
        unified_agent = None
        logger.warning("⚠️ OPENAI_API_KEY not found, UnifiedDataAgent disabled")
except Exception as e:
    unified_agent = None
    logger.error(f"❌ Failed to load UnifiedDataAgent: {e}")

app = FastAPI(
    title="Data Analyst Agent - IIT Madras Evaluation", 
    description="AI-powered data analysis service with real calculations and chart generation",
    version="3.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str

@app.get("/")
async def root():
    """Root endpoint with system status."""
    return {
        "message": "Data Analyst Agent - IIT Madras Evaluation Ready",
        "status": "active",
        "agent": "UnifiedDataAgent" if unified_agent else "None",
        "endpoints": ["/api/", "/health"]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "message": "Data Analyst Agent is running",
        "agent_status": "loaded" if unified_agent else "error"
    }

@app.post("/api/")
async def analyze_data(request: Request):
    """
    Main analysis endpoint for IIT Madras evaluation.
    Handles multipart form requests with questions.txt and CSV attachments.
    Uses UnifiedDataAgent for real calculations with exact format compliance.
    
    Expected format: curl "https://app.example.com/api/" -F "questions.txt=@questions.txt" -F "data.csv=@data.csv"
    """
    try:
        if not unified_agent:
            raise HTTPException(status_code=500, detail="UnifiedDataAgent not available")
        
        logger.info("Processing multipart form data for evaluation system")
        
        # Parse multipart form data
        form = await request.form()
        question_text = None
        data_files = {}
        
        # Process all uploaded files (matching evaluation system logic)
        for field_name, value in form.items():
            if hasattr(value, "filename") and value.filename:  # It's a file
                try:
                    file_content = await value.read()
                    file_text = file_content.decode('utf-8')
                    
                    # Check for question files (multiple variations)
                    if field_name == "question.txt" or field_name == "questions.txt":
                        question_text = file_text.strip()
                        logger.info(f"Found question in {field_name}: {question_text[:100]}...")
                    else:
                        # Save other files as data sources
                        data_files[field_name] = {
                            'filename': value.filename,
                            'content': file_text
                        }
                        logger.info(f"Found data file {field_name}: {value.filename} ({len(file_text)} chars)")
                        
                except Exception as e:
                    logger.error(f"Error reading file {field_name}: {e}")
                    raise HTTPException(status_code=400, detail=f"Error processing file {field_name}: {e}")
            else:
                # Handle non-file form fields
                if field_name in ["question", "query"] and not question_text:
                    question_text = str(value).strip()
                    logger.info(f"Found question in form field {field_name}: {question_text[:100]}...")
        
        # Fallback logic for question detection
        if not question_text and data_files:
            # Use first file as question if no explicit question file found
            first_file = list(data_files.values())[0]
            question_text = first_file['content']
            # Remove it from data_files since it's now the question
            data_files.pop(list(data_files.keys())[0])
            logger.info("Using fallback: first data file as question")
        
        # Handle simple POST body as fallback (for testing)
        if not question_text:
            try:
                body = await request.body()
                if body:
                    body_text = body.decode('utf-8').strip()
                    if body_text:
                        question_text = body_text
                        logger.info(f"Found question in POST body: {question_text[:100]}...")
            except Exception as e:
                logger.warning(f"Could not read POST body: {e}")
        
        if not question_text:
            logger.error("No question found in any format (multipart files, form fields, or POST body)")
            raise HTTPException(status_code=400, detail="No question file or content provided")
        
        logger.info(f"Final question length: {len(question_text)} characters")
        logger.info(f"Data files available: {list(data_files.keys())}")
        
        # Convert data files to DataFrame for processing
        csv_data = None
        if data_files:
            # Use the first CSV-like file for analysis
            first_data_file = list(data_files.values())[0]
            try:
                # Try to parse as CSV
                csv_data = pd.read_csv(io.StringIO(first_data_file['content']))
                logger.info(f"Loaded CSV data: {csv_data.shape} rows/cols")
                logger.info(f"CSV columns: {list(csv_data.columns)}")
            except Exception as e:
                logger.error(f"Failed to parse CSV data: {e}")
                raise HTTPException(status_code=400, detail=f"Invalid CSV data: {e}")
        else:
            # Create empty DataFrame if no data provided
            csv_data = pd.DataFrame()
            logger.warning("No data files provided, using empty DataFrame")
        
        # Process with UnifiedDataAgent
        result = unified_agent.analyze_data(question_text, csv_data)
        
        if result and isinstance(result, dict) and "error" not in result:
            logger.info(f"Success! Returning result with keys: {list(result.keys())}")
            return result
        else:
            error_msg = result.get("error", "Unknown error from UnifiedDataAgent")
            logger.error(f"UnifiedDataAgent failed: {error_msg}")
            raise HTTPException(status_code=500, detail=f"Analysis failed: {error_msg}")
            
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        logger.error(f"Unexpected error in analyze_data: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

