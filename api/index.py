import logging
import traceback
import json
import pandas as pd
import uuid
import aiofiles
from fastapi import FastAPI, HTTPException, Request, Form, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Union
import io

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import unified agent with enhanced error handling
import os

def initialize_agent():
    """Initialize UnifiedDataAgent with comprehensive error handling."""
    try:
        from unified_data_agent import UnifiedDataAgent
        openai_key = os.getenv('OPENAI_API_KEY')
        
        if not openai_key:
            logger.warning("⚠️ OPENAI_API_KEY not found, UnifiedDataAgent disabled")
            return None
            
        # Try to initialize the agent
        agent = UnifiedDataAgent(openai_api_key=openai_key)
        logger.info("✅ UnifiedDataAgent loaded successfully")
        return agent
        
    except ImportError as e:
        logger.error(f"❌ Failed to import UnifiedDataAgent: {e}")
        logger.info("📝 The agent will run in limited mode without AI capabilities")
        return None
    except Exception as e:
        logger.error(f"❌ Failed to initialize UnifiedDataAgent: {e}")
        logger.info("📝 The agent will run in limited mode")
        return None

# Initialize agent with error handling
unified_agent = initialize_agent()

app = FastAPI(
    title="Data Analyst Agent - IIT Madras Evaluation", 
    description="AI-powered data analysis service with real calculations and chart generation",
    version="3.0.0"
)

@app.on_event("startup")
async def startup_event():
    """Startup event handler for deployment verification."""
    logger.info("🚀 Data Analyst Agent startup initiated")
    logger.info(f"📊 Agent status: {'loaded' if unified_agent else 'disabled'}")
    logger.info(f"🔑 OpenAI API: {'configured' if os.getenv('OPENAI_API_KEY') else 'missing'}")
    logger.info("✅ Application startup completed successfully")

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
    """Root endpoint with system status - Fast health check for deployment."""
    try:
        # Quick health check for deployment
        status = "active"
        agent_status = "loaded" if unified_agent else "disabled"
        
        return {
            "message": "Data Analyst Agent - IIT Madras Evaluation Ready",
            "status": status,
            "agent": agent_status,
            "endpoints": ["/api/", "/health"],
            "version": "3.0.0"
        }
    except Exception as e:
        # Return basic status even if there are issues
        logger.error(f"Root endpoint error: {e}")
        return {
            "message": "Data Analyst Agent - Basic Mode",
            "status": "partial",
            "error": str(e)
        }

@app.get("/health")
async def health_check():
    """Health check endpoint - Fast response for deployment verification."""
    try:
        # Quick health check without heavy operations
        import os
        
        checks = {
            "status": "healthy",
            "message": "Data Analyst Agent is running",
            "agent_status": "loaded" if unified_agent else "disabled",
            "openai_key": "configured" if os.getenv('OPENAI_API_KEY') else "missing",
            "timestamp": str(pd.Timestamp.now())
        }
        
        return checks
    except Exception as e:
        # Always return something for health checks
        logger.error(f"Health check error: {e}")
        return {
            "status": "degraded", 
            "message": f"Health check error: {str(e)}"
        }

@app.post("/api")
async def analyze_data(request: Request):
    """
    Main analysis endpoint for IIT Madras evaluations.
    
    Handles proper evaluation format:
    1. Multipart form with question.txt + CSV files (actual evaluation format)
    2. Plain text POST body (fallback for testing)
    
    Uses UnifiedDataAgent for real calculations with exact format compliance.
    """
    try:
        if not unified_agent:
            logger.error("UnifiedDataAgent not available for analysis")
            return {
                "error": "AI analysis service unavailable",
                "message": "The analysis service is temporarily disabled. Please check configuration.",
                "status": "service_unavailable"
            }
        
        # Step 1: Create unique request folder for this analysis
        request_id = str(uuid.uuid4())
        request_folder = os.path.join("uploads", request_id)
        os.makedirs(request_folder, exist_ok=True)
        logger.info(f"🗂️ Created request folder: {request_folder}")
        
        question_text = None
        saved_files = {}
        
        content_type = request.headers.get("content-type", "").lower()
        logger.info(f"Request content-type: {content_type}")
        
        # Handle multipart form data (actual evaluation format) - FILE-TO-DISK APPROACH
        if "multipart/form-data" in content_type:
            try:
                form = await request.form()
                logger.info(f"🔍 Processing multipart form with {len(form)} files/fields")
                logger.info(f"🔍 Form field names: {list(form.keys())}")
                
                # Step 2: Save all uploaded files to disk in request folder
                for field_name, value in form.items():
                    logger.info(f"📁 Processing form field: {field_name}, type: {type(value)}")
                    
                    # Check if this is a file upload (has file attributes)
                    if hasattr(value, "filename") and hasattr(value, "read") and getattr(value, "filename", None):
                        # It's a file upload - SAVE TO DISK
                        file_content = await value.read()
                        original_filename = value.filename
                        file_path = os.path.join(request_folder, original_filename)
                        
                        # Save file to disk
                        async with aiofiles.open(file_path, "wb") as f:
                            await f.write(file_content)
                        
                        saved_files[field_name] = file_path
                        logger.info(f"💾 Saved file to disk: {file_path} (size: {len(file_content)} bytes)")
                        
                        # Extract question from question.txt files
                        if original_filename.lower() == "question.txt" or "question" in original_filename.lower():
                            async with aiofiles.open(file_path, "r") as f:
                                question_text = await f.read()
                                question_text = question_text.strip()
                            logger.info(f"✅ Found question in {original_filename}: {question_text[:150]}...")
                    
                    else:
                        # It's a regular form field
                        if field_name in ["question", "vars"] and isinstance(value, str):
                            question_text = value.strip()
                            logger.info(f"Found question in form field {field_name}: {question_text[:150]}...")
                        saved_files[field_name] = str(value) if value else ""
                
                logger.info(f"Processed files: {saved_files}")
                
                # Fallback: If no question found, look for any .txt file in saved files
                if not question_text:
                    for field_name, file_path in saved_files.items():
                        if isinstance(file_path, str) and file_path.endswith('.txt'):
                            try:
                                async with aiofiles.open(file_path, "r") as f:
                                    question_text = await f.read()
                                    question_text = question_text.strip()
                                logger.info(f"Using {file_path} as question source: {question_text[:150]}...")
                                break
                            except Exception as e:
                                logger.warning(f"Failed to read {file_path}: {e}")
                                continue
                        
            except Exception as e:
                logger.error(f"Multipart form processing failed: {e}")
                raise HTTPException(status_code=400, detail=f"Failed to process multipart form: {e}")
        
        # Handle plain text POST body (fallback for testing)
        else:
            try:
                body = await request.body()
                if body:
                    body_text = body.decode('utf-8').strip()
                    logger.info(f"Processing plain text POST body: {len(body_text)} characters")
                    
                    # Check if it's JSON or plain text
                    if body_text and not body_text.startswith('{'):
                        question_text = body_text
                        logger.info(f"Using POST body as question: {question_text[:150]}...")
                    else:
                        # Try to parse as JSON
                        try:
                            json_data = json.loads(body_text)
                            if "vars" in json_data and "question" in json_data["vars"]:
                                question_text = json_data["vars"]["question"].strip()
                                logger.info(f"Found question in JSON vars: {question_text[:150]}...")
                            elif "question" in json_data:
                                question_text = json_data["question"].strip()
                                logger.info(f"Found question in JSON: {question_text[:150]}...")
                        except Exception as json_e:
                            logger.info(f"JSON parsing failed, treating as plain text: {json_e}")
                            question_text = body_text
            except Exception as e:
                logger.info(f"POST body parsing failed: {e}")
        
        # Validate question found
        if not question_text:
            logger.error("No question found in request")
            raise HTTPException(status_code=400, detail="No question found. Expected question.txt file or plain text question.")
        
        # Step 3: Pass request folder and question to UnifiedDataAgent for file-based analysis
        logger.info(f"📂 Starting analysis with working folder: {request_folder}")
        logger.info(f"📂 Available files: {saved_files}")
        
        # NEW: Pass the working folder to agent instead of processing files in memory
        result = unified_agent.analyze_data_from_folder(question_text, request_folder, saved_files)
        
        if result and not (isinstance(result, dict) and "error" in result):
            logger.info(f"Analysis completed successfully with keys: {list(result.keys())}")
            return result
        else:
            error_msg = result.get("error", "Unknown error") if isinstance(result, dict) else "Analysis failed"
            logger.error(f"Analysis failed: {error_msg}")
            raise HTTPException(status_code=500, detail=f"Analysis failed: {error_msg}")
            
    except HTTPException:
        # Re-raise HTTP exceptions to let FastAPI handle them
        raise
    except Exception as e:
        logger.error(f"An unexpected internal server error occurred: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

@app.post("/api/upload")
async def analyze_with_upload(question: str = Form(...), file: UploadFile = File(...)):
    """
    Dedicated endpoint for testing file uploads with questions.
    Simulates how private evaluations might send question + CSV data.
    """
    try:
        if not unified_agent:
            logger.error("UnifiedDataAgent not available for upload analysis")
            return {
                "error": "AI analysis service unavailable",
                "message": "The analysis service is temporarily disabled. Please check configuration.",
                "status": "service_unavailable"
            }
        
        logger.info(f"Upload endpoint - Question: {question[:150]}...")
        logger.info(f"Upload endpoint - File: {file.filename}")
        
        # Read and process the uploaded CSV file
        file_content = await file.read()
        csv_content = file_content.decode('utf-8')
        csv_data = pd.read_csv(io.StringIO(csv_content))
        
        logger.info(f"Successfully loaded uploaded CSV: {csv_data.shape} rows/cols")
        logger.info(f"CSV columns: {list(csv_data.columns)}")
        
        # Perform analysis
        result = unified_agent.analyze_data(question, csv_data)
        
        if result and not (isinstance(result, dict) and "error" in result):
            logger.info(f"Analysis successful. Returning result with keys: {list(result.keys())}")
            return result
        else:
            error_msg = result.get("error", "Unknown error") if isinstance(result, dict) else "Analysis failed"
            logger.error(f"Analysis failed: {error_msg}")
            raise HTTPException(status_code=500, detail=f"Analysis failed: {error_msg}")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload processing error: {str(e)}")

