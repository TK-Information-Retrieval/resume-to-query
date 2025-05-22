from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import tempfile
import os
import requests
from typing import Dict, Any
import logging
import uvicorn

# Import your existing modules
from summarize import LLMModel
from pdf2text import extract_v1

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Resume Processing API",
    description="API to process resume PDFs, extract text, summarize with LLM, and retrieve results",
    version="1.0.0"
)

QUERY_TEMPLATE = """
Analyze the following resume and extract key details in the Skills.

Then, based on the extracted information, generate a concise and effective search engine query that summarizes the candidate`s professional profile and expertise in ONE SENTENCE ONLY.

Resume:
{resume}
"""

# External API endpoint
EXTERNAL_API_URL = "https://iqbal.com/retrieve"

# ========================= FUNCTIONS =========================
async def process_resume_pipeline(file_path: str) -> str:
    """
    Process resume through the existing pipeline
    """
    try:
        # Extract text from resume
        resume_text = extract_v1(file_path)
        if not resume_text or not resume_text.strip():
            raise ValueError("Failed to extract text from resume or resume is empty")
        # replace long spaces with single space
        resume_text = ' '.join(resume_text.split())
        # Clean the resume text
        resume_text = resume_text.replace("\n", " ").replace("\r", " ")
        logger.info("Resume text extracted successfully")

        # Initialize LLM model
        llm = LLMModel()
        prompt = QUERY_TEMPLATE.format(resume=resume_text)
        
        # Generate initial response
        initial_response = llm.generate_response(prompt)
        if not initial_response:
            raise ValueError("Failed to get response from LLM")
        
        # Clean the response
        clean_response = initial_response.split("</think>")[-1].strip()
        
        # Generate final summary
        final_prompt = f"summarize into one sentence: {clean_response}"
        final_response = llm.generate_response(final_prompt)
        
        if not final_response:
            raise ValueError("Failed to get final summary from LLM")
        
        return final_response.split("</think>")[-1].strip()
        
    except Exception as e:
        logger.error(f"Error in resume processing pipeline: {str(e)}")
        raise

async def call_search_engine_api(summary: str) -> Dict[Any, Any]:
    """
    Call the external API with the summary
    """
    try:
        payload = {"summary": summary}
        
        response = requests.post(
            EXTERNAL_API_URL, 
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30  # 30 second timeout
        )
        
        response.raise_for_status()  # Raises an HTTPException for bad responses
        
        return response.json()
        
    except requests.exceptions.Timeout:
        logger.error("Timeout calling external API")
        raise HTTPException(status_code=504, detail="External API timeout")
    except requests.exceptions.ConnectionError:
        logger.error("Connection error calling external API")
        raise HTTPException(status_code=503, detail="External API connection error")
    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP error calling external API: {e}")
        raise HTTPException(status_code=e.response.status_code, detail=f"External API error: {e}")
    except Exception as e:
        logger.error(f"Unexpected error calling external API: {str(e)}")
        raise HTTPException(status_code=500, detail="Unexpected error calling external API")


# ========================= API ENDPOINTS =========================

@app.post("/process-resume")
async def process_resume(file: UploadFile = File(...)):
    """
    Process a resume PDF file through the complete pipeline:
    1. Extract text from PDF
    2. Summarize with LLM
    3. Call external API with summary
    4. Return results
    """
    
    # Validate file type
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")
    
    # Create temporary file
    temp_file = None
    try:
        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        logger.info(f"Processing resume file: {file.filename}")
        
        # Step 1 & 2: Extract resume and generate summary
        summary = await process_resume_pipeline(temp_file_path)
        logger.info(f"Generated summary: {summary}")
        
        # Step 3: Call search engine API
        external_result = await call_search_engine_api(summary)
        logger.info("Successfully called search engine API")
        
        # Step 4: Return comprehensive result
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "filename": file.filename,
                "summary": summary,
                "search_engine_result": external_result,
                "message": "Resume processed successfully"
            }
        )
        
    except ValueError as e:
        logger.error(f"Processing error: {str(e)}")
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error during resume processing")
    
    finally:
        # Clean up temporary file
        if temp_file and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
                logger.info("Temporary file cleaned up")
            except Exception as e:
                logger.warning(f"Failed to clean up temporary file: {e}")


@app.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    return {"status": "healthy", "message": "Resume processing API is running"}


@app.get("/")
async def root():
    """
    Root endpoint with API information
    """
    return {
        "message": "Resume Processing API",
        "version": "1.0.0",
        "endpoints": {
            "/process-resume": "POST - Upload and process resume PDF",
            "/health": "GET - Health check",
            "/docs": "GET - API documentation"
        }
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)