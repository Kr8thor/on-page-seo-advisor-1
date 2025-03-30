"""
FastAPI application for the On-Page SEO Analyzer & Advisor.
Provides endpoints for analyzing web pages and generating SEO recommendations.
"""
import httpx
import uuid
from fastapi import FastAPI, HTTPException, BackgroundTasks, Query, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.utils import get_openapi
from typing import List, Optional, Union, Dict, Any
import os
from dotenv import load_dotenv
import logging
from datetime import datetime
import json
from pathlib import Path
from src.scraper import SEOAnalyzer, SerpApiError
from src.models import AnalysisRequest, AnalysisResponse, PageAnalysis

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG to see all logs
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="On-Page SEO Analyzer & Advisor",
    description="""
    A comprehensive API for analyzing web pages against SEO best practices and benchmarking against competitors.
    
    ## Features
    * Single page analysis
    * Competitor benchmarking
    * SEO recommendations
    * SERP analysis
    
    ## Usage
    1. Submit a URL for analysis
    2. Get detailed SEO analysis
    3. Receive actionable recommendations
    """,
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Verify required environment variables
SERP_API_KEY = os.getenv("SERP_API_KEY")
if not SERP_API_KEY:
    raise ValueError("SERP_API_KEY environment variable is not set")

# Create results directory if it doesn't exist
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# Initialize analyzer
analyzer = SEOAnalyzer()

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_page(
    request: AnalysisRequest,
    background_tasks: BackgroundTasks,
    request_id: str = Header(None)
):
    """
    Analyze a web page for SEO optimization.
    
    This endpoint performs a comprehensive SEO analysis of the provided URL,
    including benchmarking against competitors and generating recommendations.
    
    Args:
        request: AnalysisRequest containing the URL and keyword to analyze
        background_tasks: FastAPI background tasks for saving results
        request_id: Optional request ID from header
        
    Returns:
        AnalysisResponse containing the analysis results and recommendations
        
    Raises:
        HTTPException: Various HTTP exceptions for different error scenarios
    """
    # Generate request ID for tracing
    request_id = request_id or str(uuid.uuid4())
    logger.info(f"[{request_id}] Received analysis request for URL: {request.url}, Keyword: {request.keyword}")
    
    try:
        # Input validation
        if not request.url or not request.keyword:
            error_msg = "URL and keyword are required"
            logger.error(f"[{request_id}] {error_msg}")
            raise HTTPException(
                status_code=400,
                detail=error_msg
            )
            
        # Perform analysis
        logger.info(f"[{request_id}] Starting analysis for {request.url}")
        analysis = await analyzer.analyze_page_with_benchmarks(
            url=str(request.url),
            keyword=request.keyword,
            country=request.country
        )
        
        # Check analysis status
        if analysis.get('status') == 'error':
            error_msg = analysis.get('message', 'Unknown error during analysis')
            logger.error(f"[{request_id}] Analysis failed: {error_msg}")
            raise HTTPException(
                status_code=500,
                detail=error_msg
            )
            
        # Prepare successful response Pydantic model
        try:
            response_payload = AnalysisResponse(
                input=request,
                status=analysis.get('status'),
                target_analysis=analysis.get('target_analysis'),
                competitor_analyses=analysis.get('competitor_analyses'),
                benchmarks=analysis.get('benchmarks'),
                recommendations=analysis.get('recommendations'),
                warning=analysis.get('warning'),
                error_message=None
            )
        except Exception as model_error:
            logger.error(f"[{request_id}] Error creating response model from analysis results: {model_error}", exc_info=True)
            # Log the problematic dictionary for debugging
            logger.debug(f"Data passed to AnalysisResponse constructor: {analysis}")
            raise HTTPException(status_code=500, detail="Internal error processing analysis results.")
            
        # Save results to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Use a unique part of the request ID for the filename if available
        safe_request_id = request_id.split('-')[0] if request_id else timestamp
        result_file = RESULTS_DIR / f"analysis_{safe_request_id}_{timestamp}.json"
        # Pass the full AnalysisResponse model
        background_tasks.add_task(
            save_analysis_results,
            result_file,
            response_payload
        )
        
        logger.info(f"[{request_id}] Successfully completed analysis for {request.url}")
        return response_payload  # Return the Pydantic model
        
    except SerpApiError as e:
        error_msg = str(e)
        if "rate limit" in error_msg.lower():
            logger.error(f"[{request_id}] SERP API rate limit exceeded: {error_msg}")
            raise HTTPException(
                status_code=429,
                detail="SERP API rate limit exceeded. Please try again later."
            )
        else:
            logger.error(f"[{request_id}] SERP API error: {error_msg}")
            raise HTTPException(
                status_code=503,
                detail="Error accessing SERP API. Please try again later."
            )
            
    except ValueError as e:
        error_msg = str(e)
        logger.error(f"[{request_id}] Invalid input: {error_msg}")
        raise HTTPException(
            status_code=400,
            detail=error_msg
        )
        
    except httpx.TimeoutException as e:
        error_msg = f"Request timed out: {str(e)}"
        logger.error(f"[{request_id}] {error_msg}")
        raise HTTPException(
            status_code=503,
            detail="Analysis timed out. Please try again."
        )
        
    except httpx.RequestError as e:
        error_msg = f"Network error: {str(e)}"
        logger.error(f"[{request_id}] {error_msg}")
        raise HTTPException(
            status_code=503,
            detail="Network error during analysis. Please try again."
        )
        
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        logger.error(f"[{request_id}] {error_msg}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred. Please try again later."
        )

@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    
    Returns:
        Dict containing status of the API and its dependencies
    """
    return {
        "status": "healthy",
        "version": "1.0.0",
        "dependencies": {
            "serp_api": bool(SERP_API_KEY)
        }
    }

def save_analysis_results(file_path: Path, response_model: AnalysisResponse):
    """Save analysis results (as Pydantic model) to a JSON file."""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            # Convert Pydantic model to JSON-compatible dict
            json_compatible_data = response_model.model_dump(mode='json', exclude_unset=True)
            json.dump(json_compatible_data, f, indent=2, ensure_ascii=False)
        logger.info(f"Analysis results saved to {file_path}")
    except Exception as e:
        logger.error(f"Error saving analysis results to {file_path}: {e}", exc_info=True)

# Custom OpenAPI schema
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title="On-Page SEO Analyzer & Advisor API",
        version="1.0.0",
        description="API for analyzing web pages against SEO best practices",
        routes=app.routes,
    )

    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 