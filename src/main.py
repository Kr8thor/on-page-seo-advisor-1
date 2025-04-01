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
import re

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG to see all logs
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
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
    allow_origins=[
        "https://audit.mardenseo.com",  # Production domain
        "http://audit.mardenseo.com",   # Production domain (HTTP)
        "https://onpage-seo-analyzer.vercel.app",  # Vercel preview
        "https://on-page-seo-advisor-1.vercel.app",  # Vercel production
        "https://on-page-seo-advisor-1-5beemvolg-leo-corbetts-projects.vercel.app",  # Vercel deployment
        "http://localhost:8000",        # Local development
        "http://localhost:3000",        # Local development
        "http://localhost:5000",        # Local development
        "http://127.0.0.1:8000",       # Local development
        "http://127.0.0.1:3000",       # Local development
        "http://127.0.0.1:5000"        # Local development
    ],
    allow_credentials=True,            # Allow credentials (cookies, authorization headers)
    allow_methods=["*"],               # Allow all methods
    allow_headers=["*"],               # Allow all headers
    expose_headers=["*"],              # Expose all headers
    max_age=3600                       # Cache preflight requests for 1 hour
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

# Add country code validation
VALID_COUNTRY_CODES = {
    'us', 'gb', 'ca', 'au', 'nz', 'ie', 'de', 'fr', 'es', 'it',
    'jp', 'kr', 'cn', 'in', 'br', 'mx', 'ru', 'za'
}

def normalize_country_code(country: str) -> str:
    """Normalize and validate country code."""
    if not country:
        return 'us'
    
    # Clean and normalize the input
    normalized = country.lower().strip()
    
    # If it's already a valid 2-letter code
    if normalized in VALID_COUNTRY_CODES:
        return normalized
    
    # Default to 'us' if invalid
    logger.warning(f"Invalid country code '{country}', defaulting to 'us'")
    return 'us'

@app.post("/analyze")
async def analyze_page(
    request: AnalysisRequest,
    background_tasks: BackgroundTasks,
    request_id: str = Header(None),  # Keep request_id for logging
    force_refresh: bool = Query(False, description="Set to true to bypass cache")
):
    """
    Analyze a web page for SEO optimization.
    
    This endpoint performs a comprehensive SEO analysis of the provided URL,
    including benchmarking against competitors and generating recommendations.
    
    Args:
        request: AnalysisRequest containing the URL and keyword to analyze
        background_tasks: FastAPI background tasks for saving results
        request_id: Optional request ID from header
        force_refresh: If True, bypasses the cache and forces a fresh analysis
        
    Returns:
        Dict containing the analysis results
        
    Raises:
        HTTPException: Various HTTP exceptions for different error scenarios
    """
    # Generate request ID for tracing
    request_id = request_id or str(uuid.uuid4())
    
    # Normalize country code
    normalized_country = normalize_country_code(request.country)
    if normalized_country != request.country:
        logger.info(f"[{request_id}] Normalized country code from '{request.country}' to '{normalized_country}'")
    request.country = normalized_country
    
    logger.info(f"[{request_id}] Received analysis request for URL: {request.url}, Keyword: {request.keyword}, Country: {request.country}, Force refresh: {force_refresh}")
    
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
        analysis_result_dict = await analyzer.analyze_page_with_benchmarks(
            url=str(request.url),
            keyword=request.keyword,
            country=request.country,
            request_id=request_id,
            force_refresh=force_refresh  # Pass the force_refresh parameter
        )
        
        # Check analysis status
        if analysis_result_dict.get('status') != 'success':
            error_msg = analysis_result_dict.get('error_message', 'Unknown error during analysis')
            logger.error(f"[{request_id}] Analysis failed: {error_msg}")
            raise HTTPException(
                status_code=500,
                detail=error_msg
            )

        # Extract all necessary data
        target_analysis_data = analysis_result_dict.get('analysis')
        competitor_summary = analysis_result_dict.get('competitor_analysis_summary', [])
        benchmarks_dict = analysis_result_dict.get('benchmarks', {})
        recommendations_list = analysis_result_dict.get('recommendations', [])
        warning_msg = analysis_result_dict.get('warning')

        if not target_analysis_data:
            logger.error(f"[{request_id}] No target analysis data found in result dictionary")
            raise HTTPException(status_code=500, detail="Internal error: No target analysis data available.")

        # Log the data structure for debugging
        logger.debug(f"[{request_id}] Response structure:")
        logger.debug(f"- Has target analysis: {bool(target_analysis_data)}")
        logger.debug(f"- Competitor summary count: {len(competitor_summary)}")
        logger.debug(f"- Has benchmarks: {bool(benchmarks_dict)}")
        logger.debug(f"- Recommendations count: {len(recommendations_list)}")
        logger.debug(f"- Has warning: {bool(warning_msg)}")

        # Construct the complete response
        response_data = {
            "status": "success",
            "target_analysis": target_analysis_data,
            "competitor_analysis_summary": competitor_summary,
            "benchmarks": benchmarks_dict,
            "recommendations": recommendations_list,
            "warning": warning_msg
        }

        # Save results to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_request_id = request_id.split('-')[0] if request_id else timestamp
        result_file = RESULTS_DIR / f"analysis_{safe_request_id}_{timestamp}.json"
        background_tasks.add_task(
            save_analysis_results,
            result_file,
            response_data
        )

        logger.info(f"[{request_id}] Successfully completed analysis for {request.url}")
        return response_data

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