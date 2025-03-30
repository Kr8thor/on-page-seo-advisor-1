"""
Pydantic models for the On-Page SEO Analyzer & Advisor.
Defines the data structures for API requests, responses, and analysis results.
"""
from pydantic import BaseModel, HttpUrl, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

# --- Define Sub-Analysis Models FIRST ---

class TitleAnalysis(BaseModel):
    """Analysis results for the page title."""
    text: Optional[str] = None
    length: int = 0
    keyword_present: bool = False
    position: Optional[str] = None

class MetaDescriptionAnalysis(BaseModel):
    """Analysis results for the meta description."""
    text: Optional[str] = None
    length: int = 0
    keyword_present: bool = False

class HeadingDetail(BaseModel):
    """Details for a single heading."""
    text: str
    contains_keyword: bool
    level: int

class HeadingsAnalysis(BaseModel):
    """Analysis results for page headings."""
    h1: List[HeadingDetail] = []
    h2: List[HeadingDetail] = []
    h3: List[HeadingDetail] = []
    h4: List[HeadingDetail] = []
    h5: List[HeadingDetail] = []
    h6: List[HeadingDetail] = []
    h1_count: int = 0
    h1_contains_keyword: bool = False
    h2_count: int = 0
    h2_contains_keyword_count: int = 0
    h2_keywords: List[str] = []
    total_headings: int = 0
    keyword_present_in_any: bool = False

class ContentAnalysis(BaseModel):
    """Analysis results for page content."""
    word_count: int = 0
    keyword_count: int = 0
    keyword_density: float = 0.0
    readability: Dict[str, float] = {}

class LinksAnalysis(BaseModel):
    """Analysis results for page links."""
    total_links: int = 0
    internal_links: int = 0
    external_links: int = 0
    broken_links: List[str] = []

class ImagesAnalysis(BaseModel):
    """Analysis results for page images."""
    image_count: int = 0
    alts_missing: int = 0
    alts_with_keyword: int = 0
    images: List[Dict[str, str]] = []

class SchemaAnalysis(BaseModel):
    """Analysis results for schema markup."""
    types_found: List[str] = []
    schema_data: List[Dict[str, Any]] = []

# --- Define Main Page Analysis Model LAST (among these) ---

class PageAnalysis(BaseModel):
    """Detailed analysis results for a single page."""
    url: str
    title: Optional[TitleAnalysis] = None
    meta_description: Optional[MetaDescriptionAnalysis] = None
    headings: Optional[HeadingsAnalysis] = None
    content: Optional[ContentAnalysis] = None
    links: Optional[LinksAnalysis] = None
    images: Optional[ImagesAnalysis] = None
    schema: Optional[SchemaAnalysis] = None
    viewport_content: Optional[str] = None
    canonical_url: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "url": "https://example.com",
                "title": {
                    "text": "Example Page Title",
                    "length": 20,
                    "keyword_present": True,
                    "position": "start"
                },
                "meta_description": {
                    "text": "Example meta description",
                    "length": 30,
                    "keyword_present": True
                },
                "viewport_content": "width=device-width, initial-scale=1.0",
                "canonical_url": "https://example.com/canonical-page"
            }
        }

# --- API Request/Response Models ---

class SerpResult(BaseModel):
    """Model for SERP result data."""
    url: str
    title: str
    snippet: str
    position: int
    domain: str
    path: str
    full_url: str
    is_competitor: bool = False

class AnalysisRequest(BaseModel):
    """Request model for page analysis."""
    url: HttpUrl
    keyword: str
    max_competitors: int = Field(default=10, ge=1, le=20)

class AnalysisResponse(BaseModel):
    """Response model for page analysis."""
    status: str
    target_analysis: Optional[PageAnalysis] = None
    competitor_analyses: List[PageAnalysis] = []
    benchmarks: Optional[Dict[str, Any]] = None
    recommendations: Optional[List[Dict[str, Any]]] = None
    error: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "status": "success",
                "target_analysis": {
                    "url": "https://example.com",
                    "title": {
                        "text": "Example Page Title",
                        "length": 20,
                        "keyword_present": True,
                        "position": "start"
                    },
                    "meta_description": {
                        "text": "Example meta description",
                        "length": 30,
                        "keyword_present": True
                    },
                    "viewport_content": "width=device-width, initial-scale=1.0",
                    "canonical_url": "https://example.com/canonical-page"
                },
                "competitor_analyses": [],
                "benchmarks": {
                    "avg_word_count": 450,
                    "avg_seo_score": 82.0
                },
                "recommendations": [
                    {
                        "category": "Content",
                        "suggestion": "Add more relevant content"
                    }
                ],
                "error": None
            }
        } 