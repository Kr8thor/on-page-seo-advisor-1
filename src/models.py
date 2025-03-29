"""
Data models for the On-Page SEO Analyzer & Advisor.
Defines the structure of analysis results and API requests/responses.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, HttpUrl, Field

class SerpResult(BaseModel):
    """Represents a single organic result from SERP API."""
    url: str
    title: str
    snippet: Optional[str] = None

class TitleAnalysis(BaseModel):
    """Analysis results for the page title."""
    text: Optional[str] = None
    length: int = 0
    keyword_present: bool = False
    position: Optional[str] = None  # 'start', 'middle', 'end', or None

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
    h1_count: int = 0
    h1_contains_keyword: bool = False
    h2_keywords: List[str] = []  # Preserved from existing implementation

class ContentAnalysis(BaseModel):
    """Analysis results for page content."""
    word_count: int = 0
    readability_score: Optional[float] = None  # e.g., Flesch-Kincaid Grade Level
    keyword_density: float = 0.0  # Percentage
    keyword_count: int = 0

class LinksAnalysis(BaseModel):
    """Analysis results for page links."""
    internal_links: int = 0
    external_links: int = 0
    broken_links: List[str] = []  # Added from existing implementation

class ImagesAnalysis(BaseModel):
    """Analysis results for page images."""
    image_count: int = 0
    alts_missing: int = 0
    alts_with_keyword: int = 0
    images: List[Dict[str, str]] = []  # Preserved from existing implementation

class SchemaAnalysis(BaseModel):
    """Analysis results for schema.org markup."""
    types_found: List[str] = []  # List of @type values found
    schema_data: List[Dict[str, Any]] = []  # Preserved from existing implementation

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
    performance: Optional[Dict[str, Any]] = None  # Added from existing implementation
    benchmarks: Optional[Dict[str, Any]] = None
    recommendations: Optional[List[Dict[str, Any]]] = None

class AnalysisRequest(BaseModel):
    """Request model for page analysis."""
    url: HttpUrl = Field(..., description="The URL of the page to analyze")
    keyword: str = Field(..., description="The main keyword to analyze for")
    country: Optional[str] = Field('us', description="Two-letter country code for SERP analysis")
    competitor_urls: Optional[List[HttpUrl]] = Field(None, description="Optional list of competitor URLs to benchmark against")

    class Config:
        schema_extra = {
            "example": {
                "url": "https://mardenseo.com",
                "keyword": "seo tools",
                "country": "us",
                "competitor_urls": ["https://competitor1.com", "https://competitor2.com"]
            }
        }

class AnalysisResponse(BaseModel):
    """Response model containing the analysis results."""
    status: str = "success"
    target_analysis: Optional[PageAnalysis] = None  # Analysis of the target URL
    competitor_analyses: Optional[List[PageAnalysis]] = None  # Detailed analysis of competitors
    benchmarks: Optional[Dict[str, Any]] = None  # Benchmark metrics
    recommendations: Optional[List[Dict[str, Any]]] = None  # SEO recommendations
    error_message: Optional[str] = None  # Error message if status is "error" 