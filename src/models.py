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

class PerformanceAnalysis(BaseModel):
    """Analysis results for page performance metrics."""
    html_size: Optional[int] = None  # Bytes
    text_html_ratio: Optional[float] = None  # Percentage or ratio
    # Add other potential fields later if needed:
    # css_file_count: Optional[int] = None
    # js_file_count: Optional[int] = None

# --- Define Main Page Analysis Model LAST (among these) ---

class PageAnalysis(BaseModel):
    """
    Main model for page analysis results.
    
    This model represents the complete analysis of a web page, including:
    - Basic page elements (title, meta description)
    - Content structure (headings, main content)
    - Technical elements (links, images, schema)
    - Mobile optimization (viewport)
    - URL structure (canonical)
    - Performance metrics
    """
    url: str
    title: TitleAnalysis
    meta_description: MetaDescriptionAnalysis
    headings: HeadingsAnalysis
    content: ContentAnalysis
    links: LinksAnalysis
    images: ImagesAnalysis
    schema: SchemaAnalysis
    viewport_content: Optional[str] = Field(None, description="Content attribute of the viewport meta tag.")
    canonical_url: Optional[str] = Field(None, description="Absolute URL found in the canonical link tag.")
    performance: Optional[PerformanceAnalysis] = None
    benchmarks: Optional[Dict[str, Any]] = None
    recommendations: Optional[List[Dict[str, str]]] = None

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
                    "length": 25,
                    "keyword_present": True
                },
                "headings": {
                    "h1": [{"level": 1, "text": "Main Heading", "contains_keyword": True}],
                    "h2": [{"level": 2, "text": "Subheading", "contains_keyword": False}],
                    "h3": [],
                    "h1_count": 1,
                    "h1_contains_keyword": True,
                    "h2_keywords": []
                },
                "content": {
                    "word_count": 500,
                    "keyword_count": 5,
                    "keyword_density": 1.0,
                    "readability": {
                        "flesch_reading_ease": 65.0,
                        "flesch_kincaid_grade": 8.0,
                        "gunning_fog": 12.0,
                        "smog_index": 10.0,
                        "automated_readability_index": 8.0,
                        "coleman_liau_index": 8.0,
                        "linsear_write_formula": 8.0,
                        "dale_chall_readability_score": 7.0
                    }
                },
                "links": {
                    "total_links": 10,
                    "internal_links": 7,
                    "external_links": 3,
                    "broken_links": []
                },
                "images": {
                    "image_count": 5,
                    "alts_missing": 2,
                    "alts_with_keyword": 1,
                    "images": [
                        {"src": "https://example.com/image1.jpg", "alt": "Example image 1"},
                        {"src": "https://example.com/image2.jpg", "alt": ""}
                    ]
                },
                "schema": {
                    "types_found": ["Article", "WebPage"],
                    "schema_data": []
                },
                "viewport_content": "width=device-width, initial-scale=1.0",
                "canonical_url": "https://example.com/canonical-page",
                "performance": {
                    "html_size": 50000,
                    "text_html_ratio": 75.5
                }
            }
        }

# --- API Request/Response Models ---

class SerpResult(BaseModel):
    """
    Represents a single organic result from SERP API.
    
    Maps to the structure provided by ValueSERP API's organic_results array.
    Only includes fields that are actually provided by the API and needed for analysis.
    """
    url: str = Field(..., description="The full URL of the search result (maps to 'link' in API response)")
    title: str = Field(..., description="The title of the search result")
    snippet: Optional[str] = Field(None, description="The description/snippet of the search result")
    position: Optional[int] = Field(None, description="The position of the result in SERP (1-based)")
    domain: Optional[str] = Field(None, description="The domain of the result URL")

    class Config:
        json_schema_extra = {
            "example": {
                "url": "https://example.com/article",
                "title": "Example Article Title",
                "snippet": "This is an example snippet from the search result...",
                "position": 1,
                "domain": "example.com"
            }
        }

class AnalysisRequest(BaseModel):
    """Request model for page analysis."""
    url: HttpUrl = Field(..., description="The URL of the page to analyze")
    keyword: str = Field(..., description="The main keyword to analyze for")
    country: Optional[str] = Field('us', description="Optional: Two-letter country code for SERP analysis (e.g., 'us', 'gb', 'de')")
    max_competitors: int = Field(default=10, ge=1, le=20)

    class Config:
        json_schema_extra = {
            "example": {
                "url": "https://example.com",
                "keyword": "seo tools",
                "country": "us",
                "max_competitors": 10
            }
        }

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
                    "canonical_url": "https://example.com/canonical-page",
                    "performance": {
                        "html_size": 150000,
                        "text_html_ratio": 0.75
                    }
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