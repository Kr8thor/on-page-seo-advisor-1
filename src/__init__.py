"""
On-Page SEO Analyzer & Advisor
A comprehensive tool for analyzing web pages against SEO best practices and benchmarking against competitors.
"""

from .models import (
    SerpResult, TitleAnalysis, MetaDescriptionAnalysis,
    HeadingsAnalysis, ContentAnalysis, LinksAnalysis,
    ImagesAnalysis, SchemaAnalysis, PageAnalysis,
    AnalysisRequest, AnalysisResponse
)
from .scraper import SEOAnalyzer, SerpApiError, RateLimiter, Cache

__version__ = "1.0.0" 