"""
Test suite for the On-Page SEO Analyzer & Advisor scraper functionality.
"""

import pytest
import httpx
import json
from unittest.mock import patch, MagicMock, AsyncMock, PropertyMock
from pathlib import Path
import os
import sys
from typing import List

# Add src directory to path for imports if running tests from the root directory
# This might be needed depending on how you run pytest
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import necessary components AFTER adjusting path
from src.scraper import SEOAnalyzer, SerpApiError, RateLimiter, Cache
from src.models import (
    PageAnalysis, SerpResult, TitleAnalysis, MetaDescriptionAnalysis,
    HeadingsAnalysis, ContentAnalysis, LinksAnalysis, ImagesAnalysis,
    SchemaAnalysis, HeadingDetail
)

# Mock HTML content for testing analysis functions
MOCK_HTML_CONTENT = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="description" content="Meta description about test keyword">
    <title>Test Page Title with Test Keyword</title>
    <script type="application/ld+json">
        {
            "@context": "https://schema.org",
            "@type": "Article",
            "headline": "Test Article",
            "author": {
                "@type": "Person",
                "name": "Test Author"
            }
        }
    </script>
</head>
<body>
    <h1>Main H1 Heading Contains Test Keyword</h1>
    <article>
        <h2>Second H2 Heading</h2>
        <p>This is paragraph one. It contains the test keyword multiple times. test keyword!</p>
        <p>Paragraph two is here. <a href="/internal-link">Internal Link</a>.</p>
        <img src="image1.jpg" alt="Image alt text with test keyword">
        <img src="image2.png" alt=""> <!-- Missing Alt -->
        <a href="https://external.com">External Link</a>
        <h3>Third H3 Heading</h3>
    </article>
    <script> console.log('Some javascript'); </script>
</body>
</html>
"""

MOCK_URL = "https://example-test.com/test-page"
MOCK_KEYWORD = "test keyword"

class AsyncResponse:
    """Custom async response class for testing."""
    def __init__(self, status_code, headers, data, raise_error=None):
        self.status_code = status_code
        self.headers = headers
        self._data = data
        self._raise_error = raise_error
        self.text = json.dumps(data) if isinstance(data, dict) else str(data)

    async def raise_for_status(self):
        if self._raise_error:
            raise self._raise_error

    async def json(self):
        return self._data

    async def atext(self):
        return self.text

# --- Fixtures ---

@pytest.fixture(scope="module")
def analyzer_instance():
    """Provides an instance of SEOAnalyzer for tests."""
    # Ensure environment variable is set for initialization
    os.environ["SERP_API_KEY"] = "test_api_key_123"
    
    # Create a mock cache instance
    mock_cache = MagicMock(spec=Cache)
    mock_cache.cache_dir = Path("test_cache")
    mock_cache.get.return_value = None
    mock_cache.set.return_value = None
    
    # Use the mock cache when creating the analyzer
    with patch('src.scraper.Cache', return_value=mock_cache):
        analyzer = SEOAnalyzer()
        return analyzer

@pytest.fixture
def mock_httpx_client():
    """Provides a mocked httpx.AsyncClient."""
    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.get = AsyncMock()
    return mock_client

@pytest.fixture
def mock_successful_page_response():
    """Mock a successful page response."""
    response = AsyncMock(spec=httpx.Response)
    response.status_code = 200
    response.headers = {'content-type': 'text/html'}
    response.raise_for_status = AsyncMock()
    response.atext = AsyncMock(return_value=MOCK_HTML_CONTENT)
    type(response).text = PropertyMock(return_value=MOCK_HTML_CONTENT)

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.get = AsyncMock(return_value=response)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock()
    return mock_client

@pytest.fixture
def mock_failed_page_response():
    """Mock a failed page response."""
    response = AsyncMock(spec=httpx.Response)
    response.status_code = 404
    response.headers = {'content-type': 'text/plain'}
    response.raise_for_status = AsyncMock(side_effect=httpx.HTTPStatusError("Not Found", request=MagicMock(), response=response))
    response.atext = AsyncMock(return_value="Not Found Error Page")
    type(response).text = PropertyMock(return_value="Not Found Error Page")

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.get = AsyncMock(return_value=response)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock()
    return mock_client

@pytest.fixture
def mock_successful_serp_response():
    """Mock a successful SERP API response."""
    mock_data = {
        "request_info": {
            "success": True,
            "message": "Success"
        },
        "search_parameters": {
            "q": MOCK_KEYWORD,
            "location": "United States"
        },
        "organic_results": [
            {
                "link": "https://competitor1.com",
                "title": "Competitor 1 Title",
                "snippet": "Competitor 1 description"
            },
            {
                "link": "https://competitor2.com",
                "title": "Competitor 2 Title",
                "snippet": "Competitor 2 description"
            },
            {
                "link": "https://competitor3.com",
                "title": "Competitor 3 Title",
                "snippet": "Competitor 3 description"
            }
        ]
    }

    response = AsyncMock(spec=httpx.Response)
    response.status_code = 200
    response.headers = {'content-type': 'application/json'}
    response.raise_for_status = AsyncMock()
    response.json = AsyncMock(return_value=mock_data)
    type(response).text = PropertyMock(return_value=json.dumps(mock_data))

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.get = AsyncMock(return_value=response)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock()
    return mock_client

@pytest.fixture
def mock_failed_serp_response():
    """Mock a failed SERP API response."""
    response = AsyncMock(spec=httpx.Response)
    response.status_code = 401
    response.headers = {'content-type': 'application/json'}
    response.raise_for_status = AsyncMock(side_effect=httpx.HTTPStatusError("Invalid API key", request=MagicMock(), response=response))
    response.json = AsyncMock(return_value={"error": "Invalid API key"})
    type(response).text = PropertyMock(return_value='{"error": "Invalid API key"}')

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.get = AsyncMock(return_value=response)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock()
    return mock_client


# --- Test Cases ---

@pytest.mark.asyncio
async def test_fetch_page_content_success(analyzer_instance, mock_successful_page_response):
    """Test successful fetching of page content."""
    with patch('httpx.AsyncClient', return_value=mock_successful_page_response):
        html = await analyzer_instance.fetch_page_content(MOCK_URL)
        assert isinstance(html, str)
        assert html == MOCK_HTML_CONTENT
        analyzer_instance.rate_limiter.requests = []

@pytest.mark.asyncio
async def test_fetch_page_content_failure(analyzer_instance, mock_failed_page_response):
    """Test failed fetching of page content (e.g., 404)."""
    with patch('httpx.AsyncClient', return_value=mock_failed_page_response):
        html = await analyzer_instance.fetch_page_content("https://example-test.com/not-found")
        assert html is None
        analyzer_instance.rate_limiter.requests = []

@pytest.mark.asyncio
async def test_fetch_serp_results_success(analyzer_instance, mock_successful_serp_response):
    """Test successful fetching of SERP results."""
    with patch('httpx.AsyncClient', return_value=mock_successful_serp_response):
        results = await analyzer_instance.fetch_serp_results(MOCK_KEYWORD, "us")
        assert len(results) == 3
        assert all(isinstance(r, SerpResult) for r in results)
        assert results[0].url == 'https://competitor1.com'
        assert results[0].title == 'Competitor 1 Title'
        analyzer_instance.rate_limiter.requests = []

@pytest.mark.asyncio
async def test_fetch_serp_results_api_error(analyzer_instance, mock_failed_serp_response):
    """Test handling of API-level errors from SERP provider."""
    with patch('httpx.AsyncClient', return_value=mock_failed_serp_response):
        with pytest.raises(SerpApiError) as exc_info:
            await analyzer_instance.fetch_serp_results(MOCK_KEYWORD, "us")
        assert "Invalid SERP API key" in str(exc_info.value)
        analyzer_instance.rate_limiter.requests = []

# --- Test Analysis Methods ---
# We test the main analyze_page and assume it calls the helpers correctly
# Alternatively, test helper methods directly if needed for granularity

@pytest.mark.asyncio
async def test_analyze_page_full(analyzer_instance, mock_successful_page_response):
    """Test the comprehensive analyze_page method using mock HTML."""
    with patch('httpx.AsyncClient', return_value=mock_successful_page_response):
        analysis = await analyzer_instance.analyze_page(MOCK_URL, MOCK_KEYWORD)

        assert isinstance(analysis, dict)
        assert analysis['status'] == 'success'
        analysis_data = analysis['analysis']

        # Title Checks
        assert isinstance(analysis_data.title, TitleAnalysis)
        assert analysis_data.title.text == "Test Page Title with Test Keyword"
        assert analysis_data.title.length == 33
        assert analysis_data.title.keyword_present is True
        assert analysis_data.title.position == 'end'

        # Meta Description Checks
        assert isinstance(analysis_data.meta_description, MetaDescriptionAnalysis)
        assert analysis_data.meta_description.text == "Meta description about test keyword"
        assert analysis_data.meta_description.length == 35
        assert analysis_data.meta_description.keyword_present is True

        # Headings Checks
        assert isinstance(analysis_data.headings, HeadingsAnalysis)
        assert analysis_data.headings.h1_count == 1
        assert analysis_data.headings.h1_contains_keyword is True
        assert analysis_data.headings.h1[0].text == "Main H1 Heading Contains Test Keyword"
        assert len(analysis_data.headings.h2) == 1
        assert analysis_data.headings.h2[0].text == "Second H2 Heading"
        assert analysis_data.headings.h2[0].contains_keyword is False
        assert len(analysis_data.headings.h3) == 1
        assert analysis_data.headings.h3[0].text == "Third H3 Heading"

        # Content Checks
        assert isinstance(analysis_data.content, ContentAnalysis)
        assert analysis_data.content.word_count > 10
        assert analysis_data.content.keyword_count == 3
        assert analysis_data.content.keyword_density > 0
        assert analysis_data.content.readability_score is None or isinstance(analysis_data.content.readability_score, float)

        # Links Checks
        assert isinstance(analysis_data.links, LinksAnalysis)
        assert analysis_data.links.internal_links == 1
        assert analysis_data.links.external_links == 1

        # Images Checks
        assert isinstance(analysis_data.images, ImagesAnalysis)
        assert analysis_data.images.image_count == 2
        assert analysis_data.images.alts_missing == 1
        assert analysis_data.images.alts_with_keyword == 1

        # Schema Checks
        assert isinstance(analysis_data.schema, SchemaAnalysis)
        assert analysis_data.schema.types_found == ["Article"]

@pytest.mark.asyncio
async def test_analyze_page_fetch_failure(analyzer_instance, mock_failed_page_response):
    """Test analyze_page when the initial HTML fetch fails."""
    with patch('httpx.AsyncClient', return_value=mock_failed_page_response):
        analysis = await analyzer_instance.analyze_page(MOCK_URL, MOCK_KEYWORD)
        assert analysis['status'] == 'error'
        assert 'message' in analysis
        analyzer_instance.rate_limiter.requests = []

# --- Test Benchmarking and Recommendations (Basic Tests) ---

@pytest.fixture
def sample_target_analysis(analyzer_instance):
    # Create a basic PageAnalysis object for benchmarking tests
    # Note: We might need to run analyze_page on mock HTML first to get this
    # Or manually construct it based on the model
    return PageAnalysis(
        url=MOCK_URL,
        title=TitleAnalysis(text="Target Title", length=40, keyword_present=True),
        content=ContentAnalysis(word_count=500, keyword_density=2.0, readability_score=10.0),
        headings=HeadingsAnalysis(h1_count=1),
        images=ImagesAnalysis(image_count=5, alts_missing=1)
        # Add other fields as needed for rules
    )

@pytest.fixture
def sample_competitor_analyses():
    return [
        PageAnalysis(
            url="comp1.com",
            title=TitleAnalysis(length=55),
            content=ContentAnalysis(word_count=800, keyword_density=1.5, readability_score=9.0),
            headings=HeadingsAnalysis(h1_count=1),
            images=ImagesAnalysis(image_count=8, alts_missing=0)
        ),
        PageAnalysis(
            url="comp2.com",
            title=TitleAnalysis(length=60),
            content=ContentAnalysis(word_count=1200, keyword_density=2.5, readability_score=11.0),
            headings=HeadingsAnalysis(h1_count=1),
            images=ImagesAnalysis(image_count=12, alts_missing=2)
        )
    ]

def test_benchmark_analysis(analyzer_instance, sample_target_analysis, sample_competitor_analyses):
    """Test the benchmarking logic."""
    result_dict = analyzer_instance.benchmark_analysis(sample_target_analysis, sample_competitor_analyses)

    assert isinstance(result_dict, dict)
    assert 'benchmarks' in result_dict
    benchmarks = result_dict['benchmarks']
    assert 'title_length_avg' in benchmarks
    assert benchmarks['title_length_avg'] == pytest.approx(57.5) # (55+60)/2
    assert 'content_word_count_avg' in benchmarks
    assert benchmarks['content_word_count_avg'] == pytest.approx(1000.0) # (800+1200)/2
    assert 'content_readability_score_avg' in benchmarks
    assert benchmarks['content_readability_score_avg'] == pytest.approx(10.0) # (9+11)/2

def test_generate_recommendations(analyzer_instance, sample_target_analysis, sample_competitor_analyses):
    """Test the recommendation generation logic."""
    # First, get the analysis dict with benchmarks included
    analysis_with_benchmarks = analyzer_instance.benchmark_analysis(sample_target_analysis, sample_competitor_analyses)
    recommendations = analyzer_instance.generate_recommendations(analysis_with_benchmarks)

    assert isinstance(recommendations, list)
    # Check if specific recommendations are generated based on the sample data and rules
    recommendation_texts = [rec['text'] for rec in recommendations]
    assert any("Title length (40) is significantly shorter" in text for text in recommendation_texts)
    assert any("Word count (500) is significantly lower" in text for text in recommendation_texts)
    # Add more checks based on the rules you implemented


# --- Test Orchestrator (`analyze_page_with_benchmarks`) ---
# This is more of an integration test

@pytest.mark.asyncio
async def test_analyze_page_with_benchmarks_full_run(analyzer_instance):
    """Test the full run of analyze_page_with_benchmarks method."""
    # Mock target page response
    target_response = AsyncMock(spec=httpx.Response)
    target_response.status_code = 200
    target_response.headers = {'content-type': 'text/html'}
    target_response.text = PropertyMock(return_value="""
        <html>
            <head><title>Test Page Title</title></head>
            <body>
                <h1>Test H1</h1>
                <p>This is a test page with some content about SEO and optimization.</p>
            </body>
        </html>
    """)

    # Mock SERP response
    serp_response = AsyncMock(spec=httpx.Response)
    serp_response.status_code = 200
    serp_response.headers = {'content-type': 'application/json'}
    serp_response.json = AsyncMock(return_value={
        "search_metadata": {"status": "Success"},
        "organic_results": [
            {
                "link": "https://competitor1.com",
                "title": "Competitor 1 Title",
                "snippet": "Competitor 1 description"
            },
            {
                "link": "https://competitor2.com",
                "title": "Competitor 2 Title",
                "snippet": "Competitor 2 description"
            },
            {
                "link": "https://competitor3.com",
                "title": "Competitor 3 Title",
                "snippet": "Competitor 3 description"
            }
        ]
    })

    # Mock competitor responses
    competitor_response = AsyncMock(spec=httpx.Response)
    competitor_response.status_code = 200
    competitor_response.headers = {'content-type': 'text/html'}
    competitor_response.text = PropertyMock(return_value="""
        <html>
            <head><title>Competitor Page</title></head>
            <body>
                <h1>Competitor H1</h1>
                <p>This is a competitor page with relevant content about SEO and optimization.</p>
            </body>
        </html>
    """)

    # Create mock client
    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock()

    # Configure mock client responses
    async def mock_get(url, *args, **kwargs):
        if url == "https://example.com":
            return target_response
        elif "serpapi.com" in url:
            return serp_response
        else:
            return competitor_response

    mock_client.get = AsyncMock(side_effect=mock_get)

    # Run test
    with patch('httpx.AsyncClient', return_value=mock_client):
        final_result = await analyzer_instance.analyze_page_with_benchmarks(
            "https://example.com",
            "seo optimization",
            "us",
            max_competitors=3
        )

    # Assertions
    assert final_result['status'] == 'success'
    assert 'target_analysis' in final_result
    assert 'competitor_analyses' in final_result
    assert 'benchmarks' in final_result
    assert 'recommendations' in final_result

    # Check target analysis
    target_analysis = final_result['target_analysis']
    assert target_analysis['title_text'] == 'Test Page Title'
    assert target_analysis['h1_text'] == ['Test H1']
    assert target_analysis['word_count'] > 0

    # Check competitor analyses
    competitor_analyses = final_result['competitor_analyses']
    assert isinstance(competitor_analyses, list)
    assert len(competitor_analyses) > 0
    assert all(isinstance(analysis, dict) for analysis in competitor_analyses)

    # Check benchmarks
    benchmarks = final_result['benchmarks']
    assert isinstance(benchmarks, dict)
    assert 'title_length' in benchmarks
    assert 'meta_description_length' in benchmarks
    assert 'word_count' in benchmarks

    # Check recommendations
    assert isinstance(final_result['recommendations'], list)
    assert len(final_result['recommendations']) > 0

# --- Test Caching (Optional but Recommended) ---
# You would mock the Cache class methods (get, set) or file system operations

# TODO: Add tests for Cache interactions if desired
# TODO: Add tests for RateLimiter logic if desired 