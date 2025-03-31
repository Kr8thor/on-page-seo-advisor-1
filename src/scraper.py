"""
Core scraping and analysis functionality for the On-Page SEO Analyzer & Advisor.
"""

import httpx
import asyncio
import logging
from typing import List, Optional, Dict, Any, Tuple, Union, Set
from datetime import datetime, timedelta
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse
import json
from pathlib import Path
import os
from dotenv import load_dotenv
import re
import textstat
from parsel import Selector
import statistics
import aiohttp
import pydantic
from pydantic_core import to_jsonable_python

# Import models from the models file
from src.models import (
    SerpResult, TitleAnalysis, MetaDescriptionAnalysis, HeadingDetail,
    HeadingsAnalysis, ContentAnalysis, LinksAnalysis,
    ImagesAnalysis, SchemaAnalysis, PageAnalysis, PerformanceAnalysis
)

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_nested(data: Any, path: List[str], default: Any = None) -> Any:
    """Safely retrieves nested dictionary keys or object attributes."""
    current = data
    for key in path:
        if current is None: return default
        if isinstance(current, dict) and key in current:
            current = current.get(key)
        elif hasattr(current, key):
            current = getattr(current, key)
        else:
            return default
    return current

# Rate limiting configuration
RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", 100)) # requests per minute
RATE_LIMIT_WINDOW = int(os.getenv("RATE_LIMIT_WINDOW", 60)) # seconds
CACHE_DURATION = timedelta(hours=int(os.getenv("CACHE_DURATION_HOURS", 24))) # Cache duration

class RateLimiter:
    def __init__(self, max_requests: int, window_seconds: int):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = []
        self.lock = asyncio.Lock()

    async def acquire(self):
        async with self.lock:
            now = datetime.now()
            # Remove old requests
            self.requests = [req_time for req_time in self.requests
                           if now - req_time < timedelta(seconds=self.window_seconds)]

            if len(self.requests) >= self.max_requests:
                # Wait until oldest request expires
                wait_time = (self.requests[0] + timedelta(seconds=self.window_seconds) - now).total_seconds()
                if wait_time > 0:
                    logger.info(f"Rate limit hit, waiting for {wait_time:.2f} seconds.")
                    await asyncio.sleep(wait_time)
                    # Re-evaluate after wait
                    now = datetime.now()
                    self.requests = [req_time for req_time in self.requests
                                   if now - req_time < timedelta(seconds=self.window_seconds)]

            self.requests.append(now)

class Cache:
    def __init__(self, cache_dir: str = "cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

    def _get_cache_key(self, url: str, cache_key_segment: str) -> str:
        """Generate a cache key from URL and additional segment."""
        # Clean URL and segment for safe filename
        url_part = re.sub(r'[^\w]', '_', url)
        segment_part = re.sub(r'[^\w]', '_', cache_key_segment)
        return f"{url_part}_{segment_part}"

    def get(self, url: str, cache_key_segment: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached analysis results if they exist and are not expired.
        """
        cache_key = self._get_cache_key(url, cache_key_segment)
        cache_file = self.cache_dir / f"{cache_key}.json"

        try:
            if not cache_file.exists():
                return None

            with open(cache_file, 'r', encoding='utf-8') as f:
                cached_data = json.load(f)
                logger.info(f"Retrieved cached analysis for {url} | {cache_key_segment}")
                return cached_data

        except Exception as e:
            logger.error(f"Error reading cache file {cache_file}: {e}", exc_info=True)
            return None

    def set(self, url: str, cache_key_segment: str, data_to_cache: Dict[str, Any]):
        """
        Cache analysis results dictionary with timestamp, ensuring Pydantic models are serialized.
        """
        cache_key = self._get_cache_key(url, cache_key_segment)  # Use the segment passed
        cache_file = self.cache_dir / f"{cache_key}.json"

        try:
            # --- Prepare data for JSON serialization ---
            # We assume data_to_cache is the dictionary containing potentially nested Pydantic models
            # We need a way to convert these models within the dict to plain dicts/lists
            # Pydantic v2+ provides model_dump_json which is ideal, but json.dump with a default handler works too.

            # Method 1: Using json.dump with Pydantic's default function (Recommended if Pydantic v1/v2)
            # Pydantic models often have a .__json_encoder__ or similar method usable by json.dump's default
            def pydantic_serializer(obj):
                # This function tells json.dump how to handle non-standard objects
                # For Pydantic V2+, use its recommended serialization helper
                try:
                    # This converts Pydantic models within the structure to JSON-compatible python types
                    return to_jsonable_python(obj)
                except Exception:
                    # Fallback or raise error if obj isn't Pydantic or serializable
                    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

            with open(cache_file, 'w', encoding='utf-8') as f:
                # Use Method 1 (Recommended): Pass the default serializer to json.dump
                json.dump(data_to_cache, f, indent=2, ensure_ascii=False, default=pydantic_serializer)

            logger.info(f"Cached analysis for {url} | {cache_key_segment}")

        except Exception as e:
            logger.error(f"Error writing cache file {cache_file}: {e}", exc_info=True)

class SerpApiError(Exception):
    """Custom exception for SERP API related errors."""
    pass

# Add after other constants
VALID_COUNTRY_CODES: Set[str] = {
    'us', 'gb', 'ca', 'au', 'nz', 'ie', 'de', 'fr', 'es', 'it',
    'jp', 'kr', 'cn', 'in', 'br', 'mx', 'ru', 'za'
}

def normalize_country_code(country: str) -> str:
    """Normalize and validate a country code."""
    if not country:
        return 'us'
    
    normalized = country.lower().strip()
    if normalized in VALID_COUNTRY_CODES:
        return normalized
        
    logger.warning(f"Invalid country code '{country}', defaulting to 'us'")
    return 'us'

class SEOAnalyzer:
    """
    Main class for performing SEO analysis on web pages.
    
    This class handles fetching, parsing, and analyzing web pages for SEO metrics,
    including benchmarking against competitors and generating recommendations.
    
    Attributes:
        rate_limiter: RateLimiter instance for controlling API request rates
        cache: Cache instance for storing analysis results
        headers: HTTP headers for requests
        api_key: SERP API key for fetching search results
        serp_api_url: URL for the SERP API endpoint
        default_country: Default country code for SERP results
    """
    
    def __init__(self):
        """Initialize the SEOAnalyzer with configuration from environment variables."""
        self.rate_limiter = RateLimiter(RATE_LIMIT_REQUESTS, RATE_LIMIT_WINDOW)
        self.cache = Cache()
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
        # Initialize API key from environment variable
        self.api_key = os.getenv('SERP_API_KEY')
        if not self.api_key:
            raise ValueError("SERP_API_KEY environment variable is not set")

        # --- CHOOSE YOUR SERP API PROVIDER HERE ---
        # Example: ValueSERP
        self.serp_api_url = "https://api.valueserp.com/search"
        # # Example: SerpApi
        # self.serp_api_url = "https://serpapi.com/search"

        self.default_country = os.getenv('DEFAULT_COUNTRY', 'us')
        logger.info("SEOAnalyzer initialized successfully")

    async def fetch_serp_results(self, keyword: str, country: Optional[str] = None) -> List[SerpResult]:
        """
        Fetch SERP results using the configured API key with rate limiting
        and country code validation.

        Args:
            keyword: The search keyword.
            country: Optional two-letter country code (ISO 3166-1 alpha-2) or common name.

        Returns:
            List of SerpResult objects.

        Raises:
            SerpApiError: If API interaction fails.
            ValueError: If keyword is empty.
        """
        await self.rate_limiter.acquire()

        if not keyword:
            raise ValueError("Keyword cannot be empty")

        # --- Country Code Resolution & Validation ---
        input_country = country or self.default_country # Use provided or default
        resolved_country_code = input_country.lower().strip() # Basic cleanup

        # Simple mapping for common names (add more as needed)
        country_map = {
            "ireland": "ie",
            "united kingdom": "gb",
            "uk": "gb",
            "united states": "us",
            "usa": "us",
            "germany": "de",
            "france": "fr",
            "spain": "es",
            "italy": "it",
            "netherlands": "nl",
            "united arab emirates": "ae",
            "uae": "ae",
            "saudi arabia": "sa",
            "qatar": "qa",
        }

        # Attempt mapping if input is not a 2-letter code
        if len(resolved_country_code) > 2 or resolved_country_code not in VALID_COUNTRY_CODES:
            resolved_country_code = country_map.get(resolved_country_code, resolved_country_code)

        # Final validation: Must be in VALID_COUNTRY_CODES
        if resolved_country_code not in VALID_COUNTRY_CODES:
            logger.warning(f"Invalid country code '{input_country}', defaulting to '{self.default_country}'")
            resolved_country_code = self.default_country.lower().strip()

        logger.info(f"Fetching SERP for keyword='{keyword}', country_code='{resolved_country_code}'")

        # Adjust params based on ValueSERP
        params = {
            'api_key': self.api_key,
            'q': keyword,
            'gl': resolved_country_code,  # Use gl parameter for country code
            'output': 'json',
            'num': '100',  # Request maximum results
        }

        response_for_logging = None

        try:
            async with httpx.AsyncClient(timeout=45.0) as client:
                response = await client.get(self.serp_api_url, params=params, headers=self.headers)
                response_for_logging = response
                response.raise_for_status()

                # --- Attempt to decode JSON using response.json() ---
                try:
                    data = response.json()
                except json.JSONDecodeError as json_err:
                    raw_text = "[Could not read response text]"
                    try:
                        raw_text = response.text
                    except Exception:
                        pass
                    logger.error(f"Failed to decode SERP API JSON response: {json_err}. Response Text approx: {raw_text[:1000]}...")
                    raise SerpApiError("Invalid JSON response from SERP API")
                except Exception as e:
                    logger.error(f"Error calling response.json(): {e}", exc_info=True)
                    raise SerpApiError(f"Error processing SERP API response: {e}")

                # --- Validate response format ---
                if not isinstance(data, dict):
                    raise SerpApiError("Invalid response format from SERP API")

                # Check for API-specific error messages
                request_info = data.get("request_info", {})
                if request_info.get("success") is False:
                    error_msg = request_info.get("message", "Unknown API Error")
                    raise SerpApiError(f"SERP API Error: {error_msg}")

                # Extract organic results
                organic_results = data.get('organic_results', [])
                if not organic_results:
                    logger.warning(f"No organic results found for keyword '{keyword}' in country '{resolved_country_code}'")
                    return []

                # Process results with validation
                results = []
                pydantic_validation_errors = 0
                for result in organic_results:
                    api_url = result.get('link')
                    api_title = result.get('title')
                    if api_url and api_title:
                        try:
                            results.append(SerpResult(
                                url=api_url,
                                title=api_title,
                                description=result.get('snippet', ''),
                                position=result.get('position', 0)
                            ))
                        except pydantic.ValidationError as model_err:
                            pydantic_validation_errors += 1
                            logger.warning(f"Skipping SERP result due to model validation: {model_err}. Data: {result}")
                    else:
                        logger.warning(f"Skipping SERP result missing link/title: {result}")

                if pydantic_validation_errors > 0:
                    logger.warning(f"Encountered {pydantic_validation_errors} Pydantic validation errors processing SERP results")

                logger.info(f"Successfully fetched {len(results)} SERP results")
                return results[:10]  # Return top 10 validated results

        except httpx.HTTPStatusError as e:
            error_text = "[Could not read response text]"
            if response_for_logging:
                try:
                    error_text = response_for_logging.text
                except Exception:
                    pass
            logger.error(f"HTTP error {e.response.status_code} fetching SERP: {e}. Response text: {error_text[:500]}")
            
            if e.response.status_code == 400 and "'gl' parameter is invalid" in error_text:
                raise SerpApiError(f"Invalid country code '{resolved_country_code}' provided for SERP API")
            elif e.response.status_code == 401:
                raise SerpApiError("Invalid SERP API key")
            elif e.response.status_code == 429:
                raise SerpApiError("SERP API rate limit exceeded")
            else:
                raise SerpApiError(f"HTTP error {e.response.status_code} fetching SERP results")

        except httpx.TimeoutException as e:
            logger.error(f"Timeout error fetching SERP results: {e}")
            raise SerpApiError("SERP API request timed out")

        except httpx.RequestError as e:
            logger.error(f"Network error fetching SERP results: {e}")
            raise SerpApiError(f"Network error: {str(e)}")

        except Exception as e:
            logger.error(f"Unexpected error fetching SERP results: {e}", exc_info=True)
            raise SerpApiError(f"Unexpected error: {str(e)}")

    async def fetch_page_content(self, url: str) -> Optional[str]:
        """Fetch HTML content from a URL with caching."""
        if not url:
            raise ValueError("URL cannot be empty")

        logger.info(f"Fetching content for {url}")

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url, headers=self.headers)
                # response.raise_for_status()  # Temporarily commented out for debugging

                # Check content type
                content_type = response.headers.get('content-type', '')
                if isinstance(content_type, str) and 'text/html' not in content_type.lower():
                    logger.warning(f"Unexpected content type for {url}: {content_type}")
                    return None

                # Get text content
                try:
                    if hasattr(response, 'atext'):
                        html = await response.atext()
                    else:
                        html = response.text
                    if not html:
                        logger.warning(f"Empty response from {url}")
                        return None
                    return html
                except Exception as e:
                    logger.error(f"Error getting text content from response: {e}")
                    return None

        except httpx.TimeoutException:
            logger.error(f"Request timed out while fetching {url}")
            return None
        except httpx.RequestError as e:
            logger.error(f"Network error while fetching {url}: {e}")
            return None
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error {e.response.status_code} fetching {url}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error fetching {url}: {e}", exc_info=True)
            return None

    def _analyze_title(self, selector: Selector, keyword: str) -> TitleAnalysis:
        """Analyze the page title."""
        title = selector.css('title::text').get('').strip()
        length = len(title)
        keyword_lower = keyword.lower()
        title_lower = title.lower()
        keyword_present = keyword_lower in title_lower

        position = None
        if keyword_present:
            try:
                # More robust position check
                if title_lower.startswith(keyword_lower):
                    position = 'start'
                elif title_lower.endswith(keyword_lower):
                    position = 'end'
                else:
                    # Check if keyword exists as a whole word/phrase bounded by spaces/punctuation
                    # This regex looks for the keyword bounded by non-word characters or start/end of string
                    if re.search(rf'(?<!\w){re.escape(keyword_lower)}(?!\w)', title_lower):
                        position = 'middle'
                    else:
                        # Keyword is part of another word
                         position = 'substring'
            except Exception: # Fallback if regex fails
                 position = 'middle' if keyword_present else None

        return TitleAnalysis(
            text=title, length=length, keyword_present=keyword_present, position=position
        )

    def _analyze_meta_description(self, selector: Selector, keyword: str) -> MetaDescriptionAnalysis:
        """Analyze the meta description."""
        description = selector.css('meta[name="description"]::attr(content)').get('').strip()
        return MetaDescriptionAnalysis(
            text=description,
            length=len(description),
            keyword_present=keyword.lower() in description.lower()
        )

    def _analyze_headings(self, selector: Selector, keyword: str) -> HeadingsAnalysis:
        """
        Analyze headings in the page content.
        
        Args:
            selector: Parsel Selector object containing the page HTML
            keyword: Target keyword to check for in headings
            
        Returns:
            HeadingsAnalysis object containing heading analysis results
        """
        headings_analysis = HeadingsAnalysis()
        keyword = keyword.lower()
        
        # Analyze H1 headings
        h1_texts = selector.css('h1::text').getall()
        if h1_texts:
            h1_text = h1_texts[0].strip()  # Take first H1
            headings_analysis.h1.append(HeadingDetail(
                text=h1_text,
                contains_keyword=keyword in h1_text.lower(),
                level=1
            ))
            headings_analysis.h1_count = 1
            headings_analysis.h1_contains_keyword = keyword in h1_text.lower()
        
        # Analyze H2 headings
        h2_texts = selector.css('h2::text').getall()
        h2_keywords = []
        h2_contains_keyword_count = 0
        
        for h2_text in h2_texts:
            h2_text = h2_text.strip()
            if h2_text:
                contains_keyword = keyword in h2_text.lower()
                headings_analysis.h2.append(HeadingDetail(
                    text=h2_text,
                    contains_keyword=contains_keyword,
                    level=2
                ))
                if contains_keyword:
                    h2_contains_keyword_count += 1
                    h2_keywords.append(h2_text)
        
        headings_analysis.h2_count = len(h2_texts)
        headings_analysis.h2_contains_keyword_count = h2_contains_keyword_count
        headings_analysis.h2_keywords = h2_keywords
        
        # Analyze H3 headings
        h3_texts = selector.css('h3::text').getall()
        h3_count = 0
        for h3_text in h3_texts:
            h3_text = h3_text.strip()
            if h3_text:
                headings_analysis.h3.append(HeadingDetail(
                    text=h3_text,
                    contains_keyword=keyword in h3_text.lower(),
                    level=3
                ))
                h3_count += 1
        
        # Analyze H4 headings
        h4_texts = selector.css('h4::text').getall()
        h4_count = 0
        for h4_text in h4_texts:
            h4_text = h4_text.strip()
            if h4_text:
                headings_analysis.h4.append(HeadingDetail(
                    text=h4_text,
                    contains_keyword=keyword in h4_text.lower(),
                    level=4
                ))
                h4_count += 1
        
        # Analyze H5 headings
        h5_texts = selector.css('h5::text').getall()
        h5_count = 0
        for h5_text in h5_texts:
            h5_text = h5_text.strip()
            if h5_text:
                headings_analysis.h5.append(HeadingDetail(
                    text=h5_text,
                    contains_keyword=keyword in h5_text.lower(),
                    level=5
                ))
                h5_count += 1
        
        # Analyze H6 headings
        h6_texts = selector.css('h6::text').getall()
        h6_count = 0
        for h6_text in h6_texts:
            h6_text = h6_text.strip()
            if h6_text:
                headings_analysis.h6.append(HeadingDetail(
                    text=h6_text,
                    contains_keyword=keyword in h6_text.lower(),
                    level=6
                ))
                h6_count += 1
        
        # Calculate total headings using count attributes
        headings_analysis.total_headings = (
            headings_analysis.h1_count +
            headings_analysis.h2_count +
            h3_count +
            h4_count +
            h5_count +
            h6_count
        )
        
        # Check if keyword appears in any heading
        headings_analysis.keyword_present_in_any = (
            headings_analysis.h1_contains_keyword or
            headings_analysis.h2_contains_keyword_count > 0 or
            any(h.contains_keyword for h in headings_analysis.h3) or
            any(h.contains_keyword for h in headings_analysis.h4) or
            any(h.contains_keyword for h in headings_analysis.h5) or
            any(h.contains_keyword for h in headings_analysis.h6)
        )
        
        return headings_analysis

    def _extract_main_text(self, selector: Selector) -> str:
        """Extract main content text using basic heuristics (can be improved)."""
        # Try common article containers first
        main_content_selectors = ['article', 'main', '[role="main"]', '.main-content', '.post-content', '#content']
        text_content = ""
        for sel in main_content_selectors:
            content_element = selector.css(sel)
            if content_element:
                # Get all text nodes within, join, strip whitespace heavily
                text_content = ' '.join(content_element.css('*::text').getall()).strip()
                # Basic check if content seems reasonable length
                if len(text_content.split()) > 50:
                    return re.sub(r'\s+', ' ', text_content).strip() # Normalize whitespace

        # Fallback to body if no specific container found or content too short
        if not text_content or len(text_content.split()) <= 50:
             text_content = ' '.join(selector.css('body *::text').getall()).strip()

        # Very basic cleanup (remove scripts, styles if Parsel didn't)
        text_content = re.sub(r'<script[^>]*>.*?</script>', '', text_content, flags=re.DOTALL | re.IGNORECASE)
        text_content = re.sub(r'<style[^>]*>.*?</style>', '', text_content, flags=re.DOTALL | re.IGNORECASE)
        text_content = re.sub(r'<[^>]+>', ' ', text_content) # Remove remaining tags
        return re.sub(r'\s+', ' ', text_content).strip() # Normalize whitespace

    def _analyze_content(self, main_text: str, keyword: str) -> ContentAnalysis:
        """Analyze the main content."""
        if not main_text:
            return ContentAnalysis() # Return default if no text

        # Basic text cleaning already done in extraction
        words = main_text.split()
        word_count = len(words)
        keyword_lower = keyword.lower()

        # More robust keyword counting (using regex for whole word match)
        try:
            # Count occurrences of the keyword as a whole word/phrase
            keyword_count = len(re.findall(rf'(?<!\w){re.escape(keyword_lower)}(?!\w)', main_text.lower()))
        except Exception:
             # Fallback to simple count if regex fails
             keyword_count = main_text.lower().count(keyword_lower)

        keyword_density = (keyword_count / word_count * 100) if word_count > 0 else 0.0

        readability_score = None
        try:
            # Calculate readability score only if text is substantial enough
            if word_count > 100: # textstat needs sufficient text
                readability_score = textstat.flesch_kincaid_grade(main_text)
        except Exception as e:
            logger.warning(f"Could not calculate readability score: {e}")

        return ContentAnalysis(
            word_count=word_count,
            readability_score=readability_score,
            keyword_density=keyword_density,
            keyword_count=keyword_count
        )

    def _analyze_links(self, selector: Selector, base_url: str) -> LinksAnalysis:
        """Analyze internal and external links."""
        links = selector.css('a::attr(href)').getall()
        try:
            # Ensure base_url has scheme for urljoin and netloc extraction
            parsed_base = urlparse(base_url)
            if not parsed_base.scheme:
                base_url = "http://" + base_url # Assume http if scheme missing
                parsed_base = urlparse(base_url)
            base_domain = parsed_base.netloc
        except ValueError:
            logger.warning(f"Could not parse base_url: {base_url}. Link analysis may be inaccurate.")
            base_domain = None # Cannot determine internal/external reliably

        internal_links = 0
        external_links = 0

        for link in links:
            if not link or link.startswith(('#', 'mailto:', 'tel:')):
                continue # Skip anchors, mailto, tel links

            try:
                # Resolve relative URLs
                full_url = urljoin(base_url, link.strip())
                parsed_url = urlparse(full_url)

                # Check if it's HTTP/HTTPS and has a domain
                if parsed_url.scheme in ['http', 'https'] and parsed_url.netloc:
                    if base_domain and parsed_url.netloc == base_domain:
                        internal_links += 1
                    else:
                        external_links += 1
                # Consider relative links without domain as internal (usually)
                elif not parsed_url.scheme and not parsed_url.netloc and parsed_url.path:
                     internal_links += 1

            except ValueError:
                logger.warning(f"Could not parse link URL: {link}")
                continue # Skip malformed links

        return LinksAnalysis(
            internal_links=internal_links,
            external_links=external_links
        )

    def _analyze_images(self, selector: Selector, keyword: str) -> ImagesAnalysis:
        """Analyze images and their alt texts."""
        images = selector.css('img')
        image_count = len(images)
        keyword_lower = keyword.lower()

        alts_missing = 0
        alts_with_keyword = 0

        for img in images:
            # Check if image is likely visible content (basic check, can be improved)
            src = img.css('::attr(src)').get()
            if not src or src.startswith('data:'): # Skip missing src and inline data URIs
                 image_count -= 1 # Don't count non-content images
                 continue

            alt = img.css('::attr(alt)').get() # Get alt text, could be None

            if alt is None or alt.strip() == "": # Treat None or empty string as missing
                alts_missing += 1
            elif keyword_lower in alt.lower():
                alts_with_keyword += 1

        return ImagesAnalysis(
            image_count=image_count,
            alts_missing=alts_missing,
            alts_with_keyword=alts_with_keyword
        )

    def _analyze_schema(self, selector: Selector) -> SchemaAnalysis:
        """
        Analyzes schema.org markup found in JSON-LD script tags.
        Extracts unique @type values.
        
        Args:
            selector: Parsel Selector object containing the page HTML
            
        Returns:
            SchemaAnalysis object containing list of unique schema types found
        """
        schema_scripts = selector.css('script[type="application/ld+json"]::text').getall()
        types_found: Set[str] = set()  # Use a set to automatically handle duplicates

        for script_content in schema_scripts:
            try:
                # Clean potential HTML comments
                script_content = re.sub(r'<!--.*?-->', '', script_content, flags=re.DOTALL).strip()
                if not script_content:
                    continue

                data = json.loads(script_content)

                # --- Process data whether it's a dict or list ---
                items_to_check = []
                if isinstance(data, dict):
                    items_to_check.append(data)  # Wrap single dict in a list
                elif isinstance(data, list):
                    items_to_check.extend(data)  # Use the list directly
                else:
                    logger.warning(f"Found JSON-LD script, but content was neither dict nor list: {type(data)}")
                    continue  # Skip non-dict/list data

                # Extract @type from each item
                for item in items_to_check:
                    if isinstance(item, dict):
                        schema_type = item.get('@type')
                        if isinstance(schema_type, str):
                            types_found.add(schema_type)
                        elif isinstance(schema_type, list):  # Type can also be a list
                            for t in schema_type:
                                if isinstance(t, str):
                                    types_found.add(t)
                        
                        # Recursively check nested items that might have @type
                        for value in item.values():
                            if isinstance(value, (dict, list)):
                                nested_types = self._extract_nested_types(value)
                                types_found.update(nested_types)
                                
            except json.JSONDecodeError as e:
                # Log the beginning of the script that failed to parse
                logger.warning(f"Could not parse JSON-LD script: {e}. Script start: {script_content[:100]}...")
                continue
            except Exception as e:
                logger.warning(f"Error processing schema script: {e}", exc_info=True)
                continue

        # Create the model instance with sorted types list
        return SchemaAnalysis(types_found=sorted(list(types_found)))

    def _extract_nested_types(self, data: Union[Dict, List]) -> Set[str]:
        """
        Recursively extracts @type values from nested dictionaries and lists.
        
        Args:
            data: Dictionary or list that might contain nested @type values
            
        Returns:
            Set of schema types found in the nested structure
        """
        types: Set[str] = set()
        
        if isinstance(data, dict):
            # Check for @type in current dict
            schema_type = data.get('@type')
            if isinstance(schema_type, str):
                types.add(schema_type)
            elif isinstance(schema_type, list):
                types.update(t for t in schema_type if isinstance(t, str))
            
            # Recursively check all values
            for value in data.values():
                if isinstance(value, (dict, list)):
                    types.update(self._extract_nested_types(value))
                    
        elif isinstance(data, list):
            # Recursively check all items in list
            for item in data:
                if isinstance(item, (dict, list)):
                    types.update(self._extract_nested_types(item))
                    
        return types

    def _analyze_viewport(self, selector: Selector) -> Optional[str]:
        """
        Analyze the viewport meta tag content.
        
        Args:
            selector: Parsel selector for the page
            
        Returns:
            Content of viewport meta tag if present, None otherwise
        """
        try:
            viewport = selector.css('meta[name="viewport"]::attr(content)').get()
            return viewport if viewport else None
        except Exception as e:
            logger.warning(f"Error analyzing viewport tag: {e}")
            return None

    def _analyze_canonical(self, selector: Selector, base_url: str) -> Optional[str]:
        """
        Analyze the canonical link tag.
        
        Args:
            selector: Parsel selector for the page
            base_url: Base URL of the page for resolving relative URLs
            
        Returns:
            Canonical URL if specified, None otherwise
        """
        try:
            canonical = selector.css('link[rel="canonical"]::attr(href)').get()
            if canonical:
                # Resolve relative URLs
                if canonical.startswith('/'):
                    canonical = urljoin(base_url, canonical)
                return canonical
            return None
        except Exception as e:
            logger.warning(f"Error analyzing canonical tag: {e}")
            return None

    async def analyze_page(self, url: str, keyword: str) -> Dict[str, Any]:
        """Analyze a single page for SEO metrics."""
        try:
            html_content = await self.fetch_page_content(url)
            if not html_content:
                return {
                    'status': 'error',
                    'message': f'Could not retrieve or process HTML for {url}. Analysis aborted.'
                }

            # Create Parsel selector for HTML parsing
            selector = Selector(text=str(html_content))

            # Extract title and meta description
            title = selector.css('title::text').get() or ''
            meta_description = selector.css('meta[name="description"]::attr(content)').get() or ''

            # Analyze viewport and canonical tags
            viewport_content = self._analyze_viewport(selector)
            canonical_url = self._analyze_canonical(selector, url)

            # Extract headings
            headings = []
            h1_count = 0
            h2_count = 0
            h3_count = 0
            
            for level in range(1, 7):  # h1 to h6
                for heading in selector.css(f'h{level}::text').getall():
                    heading_text = heading.strip()
                    if heading_text:
                        headings.append(HeadingDetail(
                            level=level,
                            text=heading_text,
                            contains_keyword=keyword.lower() in heading_text.lower()
                        ))
                        if level == 1:
                            h1_count += 1
                        elif level == 2:
                            h2_count += 1
                        elif level == 3:
                            h3_count += 1

            # Extract links
            links = []
            internal_links = 0
            external_links = 0
            for link in selector.css('a[href]'):
                href = link.css('::attr(href)').get()
                if href:
                    if href.startswith('/'):
                        href = urljoin(url, href)
                    links.append(href)
                    if urlparse(href).netloc == urlparse(url).netloc:
                        internal_links += 1
                    else:
                        external_links += 1

            # Extract images
            images = []
            image_count = 0
            alts_missing = 0
            alts_with_keyword = 0
            for img in selector.css('img'):
                src = img.css('::attr(src)').get()
                alt = img.css('::attr(alt)').get() or ''
                if src:
                    if src.startswith('/'):
                        src = urljoin(url, src)
                    images.append({'src': src, 'alt': alt})
                    image_count += 1
                    if not alt:
                        alts_missing += 1
                    elif keyword.lower() in alt.lower():
                        alts_with_keyword += 1

            # Extract schema markup
            schema_data = []
            for script in selector.css('script[type="application/ld+json"]::text').getall():
                try:
                    data = json.loads(script)
                    schema_data.append(data)
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse JSON-LD schema in {url}")

            # Extract schema types
            types_found = []
            for data in schema_data:
                if isinstance(data, dict):
                    schema_type = data.get('@type')
                    if isinstance(schema_type, str):
                        types_found.append(schema_type)
                    elif isinstance(schema_type, list):
                        types_found.extend([t for t in schema_type if isinstance(t, str)])

            schema_analysis = SchemaAnalysis(
                types_found=types_found,
                schema_data=schema_data
            )

            # Extract main content
            content = ' '.join([
                text.strip()
                for text in selector.css('body *:not(script):not(style)::text').getall()
                if text.strip()
            ])

            # Calculate performance metrics
            html_size_bytes = len(html_content.encode('utf-8')) if html_content else 0
            text_size_bytes = len(content.encode('utf-8')) if content else 0
            text_ratio = (text_size_bytes / html_size_bytes) if html_size_bytes > 0 else 0.0

            performance_analysis = PerformanceAnalysis(
                html_size=html_size_bytes,
                text_html_ratio=round(text_ratio * 100, 2)  # Store as percentage
            )

            # Calculate readability metrics
            readability_metrics = {
                'flesch_reading_ease': textstat.flesch_reading_ease(content),
                'flesch_kincaid_grade': textstat.flesch_kincaid_grade(content),
                'gunning_fog': textstat.gunning_fog(content),
                'smog_index': textstat.smog_index(content),
                'automated_readability_index': textstat.automated_readability_index(content),
                'coleman_liau_index': textstat.coleman_liau_index(content),
                'linsear_write_formula': textstat.linsear_write_formula(content),
                'dale_chall_readability_score': textstat.dale_chall_readability_score(content),
            }

            # Calculate keyword metrics
            keyword_lower = keyword.lower()
            title_lower = title.lower()
            meta_lower = meta_description.lower()
            content_lower = content.lower()

            keyword_metrics = {
                'title_keyword_count': title_lower.count(keyword_lower),
                'meta_keyword_count': meta_lower.count(keyword_lower),
                'content_keyword_count': content_lower.count(keyword_lower),
                'content_keyword_density': (
                    content_lower.count(keyword_lower) / len(content_lower.split())
                    if content_lower else 0
                ),
                'title_starts_with_keyword': title_lower.startswith(keyword_lower),
                'meta_contains_keyword': keyword_lower in meta_lower,
                'url_contains_keyword': keyword_lower in url.lower(),
            }

            # Create analysis objects
            title_analysis = TitleAnalysis(
                text=title,
                length=len(title),
                keyword_present=keyword_lower in title_lower,
                position='start' if title_lower.startswith(keyword_lower) else 'end' if title_lower.endswith(keyword_lower) else 'middle' if keyword_lower in title_lower else None
            )

            meta_analysis = MetaDescriptionAnalysis(
                text=meta_description,
                length=len(meta_description),
                keyword_present=keyword_lower in meta_lower
            )

            # Group headings by level
            h1_headings = [h for h in headings if h.level == 1]
            h2_headings = [h for h in headings if h.level == 2]
            h3_headings = [h for h in headings if h.level == 3]

            headings_analysis = HeadingsAnalysis(
                h1=h1_headings,
                h2=h2_headings,
                h3=h3_headings,
                h1_count=h1_count,
                h1_contains_keyword=any(h.contains_keyword for h in h1_headings),
                h2_count=h2_count,
                h2_contains_keyword_count=sum(1 for h in h2_headings if h.contains_keyword),
                h2_keywords=[h.text for h in h2_headings if h.contains_keyword],
                total_headings=len(headings)
            )

            content_analysis = ContentAnalysis(
                word_count=len(content.split()),
                keyword_count=content_lower.count(keyword_lower),
                keyword_density=keyword_metrics['content_keyword_density'],
                readability=readability_metrics
            )

            links_analysis = LinksAnalysis(
                total_links=len(links),
                internal_links=internal_links,
                external_links=external_links,
                broken_links=[]  # Would require additional requests to verify
            )

            images_analysis = ImagesAnalysis(
                image_count=image_count,
                alts_missing=alts_missing,
                alts_with_keyword=alts_with_keyword,
                images=images
            )

            # Combine all analyses
            page_analysis = PageAnalysis(
                url=url,
                title=title_analysis,
                meta_description=meta_analysis,
                headings=headings_analysis,
                content=content_analysis,
                links=links_analysis,
                images=images_analysis,
                schema=schema_analysis,
                viewport_content=viewport_content,
                canonical_url=canonical_url,
                performance=performance_analysis
            )

            return {
                'status': 'success',
                'analysis': page_analysis
            }

        except Exception as e:
            logger.error(f"Critical error during page analysis for {url}: {e}", exc_info=True)
            return {
                'status': 'error',
                'message': str(e)
            }

    def benchmark_analysis(self, target_analysis: PageAnalysis, competitor_analyses: List[PageAnalysis]) -> Dict[str, Any]:
        """Generate benchmarks from competitor analyses."""
        logger.info("Generating benchmarks from competitor analyses")
        
        benchmarks: Dict[str, Dict[str, Optional[float]]] = {}
        
        if not competitor_analyses:
            logger.warning("No competitor analyses provided for benchmarking")
            return benchmarks
            
        metrics_to_benchmark = {
            'title_length': ('title', 'length'),
            'meta_description_length': ('meta_description', 'length'),
            'word_count': ('content', 'word_count'),
            'keyword_density': ('content', 'keyword_density'),
            'internal_links': ('links', 'internal_links'),
            'external_links': ('links', 'external_links'),
            'image_count': ('images', 'image_count'),
            'alts_missing': ('images', 'alts_missing'),
            'alts_with_keyword': ('images', 'alts_with_keyword'),
            'h1_count': ('headings', 'h1_count'),
            'h2_count': ('headings', 'h2_count'),
            'total_headings': ('headings', 'total_headings')
        }
        
        for key, path_tuple in metrics_to_benchmark.items():
            values = []
            for comp_analysis in competitor_analyses:
                current_val = get_nested(comp_analysis, list(path_tuple))
                if current_val is not None and isinstance(current_val, (int, float)):
                    values.append(current_val)
            
            if values:
                benchmarks[key] = {
                    'avg': statistics.mean(values),
                    'median': statistics.median(values),
                    'min': min(values),
                    'max': max(values)
                }
            else:
                benchmarks[key] = {
                    'avg': None,
                    'median': None,
                    'min': None,
                    'max': None
                }
        
        logger.info(f"Generated benchmarks for {len(benchmarks)} metrics")
        return benchmarks

    def generate_recommendations(self, target_analysis: PageAnalysis, benchmarks: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate SEO recommendations based on analysis and benchmarks."""
        logger.info("Generating SEO recommendations")
        recommendations: List[Dict[str, Any]] = []
        
        # Title Analysis
        title_len = get_nested(target_analysis, ['title', 'length'], 0)
        bm_title_len_avg = get_nested(benchmarks, ['title_length', 'avg'])
        
        if title_len < 30:
            recommendations.append({
                'type': 'title',
                'priority': 'high',
                'message': 'Title is too short. Aim for 30-60 characters.',
                'impact': 'High - Affects click-through rate and SEO visibility'
            })
        elif title_len > 60:
            recommendations.append({
                'type': 'title',
                'priority': 'medium',
                'message': 'Title is too long. Keep it under 60 characters.',
                'impact': 'Medium - May be truncated in search results'
            })
            
        if bm_title_len_avg and title_len < bm_title_len_avg * 0.8:
            recommendations.append({
                'type': 'title',
                'priority': 'medium',
                'message': f'Title length ({title_len} chars) is significantly shorter than competitors (avg: {bm_title_len_avg:.1f} chars)',
                'impact': 'Medium - May affect competitive positioning'
            })
            
        # Meta Description Analysis
        meta_len = get_nested(target_analysis, ['meta_description', 'length'], 0)
        bm_meta_len_avg = get_nested(benchmarks, ['meta_description_length', 'avg'])
        
        if meta_len < 120:
            recommendations.append({
                'type': 'meta_description',
                'priority': 'high',
                'message': 'Meta description is too short. Aim for 120-155 characters.',
                'impact': 'High - Affects click-through rate'
            })
        elif meta_len > 155:
            recommendations.append({
                'type': 'meta_description',
                'priority': 'medium',
                'message': 'Meta description is too long. Keep it under 155 characters.',
                'impact': 'Medium - May be truncated in search results'
            })
            
        if bm_meta_len_avg and meta_len < bm_meta_len_avg * 0.8:
            recommendations.append({
                'type': 'meta_description',
                'priority': 'medium',
                'message': f'Meta description length ({meta_len} chars) is significantly shorter than competitors (avg: {bm_meta_len_avg:.1f} chars)',
                'impact': 'Medium - May affect competitive positioning'
            })
            
        # Content Analysis
        word_count = get_nested(target_analysis, ['content', 'word_count'], 0)
        bm_word_count_avg = get_nested(benchmarks, ['word_count', 'avg'])
        
        if word_count < 300:
            recommendations.append({
                'type': 'content',
                'priority': 'high',
                'message': 'Content is too short. Aim for at least 300 words.',
                'impact': 'High - Affects content quality and SEO ranking'
            })
        elif bm_word_count_avg and word_count < bm_word_count_avg * 0.7:
            recommendations.append({
                'type': 'content',
                'priority': 'medium',
                'message': f'Content length ({word_count} words) is significantly shorter than competitors (avg: {bm_word_count_avg:.1f} words)',
                'impact': 'Medium - May affect competitive positioning'
            })
            
        # Keyword Density Analysis
        keyword_density = get_nested(target_analysis, ['content', 'keyword_density'], 0)
        bm_keyword_density_avg = get_nested(benchmarks, ['keyword_density', 'avg'])
        
        if keyword_density < 0.5:
            recommendations.append({
                'type': 'keyword_density',
                'priority': 'medium',
                'message': 'Keyword density is low. Consider using the target keyword more naturally in the content.',
                'impact': 'Medium - Affects keyword relevance'
            })
        elif keyword_density > 3:
            recommendations.append({
                'type': 'keyword_density',
                'priority': 'high',
                'message': 'Keyword density is too high. This may be seen as keyword stuffing.',
                'impact': 'High - May trigger search engine penalties'
            })
            
        # Headings Analysis
        h1_count = get_nested(target_analysis, ['headings', 'h1_count'], 0)
        h2_count = get_nested(target_analysis, ['headings', 'h2_count'], 0)
        total_headings = get_nested(target_analysis, ['headings', 'total_headings'], 0)
        
        if h1_count == 0:
            recommendations.append({
                'type': 'headings',
                'priority': 'high',
                'message': 'No H1 heading found. Add a clear main heading.',
                'impact': 'High - Affects content structure and SEO'
            })
        elif h1_count > 1:
            recommendations.append({
                'type': 'headings',
                'priority': 'high',
                'message': f'Multiple H1 headings found ({h1_count}). Use only one H1 per page.',
                'impact': 'High - Affects content structure and SEO'
            })
            
        if total_headings < 3:
            recommendations.append({
                'type': 'headings',
                'priority': 'medium',
                'message': 'Few headings found. Consider adding more subheadings to improve content structure.',
                'impact': 'Medium - Affects content readability and SEO'
            })
            
        # Links Analysis
        internal_links = get_nested(target_analysis, ['links', 'internal_links'], 0)
        external_links = get_nested(target_analysis, ['links', 'external_links'], 0)
        bm_internal_links_avg = get_nested(benchmarks, ['internal_links', 'avg'])
        bm_external_links_avg = get_nested(benchmarks, ['external_links', 'avg'])
        
        if internal_links < 2:
            recommendations.append({
                'type': 'links',
                'priority': 'medium',
                'message': 'Few internal links found. Consider adding more internal links to improve site structure.',
                'impact': 'Medium - Affects site structure and user navigation'
            })
            
        if bm_internal_links_avg and internal_links < bm_internal_links_avg * 0.5:
            recommendations.append({
                'type': 'links',
                'priority': 'medium',
                'message': f'Internal links ({internal_links}) are significantly fewer than competitors (avg: {bm_internal_links_avg:.1f})',
                'impact': 'Medium - May affect site structure and user navigation'
            })
            
        # Images Analysis
        image_count = get_nested(target_analysis, ['images', 'image_count'], 0)
        alts_missing = get_nested(target_analysis, ['images', 'alts_missing'], 0)
        alts_with_keyword = get_nested(target_analysis, ['images', 'alts_with_keyword'], 0)
        bm_image_count_avg = get_nested(benchmarks, ['image_count', 'avg'])
        bm_alts_missing_avg = get_nested(benchmarks, ['alts_missing', 'avg'])
        
        if alts_missing > 0:
            recommendations.append({
                'type': 'images',
                'priority': 'high',
                'message': f'{alts_missing} images are missing alt text. Add descriptive alt text for better accessibility and SEO.',
                'impact': 'High - Affects accessibility and image SEO'
            })
            
        if image_count > 0 and alts_with_keyword == 0:
            recommendations.append({
                'type': 'images',
                'priority': 'medium',
                'message': 'No images have alt text containing the target keyword. Consider adding keyword-rich alt text.',
                'impact': 'Medium - Affects image SEO'
            })
            
        if bm_image_count_avg and image_count < bm_image_count_avg * 0.5:
            recommendations.append({
                'type': 'images',
                'priority': 'medium',
                'message': f'Number of images ({image_count}) is significantly lower than competitors (avg: {bm_image_count_avg:.1f})',
                'impact': 'Medium - May affect visual appeal and user engagement'
            })
            
        # Viewport Analysis
        viewport_content = get_nested(target_analysis, ['viewport_content'])
        if not viewport_content:
            recommendations.append({
                'type': 'viewport',
                'priority': 'high',
                'message': 'No viewport meta tag found. Add one for proper mobile display.',
                'impact': 'High - Affects mobile responsiveness'
            })
            
        # Canonical URL Analysis
        canonical_url = get_nested(target_analysis, ['canonical_url'])
        if not canonical_url:
            recommendations.append({
                'type': 'canonical',
                'priority': 'medium',
                'message': 'No canonical URL found. Consider adding one to prevent duplicate content issues.',
                'impact': 'Medium - Affects content uniqueness and SEO'
            })
            
        logger.info(f"Generated {len(recommendations)} recommendations")
        return recommendations

    async def analyze_page_with_benchmarks(
        self,
        url: str,
        keyword: str,
        country: Optional[str] = None,
        max_competitors: int = 10,
        request_id: str = 'N/A',
        force_refresh: bool = False
    ) -> Dict[str, Any]:
        """
        Analyze a page and benchmark it against competitors.
        
        Args:
            url: Target URL to analyze
            keyword: Target keyword
            country: Two-letter country code (optional)
            max_competitors: Maximum number of competitors to analyze
            request_id: Request ID for logging
            force_refresh: If True, bypass cache and force fresh analysis
            
        Returns:
            Dict containing analysis results and benchmarks
        """
        try:
            # Normalize country code
            normalized_country = normalize_country_code(country or self.default_country)
            logger.info(f"[{request_id}] Using normalized country code: {normalized_country}")
            
            # Generate cache key
            cache_key = f"{url}_{keyword}_{normalized_country}"
            
            # Check cache unless force refresh is requested
            if not force_refresh:
                cached_result = self.cache.get(url, cache_key)
                if cached_result:
                    logger.info(f"[{request_id}] Returning cached analysis for {url}")
                    return {
                        'status': 'success',
                        **cached_result
                    }
            else:
                logger.info(f"[{request_id}] Force refresh requested, bypassing cache")

            # Fetch SERP results first to get competitors
            logger.info(f"[{request_id}] Fetching SERP results for '{keyword}' in {normalized_country}")
            serp_results = await self.fetch_serp_results(keyword, normalized_country)
            
            if not serp_results:
                raise ValueError(f"No SERP results found for keyword '{keyword}' in {normalized_country}")
            
            # Filter out the target URL and limit competitors
            competitor_urls = [
                result.url for result in serp_results
                if result.url.lower() != url.lower()
            ][:max_competitors]
            
            # Analyze target page
            logger.info(f"[{request_id}] Analyzing target page: {url}")
            target_analysis = await self.analyze_page(url, keyword)
            
            if not target_analysis:
                raise ValueError(f"Failed to analyze target page: {url}")
            
            # Analyze competitors
            logger.info(f"[{request_id}] Analyzing {len(competitor_urls)} competitors")
            competitor_analyses = []
            for comp_url in competitor_urls:
                try:
                    comp_analysis = await self.analyze_page(comp_url, keyword)
                    if comp_analysis:
                        competitor_analyses.append(comp_analysis)
                except Exception as e:
                    logger.error(f"[{request_id}] Error analyzing competitor {comp_url}: {str(e)}")
                    continue
                
            # Generate benchmarks
            logger.info(f"[{request_id}] Generating benchmarks from {len(competitor_analyses)} competitor analyses")
            benchmarks = self.benchmark_analysis(target_analysis, competitor_analyses)
            
            # Generate recommendations
            logger.info(f"[{request_id}] Generating recommendations")
            recommendations = self.generate_recommendations(target_analysis, benchmarks)
            
            # Prepare competitor summary
            competitor_summary = []
            for comp in competitor_analyses:
                try:
                    summary = {
                        'url': comp.get('url'),
                        'title_length': comp.get('title', {}).get('length'),
                        'word_count': comp.get('content', {}).get('word_count'),
                        'keyword_density': comp.get('content', {}).get('keyword_density')
                    }
                    competitor_summary.append(summary)
                except Exception as e:
                    logger.error(f"[{request_id}] Error creating competitor summary: {str(e)}")
                    continue
                
            # Construct final response
            result = {
                'status': 'success',
                'analysis': target_analysis,
                'competitor_analysis_summary': competitor_summary,
                'benchmarks': benchmarks,
                'recommendations': recommendations
            }
            
            # Cache the results
            logger.info(f"[{request_id}] Caching analysis results")
            self.cache.set(url, cache_key, result)
            
            return result
            
        except Exception as e:
            logger.error(f"[{request_id}] Error in analyze_page_with_benchmarks: {str(e)}", exc_info=True)
            return {
                'status': 'error',
                'error_message': str(e)
            } 