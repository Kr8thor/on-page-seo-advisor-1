"""
Core scraping and analysis functionality for the On-Page SEO Analyzer & Advisor.
"""

import httpx
import asyncio
import logging
from typing import List, Optional, Dict, Any, Tuple, Union
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

# Import models from the models file
from src.models import (
    SerpResult, TitleAnalysis, MetaDescriptionAnalysis, HeadingDetail,
    HeadingsAnalysis, ContentAnalysis, LinksAnalysis,
    ImagesAnalysis, SchemaAnalysis, PageAnalysis
)

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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

    def _get_cache_key(self, url: str, keyword: str) -> str:
        """Generate a cache key from URL and keyword."""
        # Simple slugify and hash might be better for filenames
        url_slug = re.sub(r'[^a-z0-9]+', '-', urlparse(url).netloc + urlparse(url).path).strip('-')
        kw_slug = re.sub(r'[^a-z0-9]+', '-', keyword.lower()).strip('-')
        # Limit length to avoid issues with long filenames
        return f"{url_slug[:50]}_{kw_slug[:50]}"

    def get(self, url: str, keyword: str) -> Optional[Dict[str, Any]]:
        """Get cached analysis results if they exist and are not expired."""
        cache_key = self._get_cache_key(url, keyword)
        cache_file = self.cache_dir / f"{cache_key}.json"

        if not cache_file.exists():
            logger.info(f"Cache miss for {url} | {keyword}")
            return None

        try:
            mod_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
            if datetime.now() - mod_time > CACHE_DURATION:
                logger.info(f"Cache expired for {url} | {keyword}")
                cache_file.unlink() # Remove expired file
                return None

            with open(cache_file, 'r', encoding='utf-8') as f:
                cached_data = json.load(f)
            logger.info(f"Cache hit for {url} | {keyword}")
            return cached_data # Return the full PageAnalysis dict
        except Exception as e:
            logger.error(f"Error reading cache file {cache_file}: {str(e)}")
            return None

    def set(self, url: str, keyword: str, data: Dict[str, Any]):
        """Cache analysis results (which should be a PageAnalysis dict)."""
        cache_key = self._get_cache_key(url, keyword)
        cache_file = self.cache_dir / f"{cache_key}.json"

        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                # Store the PageAnalysis dict directly
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.info(f"Cached analysis for {url} | {keyword}")
        except Exception as e:
            logger.error(f"Error writing cache file {cache_file}: {str(e)}")

class SerpApiError(Exception):
    """Custom exception for SERP API related errors."""
    pass

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
        """Fetch SERP results using the configured API key with rate limiting."""
        await self.rate_limiter.acquire()

        if not keyword:
            raise ValueError("Keyword cannot be empty")

        country = country or self.default_country
        logger.info(f"Fetching SERP for keyword='{keyword}', country='{country}'")

        # Adjust params based on ValueSERP
        params = {
            'api_key': self.api_key,
            'q': keyword,
            'location': country,
            'output': 'json',
            'num': '10',
        }

        response_for_logging = None # To hold response object for logging on error

        try:
            async with httpx.AsyncClient(timeout=45.0) as client:
                response = await client.get(self.serp_api_url, params=params, headers=self.headers)
                response_for_logging = response # Store response for potential error logging
                response.raise_for_status() # Synchronous is correct here

                # --- Attempt to decode JSON using response.json() ---
                try:
                    # response.json() is SYNCHRONOUS - REMOVE AWAIT
                    data = response.json()
                except json.JSONDecodeError as json_err:
                     # If response.json() fails, try to log the raw text if possible
                     raw_text = "[Could not read response text]"
                     try:
                         # Attempt to read text synchronously AFTER error for logging
                         raw_text = response.text
                     except Exception:
                          pass # Ignore if reading text also fails
                     logger.error(f"Failed to decode SERP API JSON response: {json_err}. Response Text approx: {raw_text[:1000]}...")
                     raise SerpApiError("Invalid JSON response from SERP API")
                except Exception as e:
                     # Catch other potential errors during .json() call
                     logger.error(f"Error calling response.json(): {e}", exc_info=True)
                     raise SerpApiError(f"Error processing SERP API response: {e}")

                # --- Proceed with validated JSON data ---
                if not isinstance(data, dict):
                     raise SerpApiError(f"Invalid API response format: Expected dict, got {type(data)}")

                if data.get("request_info", {}).get("success") is False:
                    error_message = data.get("request_info", {}).get("message", "Unknown API Error")
                    raise SerpApiError(f"SERP API Error: {error_message}")

                organic_results = data.get('organic_results', [])
                if not organic_results:
                    logger.warning(f"No organic results found for '{keyword}' in '{country}'.")
                    return []

                results = []
                for result in organic_results:
                    url_key = 'link' if 'link' in result else 'url'
                    if result.get(url_key) and result.get('title'):
                        results.append(SerpResult(
                            url=result[url_key],
                            title=result['title'],
                            snippet=result.get('snippet')
                        ))
                    else:
                         logger.warning(f"Skipping SERP result with missing URL or title: {result}")

                logger.info(f"Successfully fetched {len(results)} SERP results.")
                return results[:10]

        except httpx.TimeoutException:
            logger.error("Request timed out while fetching SERP results")
            raise SerpApiError("Request timed out while fetching SERP results")
        except httpx.RequestError as e:
            logger.error(f"Network error while fetching SERP results: {e}")
            raise SerpApiError(f"Network error while fetching SERP results: {e}")
        except httpx.HTTPStatusError as e:
            # Log response text if available on HTTP error
            error_text = "[Could not read response text]"
            if response_for_logging:
                 try:
                     error_text = response_for_logging.text
                 except Exception:
                     pass
            logger.error(f"HTTP error {e.response.status_code} fetching SERP: {e}. Response text: {error_text[:500]}")
            if e.response.status_code == 401: raise SerpApiError("Invalid SERP API key")
            if e.response.status_code == 429: raise SerpApiError("SERP API rate limit exceeded")
            raise SerpApiError(f"HTTP error {e.response.status_code} fetching SERP results")
        except SerpApiError as e:
             logger.error(f"Caught specific SerpApiError: {e}")
             raise e # Re-raise to be handled by main.py
        except Exception as e:
            error_text = "[Could not read response text]"
            if response_for_logging:
                 try:
                     error_text = response_for_logging.text
                 except Exception:
                     pass
            logger.error(f"Unexpected error processing SERP results: {e}. Response text was: {error_text[:500]}", exc_info=True)
            raise SerpApiError(f"Unexpected error processing SERP results: {e}")

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
        """Analyze page headings H1-H3."""
        keyword_lower = keyword.lower()

        def process_headings(elements: List[Any]) -> List[HeadingDetail]:
            details = []
            for el in elements:
                text = el.css('::text').get('').strip()
                if text: # Only include headings with text content
                    details.append(HeadingDetail(
                        text=text,
                        contains_keyword=(keyword_lower in text.lower())
                    ))
            return details

        h1_elements = selector.css('h1')
        h2_elements = selector.css('h2')
        h3_elements = selector.css('h3')

        h1_details = process_headings(h1_elements)
        h2_details = process_headings(h2_elements)
        h3_details = process_headings(h3_elements)

        h1_count = len(h1_details)
        h1_contains_keyword = any(hd.contains_keyword for hd in h1_details)

        return HeadingsAnalysis(
            h1=h1_details,
            h2=h2_details,
            h3=h3_details,
            h1_count=h1_count,
            h1_contains_keyword=h1_contains_keyword,
        )

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
        """Analyze schema.org markup found in JSON-LD script tags."""
        schema_scripts = selector.css('script[type="application/ld+json"]::text').getall()
        types_found = set() # Use a set to automatically handle duplicates

        for script in schema_scripts:
            try:
                # Handle potential HTML comments within script tags
                script_content = re.sub(r'<!--.*?-->', '', script, flags=re.DOTALL).strip()
                if not script_content:
                    continue

                data = json.loads(script_content)

                # Handle single object or list of objects
                items_to_check = []
                if isinstance(data, dict):
                    items_to_check.append(data)
                elif isinstance(data, list):
                    items_to_check.extend(data)

                for item in items_to_check:
                    if isinstance(item, dict):
                        schema_type = item.get('@type')
                        if isinstance(schema_type, str):
                            types_found.add(schema_type)
                        elif isinstance(schema_type, list): # Type can be a list
                            for t in schema_type:
                                if isinstance(t, str):
                                    types_found.add(t)

            except json.JSONDecodeError as e:
                logger.warning(f"Could not parse JSON-LD script: {e}")
                continue
            except Exception as e:
                 logger.warning(f"Error processing schema script: {e}")
                 continue

        return SchemaAnalysis(types_found=sorted(list(types_found)))

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

            # Extract headings
            headings = []
            for level in range(1, 7):  # h1 to h6
                for heading in selector.css(f'h{level}::text').getall():
                    heading_text = heading.strip()
                    headings.append(HeadingDetail(
                        level=level,
                        text=heading_text,
                        contains_keyword=keyword.lower() in heading_text.lower()
                    ))

            # Extract links
            links = []
            for link in selector.css('a[href]'):
                href = link.css('::attr(href)').get()
                if href:
                    if href.startswith('/'):
                        href = urljoin(url, href)
                    links.append(href)

            # Extract images
            images = []
            for img in selector.css('img'):
                src = img.css('::attr(src)').get()
                alt = img.css('::attr(alt)').get() or ''
                if src:
                    if src.startswith('/'):
                        src = urljoin(url, src)
                    images.append({'src': src, 'alt': alt})

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

            # Performance metrics (simplified)
            performance = {
                'html_size': len(html_content),
                'text_html_ratio': len(content) / len(html_content) if html_content and len(html_content) > 0 else 0,
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
                h1_count=len(h1_headings),
                h1_contains_keyword=any(h.contains_keyword for h in h1_headings),
                h2_keywords=[]  # TODO: Implement if needed
            )

            content_analysis = ContentAnalysis(
                word_count=len(content.split()),
                keyword_count=content_lower.count(keyword_lower),
                keyword_density=keyword_metrics['content_keyword_density'],
                readability=readability_metrics
            )

            links_analysis = LinksAnalysis(
                total_links=len(links),
                internal_links=len([l for l in links if urlparse(l).netloc == urlparse(url).netloc]),
                external_links=len([l for l in links if urlparse(l).netloc != urlparse(url).netloc]),
                broken_links=[]  # Would require additional requests to verify
            )

            images_analysis = ImagesAnalysis(
                image_count=len(images),
                alts_missing=len([img for img in images if not img['alt']]),
                alts_with_keyword=len([
                    img for img in images
                    if keyword_lower in img['alt'].lower()
                ]),
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
                performance=performance
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
        """
        Calculates benchmark metrics based on competitor data.

        Args:
            target_analysis: The PageAnalysis object for the target URL.
            competitor_analyses: A list of PageAnalysis objects for competitors.

        Returns:
            A dictionary containing only the calculated benchmark metrics.
        """
        logger.info(f"Benchmarking {target_analysis.url} against {len(competitor_analyses)} competitors.")

        benchmarks: Dict[str, Dict[str, Optional[float]]] = {}
        if not competitor_analyses:
            logger.warning("No competitor analyses available for benchmarking.")
            return benchmarks

        # Define metrics to benchmark
        metrics_to_benchmark = {
            'title_length': lambda x: x.title.length if x.title else 0,
            'meta_description_length': lambda x: x.meta_description.length if x.meta_description else 0,
            'word_count': lambda x: x.content.word_count if x.content else 0,
            'heading_count': lambda x: len(x.headings) if x.headings else 0,
            'image_count': lambda x: len(x.images) if x.images else 0,
            'link_count': lambda x: len(x.links) if x.links else 0
        }

        # Calculate benchmarks for each metric
        for metric_name, metric_func in metrics_to_benchmark.items():
            competitor_values = []
            for comp_analysis in competitor_analyses:
                try:
                    value = metric_func(comp_analysis)
                    if value is not None:
                        competitor_values.append(value)
                except Exception as e:
                    logger.warning(f"Error calculating {metric_name} for competitor {comp_analysis.url}: {e}")
                    continue

            if competitor_values:
                benchmarks[metric_name] = {
                    'avg': sum(competitor_values) / len(competitor_values),
                    'median': sorted(competitor_values)[len(competitor_values) // 2]
                }
            else:
                benchmarks[metric_name] = {'avg': None, 'median': None}

        logger.info(f"Finished benchmarking. Calculated {len(benchmarks)} benchmark metrics.")
        return benchmarks

    def generate_recommendations(self, target_analysis: PageAnalysis, benchmarks: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generates actionable SEO recommendations based on the target analysis object
        and calculated benchmarks dictionary.

        Args:
            target_analysis: The PageAnalysis object for the target page.
            benchmarks: A dictionary containing the calculated benchmark data.

        Returns:
            A list of recommendation dictionaries.
        """
        recommendations: List[Dict[str, Any]] = []
        url = target_analysis.url
        logger.info(f"Generating recommendations for {url}")

        # Title Recommendations
        try:
            title_len = target_analysis.title.length if target_analysis.title else 0
            bm_title_len_avg = benchmarks.get('title_length', {}).get('avg')

            if title_len == 0:
                recommendations.append({'type': 'Title', 'severity': 'High', 'text': 'Page is missing a Title tag.'})
            elif bm_title_len_avg is not None and title_len < bm_title_len_avg * 0.8:
                recommendations.append({
                    'type': 'Title',
                    'severity': 'Medium',
                    'text': f"Title length ({title_len}) is shorter than competitor average ({bm_title_len_avg:.0f}). Consider adding more relevant terms."
                })
        except Exception as e:
            logger.warning(f"Error generating title recommendations: {e}")

        # Meta Description Recommendations
        try:
            meta_desc_len = target_analysis.meta_description.length if target_analysis.meta_description else 0
            bm_meta_desc_len_avg = benchmarks.get('meta_description_length', {}).get('avg')

            if meta_desc_len == 0:
                recommendations.append({'type': 'Meta Description', 'severity': 'High', 'text': 'Page is missing a Meta Description.'})
            elif bm_meta_desc_len_avg is not None and meta_desc_len < bm_meta_desc_len_avg * 0.8:
                recommendations.append({
                    'type': 'Meta Description',
                    'severity': 'Medium',
                    'text': f"Meta description length ({meta_desc_len}) is shorter than competitor average ({bm_meta_desc_len_avg:.0f}). Consider adding more details."
                })
        except Exception as e:
            logger.warning(f"Error generating meta description recommendations: {e}")

        # Content Recommendations
        try:
            word_count = target_analysis.content.word_count if target_analysis.content else 0
            bm_word_count_avg = benchmarks.get('word_count', {}).get('avg')

            if word_count == 0:
                recommendations.append({'type': 'Content', 'severity': 'High', 'text': 'Page has no content.'})
            elif bm_word_count_avg is not None and word_count < bm_word_count_avg * 0.8:
                recommendations.append({
                    'type': 'Content',
                    'severity': 'Medium',
                    'text': f"Content length ({word_count} words) is shorter than competitor average ({bm_word_count_avg:.0f}). Consider adding more relevant content."
                })
        except Exception as e:
            logger.warning(f"Error generating content recommendations: {e}")

        # Heading Structure Recommendations
        try:
            heading_count = len(target_analysis.headings) if target_analysis.headings else 0
            bm_heading_count_avg = benchmarks.get('heading_count', {}).get('avg')

            if heading_count == 0:
                recommendations.append({'type': 'Headings', 'severity': 'High', 'text': 'Page has no headings. Add H1-H6 tags to structure your content.'})
            elif bm_heading_count_avg is not None and heading_count < bm_heading_count_avg * 0.8:
                recommendations.append({
                    'type': 'Headings',
                    'severity': 'Medium',
                    'text': f"Number of headings ({heading_count}) is lower than competitor average ({bm_heading_count_avg:.0f}). Consider improving content structure."
                })
        except Exception as e:
            logger.warning(f"Error generating heading recommendations: {e}")

        # Image Recommendations
        try:
            image_count = len(target_analysis.images) if target_analysis.images else 0
            bm_image_count_avg = benchmarks.get('image_count', {}).get('avg')

            if image_count == 0:
                recommendations.append({'type': 'Images', 'severity': 'Medium', 'text': 'Page has no images. Consider adding relevant images to improve engagement.'})
            elif bm_image_count_avg is not None and image_count < bm_image_count_avg * 0.8:
                recommendations.append({
                    'type': 'Images',
                    'severity': 'Low',
                    'text': f"Number of images ({image_count}) is lower than competitor average ({bm_image_count_avg:.0f}). Consider adding more visual content."
                })
        except Exception as e:
            logger.warning(f"Error generating image recommendations: {e}")

        # Link Recommendations
        try:
            link_count = len(target_analysis.links) if target_analysis.links else 0
            bm_link_count_avg = benchmarks.get('link_count', {}).get('avg')

            if link_count == 0:
                recommendations.append({'type': 'Links', 'severity': 'Medium', 'text': 'Page has no links. Consider adding relevant internal and external links.'})
            elif bm_link_count_avg is not None and link_count < bm_link_count_avg * 0.8:
                recommendations.append({
                    'type': 'Links',
                    'severity': 'Medium',
                    'text': f"Number of links ({link_count}) is lower than competitor average ({bm_link_count_avg:.0f}). Consider adding more relevant links."
                })
        except Exception as e:
            logger.warning(f"Error generating link recommendations: {e}")

        logger.info(f"Generated {len(recommendations)} recommendations for {url}")
        # Sort recommendations by severity (High > Medium > Low)
        severity_map = {'High': 3, 'Medium': 2, 'Low': 1}
        recommendations.sort(key=lambda x: severity_map.get(x.get('severity', 'Low'), 0), reverse=True)
        return recommendations

    async def analyze_page_with_benchmarks(
        self,
        url: str,
        keyword: str,
        country: Optional[str] = None,
        max_competitors: int = 3
    ) -> Dict[str, Any]:
        """Analyze a page and benchmark against competitors."""
        try:
            # Fetch SERP results
            serp_results = await self.fetch_serp_results(keyword, country)
            if not serp_results:
                return {
                    'status': 'error',
                    'message': 'No SERP results found',
                    'warning': 'Could not fetch competitor data'
                }

            # Analyze target page
            target_analysis = await self.analyze_page(url, keyword)
            if not target_analysis:
                return {
                    'status': 'error',
                    'message': 'Failed to analyze target page'
                }

            # Analyze competitor pages (up to max_competitors)
            competitor_analyses = []
            for result in serp_results[:max_competitors]:
                try:
                    analysis = await self.analyze_page(result.url, keyword)
                    if analysis:
                        competitor_analyses.append(analysis)
                except Exception as e:
                    logger.warning(f"Failed to analyze competitor {result.url}: {e}")
                    continue

            # Generate benchmarks
            benchmarks = self.benchmark_analysis(target_analysis, competitor_analyses)
            
            # Generate recommendations
            recommendations = self.generate_recommendations(target_analysis, benchmarks)

            # Prepare final result
            final_result = {
                'status': 'success',
                'target_analysis': target_analysis.model_dump(mode='json'),
                'competitor_analyses': [comp.model_dump(mode='json') for comp in competitor_analyses],
                'benchmarks': benchmarks,
                'recommendations': recommendations
            }

            # Add warning if some competitors couldn't be analyzed
            if len(competitor_analyses) < min(len(serp_results), max_competitors):
                final_result['warning'] = f"Only analyzed {len(competitor_analyses)} out of {min(len(serp_results), max_competitors)} competitors"

            return final_result

        except Exception as e:
            logger.error(f"Error in analyze_page_with_benchmarks: {e}", exc_info=True)
            return {
                'status': 'error',
                'message': str(e)
            } 