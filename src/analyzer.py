import httpx
from typing import List, Optional, Dict, Any
from urllib.parse import urljoin, urlparse
import textstat
from readability import Document
from parsel import Selector
import json
import re
import os
from dotenv import load_dotenv
from statistics import mean, median
from src.models import (
    SerpResult, TitleAnalysis, MetaDescriptionAnalysis,
    HeadingsAnalysis, ContentAnalysis, LinksAnalysis,
    ImagesAnalysis, SchemaAnalysis, PageAnalysis
)

# Load environment variables
load_dotenv()

class SerpApiError(Exception):
    """Custom exception for SERP API related errors."""
    pass

class SEOAnalyzer:
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        # Initialize API key from environment variable
        self.api_key = os.getenv('SERP_API_KEY')
        if not self.api_key:
            raise ValueError("SERP_API_KEY environment variable is not set")
        
        # You can customize these based on your chosen SERP API provider
        self.serp_api_url = "https://api.valueserp.com/search"  # Example using ValueSERP
        self.default_country = os.getenv('DEFAULT_COUNTRY', 'us')

    async def fetch_serp_results(self, keyword: str, country: str = None) -> List[SerpResult]:
        """
        Fetch SERP results using the configured API key.
        
        Args:
            keyword (str): The search keyword/phrase
            country (str, optional): Two-letter country code. Defaults to value from env or 'us'
            
        Returns:
            List[SerpResult]: List of search results with URL, title, and snippet
            
        Raises:
            SerpApiError: If there's an error with the API request
            ValueError: If required parameters are missing
            httpx.RequestError: If there's a network/connection error
        """
        if not keyword:
            raise ValueError("Keyword cannot be empty")
            
        country = country or self.default_country
        
        # Prepare API request parameters
        params = {
            'api_key': self.api_key,
            'q': keyword,
            'country': country,
            'output': 'json',
            'page': 1,
            'num': 10,  # Get top 10 results
            'search_type': 'organic'
        }
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(
                    self.serp_api_url,
                    params=params,
                    headers=self.headers
                )
                
                # Check for HTTP errors
                response.raise_for_status()
                
                # Parse response
                data = response.json()
                
                # Validate response structure
                if not isinstance(data, dict):
                    raise SerpApiError("Invalid API response format")
                
                # Check for API-specific error messages
                if 'error' in data:
                    raise SerpApiError(f"API Error: {data['error']}")
                
                # Extract organic results
                organic_results = data.get('organic_results', [])
                if not organic_results:
                    return []
                
                # Transform results into SerpResult objects
                results = []
                for result in organic_results:
                    try:
                        # Validate required fields
                        if not all(k in result for k in ['url', 'title', 'snippet']):
                            continue
                            
                        results.append(SerpResult(
                            url=result['url'],
                            title=result['title'],
                            snippet=result.get('snippet', '')  # Snippet might be optional
                        ))
                    except (KeyError, ValueError) as e:
                        # Log the error but continue processing other results
                        print(f"Error processing result: {str(e)}")
                        continue
                
                return results
                
        except httpx.TimeoutException:
            raise SerpApiError("Request timed out while fetching SERP results")
            
        except httpx.RequestError as e:
            raise SerpApiError(f"Network error while fetching SERP results: {str(e)}")
            
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                raise SerpApiError("Invalid API key")
            elif e.response.status_code == 429:
                raise SerpApiError("API rate limit exceeded")
            else:
                raise SerpApiError(f"HTTP error {e.response.status_code}: {str(e)}")
                
        except json.JSONDecodeError:
            raise SerpApiError("Invalid JSON response from API")
            
        except Exception as e:
            raise SerpApiError(f"Unexpected error while fetching SERP results: {str(e)}")

    async def fetch_page_content(self, url: str) -> Optional[str]:
        """Fetch HTML content from a URL."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url, headers=self.headers, timeout=30.0)
                response.raise_for_status()
                return response.text
        except Exception as e:
            print(f"Error fetching page content: {str(e)}")
            return None

    def analyze_title(self, selector: Selector, keyword: str) -> TitleAnalysis:
        """Analyze the page title."""
        title = selector.css('title::text').get('')
        keyword_present = keyword.lower() in title.lower()
        
        # Determine keyword position
        position = None
        if keyword_present:
            title_words = title.lower().split()
            keyword_index = title_words.index(keyword.lower())
            if keyword_index == 0:
                position = 'start'
            elif keyword_index == len(title_words) - 1:
                position = 'end'
            else:
                position = 'middle'
        
        return TitleAnalysis(
            text=title,
            length=len(title),
            keyword_present=keyword_present,
            position=position
        )

    def analyze_meta_description(self, selector: Selector, keyword: str) -> MetaDescriptionAnalysis:
        """Analyze the meta description."""
        description = selector.css('meta[name="description"]::attr(content)').get('')
        return MetaDescriptionAnalysis(
            text=description,
            length=len(description),
            keyword_present=keyword.lower() in description.lower()
        )

    def analyze_headings(self, selector: Selector, keyword: str) -> HeadingsAnalysis:
        """Analyze page headings."""
        h1s = selector.css('h1::text').getall()
        h2s = selector.css('h2::text').getall()
        h3s = selector.css('h3::text').getall()
        
        def process_headings(headings: List[str]) -> List[Dict[str, Any]]:
            return [{
                'text': h,
                'contains_keyword': keyword.lower() in h.lower()
            } for h in headings]
        
        h1_contains_keyword = any(keyword.lower() in h.lower() for h in h1s)
        h2_keywords = [h for h in h2s if keyword.lower() in h.lower()]
        
        return HeadingsAnalysis(
            h1=process_headings(h1s),
            h2=process_headings(h2s),
            h3=process_headings(h3s),
            h1_count=len(h1s),
            h1_contains_keyword=h1_contains_keyword,
            h2_keywords=h2_keywords
        )

    def extract_main_text(self, html_content: str) -> str:
        """Extract main content text using readability-lxml."""
        doc = Document(html_content)
        return doc.summary()

    def analyze_content(self, main_text: str, keyword: str) -> ContentAnalysis:
        """Analyze the main content."""
        # Clean text for analysis
        clean_text = re.sub(r'<[^>]+>', '', main_text)
        words = clean_text.split()
        word_count = len(words)
        
        # Calculate keyword density
        keyword_count = clean_text.lower().count(keyword.lower())
        keyword_density = keyword_count / word_count if word_count > 0 else 0
        
        # Calculate readability score (Flesch-Kincaid)
        readability_score = textstat.flesch_kincaid_grade(clean_text)
        
        return ContentAnalysis(
            word_count=word_count,
            readability_score=readability_score,
            keyword_density=keyword_density,
            keyword_count=keyword_count
        )

    def analyze_links(self, selector: Selector, base_url: str) -> LinksAnalysis:
        """Analyze internal and external links."""
        links = selector.css('a::attr(href)').getall()
        base_domain = urlparse(base_url).netloc
        
        internal_links = 0
        external_links = 0
        
        for link in links:
            if not link:
                continue
            full_url = urljoin(base_url, link)
            if urlparse(full_url).netloc == base_domain:
                internal_links += 1
            else:
                external_links += 1
        
        return LinksAnalysis(
            internal_links=internal_links,
            external_links=external_links
        )

    def analyze_images(self, selector: Selector, keyword: str) -> ImagesAnalysis:
        """Analyze images and their alt texts."""
        images = selector.css('img')
        image_count = len(images)
        
        alts_missing = 0
        alts_with_keyword = 0
        
        for img in images:
            alt = img.css('::attr(alt)').get('')
            if not alt:
                alts_missing += 1
            elif keyword.lower() in alt.lower():
                alts_with_keyword += 1
        
        return ImagesAnalysis(
            image_count=image_count,
            alts_missing=alts_missing,
            alts_with_keyword=alts_with_keyword
        )

    def analyze_schema(self, selector: Selector) -> SchemaAnalysis:
        """Analyze schema.org markup."""
        schema_scripts = selector.css('script[type="application/ld+json"]::text').getall()
        types_found = []
        
        for script in schema_scripts:
            try:
                data = json.loads(script)
                if isinstance(data, dict):
                    if '@type' in data:
                        types_found.append(data['@type'])
                elif isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict) and '@type' in item:
                            types_found.append(item['@type'])
            except json.JSONDecodeError:
                continue
        
        return SchemaAnalysis(types_found=list(set(types_found)))

    async def analyze_page(self, html_content: str, url: str, keyword: str) -> PageAnalysis:
        """Perform comprehensive page analysis."""
        selector = Selector(html_content)
        
        return PageAnalysis(
            url=url,
            title=self.analyze_title(selector, keyword),
            meta_description=self.analyze_meta_description(selector, keyword),
            headings=self.analyze_headings(selector, keyword),
            content=self.analyze_content(self.extract_main_text(html_content), keyword),
            links=self.analyze_links(selector, url),
            images=self.analyze_images(selector, keyword),
            schema=self.analyze_schema(selector)
        )

    def calculate_benchmarks(self, competitor_analyses: List[PageAnalysis]) -> Dict[str, Any]:
        """
        Calculate benchmark metrics from competitor analyses.
        
        Args:
            competitor_analyses (List[PageAnalysis]): List of competitor page analyses
            
        Returns:
            Dict[str, Any]: Dictionary containing benchmark metrics
        """
        if not competitor_analyses:
            return {}
        
        return {
            'title_length_avg': mean([c.title.length for c in competitor_analyses]),
            'meta_description_length_avg': mean([c.meta_description.length for c in competitor_analyses]),
            'word_count_avg': mean([c.content.word_count for c in competitor_analyses]),
            'keyword_density_median': median([c.content.keyword_density for c in competitor_analyses]),
            'readability_score_avg': mean([c.content.readability_score for c in competitor_analyses]),
            'internal_links_avg': mean([c.links.internal_links for c in competitor_analyses]),
            'external_links_avg': mean([c.links.external_links for c in competitor_analyses]),
            'image_count_avg': mean([c.images.image_count for c in competitor_analyses]),
            'h1_count_avg': mean([c.headings.h1_count for c in competitor_analyses])
        }

    def generate_recommendations(self, analysis: PageAnalysis, benchmarks: Dict[str, Any]) -> List[str]:
        """
        Generate SEO recommendations based on analysis and benchmarks.
        
        Args:
            analysis (PageAnalysis): Analysis of the target page
            benchmarks (Dict[str, Any]): Benchmark metrics from competitors
            
        Returns:
            List[str]: List of actionable recommendations
        """
        recommendations = []
        
        # Title recommendations
        if analysis.title.length < benchmarks.get('title_length_avg', 0) * 0.8:
            recommendations.append(
                f"Title is shorter than average top rankers (Avg: {benchmarks['title_length_avg']:.1f}). "
                "Consider expanding to include more relevant keywords and improve click-through rate."
            )
        if not analysis.title.keyword_present:
            recommendations.append(
                "Primary keyword is missing from the title. Add it to improve relevance and click-through rate."
            )
        
        # Meta description recommendations
        if analysis.meta_description.length < 120:
            recommendations.append(
                "Meta description is too short. Aim for 120-160 characters to maximize visibility in search results."
            )
        if not analysis.meta_description.keyword_present:
            recommendations.append(
                "Primary keyword is missing from meta description. Include it to improve relevance."
            )
        
        # Content recommendations
        if analysis.content.word_count < benchmarks.get('word_count_avg', 0) * 0.8:
            recommendations.append(
                f"Content length is significantly below average (Avg: {benchmarks['word_count_avg']:.1f} words). "
                "Consider adding more comprehensive content to improve authority and relevance."
            )
        
        if analysis.content.keyword_density < benchmarks.get('keyword_density_median', 0) * 0.5:
            recommendations.append(
                f"Keyword density is below median (Median: {benchmarks['keyword_density_median']:.2%}). "
                "Consider naturally incorporating the keyword more frequently."
            )
        
        # Headings recommendations
        if analysis.headings.h1_count == 0:
            recommendations.append(
                "Missing H1 tag. Add a unique H1 containing your primary keyword to improve content structure."
            )
        elif analysis.headings.h1_count > 1:
            recommendations.append(
                "Multiple H1 tags detected. Use only one H1 tag per page for optimal SEO structure."
            )
        
        if not analysis.headings.h1_contains_keyword:
            recommendations.append(
                "Primary keyword is missing from H1 tag. Include it to improve content relevance."
            )
        
        # Links recommendations
        if analysis.links.internal_links < benchmarks.get('internal_links_avg', 0) * 0.5:
            recommendations.append(
                f"Internal linking is below average (Avg: {benchmarks['internal_links_avg']:.1f}). "
                "Add more internal links to improve site structure and user navigation."
            )
        
        # Images recommendations
        if analysis.images.alts_missing > 0:
            recommendations.append(
                f"{analysis.images.alts_missing} images are missing alt text. "
                "Add descriptive alt text to improve accessibility and image SEO."
            )
        
        if analysis.images.alts_with_keyword == 0:
            recommendations.append(
                "No images have alt text containing the primary keyword. "
                "Add keyword-rich alt text to relevant images."
            )
        
        # Schema recommendations
        if not analysis.schema.types_found:
            recommendations.append(
                "No schema.org markup detected. Add relevant structured data to improve rich snippet opportunities."
            )
        
        return recommendations

    async def benchmark_analysis(
        self, 
        target_analysis: PageAnalysis, 
        competitor_analyses: List[PageAnalysis]
    ) -> PageAnalysis:
        """
        Benchmark target page against competitors and generate recommendations.
        
        Args:
            target_analysis (PageAnalysis): Analysis of the target page
            competitor_analyses (List[PageAnalysis]): List of competitor page analyses
            
        Returns:
            PageAnalysis: Updated analysis with benchmarks and recommendations
        """
        # Calculate benchmarks
        benchmarks = self.calculate_benchmarks(competitor_analyses)
        
        # Generate recommendations
        recommendations = self.generate_recommendations(target_analysis, benchmarks)
        
        # Create a new PageAnalysis with benchmarks and recommendations
        return PageAnalysis(
            url=target_analysis.url,
            title=target_analysis.title,
            meta_description=target_analysis.meta_description,
            headings=target_analysis.headings,
            content=target_analysis.content,
            links=target_analysis.links,
            images=target_analysis.images,
            schema=target_analysis.schema,
            benchmarks=benchmarks,
            recommendations=recommendations
        )

    async def analyze_page_with_benchmarks(
        self, 
        html_content: str, 
        url: str, 
        keyword: str,
        competitor_analyses: List[PageAnalysis]
    ) -> PageAnalysis:
        """
        Perform comprehensive page analysis with benchmarking.
        
        Args:
            html_content (str): HTML content of the page
            url (str): URL of the page
            keyword (str): Target keyword
            competitor_analyses (List[PageAnalysis]): List of competitor analyses
            
        Returns:
            PageAnalysis: Complete analysis with benchmarks and recommendations
        """
        # Perform basic analysis
        basic_analysis = await self.analyze_page(html_content, url, keyword)
        
        # Add benchmarks and recommendations
        return await self.benchmark_analysis(basic_analysis, competitor_analyses) 