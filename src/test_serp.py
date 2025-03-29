import asyncio
from analyzer import SEOAnalyzer
from models import SerpResult

async def test_serp_api():
    """Test the SERP API integration."""
    try:
        # Initialize the analyzer
        analyzer = SEOAnalyzer()
        
        # Test search
        keyword = "python programming"
        print(f"\nFetching SERP results for keyword: {keyword}")
        
        results = await analyzer.fetch_serp_results(keyword)
        
        print("\nResults:")
        for i, result in enumerate(results, 1):
            print(f"\n{i}. {result.title}")
            print(f"   URL: {result.url}")
            print(f"   Snippet: {result.snippet[:100]}...")
            
    except Exception as e:
        print(f"\nError: {str(e)}")

if __name__ == "__main__":
    # Run the async test
    asyncio.run(test_serp_api()) 