from typing import List, Dict, Any
from statistics import mean, median
from .models import PageAnalysis

class BenchmarkAnalyzer:
    @staticmethod
    def calculate_benchmarks(competitor_analyses: List[PageAnalysis]) -> Dict[str, Any]:
        """Calculate benchmark metrics from competitor analyses."""
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

    @staticmethod
    def generate_recommendations(analysis: PageAnalysis, benchmarks: Dict[str, Any]) -> List[str]:
        """Generate SEO recommendations based on analysis and benchmarks."""
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

    @staticmethod
    def benchmark_analysis(user_analysis: PageAnalysis, competitor_analyses: List[PageAnalysis]) -> PageAnalysis:
        """Benchmark user's page against competitors and generate recommendations."""
        benchmarks = BenchmarkAnalyzer.calculate_benchmarks(competitor_analyses)
        recommendations = BenchmarkAnalyzer.generate_recommendations(user_analysis, benchmarks)
        
        # Create a new PageAnalysis with benchmarks and recommendations
        return PageAnalysis(
            url=user_analysis.url,
            title=user_analysis.title,
            meta_description=user_analysis.meta_description,
            headings=user_analysis.headings,
            content=user_analysis.content,
            links=user_analysis.links,
            images=user_analysis.images,
            schema=user_analysis.schema,
            benchmarks=benchmarks,
            recommendations=recommendations
        ) 