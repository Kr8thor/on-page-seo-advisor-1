// Immediate console log to verify script loading
console.log('Script file loaded');

// API Configuration
const API_URL = process.env.VITE_API_URL || 'https://on-page-seo-advisor-1.onrender.com';
console.log('API URL:', API_URL);

// Initialize DOM elements
let seoForm, urlInput, keywordInput, countryInput, analyzeButton, loadingIndicator, errorDisplay, resultsSection;

// Wait for DOM to be fully loaded
document.addEventListener('DOMContentLoaded', function() {
    console.log('DOM Content Loaded');
    
    // Initialize DOM elements
    seoForm = document.getElementById('seoForm');
    urlInput = document.getElementById('urlInput');
    keywordInput = document.getElementById('keywordInput');
    countryInput = document.getElementById('countryInput');
    analyzeButton = document.getElementById('analyzeButton');
    loadingIndicator = document.getElementById('loadingIndicator');
    errorDisplay = document.getElementById('errorDisplay');
    resultsSection = document.getElementById('resultsSection');

    // Log initialization
    console.log('Initializing DOM elements:', {
        seoForm: !!seoForm,
        urlInput: !!urlInput,
        keywordInput: !!keywordInput,
        countryInput: !!countryInput,
        analyzeButton: !!analyzeButton,
        loadingIndicator: !!loadingIndicator,
        errorDisplay: !!errorDisplay,
        resultsSection: !!resultsSection
    });

    // Log the initial state of the results section
    if (resultsSection) {
        console.log('Results section initial state:', {
            display: resultsSection.style.display,
            visibility: resultsSection.style.visibility,
            opacity: resultsSection.style.opacity,
            className: resultsSection.className
        });
    }

    // Add form submit handler
    if (seoForm) {
        seoForm.addEventListener('submit', handleSubmit);
        console.log('Form submit handler attached');
    } else {
        console.error('Form element not found!');
    }
});

// Form submission handler
async function handleSubmit(event) {
    event.preventDefault();
    console.log('Form submitted');

    // Get form values
    const url = urlInput.value.trim();
    const keyword = keywordInput.value.trim();
    const country = countryInput.value.trim();

    // Log form values
    console.log('Form values:', { url, keyword, country });

    // Validate inputs
    if (!url || !keyword) {
        showError('Please enter both URL and keyword');
        return;
    }

    // Show loading indicator
    loadingIndicator.style.display = 'flex';
    errorDisplay.style.display = 'none';

    try {
        // Make API request
        console.log('Making API request to:', API_URL);
        console.log('Request payload:', { url, keyword, country });

        const response = await fetch(API_URL, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Accept': 'application/json',
                'Origin': window.location.origin
            },
            mode: 'cors',
            credentials: 'omit',
            body: JSON.stringify({ url, keyword, country })
        });

        // Log response status
        console.log('Response status:', response.status);
        console.log('Response headers:', Object.fromEntries(response.headers.entries()));

        // Check if response is ok
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        // Parse response data
        const data = await response.json();
        console.log('Response data:', data);

        // Display results
        displayResults(data);
    } catch (error) {
        console.error('Error:', error);
        showError('An error occurred while analyzing the page. Please try again.');
    } finally {
        // Hide loading indicator
        loadingIndicator.style.display = 'none';
    }
}

// API Configuration
const REQUEST_TIMEOUT = 30000; // 30 seconds

// Country code mapping
const COUNTRY_CODES = {
    'united states': 'us',
    'ireland': 'ie',
    'united kingdom': 'gb',
    'canada': 'ca',
    'australia': 'au',
    'new zealand': 'nz',
    'germany': 'de',
    'france': 'fr',
    'spain': 'es',
    'italy': 'it',
    'japan': 'jp',
    'korea': 'kr',
    'china': 'cn',
    'india': 'in',
    'brazil': 'br',
    'mexico': 'mx',
    'russia': 'ru',
    'south africa': 'za'
};

// Utility Functions
function formatNumber(value) {
    if (typeof value !== 'number') return 'N/A';
    return value.toFixed(2);
}

function formatPercentage(value) {
    if (typeof value !== 'number') return 'N/A';
    return `${value.toFixed(2)}%`;
}

function validateUrl(url) {
    try {
        new URL(url);
        return true;
    } catch {
        return false;
    }
}

function getCountryCode(countryName) {
    if (!countryName) return 'us'; // Default to US if no country specified
    
    // Clean and normalize the input
    const normalizedInput = countryName.toLowerCase().trim();
    
    // First try direct match
    if (COUNTRY_CODES[normalizedInput]) {
        return COUNTRY_CODES[normalizedInput];
    }
    
    // If input is already a valid 2-letter code, return it
    if (/^[a-z]{2}$/.test(normalizedInput)) {
        return normalizedInput;
    }
    
    // Try to find a partial match in the keys
    const matchingKey = Object.keys(COUNTRY_CODES).find(key => 
        key.includes(normalizedInput) || normalizedInput.includes(key)
    );
    
    if (matchingKey) {
        return COUNTRY_CODES[matchingKey];
    }
    
    console.warn(`Country code not found for "${countryName}", defaulting to "us"`);
    return 'us'; // Default to US if no match found
}

// Additional Utility Functions
function getMetricStatus(value, benchmark, type = 'higher') {
    if (typeof value !== 'number' || typeof benchmark !== 'number') return 'neutral';
    const percentage = ((value - benchmark) / benchmark) * 100;
    if (type === 'higher') {
        return percentage >= 10 ? 'good' : percentage <= -10 ? 'bad' : 'neutral';
    } else {
        return percentage <= -10 ? 'good' : percentage >= 10 ? 'bad' : 'neutral';
    }
}

function formatMetric(value, type = 'number') {
    if (typeof value !== 'number') return 'N/A';
    switch (type) {
        case 'percentage':
            return `${value.toFixed(2)}%`;
        case 'time':
            return `${value.toFixed(2)}s`;
        case 'count':
            return value.toLocaleString();
        default:
            return value.toFixed(2);
    }
}

// Results Display
function displayResults(data) {
    console.log('Starting displayResults function');
    console.log('Raw response data:', data);
    
    // Validate response data
    if (!data) {
        console.error('No data received from server');
        showError('No data received from server');
        return;
    }

    // Check if we're getting the raw analysis data directly
    const analysisData = data.target_analysis || data;
    console.log('Analysis data:', analysisData);

    if (!analysisData) {
        console.error('Invalid response format from server');
        showError('Invalid response format from server');
        return;
    }

    // Clear previous results and errors
    errorDisplay.style.display = 'none';
    
    // Force results section visibility
    console.log('Setting results section visibility...');
    if (!resultsSection) {
        console.error('Results section element not found!');
        showError('Error: Results section not found');
        return;
    }

    // Make sure the results section is visible
    resultsSection.style.display = 'block';
    resultsSection.style.visibility = 'visible';
    resultsSection.style.opacity = '1';
    resultsSection.classList.remove('hidden');
    resultsSection.classList.add('visible');
    
    // Log the computed styles for debugging
    const computedStyle = window.getComputedStyle(resultsSection);
    console.log('Results section styles:', {
        display: computedStyle.display,
        visibility: computedStyle.visibility,
        opacity: computedStyle.opacity,
        position: computedStyle.position,
        zIndex: computedStyle.zIndex
    });

    try {
        // Display target information
        const targetInfo = document.querySelector('#targetInfo .card-content');
        if (targetInfo) {
            console.log('Updating target info section');
            targetInfo.innerHTML = `
                <div class="info-grid">
                    <div class="info-item">
                        <span class="label">URL:</span>
                        <span class="value">${analysisData.url || 'N/A'}</span>
                    </div>
                    <div class="info-item">
                        <span class="label">Keyword:</span>
                        <span class="value">${analysisData.keyword || 'N/A'}</span>
                    </div>
                </div>
            `;
        }

        // Display title analysis
        const titleAnalysis = document.querySelector('#titleAnalysis .card-content');
        if (titleAnalysis && analysisData.title_analysis) {
            console.log('Updating title analysis section');
            titleAnalysis.innerHTML = `
                <div class="metric-grid">
                    <div class="metric-item ${getMetricStatus(analysisData.title_analysis.length, 60)}">
                        <span class="metric-label">Title Length</span>
                        <span class="metric-value">${formatMetric(analysisData.title_analysis.length, 'count')}</span>
                    </div>
                    <div class="metric-item ${getMetricStatus(analysisData.title_analysis.keyword_density, 0.5)}">
                        <span class="metric-label">Keyword Density</span>
                        <span class="metric-value">${formatMetric(analysisData.title_analysis.keyword_density, 'percentage')}</span>
                    </div>
                </div>
            `;
        }

        // Display meta description analysis
        const metaAnalysis = document.querySelector('#metaAnalysis .card-content');
        if (metaAnalysis && analysisData.meta_analysis) {
            console.log('Updating meta analysis section');
            metaAnalysis.innerHTML = `
                <div class="metric-grid">
                    <div class="metric-item ${getMetricStatus(analysisData.meta_analysis.length, 160)}">
                        <span class="metric-label">Description Length</span>
                        <span class="metric-value">${formatMetric(analysisData.meta_analysis.length, 'count')}</span>
                    </div>
                    <div class="metric-item ${getMetricStatus(analysisData.meta_analysis.keyword_density, 0.5)}">
                        <span class="metric-label">Keyword Density</span>
                        <span class="metric-value">${formatMetric(analysisData.meta_analysis.keyword_density, 'percentage')}</span>
                    </div>
                </div>
            `;
        }

        // Display content analysis
        const contentAnalysis = document.querySelector('#contentAnalysis .card-content');
        if (contentAnalysis && analysisData.content_analysis) {
            console.log('Updating content analysis section');
            contentAnalysis.innerHTML = `
                <div class="metric-grid">
                    <div class="metric-item ${getMetricStatus(analysisData.content_analysis.word_count, 300)}">
                        <span class="metric-label">Word Count</span>
                        <span class="metric-value">${formatMetric(analysisData.content_analysis.word_count, 'count')}</span>
                    </div>
                    <div class="metric-item ${getMetricStatus(analysisData.content_analysis.keyword_density, 1)}">
                        <span class="metric-label">Keyword Density</span>
                        <span class="metric-value">${formatMetric(analysisData.content_analysis.keyword_density, 'percentage')}</span>
                    </div>
                </div>
            `;
        }

        // Display recommendations if available
        const recommendationsDisplay = document.querySelector('#recommendationsDisplay .card-content');
        if (recommendationsDisplay && analysisData.recommendations) {
            console.log('Updating recommendations section');
            recommendationsDisplay.innerHTML = `
                <div class="recommendations-grid">
                    ${analysisData.recommendations.map(rec => `
                        <div class="recommendation-item">
                            <span class="recommendation-title">${rec.title}</span>
                            <p class="recommendation-description">${rec.description}</p>
                        </div>
                    `).join('')}
                </div>
            `;
        }

        // Display benchmarks if available
        const benchmarksDisplay = document.querySelector('#benchmarksDisplay .card-content');
        if (benchmarksDisplay && analysisData.benchmarks) {
            console.log('Updating benchmarks section');
            benchmarksDisplay.innerHTML = `
                <div class="benchmarks-grid">
                    ${Object.entries(analysisData.benchmarks).map(([metric, value]) => `
                        <div class="benchmark-item">
                            <span class="benchmark-label">${metric}</span>
                            <span class="benchmark-value">${formatMetric(value)}</span>
                        </div>
                    `).join('')}
                </div>
            `;
        }

        // Scroll to results section
        console.log('Scrolling to results section');
        resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });

    } catch (error) {
        console.error('Error displaying results:', error);
        console.error('Error stack:', error.stack);
        showError('Error processing the analysis results. Please try again.');
    }
}

// Loading State Management
function setLoading(isLoading) {
    loadingIndicator.style.display = isLoading ? 'flex' : 'none';
    analyzeButton.disabled = isLoading;
    if (!isLoading) {
        errorDisplay.style.display = 'none';
    }
}

// Error Display
function showError(message) {
    errorDisplay.textContent = message;
    errorDisplay.style.display = 'block';
    resultsSection.style.display = 'none';
} 