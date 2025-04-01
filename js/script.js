// API Configuration
const API_URL = 'https://on-page-seo-advisor-1.onrender.com/analyze';

// DOM Elements
const seoForm = document.getElementById('seoForm');
const loadingIndicator = document.getElementById('loadingIndicator');
const errorMessage = document.getElementById('errorMessage');
const resultsSection = document.getElementById('resultsSection');

// Form Submission Handler
seoForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    
    // Reset UI
    hideError();
    hideResults();
    showLoading();
    
    // Get form data
    const formData = {
        url: document.getElementById('url').value,
        keyword: document.getElementById('keyword').value,
        country: document.getElementById('country').value || 'us'
    };
    
    try {
        const response = await fetch(API_URL, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(formData)
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Failed to analyze page');
        }
        
        const data = await response.json();
        displayResults(data);
        
    } catch (error) {
        showError(error.message);
    } finally {
        hideLoading();
    }
});

// UI Helper Functions
function showLoading() {
    loadingIndicator.style.display = 'flex';
}

function hideLoading() {
    loadingIndicator.style.display = 'none';
}

function showError(message) {
    errorMessage.textContent = message;
    errorMessage.style.display = 'flex';
}

function hideError() {
    errorMessage.style.display = 'none';
}

function showResults() {
    resultsSection.style.display = 'grid';
}

function hideResults() {
    resultsSection.style.display = 'none';
}

// Results Rendering Functions
function displayResults(data) {
    hideError();
    showResults();
    
    // Clear previous results
    resultsSection.innerHTML = '';
    
    // Render target analysis
    if (data.target_analysis) {
        renderTargetAnalysis(data.target_analysis);
    }
    
    // Render benchmarks
    if (data.benchmarks) {
        renderBenchmarks(data.benchmarks);
    }
    
    // Render recommendations
    if (data.recommendations) {
        renderRecommendations(data.recommendations);
    }
    
    // Render competitor analysis
    if (data.competitor_analysis_summary) {
        renderCompetitorAnalysis(data.competitor_analysis_summary);
    }
}

function renderTargetAnalysis(analysis) {
    const card = document.createElement('div');
    card.className = 'results-card';
    card.innerHTML = `
        <h2>Target Analysis</h2>
        <div class="card-content">
            <p><strong>Title Length:</strong> ${analysis.title_length} characters</p>
            <p><strong>Meta Description:</strong> ${analysis.meta_description_length} characters</p>
            <p><strong>H1 Tags:</strong> ${analysis.h1_count}</p>
            <p><strong>Word Count:</strong> ${analysis.word_count}</p>
            <p><strong>Keyword Density:</strong> ${analysis.keyword_density}%</p>
        </div>
    `;
    resultsSection.appendChild(card);
}

function renderBenchmarks(benchmarks) {
    const card = document.createElement('div');
    card.className = 'results-card results-benchmarks';
    card.innerHTML = `
        <h2>Benchmarks</h2>
        <div class="card-content">
            <canvas id="benchmarksChart"></canvas>
        </div>
    `;
    resultsSection.appendChild(card);
    
    // Create chart
    const ctx = document.getElementById('benchmarksChart').getContext('2d');
    new Chart(ctx, {
        type: 'radar',
        data: {
            labels: Object.keys(benchmarks),
            datasets: [{
                label: 'Your Page',
                data: Object.values(benchmarks),
                backgroundColor: 'rgba(15, 52, 96, 0.2)',
                borderColor: 'rgba(15, 52, 96, 1)',
                pointBackgroundColor: 'rgba(15, 52, 96, 1)',
                pointBorderColor: '#fff',
                pointHoverBackgroundColor: '#fff',
                pointHoverBorderColor: 'rgba(15, 52, 96, 1)'
            }]
        },
        options: {
            scales: {
                r: {
                    beginAtZero: true,
                    ticks: {
                        stepSize: 1
                    }
                }
            }
        }
    });
}

function renderRecommendations(recommendations) {
    const card = document.createElement('div');
    card.className = 'results-card results-recommendations';
    card.innerHTML = `
        <h2>Recommendations</h2>
        <div class="card-content">
            ${recommendations.map(rec => `
                <div class="recommendation sev-${rec.severity.toLowerCase()}">
                    <h3>${rec.title}</h3>
                    <p>${rec.description}</p>
                </div>
            `).join('')}
        </div>
    `;
    resultsSection.appendChild(card);
}

function renderCompetitorAnalysis(competitors) {
    const card = document.createElement('div');
    card.className = 'results-card results-competitors';
    card.innerHTML = `
        <h2>Competitor Analysis</h2>
        <div class="card-content">
            ${competitors.map(comp => `
                <div class="competitor">
                    <h3>${comp.url}</h3>
                    <p><strong>Title Length:</strong> ${comp.title_length} characters</p>
                    <p><strong>Meta Description:</strong> ${comp.meta_description_length} characters</p>
                    <p><strong>Word Count:</strong> ${comp.word_count}</p>
                </div>
            `).join('')}
        </div>
    `;
    resultsSection.appendChild(card);
} 