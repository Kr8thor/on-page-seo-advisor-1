# On-Page SEO Analyzer & Advisor

A comprehensive API for analyzing web pages against SEO best practices and benchmarking against competitors.

## Features

- Single page analysis
- Competitor benchmarking
- SEO recommendations
- SERP analysis
- Comprehensive error handling
- Rate limiting
- Caching

## Prerequisites

- Python 3.8 or higher
- SERP API key (from ValueSERP or SerpApi)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd on-page-seo-analyzer
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the project root with your configuration:
```env
SERP_API_KEY=your_api_key_here
DEFAULT_COUNTRY=us
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_WINDOW=60
CACHE_DURATION_HOURS=24
```

## Development

Run the development server:
```bash
uvicorn src.main:app --reload --port 8000
```

## Production Deployment

The application is configured for deployment with Gunicorn and Uvicorn workers.

### Using Procfile (e.g., on Render)

The application includes a `Procfile` that configures Gunicorn with optimized settings:
```
web: gunicorn -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:$PORT --timeout 120 --keep-alive 5 --log-level info src.main:app
```

Configuration details:
- `-w 4`: 4 worker processes (adjust based on your server resources)
- `-k uvicorn.workers.UvicornWorker`: Uses Uvicorn workers for async support
- `--bind 0.0.0.0:$PORT`: Binds to all network interfaces and uses the PORT environment variable
- `--timeout 120`: 120-second timeout for worker processes
- `--keep-alive 5`: Keeps connections alive for 5 seconds
- `--log-level info`: Sets logging level to info for better monitoring

### Manual Deployment

1. Install production dependencies:
```bash
pip install -r requirements.txt
```

2. Start the production server:
```bash
gunicorn -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000 --timeout 120 --keep-alive 5 --log-level info src.main:app
```

### Worker Configuration Guidelines

- **Number of Workers**: 
  - General rule: `(2 x num_cores) + 1`
  - For Render: 2-4 workers is common
  - Adjust based on your plan's resources

- **Timeout Settings**:
  - Default: 120 seconds
  - Adjust based on your analysis requirements
  - Consider increasing for complex analyses

- **Keep-Alive**:
  - Default: 5 seconds
  - Helps maintain persistent connections
  - Adjust based on your traffic patterns

## API Documentation

Once the server is running, visit:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Environment Variables

- `SERP_API_KEY`: Your SERP API key (required)
- `DEFAULT_COUNTRY`: Default country code for SERP results (default: us)
- `RATE_LIMIT_REQUESTS`: Maximum number of requests per minute (default: 100)
- `RATE_LIMIT_WINDOW`: Time window for rate limiting in seconds (default: 60)
- `CACHE_DURATION_HOURS`: Duration to cache analysis results (default: 24)

## Security Notes

- Never commit the `.env` file to version control
- Keep your API keys secure
- The application includes rate limiting to prevent abuse
- All API responses are sanitized to prevent information leakage

## License

[Your License Here] 