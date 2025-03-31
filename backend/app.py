from flask import Flask, request, jsonify
from flask_cors import CORS
import json

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.get_json()
        print("Received request data:", data)
        
        # For testing, return a mock response
        mock_response = {
            "target_analysis": {
                "url": data.get('url', ''),
                "keyword": data.get('keyword', ''),
                "title_analysis": {
                    "length": 65,
                    "keyword_density": 0.8
                },
                "meta_analysis": {
                    "length": 155,
                    "keyword_density": 0.6
                },
                "content_analysis": {
                    "word_count": 450,
                    "keyword_density": 1.2
                },
                "recommendations": [
                    {
                        "title": "Title Tag Optimization",
                        "description": "Your title tag could be more descriptive and include the target keyword."
                    },
                    {
                        "title": "Meta Description",
                        "description": "Add a compelling meta description that includes your target keyword."
                    }
                ],
                "benchmarks": {
                    "avg_word_count": 500,
                    "avg_keyword_density": 1.0
                }
            }
        }
        
        return jsonify(mock_response)
    
    except Exception as e:
        print("Error processing request:", str(e))
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000) 