"""
Vercel serverless function entry point.
This file serves as the main entry point for Vercel deployment.
"""

import sys
import os

# Add the parent directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

try:
    from app import create_app
    
    # Create the Flask application
    app = create_app('production')
    
    # For Vercel, we need to handle the request directly
    def handler(request):
        return app(request.environ, lambda status, headers: None)
    
except Exception as e:
    # Fallback Flask app for debugging
    from flask import Flask, jsonify
    
    app = Flask(__name__)
    
    @app.route('/')
    def index():
        return jsonify({
            'error': 'Application failed to initialize',
            'message': str(e),
            'status': 'error'
        })
    
    @app.route('/<path:path>')
    def catch_all(path):
        return jsonify({
            'error': 'Application failed to initialize',
            'message': str(e),
            'path': path,
            'status': 'error'
        })

# Vercel expects the app to be available at module level
if __name__ == "__main__":
    app.run(debug=True)