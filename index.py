"""
Vercel entry point - simplified approach
"""

import os
import sys

# Ensure we can import our modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from app import create_app
    
    # Create Flask app
    app = create_app(os.environ.get('FLASK_ENV', 'production'))
    
except ImportError as e:
    # Fallback if imports fail
    from flask import Flask, jsonify
    
    app = Flask(__name__)
    
    @app.route('/')
    def home():
        return jsonify({
            'message': 'Email Spam Classifier API',
            'status': 'running',
            'error': f'Import error: {str(e)}'
        })
    
    @app.route('/health')
    def health():
        return jsonify({'status': 'ok', 'message': 'Service is running'})

except Exception as e:
    # Fallback for any other errors
    from flask import Flask, jsonify
    
    app = Flask(__name__)
    
    @app.route('/')
    def error():
        return jsonify({
            'error': 'Failed to initialize application',
            'message': str(e),
            'status': 'error'
        })

# This is what Vercel will use
if __name__ == '__main__':
    app.run(debug=True)