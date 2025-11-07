#!/usr/bin/env python3
"""
Simple script to run the Email Spam Classifier with proper setup.
"""

import os
import sys
import logging

# Set up environment
os.environ['SECRET_KEY'] = 'email-spam-classifier-secret-key'
os.environ['FLASK_ENV'] = 'development'
os.environ['FLASK_DEBUG'] = 'True'

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def main():
    """Run the application with proper setup."""
    print("🚀 Starting Email Spam Classifier...")
    print("=" * 50)
    
    try:
        # Import and create app
        from app import create_app
        
        print("✓ Creating Flask application...")
        app = create_app()
        
        # Test that services are ready
        with app.app_context():
            if hasattr(app, 'classification_service') and app.classification_service:
                is_ready = app.classification_service.is_ready()
                print(f"✓ Classification service ready: {is_ready}")
                
                if is_ready:
                    # Test classification
                    result = app.classification_service.classify_email("This is a test email.")
                    print(f"✓ Test classification: {result.prediction} (confidence: {result.confidence:.2f})")
                else:
                    print("⚠ Classification service not ready - models may need training")
            else:
                print("✗ Classification service not found")
        
        print("=" * 50)
        print("🌐 Starting web server...")
        print("📍 Open your browser and go to: http://localhost:5000")
        print("📊 Dashboard: http://localhost:5000/dashboard")
        print("🔍 API docs: http://localhost:5000/api/v1/docs")
        print("❤️ Health check: http://localhost:5000/health")
        print("=" * 50)
        print("Press Ctrl+C to stop the server")
        print()
        
        # Run the app
        app.run(
            host='0.0.0.0',
            port=5000,
            debug=True,
            use_reloader=True
        )
        
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting application: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)