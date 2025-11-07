#!/usr/bin/env python3
"""
WSGI entry point for production deployment.

This module provides the WSGI application instance for deployment
with production WSGI servers like Gunicorn or uWSGI.
"""

import os
from app import create_app

# Create application instance
application = create_app(os.environ.get('FLASK_ENV', 'production'))

# Vercel expects 'app' variable
app = application

if __name__ == "__main__":
    application.run()