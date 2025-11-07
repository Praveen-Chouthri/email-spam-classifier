#!/bin/bash

# Email Spam Classifier startup script
# This script handles the startup process for the application

set -e

echo "Starting Email Spam Classifier..."

# Load environment variables
if [ -f .env ]; then
    echo "Loading environment variables from .env file..."
    export $(cat .env | grep -v '^#' | xargs)
fi

# Set default values
export FLASK_ENV=${FLASK_ENV:-production}
export PORT=${PORT:-8000}
export WORKERS=${WORKERS:-4}

echo "Environment: $FLASK_ENV"
echo "Port: $PORT"
echo "Workers: $WORKERS"

# Check if models exist
if [ ! -f "models/trained/naive_bayes.pkl" ]; then
    echo "Trained models not found. Running training script..."
    python train_models.py
fi

# Run deployment checks
echo "Running deployment checks..."
python deploy.py

# Start the application
echo "Starting application with Gunicorn..."
exec gunicorn --config gunicorn.conf.py wsgi:application