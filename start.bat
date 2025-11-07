@echo off
REM Email Spam Classifier startup script for Windows
REM This script handles the startup process for the application

echo Starting Email Spam Classifier...

REM Set default values
if not defined FLASK_ENV set FLASK_ENV=production
if not defined PORT set PORT=8000
if not defined WORKERS set WORKERS=4

echo Environment: %FLASK_ENV%
echo Port: %PORT%
echo Workers: %WORKERS%

REM Check if models exist
if not exist "models\trained\naive_bayes.pkl" (
    echo Trained models not found. Running training script...
    python train_models.py
    if errorlevel 1 (
        echo Training failed!
        exit /b 1
    )
)

REM Run deployment checks
echo Running deployment checks...
python deploy.py
if errorlevel 1 (
    echo Deployment checks failed!
    exit /b 1
)

REM Start the application
echo Starting application with Gunicorn...
gunicorn --config gunicorn.conf.py wsgi:application