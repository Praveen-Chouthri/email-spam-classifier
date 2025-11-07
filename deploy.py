#!/usr/bin/env python3
"""
Deployment script for Email Spam Classifier.

This script handles deployment tasks including:
- Environment validation
- Model verification
- Database setup (if needed)
- Service health checks
"""

import os
import sys
import logging
import subprocess
import time
from pathlib import Path

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from config import config


def setup_logging():
    """Set up logging for deployment."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)


def check_environment():
    """Check that all required environment variables are set."""
    logger = logging.getLogger(__name__)
    
    required_vars = [
        'SECRET_KEY',
        'FLASK_ENV'
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.environ.get(var):
            missing_vars.append(var)
    
    if missing_vars:
        logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
        return False
    
    logger.info("Environment variables validated successfully")
    return True


def create_directories():
    """Create necessary directories for the application."""
    logger = logging.getLogger(__name__)
    
    app_config = config[os.environ.get('FLASK_ENV', 'production')]()
    
    directories = [
        app_config.MODEL_DIR,
        app_config.TRAINED_MODELS_DIR,
        app_config.DATA_DIR,
        app_config.UPLOAD_DIR,
        app_config.RESULTS_DIR,
        app_config.LOGS_DIR
    ]
    
    for directory in directories:
        try:
            Path(directory).mkdir(parents=True, exist_ok=True)
            logger.info(f"Created/verified directory: {directory}")
        except Exception as e:
            logger.error(f"Failed to create directory {directory}: {str(e)}")
            return False
    
    return True


def check_models():
    """Check that trained models are available."""
    logger = logging.getLogger(__name__)
    
    app_config = config[os.environ.get('FLASK_ENV', 'production')]()
    models_dir = app_config.TRAINED_MODELS_DIR
    
    required_models = ['naive_bayes.pkl', 'random_forest.pkl', 'decision_tree.pkl']
    missing_models = []
    
    for model_file in required_models:
        model_path = os.path.join(models_dir, model_file)
        if not os.path.exists(model_path):
            missing_models.append(model_file)
    
    if missing_models:
        logger.warning(f"Missing model files: {', '.join(missing_models)}")
        logger.info("You may need to run the training script: python train_models.py")
        return False
    
    logger.info("All required model files found")
    return True


def install_dependencies():
    """Install Python dependencies."""
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("Installing Python dependencies...")
        subprocess.run([
            sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'
        ], check=True, capture_output=True, text=True)
        logger.info("Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install dependencies: {e.stderr}")
        return False


def run_health_check():
    """Run a basic health check on the application."""
    logger = logging.getLogger(__name__)
    
    try:
        # Import and create app to test basic functionality
        from app import create_app
        
        app = create_app(os.environ.get('FLASK_ENV', 'production'))
        
        with app.app_context():
            # Test that services can be initialized
            if hasattr(app, 'classification_service') and app.classification_service:
                logger.info("Classification service initialized successfully")
            else:
                logger.warning("Classification service not initialized")
            
            if hasattr(app, 'model_manager') and app.model_manager:
                logger.info("Model manager initialized successfully")
            else:
                logger.warning("Model manager not initialized")
        
        logger.info("Basic health check passed")
        return True
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return False


def setup_systemd_service():
    """Create systemd service file for production deployment."""
    logger = logging.getLogger(__name__)
    
    service_content = f"""[Unit]
Description=Email Spam Classifier
After=network.target

[Service]
Type=exec
User=www-data
Group=www-data
WorkingDirectory={os.getcwd()}
Environment=PATH={os.getcwd()}/venv/bin
ExecStart={os.getcwd()}/venv/bin/gunicorn --config gunicorn.conf.py wsgi:application
ExecReload=/bin/kill -s HUP $MAINPID
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
"""
    
    service_file = '/etc/systemd/system/email-spam-classifier.service'
    
    try:
        with open('email-spam-classifier.service', 'w') as f:
            f.write(service_content)
        
        logger.info(f"Systemd service file created: email-spam-classifier.service")
        logger.info(f"To install: sudo cp email-spam-classifier.service {service_file}")
        logger.info("To enable: sudo systemctl enable email-spam-classifier")
        logger.info("To start: sudo systemctl start email-spam-classifier")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to create systemd service file: {str(e)}")
        return False


def main():
    """Main deployment function."""
    logger = setup_logging()
    logger.info("Starting Email Spam Classifier deployment")
    
    # Load environment variables from .env file if it exists
    try:
        from dotenv import load_dotenv
        if os.path.exists('.env'):
            load_dotenv()
            logger.info("Loaded environment variables from .env file")
    except ImportError:
        logger.warning("python-dotenv not available, skipping .env file loading")
    
    success = True
    
    # Step 1: Check environment
    logger.info("Step 1: Checking environment variables...")
    if not check_environment():
        success = False
    
    # Step 2: Create directories
    logger.info("Step 2: Creating necessary directories...")
    if not create_directories():
        success = False
    
    # Step 3: Install dependencies (skip in development)
    if os.environ.get('FLASK_ENV') == 'production':
        logger.info("Step 3: Installing dependencies...")
        if not install_dependencies():
            success = False
    else:
        logger.info("Step 3: Skipping dependency installation in development mode")
    
    # Step 4: Check models
    logger.info("Step 4: Checking trained models...")
    models_available = check_models()
    if not models_available:
        logger.info("Models not found. Attempting to train models...")
        try:
            subprocess.run([sys.executable, 'train_models.py'], check=True)
            logger.info("Model training completed successfully")
        except subprocess.CalledProcessError as e:
            logger.error("Model training failed")
            success = False
    
    # Step 5: Health check
    logger.info("Step 5: Running health check...")
    if not run_health_check():
        success = False
    
    # Step 6: Create systemd service (optional)
    if os.environ.get('CREATE_SYSTEMD_SERVICE', 'false').lower() == 'true':
        logger.info("Step 6: Creating systemd service file...")
        setup_systemd_service()
    
    if success:
        logger.info("✓ Deployment completed successfully!")
        logger.info("Application is ready to start.")
        logger.info("To start with Gunicorn: gunicorn --config gunicorn.conf.py wsgi:application")
        logger.info("To start with Docker: docker-compose up -d")
    else:
        logger.error("✗ Deployment failed. Please check the errors above.")
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)