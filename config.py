"""
Flask application configuration and environment management.
"""
import os
from pathlib import Path


class Config:
    """Base configuration class."""
    
    # Flask settings
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    WTF_CSRF_ENABLED = os.environ.get('WTF_CSRF_ENABLED', 'True').lower() == 'true'
    
    # Application settings
    APP_NAME = os.environ.get('APP_NAME', 'Email Spam Classifier')
    APP_VERSION = os.environ.get('APP_VERSION', '1.0.0')
    
    # Model settings
    MODEL_DIR = os.environ.get('MODEL_DIR') or os.path.join(os.path.dirname(__file__), 'models')
    TRAINED_MODELS_DIR = os.environ.get('TRAINED_MODELS_DIR') or os.path.join(MODEL_DIR, 'trained')
    DEFAULT_MODEL = os.environ.get('DEFAULT_MODEL', 'naive_bayes')
    
    # Data settings
    DATA_DIR = os.environ.get('DATA_DIR') or os.path.join(os.path.dirname(__file__), 'data')
    UPLOAD_DIR = os.environ.get('UPLOAD_DIR') or os.path.join(os.path.dirname(__file__), 'uploads')
    RESULTS_DIR = os.environ.get('RESULTS_DIR') or os.path.join(os.path.dirname(__file__), 'results')
    LOGS_DIR = os.environ.get('LOGS_DIR') or os.path.join(os.path.dirname(__file__), 'logs')
    
    # Processing limits
    MAX_BATCH_SIZE = int(os.environ.get('MAX_BATCH_SIZE', 1000))
    MAX_FILE_SIZE = int(os.environ.get('MAX_FILE_SIZE', 16 * 1024 * 1024))  # 16MB
    CLASSIFICATION_TIMEOUT = float(os.environ.get('CLASSIFICATION_TIMEOUT', 3.0))  # seconds
    
    # API settings
    API_RATE_LIMIT = os.environ.get('API_RATE_LIMIT', '100 per hour')
    API_TIMEOUT = int(os.environ.get('API_TIMEOUT', 30))
    
    # Logging settings
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
    LOG_FILE = os.environ.get('LOG_FILE', 'logs/email_classifier.log')
    LOG_MAX_BYTES = int(os.environ.get('LOG_MAX_BYTES', 10 * 1024 * 1024))  # 10MB
    LOG_BACKUP_COUNT = int(os.environ.get('LOG_BACKUP_COUNT', 10))
    
    # Security settings
    FORCE_HTTPS = os.environ.get('FORCE_HTTPS', 'False').lower() == 'true'
    SESSION_COOKIE_SECURE = os.environ.get('SESSION_COOKIE_SECURE', 'False').lower() == 'true'
    SESSION_COOKIE_HTTPONLY = os.environ.get('SESSION_COOKIE_HTTPONLY', 'True').lower() == 'true'
    SESSION_COOKIE_SAMESITE = os.environ.get('SESSION_COOKIE_SAMESITE', 'Lax')
    
    # Performance settings
    WORKERS = int(os.environ.get('WORKERS', 4))
    THREADS = int(os.environ.get('THREADS', 2))
    TIMEOUT = int(os.environ.get('TIMEOUT', 120))
    
    # Supported file formats
    ALLOWED_EXTENSIONS = {'csv', 'txt'}
    
    # Service health check settings
    SERVICE_HEALTH_CHECK_INTERVAL = int(os.environ.get('SERVICE_HEALTH_CHECK_INTERVAL', 30))  # seconds
    SERVICE_RETRY_ATTEMPTS = int(os.environ.get('SERVICE_RETRY_ATTEMPTS', 3))
    SERVICE_TIMEOUT = int(os.environ.get('SERVICE_TIMEOUT', 10))  # seconds
    
    # Service initialization settings
    SERVICE_INIT_RETRY_DELAY = int(os.environ.get('SERVICE_INIT_RETRY_DELAY', 2))  # seconds
    SERVICE_INIT_MAX_BACKOFF = int(os.environ.get('SERVICE_INIT_MAX_BACKOFF', 30))  # seconds
    
    # Enhanced error handling settings
    ERROR_TRACKING_ENABLED = os.environ.get('ERROR_TRACKING_ENABLED', 'True').lower() == 'true'
    ERROR_PATTERN_LIMIT = int(os.environ.get('ERROR_PATTERN_LIMIT', 100))  # Max errors per pattern
    ERROR_CACHE_DURATION = int(os.environ.get('ERROR_CACHE_DURATION', 300))  # seconds (5 minutes)
    
    # Dashboard settings
    DASHBOARD_REFRESH_INTERVAL = int(os.environ.get('DASHBOARD_REFRESH_INTERVAL', 30))  # seconds
    DASHBOARD_CACHE_TIMEOUT = int(os.environ.get('DASHBOARD_CACHE_TIMEOUT', 300))  # seconds
    DASHBOARD_FALLBACK_DATA_ENABLED = os.environ.get('DASHBOARD_FALLBACK_DATA_ENABLED', 'True').lower() == 'true'
    
    # Batch processing settings
    BATCH_VALIDATION_STRICT = os.environ.get('BATCH_VALIDATION_STRICT', 'True').lower() == 'true'
    BATCH_CLEANUP_INTERVAL = int(os.environ.get('BATCH_CLEANUP_INTERVAL', 3600))  # seconds (1 hour)
    BATCH_MAX_RETRIES = int(os.environ.get('BATCH_MAX_RETRIES', 2))
    BATCH_PARTIAL_RESULTS_ENABLED = os.environ.get('BATCH_PARTIAL_RESULTS_ENABLED', 'True').lower() == 'true'
    
    # Error message templates
    ERROR_MESSAGE_TEMPLATES = {
        'DASHBOARD_SERVICE_NOT_READY': os.environ.get(
            'ERROR_MSG_DASHBOARD_SERVICE_NOT_READY',
            'The system is still initializing. Please wait a moment and refresh the page.'
        ),
        'DASHBOARD_MODEL_NOT_TRAINED': os.environ.get(
            'ERROR_MSG_DASHBOARD_MODEL_NOT_TRAINED',
            'No trained models are available. Please train a model first or contact your administrator.'
        ),
        'BATCH_SERVICE_NOT_READY': os.environ.get(
            'ERROR_MSG_BATCH_SERVICE_NOT_READY',
            'The batch processing service is not ready. Please wait for system initialization to complete.'
        ),
        'BATCH_FILE_INVALID': os.environ.get(
            'ERROR_MSG_BATCH_FILE_INVALID',
            'The uploaded file is not valid. Please check the file format and try again.'
        ),
        'SERVICE_INITIALIZATION_ERROR': os.environ.get(
            'ERROR_MSG_SERVICE_INITIALIZATION_ERROR',
            'A system service failed to start properly. Please contact your administrator.'
        ),
        'INTERNAL_ERROR': os.environ.get(
            'ERROR_MSG_INTERNAL_ERROR',
            'An unexpected error occurred. Please try again or contact support if the problem persists.'
        )
    }
    
    @staticmethod
    def init_app(app):
        """Initialize application with configuration."""
        # Create necessary directories
        dirs_to_create = [
            Config.MODEL_DIR,
            Config.TRAINED_MODELS_DIR,
            Config.DATA_DIR,
            Config.UPLOAD_DIR,
            Config.RESULTS_DIR,
            Config.LOGS_DIR
        ]
        
        for directory in dirs_to_create:
            Path(directory).mkdir(parents=True, exist_ok=True)


class DevelopmentConfig(Config):
    """Development environment configuration."""
    DEBUG = True
    TESTING = False
    
    # Development-specific settings
    LOG_LEVEL = 'DEBUG'


class ProductionConfig(Config):
    """Production environment configuration."""
    DEBUG = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    TESTING = False
    
    # Production-specific settings
    SECRET_KEY = os.environ.get('SECRET_KEY')
    SESSION_COOKIE_SECURE = True
    FORCE_HTTPS = True
    
    @classmethod
    def init_app(cls, app):
        Config.init_app(app)
        
        # Ensure secret key is set in production
        if not cls.SECRET_KEY:
            raise ValueError("SECRET_KEY environment variable must be set in production")
        
        # Production-specific initialization
        import logging
        from logging.handlers import RotatingFileHandler, SMTPHandler
        
        # Set up file logging
        if not app.debug and not app.testing:
            # Ensure logs directory exists
            os.makedirs(os.path.dirname(cls.LOG_FILE), exist_ok=True)
            
            file_handler = RotatingFileHandler(
                cls.LOG_FILE,
                maxBytes=cls.LOG_MAX_BYTES,
                backupCount=cls.LOG_BACKUP_COUNT
            )
            file_handler.setFormatter(logging.Formatter(
                '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
            ))
            
            # Set log level from configuration
            log_level = getattr(logging, cls.LOG_LEVEL.upper(), logging.INFO)
            file_handler.setLevel(log_level)
            app.logger.addHandler(file_handler)
            app.logger.setLevel(log_level)
            
            # Add email handler for critical errors (optional)
            mail_handler = None
            if os.environ.get('MAIL_SERVER'):
                mail_handler = SMTPHandler(
                    mailhost=(os.environ.get('MAIL_SERVER'), os.environ.get('MAIL_PORT', 587)),
                    fromaddr=os.environ.get('MAIL_FROM_ADDR', 'noreply@example.com'),
                    toaddrs=os.environ.get('MAIL_TO_ADDRS', '').split(','),
                    subject='Email Spam Classifier Error',
                    credentials=(os.environ.get('MAIL_USERNAME'), os.environ.get('MAIL_PASSWORD')),
                    secure=()
                )
                mail_handler.setLevel(logging.ERROR)
                mail_handler.setFormatter(logging.Formatter(
                    '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
                ))
                app.logger.addHandler(mail_handler)
            
            app.logger.info('Email Spam Classifier production startup')


class TestingConfig(Config):
    """Testing environment configuration."""
    TESTING = True
    DEBUG = True
    
    # Testing-specific settings
    WTF_CSRF_ENABLED = False
    LOG_LEVEL = 'DEBUG'


# Configuration mapping
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}