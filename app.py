"""
Main Flask application entry point for Email Spam Classifier.
"""
import os
import logging
from flask import Flask, jsonify, render_template
from datetime import datetime
from config import config


def create_app(config_name=None):
    """
    Application factory pattern for creating Flask app instances.
    
    Args:
        config_name: Configuration environment name ('development', 'production', 'testing')
        
    Returns:
        Flask application instance
    """
    if config_name is None:
        config_name = os.environ.get('FLASK_ENV', 'default')
    
    app = Flask(__name__)
    
    # Load configuration
    app.config.from_object(config[config_name])
    config[config_name].init_app(app)
    
    # Initialize logging
    _init_logging(app)
    
    # Register blueprints
    _register_blueprints(app)
    
    # Register error handlers
    _register_error_handlers(app)
    
    # Initialize services after app context is available
    with app.app_context():
        _init_services(app)
    
    return app


def _init_logging(app):
    """Initialize application logging."""
    if not app.debug and not app.testing:
        # Set up console logging for production
        if not app.logger.handlers:
            stream_handler = logging.StreamHandler()
            stream_handler.setLevel(logging.INFO)
            formatter = logging.Formatter(
                '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
            )
            stream_handler.setFormatter(formatter)
            app.logger.addHandler(stream_handler)
            app.logger.setLevel(logging.INFO)
            app.logger.info('Email Spam Classifier startup')


def _register_blueprints(app):
    """Register application blueprints."""
    from routes.web import web_bp
    from routes.api import api_bp
    
    app.register_blueprint(web_bp)
    app.register_blueprint(api_bp)


def _register_error_handlers(app):
    """Register global error handlers with enhanced error handling."""
    from src.error_handling import (
        enhanced_error_handler, ValidationError, ServiceNotReadyError, 
        DashboardErrorContext, BatchErrorContext
    )
    
    @app.errorhandler(ValidationError)
    def handle_validation_error(error):
        """Handle validation errors with enhanced context."""
        from flask import request
        if request.path.startswith('/api/'):
            return enhanced_error_handler.handle_api_error(error, 400)
        
        # Determine context based on route
        if 'batch' in request.path:
            context = BatchErrorContext(
                route=request.path,
                processing_stage="validation"
            )
            return enhanced_error_handler.handle_batch_error(error, context)
        elif 'dashboard' in request.path:
            context = DashboardErrorContext(
                route=request.path,
                user_action="validation"
            )
            return enhanced_error_handler.handle_dashboard_error(error, context)
        else:
            return enhanced_error_handler.handle_web_error(error, 'errors/400.html')
    
    @app.errorhandler(ServiceNotReadyError)
    def handle_service_not_ready_error(error):
        """Handle service not ready errors with enhanced context."""
        from flask import request
        if request.path.startswith('/api/'):
            return enhanced_error_handler.handle_api_error(error, 503)
        
        # Determine context based on route
        if 'batch' in request.path:
            context = BatchErrorContext(
                route=request.path,
                processing_stage="service_check"
            )
            return enhanced_error_handler.handle_batch_error(error, context)
        elif 'dashboard' in request.path:
            context = DashboardErrorContext(
                route=request.path,
                user_action="service_check"
            )
            return enhanced_error_handler.handle_dashboard_error(error, context)
        else:
            return enhanced_error_handler.handle_web_error(error, 'errors/503.html')
    
    @app.errorhandler(404)
    def not_found_error(error):
        """Handle 404 errors globally."""
        from flask import request
        if request.path.startswith('/api/'):
            return jsonify({
                'error': True,
                'message': 'API endpoint not found',
                'code': 'ENDPOINT_NOT_FOUND',
                'timestamp': datetime.now().isoformat()
            }), 404
        return render_template('errors/404.html'), 404
    
    @app.errorhandler(500)
    def internal_error(error):
        """Handle 500 errors globally with enhanced logging."""
        from flask import request
        
        # Log error with enhanced context
        enhanced_error_handler.log_error_with_context(
            error, 
            "GLOBAL_ERROR_HANDLER", 
            {
                'path': request.path,
                'method': request.method,
                'client_ip': request.remote_addr
            }
        )
        
        if request.path.startswith('/api/'):
            return jsonify({
                'error': True,
                'message': 'Internal server error',
                'code': 'INTERNAL_ERROR',
                'timestamp': datetime.now().isoformat()
            }), 500
        return render_template('errors/500.html'), 500
    
    @app.errorhandler(413)
    def request_entity_too_large(error):
        """Handle file size limit exceeded."""
        from flask import request
        if request.path.startswith('/api/'):
            return jsonify({
                'error': True,
                'message': 'Request entity too large',
                'code': 'REQUEST_TOO_LARGE',
                'timestamp': datetime.now().isoformat()
            }), 413
        return render_template('errors/413.html'), 413
    
    @app.errorhandler(400)
    def bad_request_error(error):
        """Handle 400 errors globally."""
        from flask import request
        if request.path.startswith('/api/'):
            return jsonify({
                'error': True,
                'message': 'Bad request',
                'code': 'BAD_REQUEST',
                'timestamp': datetime.now().isoformat()
            }), 400
        return render_template('errors/400.html'), 400


def _init_services(app):
    """Initialize application services with proper dependency injection and retry mechanisms."""
    import time
    
    retry_attempts = app.config.get('SERVICE_RETRY_ATTEMPTS', 3)
    retry_delay = 2  # seconds between retries
    
    # Initialize service health manager first (always succeeds)
    app.logger.info("Initializing ServiceHealthManager...")
    from src.services.service_health_manager import ServiceHealthManager
    service_health_manager = ServiceHealthManager(
        check_interval=app.config['SERVICE_HEALTH_CHECK_INTERVAL'],
        retry_attempts=app.config['SERVICE_RETRY_ATTEMPTS'],
        timeout=app.config['SERVICE_TIMEOUT']
    )
    
    # Store health manager immediately so it's always available
    app.service_health_manager = service_health_manager
    
    # Initialize enhanced error handler and make it available globally
    from src.error_handling import enhanced_error_handler
    app.enhanced_error_handler = enhanced_error_handler
    
    # Initialize services with retry logic
    services_initialized = False
    last_error = None
    
    for attempt in range(1, retry_attempts + 1):
        try:
            app.logger.info(f"Starting service initialization attempt {attempt}/{retry_attempts}...")
            
            # Import service classes
            app.logger.debug("Importing service classes...")
            from src.services.classification_service import ClassificationService
            from src.services.batch_processor import BatchProcessor
            from src.models.model_manager import ModelManager
            from src.preprocessing import PreprocessingPipeline
            
            # Initialize core components with individual error handling
            model_manager = _init_model_manager(app)
            preprocessing_pipeline = _init_preprocessing_pipeline(app)
            classification_service = _init_classification_service(app, model_manager, preprocessing_pipeline)
            batch_processor = _init_batch_processor(app, classification_service)
            
            # Set service references in health manager
            app.logger.info("Configuring ServiceHealthManager with service references...")
            service_health_manager.set_services(
                classification_service=classification_service,
                batch_processor=batch_processor,
                model_manager=model_manager,
                preprocessing_pipeline=preprocessing_pipeline
            )
            
            # Store services in app context for access by routes
            app.logger.debug("Storing services in app context...")
            app.classification_service = classification_service
            app.batch_processor = batch_processor
            app.model_manager = model_manager
            app.preprocessing_pipeline = preprocessing_pipeline
            
            # Try to load existing models (non-critical operation)
            _load_existing_models(app, model_manager)
            
            # Perform initial health check
            app.logger.info("Performing initial system health check...")
            health_summary = service_health_manager.check_all_services()
            app.logger.info(f"Initial system health status: {health_summary.overall_status.value}")
            
            # Verify service readiness
            app.logger.info("Verifying service readiness...")
            if service_health_manager.is_system_ready():
                app.logger.info("System is ready for use")
            else:
                app.logger.warning(f"System not fully ready: {service_health_manager.get_system_readiness_message()}")
            
            app.logger.info(f'Services initialized successfully on attempt {attempt}')
            services_initialized = True
            break
            
        except Exception as e:
            last_error = e
            app.logger.warning(f'Service initialization attempt {attempt} failed: {str(e)}')
            
            if attempt < retry_attempts:
                app.logger.info(f'Retrying service initialization in {retry_delay} seconds...')
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                app.logger.error(f'All {retry_attempts} service initialization attempts failed', exc_info=True)
    
    # If initialization failed, create fallback services
    if not services_initialized:
        app.logger.error(f'Failed to initialize services after {retry_attempts} attempts. Last error: {str(last_error)}')
        _create_fallback_services(app, service_health_manager)


def _init_model_manager(app):
    """Initialize ModelManager with error handling."""
    try:
        app.logger.info(f"Initializing ModelManager with models_dir: {app.config['TRAINED_MODELS_DIR']}")
        from src.models.model_manager import ModelManager
        return ModelManager(models_dir=app.config['TRAINED_MODELS_DIR'])
    except Exception as e:
        app.logger.error(f"Failed to initialize ModelManager: {str(e)}")
        raise


def _init_preprocessing_pipeline(app):
    """Initialize PreprocessingPipeline with error handling."""
    try:
        app.logger.info("Initializing PreprocessingPipeline...")
        from src.preprocessing import PreprocessingPipeline
        return PreprocessingPipeline()
    except Exception as e:
        app.logger.error(f"Failed to initialize PreprocessingPipeline: {str(e)}")
        raise


def _init_classification_service(app, model_manager, preprocessing_pipeline):
    """Initialize ClassificationService with error handling."""
    try:
        app.logger.info("Initializing ClassificationService...")
        from src.services.classification_service import ClassificationService
        return ClassificationService(
            model_manager=model_manager,
            preprocessing_pipeline=preprocessing_pipeline
        )
    except Exception as e:
        app.logger.error(f"Failed to initialize ClassificationService: {str(e)}")
        raise


def _init_batch_processor(app, classification_service):
    """Initialize BatchProcessor with error handling."""
    try:
        app.logger.info("Initializing BatchProcessor...")
        from src.services.batch_processor import BatchProcessor
        return BatchProcessor(
            classification_service=classification_service,
            results_dir=app.config['RESULTS_DIR'],
            max_batch_size=app.config['MAX_BATCH_SIZE']
        )
    except Exception as e:
        app.logger.error(f"Failed to initialize BatchProcessor: {str(e)}")
        raise


def _load_existing_models(app, model_manager):
    """Load existing models with error handling (non-critical operation)."""
    try:
        app.logger.info("Attempting to load existing models...")
        loaded_models = model_manager.load_models()
        if loaded_models:
            app.logger.info(f'Successfully loaded {len(loaded_models)} existing models: {list(loaded_models.keys())}')
        else:
            app.logger.warning('No existing models found. System will need training before use.')
    except Exception as e:
        app.logger.warning(f"Failed to load existing models (non-critical): {str(e)}")


def _create_fallback_services(app, service_health_manager):
    """Create minimal fallback services to prevent startup failure."""
    app.logger.warning("Creating fallback services to prevent startup failure")
    
    try:
        # Set fallback services in app context
        app.classification_service = service_health_manager.get_fallback_classification_service()
        app.batch_processor = service_health_manager.get_fallback_batch_processor()
        app.model_manager = None
        app.preprocessing_pipeline = None
        
        app.logger.info("Fallback services created successfully")
        
    except Exception as e:
        app.logger.critical(f"Failed to create fallback services: {str(e)}", exc_info=True)
        # Set minimal None services as last resort
        app.classification_service = None
        app.batch_processor = None
        app.model_manager = None
        app.preprocessing_pipeline = None


if __name__ == '__main__':
    app = create_app()
    app.run(debug=True)