"""
REST API routes for the Email Spam Classifier.
"""
from flask import Blueprint, request, jsonify, current_app
from functools import wraps
import time
from datetime import datetime, timedelta
import uuid
import os

from src.error_handling import (
    error_handler, enhanced_error_handler, input_validator, health_checker, enhanced_logger,
    ValidationError, ServiceNotReadyError, ProcessingError
)

# Create blueprint
api_bp = Blueprint('api', __name__, url_prefix='/api/v1')

# Simple in-memory rate limiting (in production, use Redis or database)
rate_limit_storage = {}


def get_service_health_manager():
    """Get service health manager from current app context."""
    from flask import current_app
    if not hasattr(current_app, 'service_health_manager') or current_app.service_health_manager is None:
        raise ServiceNotReadyError("Service Health Manager", "Service health manager not initialized during application startup")
    return current_app.service_health_manager


def get_classification_service():
    """Get classification service from current app context with readiness validation."""
    from flask import current_app
    if not hasattr(current_app, 'classification_service') or current_app.classification_service is None:
        raise ServiceNotReadyError("Classification Service", "Classification service not initialized. Please check application startup.")
    return current_app.classification_service


def get_batch_processor():
    """Get batch processor from current app context with readiness validation."""
    from flask import current_app
    if not hasattr(current_app, 'batch_processor') or current_app.batch_processor is None:
        raise ServiceNotReadyError("Batch Processor", "Batch processor not initialized. Please check application startup.")
    return current_app.batch_processor


def validate_service_readiness(service_name: str, require_ready: bool = True):
    """
    Validate service readiness using the service health manager.
    
    Args:
        service_name: Name of the service to validate
        require_ready: If True, raises exception if service not ready
        
    Returns:
        Tuple of (is_ready, message)
        
    Raises:
        ServiceNotReadyError: If require_ready=True and service is not ready
    """
    try:
        health_manager = get_service_health_manager()
        is_ready, message = health_manager.validate_service_readiness(service_name)
        
        if require_ready and not is_ready:
            raise ServiceNotReadyError(service_name, message)
        
        return is_ready, message
        
    except ServiceNotReadyError:
        # Re-raise service not ready errors
        raise
    except Exception as e:
        # Handle other errors (health manager issues)
        error_msg = f"Unable to validate service readiness: {str(e)}"
        if require_ready:
            raise ServiceNotReadyError(service_name, error_msg)
        return False, error_msg


def check_classification_service_readiness():
    """Check if classification service is ready for operation."""
    validate_service_readiness('classification_service', require_ready=True)


def check_batch_processor_readiness():
    """Check if batch processor is ready for operation."""
    validate_service_readiness('batch_processor', require_ready=True)


@api_bp.before_request
def before_request():
    """Pre-process API requests."""
    # Add CORS headers
    if request.method == 'OPTIONS':
        response = jsonify({'status': 'OK'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
        return response
    
    # Enhanced API request logging
    enhanced_logger.log_operation_start(
        component="api",
        operation=f"{request.method}_{request.endpoint or 'unknown'}",
        context_data={
            'method': request.method,
            'path': request.path,
            'client_ip': request.remote_addr,
            'user_agent': request.headers.get('User-Agent', 'Unknown'),
            'content_type': request.headers.get('Content-Type'),
            'content_length': request.headers.get('Content-Length')
        }
    )


@api_bp.after_request
def after_request(response):
    """Post-process API responses."""
    # Add CORS headers to all responses
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
    
    # Add security headers
    response.headers.add('X-Content-Type-Options', 'nosniff')
    response.headers.add('X-Frame-Options', 'DENY')
    response.headers.add('X-XSS-Protection', '1; mode=block')
    
    # Enhanced API response logging
    enhanced_logger.log_operation_end(
        operation_id="api_request",  # This would ideally be stored from before_request
        component="api",
        operation=f"{request.method}_{request.endpoint or 'unknown'}",
        success=response.status_code < 400,
        result_data={
            'status_code': response.status_code,
            'content_type': response.headers.get('Content-Type'),
            'content_length': response.headers.get('Content-Length')
        }
    )
    
    return response





def rate_limit(max_requests=100, window_minutes=60):
    """Rate limiting decorator."""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            client_ip = request.remote_addr
            now = datetime.now()
            window_start = now - timedelta(minutes=window_minutes)
            
            # Clean old entries
            if client_ip in rate_limit_storage:
                rate_limit_storage[client_ip] = [
                    req_time for req_time in rate_limit_storage[client_ip]
                    if req_time > window_start
                ]
            else:
                rate_limit_storage[client_ip] = []
            
            # Check rate limit
            if len(rate_limit_storage[client_ip]) >= max_requests:
                return jsonify({
                    'error': True,
                    'message': 'Rate limit exceeded',
                    'code': 'RATE_LIMIT_EXCEEDED',
                    'timestamp': now.isoformat()
                }), 429
            
            # Add current request
            rate_limit_storage[client_ip].append(now)
            
            return f(*args, **kwargs)
        return decorated_function
    return decorator


def require_auth(f):
    """Simple authentication decorator (placeholder for real auth)."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            return jsonify({
                'error': True,
                'message': 'Authentication required',
                'code': 'AUTH_REQUIRED',
                'timestamp': datetime.now().isoformat()
            }), 401
        
        # In production, validate the token properly
        token = auth_header.split(' ')[1]
        if token != 'demo-token':  # Placeholder validation
            return jsonify({
                'error': True,
                'message': 'Invalid authentication token',
                'code': 'INVALID_TOKEN',
                'timestamp': datetime.now().isoformat()
            }), 401
        
        return f(*args, **kwargs)
    return decorated_function


def validate_json(required_fields=None):
    """Validate JSON request data."""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if not request.is_json:
                return jsonify({
                    'error': True,
                    'message': 'Content-Type must be application/json',
                    'code': 'INVALID_CONTENT_TYPE',
                    'timestamp': datetime.now().isoformat()
                }), 400
            
            try:
                data = request.get_json()
                if data is None:
                    raise ValueError("Invalid JSON")
            except Exception:
                return jsonify({
                    'error': True,
                    'message': 'Invalid JSON format',
                    'code': 'INVALID_JSON',
                    'timestamp': datetime.now().isoformat()
                }), 400
            
            # Check required fields
            if required_fields:
                missing_fields = [field for field in required_fields if field not in data or not data[field]]
                if missing_fields:
                    return jsonify({
                        'error': True,
                        'message': f'Missing required fields: {", ".join(missing_fields)}',
                        'code': 'MISSING_REQUIRED_FIELDS',
                        'timestamp': datetime.now().isoformat()
                    }), 400
            
            return f(*args, **kwargs)
        return decorated_function
    return decorator


@api_bp.route('/classify', methods=['POST'])
@rate_limit(max_requests=100, window_minutes=60)
@validate_json(required_fields=['email_text'])
def classify_email():
    """Classify a single email via API."""
    try:
        data = request.get_json()
        email_text = data['email_text']
        
        # Validate input
        input_validator.validate_email_text(email_text)
        
        # Check service readiness
        check_classification_service_readiness()
        
        # Get service (already validated as ready)
        classification_service = get_classification_service()
        
        # Classify the email
        result = classification_service.classify_email(email_text)
        
        # Prepare response data
        response_data = {
            'prediction': result.prediction,
            'confidence': result.confidence,
            'model_used': result.model_used,
            'processing_time': result.processing_time,
            'timestamp': result.timestamp.isoformat()
        }
        
        # Add balancing information if available
        if hasattr(result, 'model_balanced'):
            response_data['model_balanced'] = result.model_balanced
            if hasattr(result, 'balancing_method'):
                response_data['balancing_method'] = result.balancing_method
            if hasattr(result, 'false_negative_rate'):
                response_data['false_negative_rate'] = result.false_negative_rate
        
        return jsonify({
            'error': False,
            'data': response_data
        }), 200
    
    except ValidationError as e:
        return error_handler.handle_api_error(e, 400)
    
    except ServiceNotReadyError as e:
        return error_handler.handle_api_error(e, 503)
    
    except Exception as e:
        return error_handler.handle_api_error(e, 500)


@api_bp.route('/classify/batch', methods=['POST'])
@rate_limit(max_requests=10, window_minutes=60)
@require_auth
@validate_json(required_fields=['emails'])
def classify_batch():
    """Start batch classification job via API."""
    try:
        data = request.get_json()
        emails = data['emails']
        
        # Validate batch
        max_batch_size = current_app.config.get('MAX_BATCH_SIZE', 1000)
        input_validator.validate_email_batch(emails, max_batch_size)
        
        # Check service readiness
        check_classification_service_readiness()
        
        # Get service (already validated as ready)
        classification_service = get_classification_service()
        
        # Process batch
        job_id = str(uuid.uuid4())
        results = classification_service.classify_batch(emails)
        
        # Prepare batch results with balancing information
        batch_results = []
        for result in results:
            result_data = {
                'prediction': result.prediction,
                'confidence': result.confidence,
                'model_used': result.model_used,
                'processing_time': result.processing_time,
                'timestamp': result.timestamp.isoformat()
            }
            
            # Add balancing information if available
            if hasattr(result, 'model_balanced'):
                result_data['model_balanced'] = result.model_balanced
                if hasattr(result, 'balancing_method'):
                    result_data['balancing_method'] = result.balancing_method
                if hasattr(result, 'false_negative_rate'):
                    result_data['false_negative_rate'] = result.false_negative_rate
            
            batch_results.append(result_data)
        
        return jsonify({
            'error': False,
            'data': {
                'job_id': job_id,
                'total_emails': len(emails),
                'results': batch_results
            }
        }), 200
    
    except ValidationError as e:
        return error_handler.handle_api_error(e, 400)
    
    except ServiceNotReadyError as e:
        return error_handler.handle_api_error(e, 503)
    
    except Exception as e:
        return error_handler.handle_api_error(e, 500)


@api_bp.route('/models', methods=['GET'])
@rate_limit(max_requests=50, window_minutes=60)
def get_models():
    """Get information about available models with balancing status."""
    try:
        # Check service readiness (graceful fallback if not ready)
        try:
            check_classification_service_readiness()
            classification_service = get_classification_service()
            active_model = classification_service.get_active_model()
            metrics = classification_service.get_model_metrics()
            
            # Get balancing information from model manager
            model_manager = getattr(classification_service, 'model_manager', None)
            balanced_models = {}
            unbalanced_models = {}
            
            if model_manager:
                balanced_models = {name: metrics[name] for name, model_metrics in model_manager.get_all_metrics().items() 
                                 if model_metrics.class_balancing_enabled and name in metrics}
                unbalanced_models = {name: metrics[name] for name, model_metrics in model_manager.get_all_metrics().items() 
                                   if not model_metrics.class_balancing_enabled and name in metrics}
            
        except ServiceNotReadyError:
            # Service not ready, return empty data
            active_model = None
            metrics = {}
            balanced_models = {}
            unbalanced_models = {}
        
        return jsonify({
            'error': False,
            'data': {
                'active_model': active_model,
                'available_models': list(metrics.keys()) if metrics else [],
                'metrics': metrics,
                'service_ready': active_model is not None,
                'balancing_summary': {
                    'balanced_models_count': len(balanced_models),
                    'unbalanced_models_count': len(unbalanced_models),
                    'balanced_models': list(balanced_models.keys()),
                    'unbalanced_models': list(unbalanced_models.keys()),
                    'active_model_balanced': active_model in balanced_models if active_model else False
                }
            }
        }), 200
    
    except Exception as e:
        return error_handler.handle_api_error(e, 500)


@api_bp.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint with comprehensive service status."""
    try:
        # Get service health manager for detailed health information
        try:
            health_manager = get_service_health_manager()
            health_summary = health_manager.check_all_services()
            service_status = health_manager.get_service_status_summary()
            
            # Determine overall status
            if health_summary.overall_status.value == 'healthy':
                overall_status = 'healthy'
                status_code = 200
            elif health_summary.overall_status.value in ['degraded', 'initializing']:
                overall_status = 'degraded'
                status_code = 200
            else:
                overall_status = 'unhealthy'
                status_code = 503
            
            status = {
                'status': overall_status,
                'timestamp': datetime.now().isoformat(),
                'version': current_app.config.get('APP_VERSION', '1.0.0'),
                'system_status': health_summary.overall_status.value,
                'services': {
                    'classification_service': service_status.get('classification_service', False),
                    'batch_processor': service_status.get('batch_processor', False),
                    'model_availability': service_status.get('model_availability', False),
                    'preprocessing_pipeline': service_status.get('preprocessing_pipeline', False)
                },
                'readiness_message': health_manager.get_system_readiness_message()
            }
            
        except ServiceNotReadyError:
            # Health manager not available - basic fallback
            status = {
                'status': 'unhealthy',
                'timestamp': datetime.now().isoformat(),
                'version': current_app.config.get('APP_VERSION', '1.0.0'),
                'system_status': 'not_ready',
                'services': {
                    'classification_service': hasattr(current_app, 'classification_service') and current_app.classification_service is not None,
                    'batch_processor': hasattr(current_app, 'batch_processor') and current_app.batch_processor is not None,
                    'model_availability': False,
                    'preprocessing_pipeline': False
                },
                'readiness_message': 'Service health manager not available'
            }
            status_code = 503
        
        return jsonify(status), status_code
    
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500


@api_bp.route('/status', methods=['GET'])
def detailed_status():
    """Detailed service status endpoint for API consumers."""
    try:
        # Get comprehensive service health information
        try:
            health_manager = get_service_health_manager()
            health_summary = health_manager.check_all_services()
            detailed_status = health_manager.get_detailed_health_status()
            
            status_data = {
                'error': False,
                'data': {
                    'overall_status': health_summary.overall_status.value,
                    'timestamp': datetime.now().isoformat(),
                    'version': current_app.config.get('APP_VERSION', '1.0.0'),
                    'readiness_message': health_manager.get_system_readiness_message(),
                    'system_ready': health_manager.is_system_ready(),
                    'services': {}
                }
            }
            
            # Add detailed service information
            for service_name, status in detailed_status.items():
                status_data['data']['services'][service_name] = {
                    'status': status.status.value,
                    'is_ready': status.is_ready,
                    'last_check': status.last_check.isoformat(),
                    'error_message': status.error_message,
                    'details': status.details
                }
            
            return jsonify(status_data), 200
            
        except ServiceNotReadyError as e:
            # Health manager not available
            return jsonify({
                'error': True,
                'message': f'Service health manager not available: {e.message}',
                'code': 'HEALTH_MANAGER_NOT_READY',
                'timestamp': datetime.now().isoformat(),
                'data': {
                    'overall_status': 'not_ready',
                    'system_ready': False,
                    'services': {}
                }
            }), 503
            
    except Exception as e:
        current_app.logger.error(f'Error getting detailed API status: {str(e)}', exc_info=True)
        return jsonify({
            'error': True,
            'message': 'Internal error while retrieving status',
            'code': 'STATUS_ERROR',
            'timestamp': datetime.now().isoformat()
        }), 500


@api_bp.route('/models/balancing', methods=['GET'])
@rate_limit(max_requests=50, window_minutes=60)
def get_model_balancing_info():
    """Get detailed class balancing information for all models."""
    try:
        # Check service readiness
        try:
            check_classification_service_readiness()
            classification_service = get_classification_service()
            model_manager = getattr(classification_service, 'model_manager', None)
            
            if not model_manager:
                return jsonify({
                    'error': True,
                    'message': 'Model manager not available',
                    'code': 'MODEL_MANAGER_NOT_AVAILABLE',
                    'timestamp': datetime.now().isoformat()
                }), 503
            
            # Get balancing information for all models
            balancing_info = {}
            for model_name in model_manager.models.keys():
                model_balancing_info = model_manager.get_model_balancing_info(model_name)
                if model_balancing_info:
                    balancing_info[model_name] = model_balancing_info
            
            # Get balanced and unbalanced model lists
            balanced_models = model_manager.get_balanced_models()
            unbalanced_models = model_manager.get_unbalanced_models()
            
            # Calculate summary statistics
            total_models = len(model_manager.models)
            balanced_count = len(balanced_models)
            unbalanced_count = len(unbalanced_models)
            
            # Calculate average false negative rate for balanced models
            avg_fnr_balanced = 0.0
            if balanced_models:
                fnr_values = [metrics.false_negative_rate for metrics in balanced_models.values()]
                avg_fnr_balanced = sum(fnr_values) / len(fnr_values) if fnr_values else 0.0
            
            return jsonify({
                'error': False,
                'data': {
                    'balancing_info': balancing_info,
                    'summary': {
                        'total_models': total_models,
                        'balanced_models_count': balanced_count,
                        'unbalanced_models_count': unbalanced_count,
                        'balanced_models': list(balanced_models.keys()),
                        'unbalanced_models': list(unbalanced_models.keys()),
                        'avg_false_negative_rate_balanced': avg_fnr_balanced,
                        'balancing_coverage_percentage': (balanced_count / total_models * 100) if total_models > 0 else 0.0
                    },
                    'timestamp': datetime.now().isoformat()
                }
            }), 200
            
        except ServiceNotReadyError as e:
            return jsonify({
                'error': True,
                'message': f'Service not ready: {e.message}',
                'code': 'SERVICE_NOT_READY',
                'timestamp': datetime.now().isoformat()
            }), 503
    
    except Exception as e:
        current_app.logger.error(f'Error getting model balancing info: {str(e)}', exc_info=True)
        return jsonify({
            'error': True,
            'message': 'Internal error while retrieving balancing information',
            'code': 'BALANCING_INFO_ERROR',
            'timestamp': datetime.now().isoformat()
        }), 500


@api_bp.route('/models/<model_name>/balancing', methods=['GET'])
@rate_limit(max_requests=50, window_minutes=60)
def get_specific_model_balancing_info(model_name: str):
    """Get detailed class balancing information for a specific model."""
    try:
        # Check service readiness
        try:
            check_classification_service_readiness()
            classification_service = get_classification_service()
            model_manager = getattr(classification_service, 'model_manager', None)
            
            if not model_manager:
                return jsonify({
                    'error': True,
                    'message': 'Model manager not available',
                    'code': 'MODEL_MANAGER_NOT_AVAILABLE',
                    'timestamp': datetime.now().isoformat()
                }), 503
            
            # Check if model exists
            if model_name not in model_manager.models:
                return jsonify({
                    'error': True,
                    'message': f'Model "{model_name}" not found',
                    'code': 'MODEL_NOT_FOUND',
                    'available_models': list(model_manager.models.keys()),
                    'timestamp': datetime.now().isoformat()
                }), 404
            
            # Get balancing information for the specific model
            balancing_info = model_manager.get_model_balancing_info(model_name)
            model_metrics = model_manager.get_model_metrics(model_name)
            
            if not balancing_info:
                return jsonify({
                    'error': True,
                    'message': f'No balancing information available for model "{model_name}"',
                    'code': 'BALANCING_INFO_NOT_AVAILABLE',
                    'timestamp': datetime.now().isoformat()
                }), 404
            
            # Add performance metrics to balancing info
            response_data = {
                'model_name': model_name,
                'balancing_info': balancing_info,
                'performance_metrics': {
                    'accuracy': model_metrics.accuracy if model_metrics else 0.0,
                    'precision': model_metrics.precision if model_metrics else 0.0,
                    'recall': model_metrics.recall if model_metrics else 0.0,
                    'f1_score': model_metrics.f1_score if model_metrics else 0.0,
                    'false_negative_rate': model_metrics.false_negative_rate if model_metrics else 0.0,
                    'test_samples': model_metrics.test_samples if model_metrics else 0,
                    'training_date': model_metrics.training_date.isoformat() if model_metrics and model_metrics.training_date else None
                },
                'timestamp': datetime.now().isoformat()
            }
            
            return jsonify({
                'error': False,
                'data': response_data
            }), 200
            
        except ServiceNotReadyError as e:
            return jsonify({
                'error': True,
                'message': f'Service not ready: {e.message}',
                'code': 'SERVICE_NOT_READY',
                'timestamp': datetime.now().isoformat()
            }), 503
    
    except Exception as e:
        current_app.logger.error(f'Error getting balancing info for model {model_name}: {str(e)}', exc_info=True)
        return jsonify({
            'error': True,
            'message': 'Internal error while retrieving model balancing information',
            'code': 'MODEL_BALANCING_INFO_ERROR',
            'timestamp': datetime.now().isoformat()
        }), 500


@api_bp.route('/docs', methods=['GET'])
def api_documentation():
    """API documentation endpoint."""
    docs = {
        'title': 'Email Spam Classifier API',
        'version': '1.0.0',
        'description': 'REST API for email spam classification using machine learning',
        'base_url': request.url_root + 'api/v1',
        'authentication': {
            'type': 'Bearer Token',
            'header': 'Authorization: Bearer <token>',
            'note': 'Required for batch processing endpoints'
        },
        'rate_limits': {
            'classify': '100 requests per hour',
            'batch': '10 requests per hour (authenticated)',
            'models': '50 requests per hour',
            'health': 'No limit'
        },
        'endpoints': {
            'POST /classify': {
                'description': 'Classify a single email',
                'authentication': 'Not required',
                'request': {
                    'content_type': 'application/json',
                    'body': {
                        'email_text': 'string (required) - Email content to classify'
                    }
                },
                'response': {
                    'success': {
                        'error': False,
                        'data': {
                            'prediction': 'string - "Spam" or "Legitimate"',
                            'confidence': 'float - Confidence score (0.0-1.0)',
                            'model_used': 'string - Name of ML model used',
                            'processing_time': 'float - Processing time in seconds',
                            'timestamp': 'string - ISO timestamp',
                            'model_balanced': 'boolean - Whether model was trained with class balancing (optional)',
                            'balancing_method': 'string - Balancing method used (optional)',
                            'false_negative_rate': 'float - Model false negative rate (optional)'
                        }
                    },
                    'error': {
                        'error': True,
                        'message': 'string - Error description',
                        'code': 'string - Error code',
                        'timestamp': 'string - ISO timestamp'
                    }
                }
            },
            'POST /classify/batch': {
                'description': 'Classify multiple emails in batch',
                'authentication': 'Required',
                'request': {
                    'content_type': 'application/json',
                    'body': {
                        'emails': 'array - List of email strings (max 1000)'
                    }
                },
                'response': {
                    'success': {
                        'error': False,
                        'data': {
                            'job_id': 'string - Unique job identifier',
                            'total_emails': 'integer - Number of emails processed',
                            'results': 'array - Classification results for each email'
                        }
                    }
                }
            },
            'GET /models': {
                'description': 'Get information about available models with balancing status',
                'authentication': 'Not required',
                'response': {
                    'success': {
                        'error': False,
                        'data': {
                            'active_model': 'string - Currently active model name',
                            'available_models': 'array - List of available model names',
                            'metrics': 'object - Performance metrics for each model (includes balancing info)',
                            'balancing_summary': 'object - Summary of class balancing across all models'
                        }
                    }
                }
            },
            'GET /models/balancing': {
                'description': 'Get detailed class balancing information for all models',
                'authentication': 'Not required',
                'response': {
                    'success': {
                        'error': False,
                        'data': {
                            'balancing_info': 'object - Detailed balancing info for each model',
                            'summary': 'object - Balancing summary statistics',
                            'timestamp': 'string - ISO timestamp'
                        }
                    }
                }
            },
            'GET /models/<model_name>/balancing': {
                'description': 'Get detailed class balancing information for a specific model',
                'authentication': 'Not required',
                'parameters': {
                    'model_name': 'string - Name of the model (path parameter)'
                },
                'response': {
                    'success': {
                        'error': False,
                        'data': {
                            'model_name': 'string - Model name',
                            'balancing_info': 'object - Detailed balancing information',
                            'performance_metrics': 'object - Model performance metrics',
                            'timestamp': 'string - ISO timestamp'
                        }
                    }
                }
            },
            'GET /health': {
                'description': 'Health check endpoint',
                'authentication': 'Not required',
                'response': {
                    'success': {
                        'status': 'string - "healthy" or "unhealthy"',
                        'timestamp': 'string - ISO timestamp',
                        'version': 'string - API version',
                        'services': 'object - Service availability status'
                    }
                }
            },
            'GET /docs': {
                'description': 'This API documentation',
                'authentication': 'Not required'
            }
        },
        'error_codes': {
            'INVALID_CONTENT_TYPE': 'Request must use application/json content type',
            'MISSING_EMAIL_TEXT': 'email_text field is required',
            'MISSING_EMAILS': 'emails field is required for batch processing',
            'BATCH_SIZE_EXCEEDED': 'Batch size exceeds maximum limit',
            'AUTH_REQUIRED': 'Authentication token required',
            'INVALID_TOKEN': 'Invalid authentication token',
            'RATE_LIMIT_EXCEEDED': 'Rate limit exceeded',
            'CLASSIFICATION_ERROR': 'Error during email classification',
            'BATCH_CLASSIFICATION_ERROR': 'Error during batch classification',
            'MODELS_ERROR': 'Error retrieving model information',
            'ENDPOINT_NOT_FOUND': 'API endpoint not found',
            'METHOD_NOT_ALLOWED': 'HTTP method not allowed',
            'INTERNAL_ERROR': 'Internal server error'
        },
        'examples': {
            'classify_request': {
                'email_text': 'Subject: Special Offer! Get 50% off now! Click here to claim your discount and win amazing prizes!'
            },
            'classify_response': {
                'error': False,
                'data': {
                    'prediction': 'Spam',
                    'confidence': 0.95,
                    'model_used': 'Naive Bayes',
                    'processing_time': 0.023,
                    'timestamp': '2024-01-01T12:00:00Z',
                    'model_balanced': True,
                    'balancing_method': 'smote',
                    'false_negative_rate': 0.032
                }
            },
            'batch_request': {
                'emails': [
                    'Subject: Meeting Tomorrow Hi John, reminder about our meeting...',
                    'Subject: URGENT! Click now to claim your prize!'
                ]
            }
        }
    }
    
    return jsonify(docs), 200


@api_bp.errorhandler(404)
def api_not_found(error):
    """Handle API 404 errors."""
    return jsonify({
        'error': True,
        'message': 'API endpoint not found',
        'code': 'ENDPOINT_NOT_FOUND',
        'timestamp': datetime.now().isoformat()
    }), 404


@api_bp.errorhandler(405)
def method_not_allowed(error):
    """Handle method not allowed errors."""
    return jsonify({
        'error': True,
        'message': 'HTTP method not allowed for this endpoint',
        'code': 'METHOD_NOT_ALLOWED',
        'timestamp': datetime.now().isoformat()
    }), 405


@api_bp.errorhandler(500)
def api_internal_error(error):
    """Handle API 500 errors."""
    return jsonify({
        'error': True,
        'message': 'Internal server error',
        'code': 'INTERNAL_ERROR',
        'timestamp': datetime.now().isoformat()
    }), 500