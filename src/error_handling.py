"""
Centralized error handling utilities for the Email Spam Classifier.

This module provides custom exceptions, error formatters, and validation utilities
for consistent error handling across the application.
"""

import logging
import logging.handlers
import traceback
import sys
from datetime import datetime
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, asdict
from enum import Enum
from flask import jsonify, render_template, request, has_request_context


@dataclass
class ErrorResponse:
    """Standardized error response format."""
    error: bool = True
    message: str = ""
    code: str = ""
    timestamp: str = ""
    details: Optional[Dict[str, Any]] = None


class ErrorCategory(Enum):
    """Error categories for classification and tracking."""
    VALIDATION = "validation"
    SERVICE_INITIALIZATION = "service_initialization"
    MODEL_OPERATION = "model_operation"
    DATA_PROCESSING = "data_processing"
    FILE_OPERATION = "file_operation"
    NETWORK = "network"
    AUTHENTICATION = "authentication"
    RATE_LIMITING = "rate_limiting"
    SYSTEM_RESOURCE = "system_resource"
    CONFIGURATION = "configuration"
    UNKNOWN = "unknown"


class ErrorSeverity(Enum):
    """Error severity levels for logging and alerting."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ErrorContext:
    """Enhanced error context with categorization and tracking information."""
    error_id: str
    category: ErrorCategory
    severity: ErrorSeverity
    component: str
    operation: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    client_ip: Optional[str] = None
    user_agent: Optional[str] = None
    timestamp: datetime = None
    additional_data: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.additional_data is None:
            self.additional_data = {}


class EmailClassifierError(Exception):
    """Base exception for Email Classifier application."""
    
    def __init__(self, message: str, code: str = "GENERAL_ERROR", details: Optional[Dict] = None,
                 category: ErrorCategory = ErrorCategory.UNKNOWN, severity: ErrorSeverity = ErrorSeverity.MEDIUM):
        self.message = message
        self.code = code
        self.details = details or {}
        self.category = category
        self.severity = severity
        super().__init__(self.message)


class ValidationError(EmailClassifierError):
    """Exception for input validation errors."""
    
    def __init__(self, message: str, field: str = None, details: Optional[Dict] = None):
        self.field = field
        if details is None:
            details = {}
        if field:
            details["field"] = field
        super().__init__(message, "VALIDATION_ERROR", details, ErrorCategory.VALIDATION, ErrorSeverity.LOW)


class ServiceNotReadyError(EmailClassifierError):
    """Exception when service is not ready for operation."""
    
    def __init__(self, service_name: str, reason: str = "Service not initialized"):
        message = f"{service_name} is not ready: {reason}"
        super().__init__(message, "SERVICE_NOT_READY", {"service": service_name, "reason": reason},
                        ErrorCategory.SERVICE_INITIALIZATION, ErrorSeverity.HIGH)


class ModelError(EmailClassifierError):
    """Exception for ML model related errors."""
    
    def __init__(self, message: str, model_name: str = None, details: Optional[Dict] = None):
        self.model_name = model_name
        code = "MODEL_ERROR"
        if details is None:
            details = {}
        if model_name:
            details["model_name"] = model_name
        super().__init__(message, code, details, ErrorCategory.MODEL_OPERATION, ErrorSeverity.HIGH)


class ProcessingError(EmailClassifierError):
    """Exception for processing related errors."""
    
    def __init__(self, message: str, processing_stage: str = None, details: Optional[Dict] = None):
        self.processing_stage = processing_stage
        code = "PROCESSING_ERROR"
        if details is None:
            details = {}
        if processing_stage:
            details["processing_stage"] = processing_stage
        super().__init__(message, code, details, ErrorCategory.DATA_PROCESSING, ErrorSeverity.MEDIUM)


class ErrorHandler:
    """Centralized error handling and formatting."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def format_error_response(self, 
                            error: Exception, 
                            default_message: str = "An error occurred",
                            include_details: bool = False) -> ErrorResponse:
        """
        Format an exception into a standardized error response.
        
        Args:
            error: The exception to format
            default_message: Default message if error message is empty
            include_details: Whether to include error details in response
            
        Returns:
            ErrorResponse object
        """
        if isinstance(error, EmailClassifierError):
            message = error.message or default_message
            code = error.code
            details = error.details if include_details else None
        else:
            message = str(error) or default_message
            code = "INTERNAL_ERROR"
            details = {"type": type(error).__name__} if include_details else None
        
        return ErrorResponse(
            error=True,
            message=message,
            code=code,
            timestamp=datetime.now().isoformat(),
            details=details
        )
    
    def handle_api_error(self, error: Exception, status_code: int = 500) -> tuple:
        """
        Handle API errors and return JSON response.
        
        Args:
            error: The exception to handle
            status_code: HTTP status code
            
        Returns:
            Tuple of (JSON response, status code)
        """
        # Log the error
        self.logger.error(f"API Error: {str(error)}", exc_info=True)
        
        # Format error response
        error_response = self.format_error_response(error, include_details=False)
        
        # Convert to dictionary for JSON response
        response_dict = {
            'error': error_response.error,
            'message': error_response.message,
            'code': error_response.code,
            'timestamp': error_response.timestamp
        }
        
        return jsonify(response_dict), status_code
    
    def handle_web_error(self, error: Exception, template: str = 'errors/500.html') -> tuple:
        """
        Handle web interface errors and return HTML response.
        
        Args:
            error: The exception to handle
            template: Error template to render
            
        Returns:
            Tuple of (HTML response, status code)
        """
        # Log the error
        self.logger.error(f"Web Error: {str(error)}", exc_info=True)
        
        # Format error response
        error_response = self.format_error_response(error)
        
        # Determine status code
        status_code = 500
        if isinstance(error, ValidationError):
            status_code = 400
            template = 'errors/400.html'
        elif isinstance(error, ServiceNotReadyError):
            status_code = 503
            template = 'errors/503.html'
        
        return render_template(template, error=error_response), status_code


class EnhancedLogger:
    """
    Enhanced logging system with contextual error information and categorization.
    
    Provides structured logging with error categorization, severity levels,
    and detailed context information for debugging purposes.
    """
    
    def __init__(self, logger_name: str = __name__):
        self.logger = logging.getLogger(logger_name)
        self._setup_structured_logging()
    
    def _setup_structured_logging(self):
        """Setup structured logging with custom formatters."""
        # Create custom formatter for structured logs
        formatter = logging.Formatter(
            '%(asctime)s [%(levelname)s] %(name)s: %(message)s | '
            'Context: %(context)s | Error_ID: %(error_id)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Only add handler if not already present
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def log_error_with_context(self, 
                              error: Exception, 
                              context: ErrorContext,
                              include_traceback: bool = True) -> str:
        """
        Log error with detailed contextual information.
        
        Args:
            error: The exception to log
            context: Error context with categorization and metadata
            include_traceback: Whether to include full stack trace
            
        Returns:
            Error ID for tracking
        """
        # Generate error ID if not provided
        if not context.error_id:
            context.error_id = self._generate_error_id()
        
        # Prepare structured log data
        log_data = {
            'error_id': context.error_id,
            'error_type': type(error).__name__,
            'error_message': str(error),
            'category': context.category.value,
            'severity': context.severity.value,
            'component': context.component,
            'operation': context.operation,
            'timestamp': context.timestamp.isoformat(),
        }
        
        # Add request context if available
        if has_request_context():
            try:
                log_data.update({
                    'request_method': request.method,
                    'request_path': request.path,
                    'request_args': dict(request.args),
                    'client_ip': request.remote_addr,
                    'user_agent': request.headers.get('User-Agent', 'Unknown')
                })
            except Exception:
                pass  # Ignore if request context is not available
        
        # Add custom context data
        if context.additional_data:
            log_data['additional_data'] = context.additional_data
        
        # Add error-specific details
        if hasattr(error, 'details') and error.details:
            log_data['error_details'] = error.details
        
        # Prepare log message
        log_message = f"{context.category.value.upper()} ERROR in {context.component}.{context.operation}: {str(error)}"
        
        # Add traceback if requested
        if include_traceback:
            log_data['traceback'] = traceback.format_exc()
        
        # Log at appropriate level based on severity
        log_level = self._get_log_level_for_severity(context.severity)
        
        # Use extra parameter to pass structured data
        self.logger.log(
            log_level,
            log_message,
            extra={
                'context': log_data,
                'error_id': context.error_id
            }
        )
        
        return context.error_id
    
    def log_operation_start(self, 
                           component: str, 
                           operation: str, 
                           context_data: Optional[Dict] = None) -> str:
        """
        Log the start of an operation with context.
        
        Args:
            component: Component name (e.g., 'dashboard', 'batch_processor')
            operation: Operation name (e.g., 'load_data', 'process_file')
            context_data: Additional context information
            
        Returns:
            Operation ID for tracking
        """
        operation_id = self._generate_error_id()
        
        log_data = {
            'operation_id': operation_id,
            'component': component,
            'operation': operation,
            'phase': 'START',
            'timestamp': datetime.now().isoformat()
        }
        
        if context_data:
            log_data['context'] = context_data
        
        if has_request_context():
            try:
                log_data.update({
                    'request_method': request.method,
                    'request_path': request.path,
                    'client_ip': request.remote_addr
                })
            except Exception:
                pass
        
        self.logger.info(
            f"OPERATION START: {component}.{operation}",
            extra={
                'context': log_data,
                'error_id': operation_id
            }
        )
        
        return operation_id
    
    def log_operation_end(self, 
                         operation_id: str,
                         component: str, 
                         operation: str, 
                         success: bool = True,
                         duration: Optional[float] = None,
                         result_data: Optional[Dict] = None) -> None:
        """
        Log the end of an operation with results.
        
        Args:
            operation_id: Operation ID from log_operation_start
            component: Component name
            operation: Operation name
            success: Whether operation succeeded
            duration: Operation duration in seconds
            result_data: Additional result information
        """
        log_data = {
            'operation_id': operation_id,
            'component': component,
            'operation': operation,
            'phase': 'END',
            'success': success,
            'timestamp': datetime.now().isoformat()
        }
        
        if duration is not None:
            log_data['duration_seconds'] = duration
        
        if result_data:
            log_data['result'] = result_data
        
        status = "SUCCESS" if success else "FAILURE"
        log_level = logging.INFO if success else logging.WARNING
        
        self.logger.log(
            log_level,
            f"OPERATION {status}: {component}.{operation}",
            extra={
                'context': log_data,
                'error_id': operation_id
            }
        )
    
    def log_dashboard_access(self, 
                           user_action: str,
                           service_status: Optional[Dict] = None,
                           data_retrieved: bool = True,
                           error_occurred: bool = False) -> None:
        """
        Log dashboard access with specific context.
        
        Args:
            user_action: Action performed by user
            service_status: Status of various services
            data_retrieved: Whether data was successfully retrieved
            error_occurred: Whether an error occurred
        """
        log_data = {
            'component': 'dashboard',
            'user_action': user_action,
            'data_retrieved': data_retrieved,
            'error_occurred': error_occurred,
            'timestamp': datetime.now().isoformat()
        }
        
        if service_status:
            log_data['service_status'] = service_status
        
        if has_request_context():
            try:
                log_data.update({
                    'request_path': request.path,
                    'client_ip': request.remote_addr,
                    'user_agent': request.headers.get('User-Agent', 'Unknown')
                })
            except Exception:
                pass
        
        log_level = logging.INFO
        if error_occurred:
            log_level = logging.WARNING
        
        self.logger.log(
            log_level,
            f"DASHBOARD ACCESS: {user_action} - {'SUCCESS' if data_retrieved and not error_occurred else 'PARTIAL' if data_retrieved else 'FAILED'}",
            extra={
                'context': log_data,
                'error_id': self._generate_error_id()
            }
        )
    
    def log_batch_operation(self, 
                           operation: str,
                           filename: Optional[str] = None,
                           file_size: Optional[int] = None,
                           job_id: Optional[str] = None,
                           processed_count: Optional[int] = None,
                           total_count: Optional[int] = None,
                           success: bool = True,
                           error_details: Optional[str] = None) -> None:
        """
        Log batch processing operations with detailed context.
        
        Args:
            operation: Batch operation type
            filename: Name of file being processed
            file_size: Size of file in bytes
            job_id: Batch job identifier
            processed_count: Number of items processed
            total_count: Total number of items
            success: Whether operation succeeded
            error_details: Error details if operation failed
        """
        log_data = {
            'component': 'batch_processor',
            'operation': operation,
            'success': success,
            'timestamp': datetime.now().isoformat()
        }
        
        if filename:
            log_data['filename'] = filename
        if file_size is not None:
            log_data['file_size_bytes'] = file_size
            log_data['file_size_mb'] = round(file_size / (1024 * 1024), 2)
        if job_id:
            log_data['job_id'] = job_id
        if processed_count is not None:
            log_data['processed_count'] = processed_count
        if total_count is not None:
            log_data['total_count'] = total_count
            if processed_count is not None:
                log_data['completion_percentage'] = round((processed_count / total_count) * 100, 2)
        if error_details:
            log_data['error_details'] = error_details
        
        if has_request_context():
            try:
                log_data.update({
                    'request_path': request.path,
                    'client_ip': request.remote_addr
                })
            except Exception:
                pass
        
        log_level = logging.INFO if success else logging.ERROR
        status = "SUCCESS" if success else "FAILED"
        
        self.logger.log(
            log_level,
            f"BATCH {operation.upper()}: {status}" + (f" - {filename}" if filename else ""),
            extra={
                'context': log_data,
                'error_id': self._generate_error_id()
            }
        )
    
    def log_service_health_check(self, 
                                service_name: str,
                                is_healthy: bool,
                                check_details: Optional[Dict] = None,
                                response_time: Optional[float] = None) -> None:
        """
        Log service health check results.
        
        Args:
            service_name: Name of service being checked
            is_healthy: Whether service is healthy
            check_details: Additional health check details
            response_time: Health check response time in seconds
        """
        log_data = {
            'component': 'service_health',
            'service_name': service_name,
            'is_healthy': is_healthy,
            'timestamp': datetime.now().isoformat()
        }
        
        if check_details:
            log_data['check_details'] = check_details
        if response_time is not None:
            log_data['response_time_seconds'] = response_time
        
        log_level = logging.INFO if is_healthy else logging.WARNING
        status = "HEALTHY" if is_healthy else "UNHEALTHY"
        
        self.logger.log(
            log_level,
            f"SERVICE HEALTH: {service_name} - {status}",
            extra={
                'context': log_data,
                'error_id': self._generate_error_id()
            }
        )
    
    def _get_log_level_for_severity(self, severity: ErrorSeverity) -> int:
        """Map error severity to logging level."""
        severity_mapping = {
            ErrorSeverity.LOW: logging.INFO,
            ErrorSeverity.MEDIUM: logging.WARNING,
            ErrorSeverity.HIGH: logging.ERROR,
            ErrorSeverity.CRITICAL: logging.CRITICAL
        }
        return severity_mapping.get(severity, logging.ERROR)
    
    def _generate_error_id(self) -> str:
        """Generate unique error ID for tracking."""
        import uuid
        return str(uuid.uuid4())[:8]


class ErrorTracker:
    """
    Error tracking and categorization system for monitoring and analysis.
    
    Tracks error patterns, frequencies, and provides insights for debugging
    and system improvement.
    """
    
    def __init__(self):
        self.logger = EnhancedLogger(f"{__name__}.ErrorTracker")
        self._error_counts = {}
        self._error_patterns = {}
        
        # Load configuration
        self._load_configuration()
    
    def _load_configuration(self) -> None:
        """Load configuration settings."""
        try:
            from flask import current_app
            if current_app:
                self.pattern_limit = current_app.config.get('ERROR_PATTERN_LIMIT', 100)
                self.tracking_enabled = current_app.config.get('ERROR_TRACKING_ENABLED', True)
            else:
                self.pattern_limit = 100
                self.tracking_enabled = True
        except RuntimeError:
            # Outside of application context
            self.pattern_limit = 100
            self.tracking_enabled = True
    
    def track_error(self, error: Exception, context: ErrorContext) -> None:
        """
        Track error occurrence for pattern analysis.
        
        Args:
            error: The exception that occurred
            context: Error context with categorization
        """
        # Skip tracking if disabled
        if not self.tracking_enabled:
            return
        
        error_key = f"{context.category.value}:{type(error).__name__}"
        
        # Update error counts
        if error_key not in self._error_counts:
            self._error_counts[error_key] = {
                'count': 0,
                'first_seen': context.timestamp,
                'last_seen': context.timestamp,
                'severity': context.severity.value,
                'component': context.component
            }
        
        self._error_counts[error_key]['count'] += 1
        self._error_counts[error_key]['last_seen'] = context.timestamp
        
        # Track error patterns
        pattern_key = f"{context.component}:{context.operation}"
        if pattern_key not in self._error_patterns:
            self._error_patterns[pattern_key] = []
        
        self._error_patterns[pattern_key].append({
            'error_type': type(error).__name__,
            'category': context.category.value,
            'severity': context.severity.value,
            'timestamp': context.timestamp,
            'error_id': context.error_id
        })
        
        # Keep only recent patterns (use configured limit)
        if len(self._error_patterns[pattern_key]) > self.pattern_limit:
            self._error_patterns[pattern_key] = self._error_patterns[pattern_key][-self.pattern_limit:]
        
        # Log error tracking
        self.logger.log_error_with_context(
            error,
            ErrorContext(
                error_id=context.error_id,
                category=ErrorCategory.UNKNOWN,
                severity=ErrorSeverity.LOW,
                component="error_tracker",
                operation="track_error",
                additional_data={
                    'original_category': context.category.value,
                    'original_component': context.component,
                    'error_count': self._error_counts[error_key]['count']
                }
            ),
            include_traceback=False
        )
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of tracked errors."""
        return {
            'total_error_types': len(self._error_counts),
            'error_counts': dict(self._error_counts),
            'error_patterns': dict(self._error_patterns),
            'summary_generated': datetime.now().isoformat()
        }
    
    def get_frequent_errors(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get most frequent errors."""
        sorted_errors = sorted(
            self._error_counts.items(),
            key=lambda x: x[1]['count'],
            reverse=True
        )
        
        return [
            {
                'error_type': error_key,
                'count': data['count'],
                'first_seen': data['first_seen'].isoformat(),
                'last_seen': data['last_seen'].isoformat(),
                'severity': data['severity'],
                'component': data['component']
            }
            for error_key, data in sorted_errors[:limit]
        ]


@dataclass
class DashboardErrorContext:
    """Context information for dashboard errors."""
    route: str
    user_action: str
    service_status: Optional[Dict[str, bool]] = None
    fallback_data: Optional[Dict[str, Any]] = None


@dataclass
class BatchErrorContext:
    """Context information for batch processing errors."""
    route: str
    filename: Optional[str] = None
    file_size: Optional[int] = None
    processing_stage: Optional[str] = None
    processed_count: Optional[int] = None
    total_count: Optional[int] = None


class EnhancedErrorHandler(ErrorHandler):
    """
    Enhanced error handler with specialized methods for different error types,
    user-friendly message generation, and consistent response formats.
    
    Extends the base ErrorHandler with dashboard and batch processing specific
    error handling capabilities as required by Requirements 4.1, 4.2, 4.3.
    """
    
    def __init__(self):
        super().__init__()
        self.user_friendly_messages = self._init_user_friendly_messages()
        self.enhanced_logger = EnhancedLogger(f"{__name__}.EnhancedErrorHandler")
        self.error_tracker = ErrorTracker()
        
        # Load configuration settings
        self._load_configuration()
    
    def _load_configuration(self) -> None:
        """Load configuration settings from Flask app config."""
        try:
            from flask import current_app
            if current_app:
                # Load error tracking settings
                self.error_tracking_enabled = current_app.config.get('ERROR_TRACKING_ENABLED', True)
                self.error_pattern_limit = current_app.config.get('ERROR_PATTERN_LIMIT', 100)
                self.error_cache_duration = current_app.config.get('ERROR_CACHE_DURATION', 300)
                
                # Load dashboard settings
                self.dashboard_refresh_interval = current_app.config.get('DASHBOARD_REFRESH_INTERVAL', 30)
                self.dashboard_cache_timeout = current_app.config.get('DASHBOARD_CACHE_TIMEOUT', 300)
                self.dashboard_fallback_enabled = current_app.config.get('DASHBOARD_FALLBACK_DATA_ENABLED', True)
                
                # Load batch processing settings
                self.batch_validation_strict = current_app.config.get('BATCH_VALIDATION_STRICT', True)
                self.batch_cleanup_interval = current_app.config.get('BATCH_CLEANUP_INTERVAL', 3600)
                self.batch_max_retries = current_app.config.get('BATCH_MAX_RETRIES', 2)
                self.batch_partial_results_enabled = current_app.config.get('BATCH_PARTIAL_RESULTS_ENABLED', True)
                
                # Load custom error message templates
                custom_templates = current_app.config.get('ERROR_MESSAGE_TEMPLATES', {})
                self.user_friendly_messages.update(custom_templates)
                
                self.logger.info("Enhanced error handler configuration loaded from Flask app")
            else:
                self._load_default_configuration()
        except RuntimeError:
            # Outside of application context
            self._load_default_configuration()
    
    def _load_default_configuration(self) -> None:
        """Load default configuration when Flask app context is not available."""
        self.error_tracking_enabled = True
        self.error_pattern_limit = 100
        self.error_cache_duration = 300
        self.dashboard_refresh_interval = 30
        self.dashboard_cache_timeout = 300
        self.dashboard_fallback_enabled = True
        self.batch_validation_strict = True
        self.batch_cleanup_interval = 3600
        self.batch_max_retries = 2
        self.batch_partial_results_enabled = True
        self.logger.info("Enhanced error handler using default configuration")
    
    def _init_user_friendly_messages(self) -> Dict[str, str]:
        """Initialize user-friendly error message templates."""
        return {
            # Dashboard specific messages
            'DASHBOARD_SERVICE_NOT_READY': "The system is still initializing. Please wait a moment and refresh the page.",
            'DASHBOARD_MODEL_NOT_TRAINED': "No trained models are available. Please train a model first or contact your administrator.",
            'DASHBOARD_DATA_UNAVAILABLE': "Dashboard data is temporarily unavailable. The system will retry automatically.",
            'DASHBOARD_METRICS_ERROR': "Unable to retrieve performance metrics at this time. Please try again later.",
            
            # Batch processing specific messages
            'BATCH_FILE_INVALID': "The uploaded file is not valid. Please check the file format and try again.",
            'BATCH_SERVICE_NOT_READY': "The batch processing service is not ready. Please wait for system initialization to complete.",
            'BATCH_PROCESSING_FAILED': "Batch processing encountered an error. Some emails may not have been processed.",
            'BATCH_FILE_TOO_LARGE': "The uploaded file is too large. Please reduce the file size or split it into smaller files.",
            'BATCH_INVALID_CSV': "The CSV file format is invalid. Please ensure it contains email data in the correct format.",
            
            # General service messages
            'SERVICE_INITIALIZATION_ERROR': "A system service failed to start properly. Please contact your administrator.",
            'MODEL_LOADING_ERROR': "Unable to load the machine learning model. Please check if models are properly trained.",
            'PREPROCESSING_ERROR': "Error occurred while preparing data for processing. Please check your input format.",
            
            # Generic fallbacks
            'INTERNAL_ERROR': "An unexpected error occurred. Please try again or contact support if the problem persists.",
            'VALIDATION_ERROR': "The provided data is not valid. Please check your input and try again.",
            'PROCESSING_ERROR': "An error occurred while processing your request. Please try again."
        }
    
    def handle_dashboard_error(self, error: Exception, context: Optional[DashboardErrorContext] = None) -> tuple:
        """
        Handle dashboard-specific errors with enhanced context and user-friendly messages.
        
        Args:
            error: The exception to handle
            context: Dashboard-specific error context
            
        Returns:
            Tuple of (response, status_code) for Flask
        """
        # Create enhanced error context
        error_context = self._create_error_context(
            error=error,
            component="dashboard",
            operation=context.user_action if context else "unknown",
            additional_data={
                'route': context.route if context else 'unknown',
                'service_status': context.service_status if context else None,
                'has_fallback_data': context.fallback_data is not None if context else False
            }
        )
        
        # Log error with enhanced context
        error_id = self.enhanced_logger.log_error_with_context(error, error_context)
        
        # Track error for pattern analysis
        self.error_tracker.track_error(error, error_context)
        
        # Log dashboard access with error details
        self.enhanced_logger.log_dashboard_access(
            user_action=context.user_action if context else "unknown",
            service_status=context.service_status if context else None,
            data_retrieved=context.fallback_data is not None if context else False,
            error_occurred=True
        )
        
        # Determine error type and generate user-friendly message
        user_message, error_code, status_code = self._categorize_dashboard_error(error)
        
        # Create enhanced error response
        error_response = ErrorResponse(
            error=True,
            message=user_message,
            code=error_code,
            timestamp=datetime.now().isoformat(),
            details=self._create_dashboard_error_details(error, context, error_id)
        )
        
        # Determine if this is an API or web request
        if self._is_api_request():
            return self._create_api_response(error_response, status_code)
        else:
            return self._create_dashboard_web_response(error_response, status_code, context)
    
    def handle_batch_error(self, error: Exception, context: Optional[BatchErrorContext] = None) -> tuple:
        """
        Handle batch processing specific errors with enhanced context and recovery guidance.
        
        Args:
            error: The exception to handle
            context: Batch processing specific error context
            
        Returns:
            Tuple of (response, status_code) for Flask
        """
        # Create enhanced error context
        error_context = self._create_error_context(
            error=error,
            component="batch_processor",
            operation=context.processing_stage if context else "unknown",
            additional_data={
                'route': context.route if context else 'unknown',
                'filename': context.filename if context else None,
                'file_size': context.file_size if context else None,
                'processed_count': context.processed_count if context else None,
                'total_count': context.total_count if context else None
            }
        )
        
        # Log error with enhanced context
        error_id = self.enhanced_logger.log_error_with_context(error, error_context)
        
        # Track error for pattern analysis
        self.error_tracker.track_error(error, error_context)
        
        # Log batch operation with error details
        self.enhanced_logger.log_batch_operation(
            operation=context.processing_stage if context else "unknown",
            filename=context.filename if context else None,
            file_size=context.file_size if context else None,
            processed_count=context.processed_count if context else None,
            total_count=context.total_count if context else None,
            success=False,
            error_details=str(error)
        )
        
        # Determine error type and generate user-friendly message
        user_message, error_code, status_code = self._categorize_batch_error(error, context)
        
        # Create enhanced error response
        error_response = ErrorResponse(
            error=True,
            message=user_message,
            code=error_code,
            timestamp=datetime.now().isoformat(),
            details=self._create_batch_error_details(error, context, error_id)
        )
        
        # Determine if this is an API or web request
        if self._is_api_request():
            return self._create_api_response(error_response, status_code)
        else:
            return self._create_batch_web_response(error_response, status_code, context)
    
    def log_error_with_context(self, error: Exception, context_type: str, context_data: Optional[Dict] = None) -> str:
        """
        Log detailed error information with contextual data for debugging.
        
        Args:
            error: The exception to log
            context_type: Type of context (DASHBOARD, BATCH, API, etc.)
            context_data: Additional context information
            
        Returns:
            Error ID for tracking
        """
        # Create error context
        error_context = self._create_error_context(
            error=error,
            component=context_type.lower(),
            operation="unknown",
            additional_data=context_data or {}
        )
        
        # Log with enhanced logger
        error_id = self.enhanced_logger.log_error_with_context(error, error_context)
        
        # Track error
        self.error_tracker.track_error(error, error_context)
        
        return error_id
    
    def create_user_friendly_message(self, error: Exception, error_type: str = None) -> str:
        """
        Generate user-friendly error messages from technical exceptions.
        
        Args:
            error: The exception to convert
            error_type: Optional error type hint for better message selection
            
        Returns:
            User-friendly error message
        """
        # Handle custom application errors
        if isinstance(error, EmailClassifierError):
            return self._get_user_friendly_message_for_custom_error(error)
        
        # Handle common Python exceptions
        if isinstance(error, FileNotFoundError):
            return "The requested file could not be found. Please check the file path and try again."
        elif isinstance(error, PermissionError):
            return "Permission denied. Please check file permissions or contact your administrator."
        elif isinstance(error, MemoryError):
            return "The system is running low on memory. Please try with a smaller file or contact support."
        elif isinstance(error, TimeoutError):
            return "The operation timed out. Please try again or contact support if the problem persists."
        elif isinstance(error, ConnectionError):
            return "Unable to connect to required services. Please check your connection and try again."
        
        # Fallback to generic message
        return self.user_friendly_messages.get('INTERNAL_ERROR')
    
    def _create_error_context(self, 
                             error: Exception, 
                             component: str, 
                             operation: str,
                             additional_data: Optional[Dict] = None) -> ErrorContext:
        """
        Create enhanced error context from exception and metadata.
        
        Args:
            error: The exception
            component: Component where error occurred
            operation: Operation being performed
            additional_data: Additional context data
            
        Returns:
            ErrorContext with categorization and metadata
        """
        # Determine category and severity from error type
        category = ErrorCategory.UNKNOWN
        severity = ErrorSeverity.MEDIUM
        
        if isinstance(error, EmailClassifierError):
            category = error.category
            severity = error.severity
        else:
            # Categorize standard Python exceptions
            if isinstance(error, (ValueError, TypeError)):
                category = ErrorCategory.VALIDATION
                severity = ErrorSeverity.LOW
            elif isinstance(error, (FileNotFoundError, PermissionError)):
                category = ErrorCategory.FILE_OPERATION
                severity = ErrorSeverity.MEDIUM
            elif isinstance(error, (ConnectionError, TimeoutError)):
                category = ErrorCategory.NETWORK
                severity = ErrorSeverity.HIGH
            elif isinstance(error, MemoryError):
                category = ErrorCategory.SYSTEM_RESOURCE
                severity = ErrorSeverity.CRITICAL
        
        # Get request context if available
        client_ip = None
        user_agent = None
        request_id = None
        
        if has_request_context():
            try:
                client_ip = request.remote_addr
                user_agent = request.headers.get('User-Agent')
                request_id = request.headers.get('X-Request-ID')
            except Exception:
                pass
        
        return ErrorContext(
            error_id=self.enhanced_logger._generate_error_id(),
            category=category,
            severity=severity,
            component=component,
            operation=operation,
            client_ip=client_ip,
            user_agent=user_agent,
            request_id=request_id,
            additional_data=additional_data or {}
        )
    
    def _categorize_dashboard_error(self, error: Exception) -> tuple:
        """Categorize dashboard errors and return appropriate message, code, and status."""
        if isinstance(error, ServiceNotReadyError):
            if "model" in error.message.lower():
                return (
                    self.user_friendly_messages['DASHBOARD_MODEL_NOT_TRAINED'],
                    'DASHBOARD_MODEL_NOT_TRAINED',
                    503
                )
            else:
                return (
                    self.user_friendly_messages['DASHBOARD_SERVICE_NOT_READY'],
                    'DASHBOARD_SERVICE_NOT_READY',
                    503
                )
        elif isinstance(error, ModelError):
            return (
                self.user_friendly_messages['DASHBOARD_MODEL_NOT_TRAINED'],
                'DASHBOARD_MODEL_NOT_TRAINED',
                503
            )
        elif isinstance(error, ProcessingError):
            return (
                self.user_friendly_messages['DASHBOARD_METRICS_ERROR'],
                'DASHBOARD_METRICS_ERROR',
                500
            )
        elif isinstance(error, ValidationError):
            return (
                self.user_friendly_messages['VALIDATION_ERROR'],
                'VALIDATION_ERROR',
                400
            )
        else:
            return (
                self.user_friendly_messages['DASHBOARD_DATA_UNAVAILABLE'],
                'DASHBOARD_DATA_UNAVAILABLE',
                500
            )
    
    def _categorize_batch_error(self, error: Exception, context: Optional[BatchErrorContext] = None) -> tuple:
        """Categorize batch processing errors and return appropriate message, code, and status."""
        if isinstance(error, ValidationError):
            if error.field == 'file':
                return (
                    self.user_friendly_messages['BATCH_FILE_INVALID'],
                    'BATCH_FILE_INVALID',
                    400
                )
            else:
                return (
                    self.user_friendly_messages['BATCH_INVALID_CSV'],
                    'BATCH_INVALID_CSV',
                    400
                )
        elif isinstance(error, ServiceNotReadyError):
            return (
                self.user_friendly_messages['BATCH_SERVICE_NOT_READY'],
                'BATCH_SERVICE_NOT_READY',
                503
            )
        elif isinstance(error, ProcessingError):
            return (
                self.user_friendly_messages['BATCH_PROCESSING_FAILED'],
                'BATCH_PROCESSING_FAILED',
                500
            )
        elif isinstance(error, MemoryError):
            return (
                self.user_friendly_messages['BATCH_FILE_TOO_LARGE'],
                'BATCH_FILE_TOO_LARGE',
                413
            )
        else:
            return (
                self.user_friendly_messages['BATCH_PROCESSING_FAILED'],
                'BATCH_PROCESSING_FAILED',
                500
            )
    
    def _get_user_friendly_message_for_custom_error(self, error: EmailClassifierError) -> str:
        """Get user-friendly message for custom application errors."""
        error_code = error.code
        
        # Map error codes to user-friendly messages
        code_mapping = {
            'SERVICE_NOT_READY': 'SERVICE_INITIALIZATION_ERROR',
            'MODEL_ERROR': 'MODEL_LOADING_ERROR',
            'PROCESSING_ERROR': 'PREPROCESSING_ERROR',
            'VALIDATION_ERROR': 'VALIDATION_ERROR'
        }
        
        message_key = code_mapping.get(error_code, 'INTERNAL_ERROR')
        return self.user_friendly_messages.get(message_key, error.message)
    
    def _create_dashboard_error_details(self, error: Exception, context: Optional[DashboardErrorContext], error_id: str = None) -> Dict[str, Any]:
        """Create detailed error information for dashboard errors."""
        details = {
            'error_type': type(error).__name__,
            'component': 'dashboard',
            'error_id': error_id
        }
        
        if context:
            details.update({
                'route': context.route,
                'user_action': context.user_action
            })
            
            if context.service_status:
                details['service_status'] = context.service_status
            
            if context.fallback_data:
                details['fallback_available'] = True
        
        # Add error categorization
        if isinstance(error, EmailClassifierError):
            details.update({
                'category': error.category.value,
                'severity': error.severity.value
            })
        
        return details
    
    def _create_batch_error_details(self, error: Exception, context: Optional[BatchErrorContext], error_id: str = None) -> Dict[str, Any]:
        """Create detailed error information for batch processing errors."""
        details = {
            'error_type': type(error).__name__,
            'component': 'batch_processing',
            'error_id': error_id
        }
        
        if context:
            details.update({
                'route': context.route,
                'filename': context.filename,
                'processing_stage': context.processing_stage
            })
            
            if context.processed_count is not None and context.total_count is not None:
                details.update({
                    'processed_count': context.processed_count,
                    'total_count': context.total_count,
                    'completion_percentage': round((context.processed_count / context.total_count) * 100, 2)
                })
        
        # Add error categorization
        if isinstance(error, EmailClassifierError):
            details.update({
                'category': error.category.value,
                'severity': error.severity.value
            })
        
        return details
    
    def _is_api_request(self) -> bool:
        """Check if the current request is an API request."""
        try:
            from flask import request
            return request.path.startswith('/api/')
        except RuntimeError:
            # Outside of request context
            return False
    
    def _create_api_response(self, error_response: ErrorResponse, status_code: int) -> tuple:
        """Create JSON API response."""
        response_dict = {
            'error': error_response.error,
            'message': error_response.message,
            'code': error_response.code,
            'timestamp': error_response.timestamp
        }
        
        if error_response.details:
            response_dict['details'] = error_response.details
        
        return jsonify(response_dict), status_code
    
    def _create_dashboard_web_response(self, error_response: ErrorResponse, status_code: int, context: Optional[DashboardErrorContext]) -> tuple:
        """Create HTML response for dashboard errors."""
        template = 'errors/500.html'
        
        if status_code == 400:
            template = 'errors/400.html'
        elif status_code == 503:
            template = 'errors/503.html'
        
        template_data = {
            'error': error_response,
            'show_refresh': status_code == 503,
            'fallback_data': context.fallback_data if context else None
        }
        
        return render_template(template, **template_data), status_code
    
    def _create_batch_web_response(self, error_response: ErrorResponse, status_code: int, context: Optional[BatchErrorContext]) -> tuple:
        """Create HTML response for batch processing errors."""
        template = 'errors/500.html'
        
        if status_code == 400:
            template = 'errors/400.html'
        elif status_code == 413:
            template = 'errors/413.html'
        elif status_code == 503:
            template = 'errors/503.html'
        
        template_data = {
            'error': error_response,
            'show_retry': status_code in [500, 503],
            'partial_results': context and context.processed_count and context.processed_count > 0
        }
        
        return render_template(template, **template_data), status_code


class InputValidator:
    """Input validation utilities."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def validate_email_text(self, text: str, field_name: str = "email_text") -> None:
        """
        Validate email text input.
        
        Args:
            text: Email text to validate
            field_name: Name of the field for error messages
            
        Raises:
            ValidationError: If validation fails
        """
        if text is None:
            raise ValidationError(f"{field_name} cannot be None", field_name)
        
        if not isinstance(text, str):
            raise ValidationError(f"{field_name} must be a string", field_name)
        
        text = text.strip()
        if not text:
            raise ValidationError(f"{field_name} cannot be empty", field_name)
        
        if len(text) < 10:
            raise ValidationError(f"{field_name} is too short (minimum 10 characters)", field_name)
        
        if len(text) > 10000:
            raise ValidationError(f"{field_name} is too long (maximum 10,000 characters)", field_name)
    
    def validate_email_batch(self, emails: List[str], max_batch_size: int = 1000) -> None:
        """
        Validate a batch of emails.
        
        Args:
            emails: List of email texts
            max_batch_size: Maximum allowed batch size
            
        Raises:
            ValidationError: If validation fails
        """
        if not isinstance(emails, list):
            raise ValidationError("emails must be a list", "emails")
        
        if not emails:
            raise ValidationError("emails list cannot be empty", "emails")
        
        if len(emails) > max_batch_size:
            raise ValidationError(
                f"Batch size {len(emails)} exceeds maximum limit of {max_batch_size}",
                "emails"
            )
        
        # Validate each email
        for i, email in enumerate(emails):
            try:
                self.validate_email_text(email, f"emails[{i}]")
            except ValidationError as e:
                raise ValidationError(f"Email at index {i}: {e.message}", "emails")
    
    def validate_file_upload(self, file, allowed_extensions: set = None) -> None:
        """
        Enhanced file upload validation with detailed error messages.
        
        Args:
            file: Uploaded file object
            allowed_extensions: Set of allowed file extensions
            
        Raises:
            ValidationError: If validation fails
        """
        if allowed_extensions is None:
            allowed_extensions = {'csv', 'txt'}
        
        if not file or file.filename == '':
            raise ValidationError("No file selected. Please choose a file to upload.", "file")
        
        filename = file.filename.strip()
        if not filename:
            raise ValidationError("Invalid filename. Please use a file with a valid name.", "file")
        
        # Check for dangerous characters in filename
        dangerous_chars = ['<', '>', ':', '"', '|', '?', '*', '\\', '/']
        if any(char in filename for char in dangerous_chars):
            raise ValidationError(
                "Invalid filename. Please use a filename without special characters like < > : \" | ? * \\ /",
                "file"
            )
        
        # Check file extension
        if not self._allowed_file(filename, allowed_extensions):
            file_ext = filename.rsplit('.', 1)[1].lower() if '.' in filename else 'none'
            raise ValidationError(
                f"Invalid file format '{file_ext}'. Please upload a file with one of these formats: {', '.join(sorted(allowed_extensions))}",
                "file"
            )
        
        # Additional CSV-specific validation if it's a CSV file
        if filename.lower().endswith('.csv'):
            self._validate_csv_file_basic(file)
    
    def _allowed_file(self, filename: str, allowed_extensions: set) -> bool:
        """Check if file extension is allowed."""
        return ('.' in filename and 
                filename.rsplit('.', 1)[1].lower() in allowed_extensions)
    
    def _validate_csv_file_basic(self, file) -> None:
        """
        Perform basic CSV file validation.
        
        Args:
            file: Uploaded file object
            
        Raises:
            ValidationError: If CSV validation fails
        """
        import csv
        import io
        
        # Save current position
        current_pos = file.tell()
        
        try:
            # Read first few lines to validate CSV format
            file.seek(0)
            content = file.read(1024).decode('utf-8', errors='ignore')  # Read first 1KB
            
            if not content.strip():
                raise ValidationError(
                    "The CSV file appears to be empty. Please upload a file with email data.",
                    "file"
                )
            
            # Try to parse as CSV
            csv_reader = csv.reader(io.StringIO(content))
            try:
                first_row = next(csv_reader)
                if not first_row or all(cell.strip() == '' for cell in first_row):
                    raise ValidationError(
                        "The CSV file has no valid data in the first row. Please ensure your file contains email data.",
                        "file"
                    )
            except StopIteration:
                raise ValidationError(
                    "The CSV file appears to have no readable rows. Please check the file format.",
                    "file"
                )
            except csv.Error as e:
                raise ValidationError(
                    f"Invalid CSV format: {str(e)}. Please ensure your file is properly formatted as CSV.",
                    "file"
                )
        
        except UnicodeDecodeError:
            raise ValidationError(
                "The file contains invalid characters. Please ensure your CSV file is saved with UTF-8 encoding.",
                "file"
            )
        except Exception as e:
            if isinstance(e, ValidationError):
                raise
            raise ValidationError(
                "Unable to read the CSV file. Please check the file format and try again.",
                "file"
            )
        finally:
            # Restore file position
            file.seek(current_pos)


class ServiceHealthChecker:
    """Check service health and readiness."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def check_classification_service(self, service) -> None:
        """
        Check if classification service is ready.
        
        Args:
            service: Classification service instance
            
        Raises:
            ServiceNotReadyError: If service is not ready
        """
        if service is None:
            raise ServiceNotReadyError("Classification Service", "Service not initialized")
        
        if not hasattr(service, 'is_ready') or not service.is_ready():
            raise ServiceNotReadyError(
                "Classification Service", 
                "No trained models available or preprocessing pipeline not fitted"
            )
    
    def check_batch_processor(self, processor) -> None:
        """
        Check if batch processor is ready.
        
        Args:
            processor: Batch processor instance
            
        Raises:
            ServiceNotReadyError: If processor is not ready
        """
        if processor is None:
            raise ServiceNotReadyError("Batch Processor", "Processor not initialized")
        
        if processor.classification_service is None:
            raise ServiceNotReadyError(
                "Batch Processor", 
                "Classification service not available"
            )


# Global instances
error_handler = ErrorHandler()
enhanced_error_handler = EnhancedErrorHandler()
input_validator = InputValidator()
health_checker = ServiceHealthChecker()
enhanced_logger = EnhancedLogger()
error_tracker = ErrorTracker()