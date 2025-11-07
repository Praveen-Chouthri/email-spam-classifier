"""
Service Health Management System for Email Spam Classification.

This module provides the ServiceHealthManager class that monitors service
initialization status, performs health checks, and provides fallback services
when components are not ready.
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Dict, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum

from src.models.data_models import ClassificationResult
from src.error_handling import ServiceNotReadyError


class ServiceStatus(Enum):
    """Service status enumeration."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    INITIALIZING = "initializing"
    NOT_READY = "not_ready"


@dataclass
class ServiceHealthStatus:
    """Health status information for a service."""
    service_name: str
    status: ServiceStatus
    is_ready: bool
    last_check: datetime
    error_message: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


@dataclass
class SystemHealthSummary:
    """Overall system health summary."""
    overall_status: ServiceStatus
    classification_service: ServiceHealthStatus
    batch_processor: ServiceHealthStatus
    model_availability: ServiceHealthStatus
    preprocessing_pipeline: ServiceHealthStatus
    last_updated: datetime


class FallbackClassificationService:
    """Fallback service when classification service is not ready."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def is_ready(self) -> bool:
        """Always returns False as this is a fallback service."""
        return False
    
    def classify_email(self, email_text: str) -> ClassificationResult:
        """Provide fallback classification result."""
        self.logger.warning("Using fallback classification service - returning default result")
        return ClassificationResult(
            prediction="Service Not Ready",
            confidence=0.0,
            model_used="Fallback Service",
            processing_time=0.0,
            timestamp=datetime.now()
        )
    
    def get_active_model(self) -> str:
        """Return fallback model name."""
        return "No Model Available"
    
    def get_model_metrics(self) -> dict:
        """Return empty metrics."""
        return {}


class FallbackBatchProcessor:
    """Fallback service when batch processor is not ready."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.classification_service = FallbackClassificationService()
    
    def process_csv(self, file_path: str, job_id: Optional[str] = None) -> str:
        """Reject batch processing when service not ready."""
        raise ServiceNotReadyError(
            "Batch Processor", 
            "System is not ready for batch processing. Please wait for initialization to complete."
        )
    
    def get_processing_status(self, job_id: str) -> None:
        """Return None for any job status request."""
        return None
    
    def list_active_jobs(self) -> list:
        """Return empty job list."""
        return []


class ServiceHealthManager:
    """
    Manages health checking and monitoring for all application services.
    
    Provides health check methods for classification service, batch processor,
    and model availability, along with fallback service creation when services
    are not ready.
    """
    
    def __init__(self, 
                 check_interval: int = 30,
                 retry_attempts: int = 3,
                 timeout: int = 10):
        """
        Initialize the service health manager.
        
        Args:
            check_interval: Interval between health checks in seconds
            retry_attempts: Number of retry attempts for failed checks
            timeout: Timeout for health check operations in seconds
        """
        self.check_interval = check_interval
        self.retry_attempts = retry_attempts
        self.timeout = timeout
        self.logger = logging.getLogger(__name__)
        
        # Health status tracking
        self._health_status: Dict[str, ServiceHealthStatus] = {}
        self._last_system_check = datetime.now()
        
        # Fallback services
        self._fallback_classification_service = FallbackClassificationService()
        self._fallback_batch_processor = FallbackBatchProcessor()
        
        # Service references (will be set by application)
        self._classification_service = None
        self._batch_processor = None
        self._model_manager = None
        self._preprocessing_pipeline = None
        
        self.logger.info(f"ServiceHealthManager initialized with check_interval={check_interval}s, retry_attempts={retry_attempts}, timeout={timeout}s")
    
    def set_services(self, 
                    classification_service=None,
                    batch_processor=None,
                    model_manager=None,
                    preprocessing_pipeline=None) -> None:
        """
        Set service references for health checking.
        
        Args:
            classification_service: ClassificationService instance
            batch_processor: BatchProcessor instance
            model_manager: ModelManager instance
            preprocessing_pipeline: PreprocessingPipeline instance
        """
        self._classification_service = classification_service
        self._batch_processor = batch_processor
        self._model_manager = model_manager
        self._preprocessing_pipeline = preprocessing_pipeline
        
        self.logger.info("Service references updated in ServiceHealthManager")
        
        # Perform initial health check
        self.check_all_services()
    
    def check_classification_service(self) -> bool:
        """
        Check if classification service is ready and healthy.
        
        Returns:
            True if service is ready, False otherwise
        """
        service_name = "classification_service"
        
        try:
            if self._classification_service is None:
                self._update_service_status(
                    service_name,
                    ServiceStatus.NOT_READY,
                    False,
                    "Classification service not initialized"
                )
                return False
            
            # Check if service has is_ready method and call it
            if hasattr(self._classification_service, 'is_ready'):
                is_ready = self._classification_service.is_ready()
                
                if is_ready:
                    # Additional checks for service health
                    try:
                        # Try to get active model
                        active_model = self._classification_service.get_active_model()
                        
                        # Try to get model metrics
                        metrics = self._classification_service.get_model_metrics()
                        
                        self._update_service_status(
                            service_name,
                            ServiceStatus.HEALTHY,
                            True,
                            None,
                            {
                                "active_model": active_model,
                                "available_models": len(metrics) if metrics else 0
                            }
                        )
                        return True
                        
                    except Exception as e:
                        self.logger.warning(f"Classification service partially ready but has issues: {str(e)}")
                        self._update_service_status(
                            service_name,
                            ServiceStatus.DEGRADED,
                            True,
                            f"Service ready but with issues: {str(e)}"
                        )
                        return True
                else:
                    self._update_service_status(
                        service_name,
                        ServiceStatus.NOT_READY,
                        False,
                        "Service reports not ready - models may not be trained"
                    )
                    return False
            else:
                self._update_service_status(
                    service_name,
                    ServiceStatus.UNHEALTHY,
                    False,
                    "Service does not implement readiness check"
                )
                return False
                
        except Exception as e:
            self.logger.error(f"Error checking classification service: {str(e)}")
            self._update_service_status(
                service_name,
                ServiceStatus.UNHEALTHY,
                False,
                f"Health check failed: {str(e)}"
            )
            return False
    
    def check_batch_processor(self) -> bool:
        """
        Check if batch processor is ready and healthy.
        
        Returns:
            True if processor is ready, False otherwise
        """
        service_name = "batch_processor"
        
        try:
            if self._batch_processor is None:
                self._update_service_status(
                    service_name,
                    ServiceStatus.NOT_READY,
                    False,
                    "Batch processor not initialized"
                )
                return False
            
            # Check if batch processor has classification service
            if not hasattr(self._batch_processor, 'classification_service') or \
               self._batch_processor.classification_service is None:
                self._update_service_status(
                    service_name,
                    ServiceStatus.NOT_READY,
                    False,
                    "Batch processor missing classification service"
                )
                return False
            
            # Check if classification service is ready
            classification_ready = self.check_classification_service()
            
            if classification_ready:
                # Check batch processor specific functionality
                try:
                    # Check if results directory exists and is writable
                    results_dir = getattr(self._batch_processor, 'results_dir', None)
                    if results_dir:
                        import os
                        if not os.path.exists(results_dir):
                            os.makedirs(results_dir, exist_ok=True)
                    
                    # Get active jobs count
                    active_jobs = []
                    if hasattr(self._batch_processor, 'list_active_jobs'):
                        active_jobs = self._batch_processor.list_active_jobs()
                    
                    self._update_service_status(
                        service_name,
                        ServiceStatus.HEALTHY,
                        True,
                        None,
                        {
                            "active_jobs": len(active_jobs),
                            "results_dir": results_dir
                        }
                    )
                    return True
                    
                except Exception as e:
                    self.logger.warning(f"Batch processor has issues: {str(e)}")
                    self._update_service_status(
                        service_name,
                        ServiceStatus.DEGRADED,
                        True,
                        f"Processor ready but with issues: {str(e)}"
                    )
                    return True
            else:
                self._update_service_status(
                    service_name,
                    ServiceStatus.NOT_READY,
                    False,
                    "Batch processor not ready - classification service not available"
                )
                return False
                
        except Exception as e:
            self.logger.error(f"Error checking batch processor: {str(e)}")
            self._update_service_status(
                service_name,
                ServiceStatus.UNHEALTHY,
                False,
                f"Health check failed: {str(e)}"
            )
            return False
    
    def check_model_availability(self) -> bool:
        """
        Check if trained models are available and ready.
        
        Returns:
            True if models are available, False otherwise
        """
        service_name = "model_availability"
        
        try:
            if self._model_manager is None:
                self._update_service_status(
                    service_name,
                    ServiceStatus.NOT_READY,
                    False,
                    "Model manager not initialized"
                )
                return False
            
            # Check if models are loaded
            try:
                # Try to get best model
                model_name, model = self._model_manager.get_best_model()
                
                if model is None:
                    self._update_service_status(
                        service_name,
                        ServiceStatus.NOT_READY,
                        False,
                        "No models available"
                    )
                    return False
                
                # Check if model is trained
                is_trained = hasattr(model, 'is_trained') and model.is_trained
                
                if is_trained:
                    # Get model metrics
                    all_metrics = self._model_manager.get_all_metrics()
                    
                    self._update_service_status(
                        service_name,
                        ServiceStatus.HEALTHY,
                        True,
                        None,
                        {
                            "best_model": model_name,
                            "total_models": len(all_metrics),
                            "trained_models": sum(1 for m in self._model_manager.models.values() 
                                                if hasattr(m, 'is_trained') and m.is_trained)
                        }
                    )
                    return True
                else:
                    self._update_service_status(
                        service_name,
                        ServiceStatus.NOT_READY,
                        False,
                        f"Best model ({model_name}) is not trained"
                    )
                    return False
                    
            except Exception as e:
                self.logger.warning(f"Error accessing models: {str(e)}")
                self._update_service_status(
                    service_name,
                    ServiceStatus.UNHEALTHY,
                    False,
                    f"Model access failed: {str(e)}"
                )
                return False
                
        except Exception as e:
            self.logger.error(f"Error checking model availability: {str(e)}")
            self._update_service_status(
                service_name,
                ServiceStatus.UNHEALTHY,
                False,
                f"Health check failed: {str(e)}"
            )
            return False
    
    def check_preprocessing_pipeline(self) -> bool:
        """
        Check if preprocessing pipeline is fitted and ready.
        
        Returns:
            True if pipeline is ready, False otherwise
        """
        service_name = "preprocessing_pipeline"
        
        try:
            # Check preprocessing pipeline through classification service if available
            if self._classification_service is not None:
                try:
                    # Get the preprocessing pipeline from the classification service
                    pipeline = getattr(self._classification_service, 'preprocessing_pipeline', None)
                    
                    if pipeline is None:
                        self._update_service_status(
                            service_name,
                            ServiceStatus.NOT_READY,
                            False,
                            "Preprocessing pipeline not available in classification service"
                        )
                        return False
                    
                    # Check if pipeline is fitted
                    is_fitted = hasattr(pipeline, '_is_fitted') and pipeline._is_fitted
                    
                    if is_fitted:
                        self._update_service_status(
                            service_name,
                            ServiceStatus.HEALTHY,
                            True,
                            None,
                            {
                                "fitted": True,
                                "pipeline_type": type(pipeline).__name__,
                                "source": "classification_service"
                            }
                        )
                        return True
                    else:
                        self._update_service_status(
                            service_name,
                            ServiceStatus.NOT_READY,
                            False,
                            "Preprocessing pipeline not fitted"
                        )
                        return False
                        
                except Exception as e:
                    self.logger.warning(f"Error checking pipeline through classification service: {str(e)}")
                    # Fall back to direct pipeline check
            
            # Fallback: check direct preprocessing pipeline reference
            if self._preprocessing_pipeline is None:
                self._update_service_status(
                    service_name,
                    ServiceStatus.NOT_READY,
                    False,
                    "Preprocessing pipeline not initialized"
                )
                return False
            
            # Check if pipeline is fitted
            is_fitted = hasattr(self._preprocessing_pipeline, '_is_fitted') and \
                       self._preprocessing_pipeline._is_fitted
            
            if is_fitted:
                self._update_service_status(
                    service_name,
                    ServiceStatus.HEALTHY,
                    True,
                    None,
                    {
                        "fitted": True,
                        "pipeline_type": type(self._preprocessing_pipeline).__name__,
                        "source": "direct_reference"
                    }
                )
                return True
            else:
                self._update_service_status(
                    service_name,
                    ServiceStatus.NOT_READY,
                    False,
                    "Preprocessing pipeline not fitted"
                )
                return False
                
        except Exception as e:
            self.logger.error(f"Error checking preprocessing pipeline: {str(e)}")
            self._update_service_status(
                service_name,
                ServiceStatus.UNHEALTHY,
                False,
                f"Health check failed: {str(e)}"
            )
            return False
    
    def check_all_services(self) -> SystemHealthSummary:
        """
        Check health of all services and return system summary.
        
        Returns:
            SystemHealthSummary with overall system health
        """
        self.logger.debug("Performing comprehensive health check of all services")
        
        # Check individual services
        classification_ready = self.check_classification_service()
        batch_ready = self.check_batch_processor()
        models_ready = self.check_model_availability()
        pipeline_ready = self.check_preprocessing_pipeline()
        
        # Determine overall system status
        overall_status = self._determine_overall_status(
            classification_ready, batch_ready, models_ready, pipeline_ready
        )
        
        # Create summary
        summary = SystemHealthSummary(
            overall_status=overall_status,
            classification_service=self._health_status.get("classification_service"),
            batch_processor=self._health_status.get("batch_processor"),
            model_availability=self._health_status.get("model_availability"),
            preprocessing_pipeline=self._health_status.get("preprocessing_pipeline"),
            last_updated=datetime.now()
        )
        
        self._last_system_check = datetime.now()
        
        self.logger.info(f"System health check completed. Overall status: {overall_status.value}")
        
        return summary
    
    def get_service_status_summary(self) -> Dict[str, bool]:
        """
        Get a simple summary of service readiness status.
        
        Returns:
            Dictionary mapping service names to readiness status
        """
        return {
            "classification_service": self._health_status.get("classification_service", 
                ServiceHealthStatus("classification_service", ServiceStatus.NOT_READY, False, datetime.now())).is_ready,
            "batch_processor": self._health_status.get("batch_processor",
                ServiceHealthStatus("batch_processor", ServiceStatus.NOT_READY, False, datetime.now())).is_ready,
            "model_availability": self._health_status.get("model_availability",
                ServiceHealthStatus("model_availability", ServiceStatus.NOT_READY, False, datetime.now())).is_ready,
            "preprocessing_pipeline": self._health_status.get("preprocessing_pipeline",
                ServiceHealthStatus("preprocessing_pipeline", ServiceStatus.NOT_READY, False, datetime.now())).is_ready
        }
    
    def get_fallback_classification_service(self):
        """
        Get fallback classification service for when main service is not ready.
        
        Returns:
            FallbackClassificationService instance
        """
        return self._fallback_classification_service
    
    def get_fallback_batch_processor(self):
        """
        Get fallback batch processor for when main processor is not ready.
        
        Returns:
            FallbackBatchProcessor instance
        """
        return self._fallback_batch_processor
    
    def should_use_fallback_services(self) -> bool:
        """
        Determine if fallback services should be used.
        
        Returns:
            True if fallback services should be used
        """
        classification_ready = self._health_status.get("classification_service", 
            ServiceHealthStatus("classification_service", ServiceStatus.NOT_READY, False, datetime.now())).is_ready
        
        return not classification_ready
    
    def _update_service_status(self, 
                              service_name: str,
                              status: ServiceStatus,
                              is_ready: bool,
                              error_message: Optional[str] = None,
                              details: Optional[Dict[str, Any]] = None) -> None:
        """Update the health status for a service."""
        self._health_status[service_name] = ServiceHealthStatus(
            service_name=service_name,
            status=status,
            is_ready=is_ready,
            last_check=datetime.now(),
            error_message=error_message,
            details=details or {}
        )
    
    def _determine_overall_status(self, 
                                 classification_ready: bool,
                                 batch_ready: bool,
                                 models_ready: bool,
                                 pipeline_ready: bool) -> ServiceStatus:
        """Determine overall system status based on individual service status."""
        
        # Count ready services
        ready_services = sum([classification_ready, batch_ready, models_ready, pipeline_ready])
        
        if ready_services == 4:
            return ServiceStatus.HEALTHY
        elif ready_services >= 2:
            return ServiceStatus.DEGRADED
        elif ready_services >= 1:
            return ServiceStatus.UNHEALTHY
        else:
            return ServiceStatus.NOT_READY
    
    def get_detailed_health_status(self) -> Dict[str, ServiceHealthStatus]:
        """
        Get detailed health status for all services.
        
        Returns:
            Dictionary mapping service names to their health status
        """
        return self._health_status.copy()
    
    def is_system_ready(self) -> bool:
        """
        Check if the overall system is ready for operation.
        
        Returns:
            True if system is ready (at least classification service is ready)
        """
        classification_status = self._health_status.get("classification_service")
        return classification_status is not None and classification_status.is_ready
    
    def get_system_readiness_message(self) -> str:
        """
        Get a user-friendly message about system readiness.
        
        Returns:
            Human-readable message about system status
        """
        if self.is_system_ready():
            return "System is ready for email classification"
        
        # Check what's missing
        missing_components = []
        
        for service_name, status in self._health_status.items():
            if not status.is_ready:
                missing_components.append(service_name.replace("_", " ").title())
        
        if missing_components:
            return f"System is not ready. Missing: {', '.join(missing_components)}"
        else:
            return "System is initializing. Please wait..."
    
    def validate_service_readiness(self, service_name: str) -> Tuple[bool, str]:
        """
        Validate if a specific service is ready for operation.
        
        Args:
            service_name: Name of the service to check
            
        Returns:
            Tuple of (is_ready, message)
        """
        status = self._health_status.get(service_name)
        
        if status is None:
            return False, f"Service '{service_name}' not found"
        
        if status.is_ready:
            return True, f"Service '{service_name}' is ready"
        else:
            error_msg = status.error_message or "Service not ready"
            return False, f"Service '{service_name}' is not ready: {error_msg}"
    
    def get_service_readiness_for_routes(self) -> Dict[str, Tuple[bool, str]]:
        """
        Get service readiness information formatted for route validation.
        
        Returns:
            Dictionary mapping service names to (is_ready, message) tuples
        """
        return {
            service_name: self.validate_service_readiness(service_name)
            for service_name in ['classification_service', 'batch_processor', 'model_availability']
        }
    
    def create_service_not_ready_response(self, service_name: str, request_type: str = 'web') -> dict:
        """
        Create a standardized response for when services are not ready.
        
        Args:
            service_name: Name of the service that's not ready
            request_type: Type of request ('web' or 'api')
            
        Returns:
            Dictionary with error response data
        """
        is_ready, message = self.validate_service_readiness(service_name)
        
        if request_type == 'api':
            return {
                'error': True,
                'message': message,
                'code': 'SERVICE_NOT_READY',
                'service': service_name,
                'timestamp': datetime.now().isoformat(),
                'system_status': self.get_service_status_summary()
            }
        else:
            return {
                'error_message': message,
                'service_name': service_name,
                'system_status': self.get_service_status_summary(),
                'readiness_message': self.get_system_readiness_message()
            }