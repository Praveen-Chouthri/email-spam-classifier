"""
Dashboard Data Provider with Fallback Support.

This module provides the DashboardDataProvider class that safely retrieves
dashboard data with comprehensive fallback mechanisms when services are
unavailable or not ready.
"""

import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict

from src.services.service_health_manager import ServiceHealthManager, SystemHealthSummary
from src.error_handling import ServiceNotReadyError


@dataclass
class DashboardData:
    """Complete dashboard data structure with fallback support."""
    active_model: str
    metrics: Dict[str, Any]
    system_status: str
    processed_today: int
    avg_response_time: float
    service_ready: bool
    last_updated: str
    error_message: Optional[str] = None
    health_summary: Optional[SystemHealthSummary] = None
    service_status: Optional[Dict[str, bool]] = None
    processing_time_distribution: Optional[Dict[str, int]] = None
    classification_results_distribution: Optional[Dict[str, int]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for template rendering."""
        return asdict(self)


@dataclass
class ModelMetrics:
    """Model performance metrics with safe defaults."""
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    test_samples: int = 0
    training_date: Optional[str] = None
    class_balancing_enabled: bool = False
    original_spam_ratio: float = 0.0
    balanced_spam_ratio: float = 0.0
    false_negative_rate: float = 0.0
    synthetic_samples_used: int = 0
    balancing_method: str = 'none'
    
    @classmethod
    def create_fallback(cls) -> 'ModelMetrics':
        """Create fallback metrics when real data is unavailable."""
        return cls(
            accuracy=0.0,
            precision=0.0,
            recall=0.0,
            f1_score=0.0,
            test_samples=0,
            training_date=None,
            class_balancing_enabled=False,
            original_spam_ratio=0.0,
            balanced_spam_ratio=0.0,
            false_negative_rate=0.0,
            synthetic_samples_used=0,
            balancing_method='none'
        )


class DashboardDataProvider:
    """
    Provides dashboard data with comprehensive fallback mechanisms.
    
    This class safely retrieves dashboard data from various services and
    provides appropriate fallback values when services are not ready or
    when errors occur during data retrieval.
    """
    
    def __init__(self, service_health_manager: ServiceHealthManager):
        """
        Initialize the dashboard data provider.
        
        Args:
            service_health_manager: ServiceHealthManager instance for health checks
        """
        self.service_health_manager = service_health_manager
        self.logger = logging.getLogger(__name__)
        
        # Cache for fallback data
        self._fallback_data_cache = None
        self._last_successful_data = None
        
        self.logger.info("DashboardDataProvider initialized")
    
    def get_dashboard_data(self, 
                          classification_service=None,
                          batch_processor=None) -> DashboardData:
        """
        Get complete dashboard data with fallback support.
        
        Args:
            classification_service: Optional classification service instance
            batch_processor: Optional batch processor instance
            
        Returns:
            DashboardData with all dashboard information
        """
        self.logger.debug("Retrieving dashboard data...")
        
        try:
            # Perform health check first
            health_summary = self.service_health_manager.check_all_services()
            
            # Check if system is ready
            if not self.service_health_manager.is_system_ready():
                return self._create_fallback_dashboard_data(
                    health_summary=health_summary,
                    error_message=self.service_health_manager.get_system_readiness_message()
                )
            
            # System is ready - try to get real data
            return self._get_real_dashboard_data(
                classification_service, 
                batch_processor, 
                health_summary
            )
            
        except Exception as e:
            self.logger.error(f"Error getting dashboard data: {str(e)}", exc_info=True)
            return self._create_fallback_dashboard_data(
                error_message=f"Error retrieving dashboard data: {str(e)}"
            )
    
    def _get_real_dashboard_data(self, 
                                classification_service,
                                batch_processor,
                                health_summary: SystemHealthSummary) -> DashboardData:
        """Get real dashboard data from services."""
        self.logger.debug("Attempting to retrieve real dashboard data...")
        
        # Initialize data with safe defaults
        active_model = "No Model Available"
        metrics = {}
        processed_today = 0
        avg_response_time = 0.0
        service_status = self.service_health_manager.get_service_status_summary()
        
        # Get model information
        if classification_service:
            try:
                active_model = self._get_active_model_safe(classification_service)
                metrics = self._get_model_metrics_safe(classification_service)
            except Exception as e:
                self.logger.warning(f"Error getting model data: {str(e)}")
        
        # Get processing statistics
        try:
            processed_today, avg_response_time = self._get_processing_stats_safe(
                classification_service, batch_processor
            )
        except Exception as e:
            self.logger.warning(f"Error getting processing stats: {str(e)}")
        
        # Get processing time distribution
        processing_time_distribution = self._get_processing_time_distribution(classification_service)
        
        # Get classification results distribution
        classification_results_distribution = self._get_classification_results_distribution(classification_service)
        
        # Create dashboard data
        dashboard_data = DashboardData(
            active_model=active_model,
            metrics=metrics,
            system_status=health_summary.overall_status.value,
            processed_today=processed_today,
            avg_response_time=avg_response_time,
            service_ready=True,
            last_updated=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            health_summary=health_summary,
            service_status=service_status,
            processing_time_distribution=processing_time_distribution,
            classification_results_distribution=classification_results_distribution
        )
        
        # Cache successful data for future fallback
        self._last_successful_data = dashboard_data
        
        self.logger.info("Successfully retrieved real dashboard data")
        return dashboard_data
    
    def _get_active_model_safe(self, classification_service) -> str:
        """Safely get active model name."""
        try:
            if hasattr(classification_service, 'get_active_model'):
                active_model = classification_service.get_active_model()
                return active_model if active_model else "No Model Set"
            else:
                return "Model Service Unavailable"
        except Exception as e:
            self.logger.debug(f"Could not get active model: {str(e)}")
            return "Unknown Model"
    
    def _get_model_metrics_safe(self, classification_service) -> Dict[str, Any]:
        """Safely get model metrics."""
        try:
            if hasattr(classification_service, 'get_model_metrics'):
                metrics = classification_service.get_model_metrics()
                
                # Validate and sanitize metrics
                if isinstance(metrics, dict):
                    sanitized_metrics = {}
                    for model_name, model_metrics in metrics.items():
                        sanitized_metrics[model_name] = self._sanitize_model_metrics(model_metrics)
                    return sanitized_metrics
                else:
                    self.logger.warning("Model metrics returned in unexpected format")
                    return {}
            else:
                return {}
        except Exception as e:
            self.logger.debug(f"Could not get model metrics: {str(e)}")
            return {}
    
    def _sanitize_model_metrics(self, metrics) -> ModelMetrics:
        """Sanitize and validate model metrics."""
        try:
            if hasattr(metrics, 'accuracy'):
                # Metrics object with attributes
                return ModelMetrics(
                    accuracy=getattr(metrics, 'accuracy', 0.0),
                    precision=getattr(metrics, 'precision', 0.0),
                    recall=getattr(metrics, 'recall', 0.0),
                    f1_score=getattr(metrics, 'f1_score', 0.0),
                    test_samples=getattr(metrics, 'test_samples', 0),
                    training_date=getattr(metrics, 'training_date', None),
                    class_balancing_enabled=getattr(metrics, 'class_balancing_enabled', False),
                    original_spam_ratio=getattr(metrics, 'original_spam_ratio', 0.0),
                    balanced_spam_ratio=getattr(metrics, 'balanced_spam_ratio', 0.0),
                    false_negative_rate=getattr(metrics, 'false_negative_rate', 0.0),
                    synthetic_samples_used=getattr(metrics, 'synthetic_samples_used', 0),
                    balancing_method=getattr(metrics, 'balancing_method', 'none')
                )
            elif isinstance(metrics, dict):
                # Dictionary format
                return ModelMetrics(
                    accuracy=metrics.get('accuracy', 0.0),
                    precision=metrics.get('precision', 0.0),
                    recall=metrics.get('recall', 0.0),
                    f1_score=metrics.get('f1_score', 0.0),
                    test_samples=metrics.get('test_samples', 0),
                    training_date=metrics.get('training_date', None),
                    class_balancing_enabled=metrics.get('class_balancing_enabled', False),
                    original_spam_ratio=metrics.get('original_spam_ratio', 0.0),
                    balanced_spam_ratio=metrics.get('balanced_spam_ratio', 0.0),
                    false_negative_rate=metrics.get('false_negative_rate', 0.0),
                    synthetic_samples_used=metrics.get('synthetic_samples_used', 0),
                    balancing_method=metrics.get('balancing_method', 'none')
                )
            else:
                return ModelMetrics.create_fallback()
        except Exception as e:
            self.logger.debug(f"Error sanitizing metrics: {str(e)}")
            return ModelMetrics.create_fallback()
    
    def _get_processing_stats_safe(self, 
                                  classification_service, 
                                  batch_processor) -> tuple[int, float]:
        """Safely get processing statistics."""
        processed_today = 0
        avg_response_time = 0.0
        
        try:
            # Try to get stats from classification service
            if classification_service and hasattr(classification_service, 'get_processing_stats'):
                stats = classification_service.get_processing_stats()
                if isinstance(stats, dict):
                    processed_today = stats.get('processed_today', 0)
                    avg_response_time = stats.get('avg_response_time', 0.0)
        except Exception as e:
            self.logger.debug(f"Could not get processing stats from classification service: {str(e)}")
        
        try:
            # Try to get additional stats from batch processor
            if batch_processor and hasattr(batch_processor, 'get_daily_stats'):
                batch_stats = batch_processor.get_daily_stats()
                if isinstance(batch_stats, dict):
                    batch_processed = batch_stats.get('processed_today', 0)
                    processed_today += batch_processed
        except Exception as e:
            self.logger.debug(f"Could not get batch processing stats: {str(e)}")
        
        return processed_today, avg_response_time
    
    def _get_processing_time_distribution(self, classification_service) -> Dict[str, int]:
        """Get processing time distribution for charts."""
        try:
            if classification_service and hasattr(classification_service, '_response_times'):
                response_times = list(classification_service._response_times)
                
                if not response_times:
                    # Return empty distribution if no data
                    return {
                        '0-1s': 0,
                        '1-2s': 0, 
                        '2-3s': 0,
                        '3-4s': 0,
                        '4s+': 0
                    }
                
                # Categorize response times
                distribution = {
                    '0-1s': 0,
                    '1-2s': 0,
                    '2-3s': 0, 
                    '3-4s': 0,
                    '4s+': 0
                }
                
                for time in response_times:
                    if time < 1.0:
                        distribution['0-1s'] += 1
                    elif time < 2.0:
                        distribution['1-2s'] += 1
                    elif time < 3.0:
                        distribution['2-3s'] += 1
                    elif time < 4.0:
                        distribution['3-4s'] += 1
                    else:
                        distribution['4s+'] += 1
                
                return distribution
                
        except Exception as e:
            self.logger.debug(f"Could not get processing time distribution: {str(e)}")
        
        # Return empty distribution on error
        return {
            '0-1s': 0,
            '1-2s': 0,
            '2-3s': 0,
            '3-4s': 0,
            '4s+': 0
        }
    
    def _get_classification_results_distribution(self, classification_service) -> Dict[str, int]:
        """Get classification results distribution for charts."""
        try:
            if classification_service and hasattr(classification_service, 'get_classification_distribution'):
                distribution = classification_service.get_classification_distribution()
                
                # Validate the distribution data
                if isinstance(distribution, dict):
                    return {
                        'spam_count': distribution.get('spam_count', 0),
                        'legitimate_count': distribution.get('legitimate_count', 0),
                        'total_count': distribution.get('total_count', 0),
                        'spam_percentage': distribution.get('spam_percentage', 0.0),
                        'legitimate_percentage': distribution.get('legitimate_percentage', 0.0)
                    }
                
        except Exception as e:
            self.logger.debug(f"Could not get classification results distribution: {str(e)}")
        
        # Return empty distribution on error
        return {
            'spam_count': 0,
            'legitimate_count': 0,
            'total_count': 0,
            'spam_percentage': 0.0,
            'legitimate_percentage': 0.0
        }
    
    def _create_fallback_dashboard_data(self, 
                                       health_summary: Optional[SystemHealthSummary] = None,
                                       error_message: Optional[str] = None) -> DashboardData:
        """Create fallback dashboard data when services are not ready."""
        self.logger.debug("Creating fallback dashboard data")
        
        # Determine system status
        system_status = "not_ready"
        if health_summary:
            system_status = health_summary.overall_status.value
        
        # Use cached data if available and recent
        if self._last_successful_data and self._is_cached_data_recent():
            self.logger.debug("Using cached dashboard data as fallback")
            cached_data = self._last_successful_data
            return DashboardData(
                active_model=f"{cached_data.active_model} (Cached)",
                metrics=cached_data.metrics,
                system_status=system_status,
                processed_today=cached_data.processed_today,
                avg_response_time=cached_data.avg_response_time,
                service_ready=False,
                last_updated=cached_data.last_updated,
                error_message=error_message,
                health_summary=health_summary,
                service_status=self.service_health_manager.get_service_status_summary(),
                processing_time_distribution=cached_data.processing_time_distribution,
                classification_results_distribution=cached_data.classification_results_distribution
            )
        
        # Create complete fallback data
        fallback_data = DashboardData(
            active_model="System Not Ready",
            metrics=self._create_fallback_metrics(),
            system_status=system_status,
            processed_today=0,
            avg_response_time=0.0,
            service_ready=False,
            last_updated=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            error_message=error_message or "System is initializing. Please wait...",
            health_summary=health_summary,
            service_status=self.service_health_manager.get_service_status_summary(),
            processing_time_distribution={
                '0-1s': 0,
                '1-2s': 0,
                '2-3s': 0,
                '3-4s': 0,
                '4s+': 0
            },
            classification_results_distribution={
                'spam_count': 0,
                'legitimate_count': 0,
                'total_count': 0,
                'spam_percentage': 0.0,
                'legitimate_percentage': 0.0
            }
        )
        
        self.logger.info("Created fallback dashboard data")
        return fallback_data
    
    def _create_fallback_metrics(self) -> Dict[str, ModelMetrics]:
        """Create fallback metrics for when no models are available."""
        return {
            "No Models Available": ModelMetrics.create_fallback()
        }
    
    def _is_cached_data_recent(self) -> bool:
        """Check if cached data is recent enough to use as fallback."""
        if not self._last_successful_data:
            return False
        
        try:
            # Parse the last updated timestamp
            last_updated = datetime.strptime(
                self._last_successful_data.last_updated, 
                '%Y-%m-%d %H:%M:%S'
            )
            
            # Consider data recent if it's less than 5 minutes old
            time_diff = datetime.now() - last_updated
            return time_diff.total_seconds() < 300  # 5 minutes
            
        except Exception as e:
            self.logger.debug(f"Error checking cached data age: {str(e)}")
            return False
    
    def get_system_health_summary(self) -> SystemHealthSummary:
        """Get current system health summary."""
        return self.service_health_manager.check_all_services()
    
    def get_service_readiness_message(self) -> str:
        """Get user-friendly service readiness message."""
        return self.service_health_manager.get_system_readiness_message()
    
    def clear_cache(self) -> None:
        """Clear cached dashboard data."""
        self._last_successful_data = None
        self._fallback_data_cache = None
        self.logger.info("Dashboard data cache cleared")