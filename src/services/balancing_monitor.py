"""
Production monitoring system for class balancing operations.

This module provides comprehensive monitoring, alerting, and health checks
for class balancing systems in production spam classification deployments.
"""
import logging
import time
import json
import os
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import threading
from collections import deque


class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class BalancingAlert:
    """Alert for balancing system issues."""
    timestamp: datetime
    level: AlertLevel
    component: str
    message: str
    metrics: Dict[str, Any]
    resolved: bool = False
    resolution_time: Optional[datetime] = None


@dataclass
class HealthCheckResult:
    """Result from a health check operation."""
    component: str
    status: str  # "healthy", "degraded", "unhealthy"
    timestamp: datetime
    response_time_ms: float
    details: Dict[str, Any]
    error_message: Optional[str] = None


@dataclass
class PerformanceMetrics:
    """Performance metrics for monitoring."""
    timestamp: datetime
    processing_time_ms: float
    memory_usage_mb: float
    samples_processed: int
    false_negative_rate: float
    accuracy: float
    synthetic_samples_created: int
    balancing_method: str
    validation_passed: bool


class BalancingMonitor:
    """
    Production monitoring system for class balancing operations.
    
    Provides real-time monitoring, alerting, and health checks for
    class balancing components in production environments.
    """
    
    def __init__(self, 
                 alert_handlers: Optional[List[Callable]] = None,
                 metrics_retention_hours: int = 24,
                 health_check_interval_seconds: int = 300):
        """
        Initialize the balancing monitor.
        
        Args:
            alert_handlers: List of functions to handle alerts
            metrics_retention_hours: How long to retain metrics in memory
            health_check_interval_seconds: Interval between health checks
        """
        self.logger = logging.getLogger(__name__)
        
        # Alert system
        self.alert_handlers = alert_handlers or []
        self.active_alerts: List[BalancingAlert] = []
        self.alert_history: deque = deque(maxlen=1000)
        
        # Metrics storage
        self.metrics_retention_hours = metrics_retention_hours
        self.performance_metrics: deque = deque(maxlen=10000)
        self.health_check_results: deque = deque(maxlen=1000)
        
        # Health check system
        self.health_check_interval = health_check_interval_seconds
        self.health_checks: Dict[str, Callable] = {}
        self.monitoring_active = False
        self.monitoring_thread: Optional[threading.Thread] = None
        
        # Performance thresholds
        self.thresholds = {
            'max_processing_time_ms': 30000,  # 30 seconds
            'max_memory_usage_mb': 4096,      # 4GB
            'min_accuracy': 0.90,             # 90%
            'max_fnr': 0.10,                  # 10%
            'max_synthetic_ratio': 2.0        # 2x original samples
        }
        
        # Component status
        self.component_status = {
            'class_balancer': 'unknown',
            'smote_processor': 'unknown',
            'balancing_validator': 'unknown',
            'performance_optimizer': 'unknown'
        }
        
        self.logger.info("Balancing monitor initialized")
    
    def start_monitoring(self) -> None:
        """Start continuous monitoring in background thread."""
        if self.monitoring_active:
            self.logger.warning("Monitoring already active")
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        self.logger.info(f"Started continuous monitoring with {self.health_check_interval}s interval")
    
    def stop_monitoring(self) -> None:
        """Stop continuous monitoring."""
        self.monitoring_active = False
        
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5.0)
        
        self.logger.info("Stopped continuous monitoring")
    
    def record_performance_metrics(self, 
                                 processing_time_ms: float,
                                 memory_usage_mb: float,
                                 samples_processed: int,
                                 false_negative_rate: float,
                                 accuracy: float,
                                 synthetic_samples_created: int = 0,
                                 balancing_method: str = "none",
                                 validation_passed: bool = True) -> None:
        """
        Record performance metrics for monitoring.
        
        Args:
            processing_time_ms: Processing time in milliseconds
            memory_usage_mb: Memory usage in MB
            samples_processed: Number of samples processed
            false_negative_rate: False negative rate achieved
            accuracy: Model accuracy
            synthetic_samples_created: Number of synthetic samples created
            balancing_method: Balancing method used
            validation_passed: Whether validation checks passed
        """
        metrics = PerformanceMetrics(
            timestamp=datetime.now(),
            processing_time_ms=processing_time_ms,
            memory_usage_mb=memory_usage_mb,
            samples_processed=samples_processed,
            false_negative_rate=false_negative_rate,
            accuracy=accuracy,
            synthetic_samples_created=synthetic_samples_created,
            balancing_method=balancing_method,
            validation_passed=validation_passed
        )
        
        self.performance_metrics.append(metrics)
        
        # Check for threshold violations
        self._check_performance_thresholds(metrics)
        
        # Clean up old metrics
        self._cleanup_old_metrics()
        
        self.logger.debug(f"Recorded performance metrics: "
                         f"processing_time={processing_time_ms:.1f}ms, "
                         f"accuracy={accuracy:.3f}, fnr={false_negative_rate:.3f}")
    
    def record_balancing_operation(self, 
                                 operation_type: str,
                                 success: bool,
                                 duration_ms: float,
                                 details: Dict[str, Any]) -> None:
        """
        Record a balancing operation for monitoring.
        
        Args:
            operation_type: Type of operation (e.g., "smote_generation", "class_weighting")
            success: Whether the operation succeeded
            duration_ms: Operation duration in milliseconds
            details: Additional operation details
        """
        timestamp = datetime.now()
        
        operation_record = {
            'timestamp': timestamp.isoformat(),
            'operation_type': operation_type,
            'success': success,
            'duration_ms': duration_ms,
            'details': details
        }
        
        # Log operation
        if success:
            self.logger.info(f"Balancing operation completed: {operation_type} ({duration_ms:.1f}ms)")
        else:
            self.logger.error(f"Balancing operation failed: {operation_type} - {details.get('error', 'Unknown error')}")
            
            # Create alert for failed operations
            self._create_alert(
                AlertLevel.ERROR,
                f"balancing_{operation_type}",
                f"Balancing operation failed: {operation_type}",
                details
            )
    
    def add_health_check(self, name: str, check_function: Callable[[], HealthCheckResult]) -> None:
        """
        Add a health check function.
        
        Args:
            name: Name of the health check
            check_function: Function that returns HealthCheckResult
        """
        self.health_checks[name] = check_function
        self.logger.info(f"Added health check: {name}")
    
    def run_health_checks(self) -> Dict[str, HealthCheckResult]:
        """
        Run all registered health checks.
        
        Returns:
            Dictionary of health check results
        """
        results = {}
        
        for name, check_function in self.health_checks.items():
            try:
                start_time = time.time()
                result = check_function()
                result.response_time_ms = (time.time() - start_time) * 1000
                
                results[name] = result
                self.health_check_results.append(result)
                
                # Update component status
                if result.component in self.component_status:
                    self.component_status[result.component] = result.status
                
                # Create alerts for unhealthy components
                if result.status == "unhealthy":
                    self._create_alert(
                        AlertLevel.ERROR,
                        result.component,
                        f"Health check failed: {name}",
                        {'error': result.error_message, 'details': result.details}
                    )
                elif result.status == "degraded":
                    self._create_alert(
                        AlertLevel.WARNING,
                        result.component,
                        f"Health check degraded: {name}",
                        {'details': result.details}
                    )
                
            except Exception as e:
                error_result = HealthCheckResult(
                    component=name,
                    status="unhealthy",
                    timestamp=datetime.now(),
                    response_time_ms=0.0,
                    details={},
                    error_message=str(e)
                )
                results[name] = error_result
                self.health_check_results.append(error_result)
                
                self.logger.error(f"Health check {name} failed: {str(e)}")
        
        return results
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get comprehensive system status.
        
        Returns:
            Dictionary with system status information
        """
        current_time = datetime.now()
        
        # Recent metrics (last hour)
        recent_metrics = [
            m for m in self.performance_metrics 
            if (current_time - m.timestamp).total_seconds() < 3600
        ]
        
        # Active alerts
        active_alerts = [a for a in self.active_alerts if not a.resolved]
        
        # Component health
        overall_health = "healthy"
        if any(status == "unhealthy" for status in self.component_status.values()):
            overall_health = "unhealthy"
        elif any(status == "degraded" for status in self.component_status.values()):
            overall_health = "degraded"
        
        # Performance summary
        performance_summary = {}
        if recent_metrics:
            performance_summary = {
                'avg_processing_time_ms': sum(m.processing_time_ms for m in recent_metrics) / len(recent_metrics),
                'avg_memory_usage_mb': sum(m.memory_usage_mb for m in recent_metrics) / len(recent_metrics),
                'avg_accuracy': sum(m.accuracy for m in recent_metrics) / len(recent_metrics),
                'avg_fnr': sum(m.false_negative_rate for m in recent_metrics) / len(recent_metrics),
                'total_samples_processed': sum(m.samples_processed for m in recent_metrics),
                'operations_count': len(recent_metrics)
            }
        
        return {
            'timestamp': current_time.isoformat(),
            'overall_health': overall_health,
            'component_status': self.component_status.copy(),
            'active_alerts_count': len(active_alerts),
            'recent_performance': performance_summary,
            'monitoring_active': self.monitoring_active,
            'metrics_retention_hours': self.metrics_retention_hours,
            'thresholds': self.thresholds.copy()
        }
    
    def get_performance_trends(self, hours: int = 24) -> Dict[str, Any]:
        """
        Get performance trends over specified time period.
        
        Args:
            hours: Number of hours to analyze
            
        Returns:
            Dictionary with trend analysis
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        # Filter metrics by time period
        period_metrics = [
            m for m in self.performance_metrics 
            if m.timestamp >= cutoff_time
        ]
        
        if not period_metrics:
            return {'error': 'No metrics available for specified period'}
        
        # Calculate trends
        processing_times = [m.processing_time_ms for m in period_metrics]
        memory_usage = [m.memory_usage_mb for m in period_metrics]
        accuracies = [m.accuracy for m in period_metrics]
        fnrs = [m.false_negative_rate for m in period_metrics]
        
        return {
            'period_hours': hours,
            'total_operations': len(period_metrics),
            'processing_time': {
                'min': min(processing_times),
                'max': max(processing_times),
                'avg': sum(processing_times) / len(processing_times),
                'trend': self._calculate_trend(processing_times)
            },
            'memory_usage': {
                'min': min(memory_usage),
                'max': max(memory_usage),
                'avg': sum(memory_usage) / len(memory_usage),
                'trend': self._calculate_trend(memory_usage)
            },
            'accuracy': {
                'min': min(accuracies),
                'max': max(accuracies),
                'avg': sum(accuracies) / len(accuracies),
                'trend': self._calculate_trend(accuracies)
            },
            'false_negative_rate': {
                'min': min(fnrs),
                'max': max(fnrs),
                'avg': sum(fnrs) / len(fnrs),
                'trend': self._calculate_trend(fnrs)
            }
        }
    
    def export_metrics(self, file_path: str, hours: int = 24) -> None:
        """
        Export metrics to JSON file.
        
        Args:
            file_path: Path to save metrics
            hours: Number of hours of metrics to export
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        # Filter and convert metrics
        export_metrics = []
        for metric in self.performance_metrics:
            if metric.timestamp >= cutoff_time:
                metric_dict = asdict(metric)
                metric_dict['timestamp'] = metric.timestamp.isoformat()
                export_metrics.append(metric_dict)
        
        # Export data
        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'period_hours': hours,
            'metrics_count': len(export_metrics),
            'system_status': self.get_system_status(),
            'performance_trends': self.get_performance_trends(hours),
            'metrics': export_metrics
        }
        
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            self.logger.info(f"Exported {len(export_metrics)} metrics to {file_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to export metrics: {str(e)}")
    
    def add_alert_handler(self, handler: Callable[[BalancingAlert], None]) -> None:
        """
        Add an alert handler function.
        
        Args:
            handler: Function that processes alerts
        """
        self.alert_handlers.append(handler)
        self.logger.info("Added alert handler")
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop running in background thread."""
        self.logger.info("Started monitoring loop")
        
        while self.monitoring_active:
            try:
                # Run health checks
                self.run_health_checks()
                
                # Clean up old data
                self._cleanup_old_metrics()
                self._cleanup_resolved_alerts()
                
                # Sleep until next check
                time.sleep(self.health_check_interval)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {str(e)}")
                time.sleep(60)  # Wait a minute before retrying
    
    def _check_performance_thresholds(self, metrics: PerformanceMetrics) -> None:
        """Check if performance metrics exceed thresholds."""
        alerts_created = []
        
        # Processing time threshold
        if metrics.processing_time_ms > self.thresholds['max_processing_time_ms']:
            alert = self._create_alert(
                AlertLevel.WARNING,
                "performance",
                f"Processing time exceeded threshold: {metrics.processing_time_ms:.1f}ms > {self.thresholds['max_processing_time_ms']}ms",
                {'processing_time_ms': metrics.processing_time_ms}
            )
            alerts_created.append(alert)
        
        # Memory usage threshold
        if metrics.memory_usage_mb > self.thresholds['max_memory_usage_mb']:
            alert = self._create_alert(
                AlertLevel.WARNING,
                "performance",
                f"Memory usage exceeded threshold: {metrics.memory_usage_mb:.1f}MB > {self.thresholds['max_memory_usage_mb']}MB",
                {'memory_usage_mb': metrics.memory_usage_mb}
            )
            alerts_created.append(alert)
        
        # Accuracy threshold
        if metrics.accuracy < self.thresholds['min_accuracy']:
            alert = self._create_alert(
                AlertLevel.ERROR,
                "performance",
                f"Accuracy below threshold: {metrics.accuracy:.3f} < {self.thresholds['min_accuracy']:.3f}",
                {'accuracy': metrics.accuracy}
            )
            alerts_created.append(alert)
        
        # False negative rate threshold
        if metrics.false_negative_rate > self.thresholds['max_fnr']:
            alert = self._create_alert(
                AlertLevel.ERROR,
                "performance",
                f"False negative rate exceeded threshold: {metrics.false_negative_rate:.3f} > {self.thresholds['max_fnr']:.3f}",
                {'false_negative_rate': metrics.false_negative_rate}
            )
            alerts_created.append(alert)
        
        # Synthetic sample ratio threshold
        if metrics.samples_processed > 0:
            synthetic_ratio = metrics.synthetic_samples_created / metrics.samples_processed
            if synthetic_ratio > self.thresholds['max_synthetic_ratio']:
                alert = self._create_alert(
                    AlertLevel.WARNING,
                    "balancing",
                    f"Synthetic sample ratio exceeded threshold: {synthetic_ratio:.2f} > {self.thresholds['max_synthetic_ratio']:.2f}",
                    {'synthetic_ratio': synthetic_ratio}
                )
                alerts_created.append(alert)
    
    def _create_alert(self, level: AlertLevel, component: str, message: str, metrics: Dict[str, Any]) -> BalancingAlert:
        """Create and process a new alert."""
        alert = BalancingAlert(
            timestamp=datetime.now(),
            level=level,
            component=component,
            message=message,
            metrics=metrics
        )
        
        self.active_alerts.append(alert)
        self.alert_history.append(alert)
        
        # Process alert through handlers
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                self.logger.error(f"Error in alert handler: {str(e)}")
        
        # Log alert
        log_level = {
            AlertLevel.INFO: logging.INFO,
            AlertLevel.WARNING: logging.WARNING,
            AlertLevel.ERROR: logging.ERROR,
            AlertLevel.CRITICAL: logging.CRITICAL
        }.get(level, logging.INFO)
        
        self.logger.log(log_level, f"ALERT [{level.value.upper()}] {component}: {message}")
        
        return alert
    
    def _cleanup_old_metrics(self) -> None:
        """Remove old metrics beyond retention period."""
        cutoff_time = datetime.now() - timedelta(hours=self.metrics_retention_hours)
        
        # Clean performance metrics
        while (self.performance_metrics and 
               self.performance_metrics[0].timestamp < cutoff_time):
            self.performance_metrics.popleft()
        
        # Clean health check results
        while (self.health_check_results and 
               self.health_check_results[0].timestamp < cutoff_time):
            self.health_check_results.popleft()
    
    def _cleanup_resolved_alerts(self) -> None:
        """Remove resolved alerts older than 1 hour."""
        cutoff_time = datetime.now() - timedelta(hours=1)
        
        self.active_alerts = [
            alert for alert in self.active_alerts
            if not (alert.resolved and alert.resolution_time and alert.resolution_time < cutoff_time)
        ]
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction for a list of values."""
        if len(values) < 2:
            return "stable"
        
        # Simple linear trend calculation
        n = len(values)
        x_sum = sum(range(n))
        y_sum = sum(values)
        xy_sum = sum(i * values[i] for i in range(n))
        x2_sum = sum(i * i for i in range(n))
        
        slope = (n * xy_sum - x_sum * y_sum) / (n * x2_sum - x_sum * x_sum)
        
        if abs(slope) < 0.01:  # Threshold for considering stable
            return "stable"
        elif slope > 0:
            return "increasing"
        else:
            return "decreasing"


# Default health check functions
def create_class_balancer_health_check(class_balancer) -> Callable[[], HealthCheckResult]:
    """Create health check for ClassBalancer component."""
    def health_check() -> HealthCheckResult:
        try:
            # Test basic functionality
            test_data = np.array([[1, 2], [3, 4], [5, 6]])
            test_labels = np.array([0, 1, 0])
            
            start_time = time.time()
            imbalance_detected = class_balancer.detect_imbalance(test_data, test_labels)
            response_time = (time.time() - start_time) * 1000
            
            return HealthCheckResult(
                component="class_balancer",
                status="healthy",
                timestamp=datetime.now(),
                response_time_ms=response_time,
                details={
                    'imbalance_detection_working': True,
                    'config_enabled': class_balancer.config.enabled,
                    'method': class_balancer.config.method
                }
            )
            
        except Exception as e:
            return HealthCheckResult(
                component="class_balancer",
                status="unhealthy",
                timestamp=datetime.now(),
                response_time_ms=0.0,
                details={},
                error_message=str(e)
            )
    
    return health_check


def create_smote_processor_health_check(smote_processor) -> Callable[[], HealthCheckResult]:
    """Create health check for SMOTEProcessor component."""
    def health_check() -> HealthCheckResult:
        try:
            # Test SMOTE processor info retrieval
            start_time = time.time()
            smote_info = smote_processor.get_smote_info()
            response_time = (time.time() - start_time) * 1000
            
            return HealthCheckResult(
                component="smote_processor",
                status="healthy",
                timestamp=datetime.now(),
                response_time_ms=response_time,
                details={
                    'k_neighbors': smote_info.get('k_neighbors', 0),
                    'target_ratio': smote_info.get('target_ratio', 0.0),
                    'is_fitted': smote_info.get('is_fitted', False)
                }
            )
            
        except Exception as e:
            return HealthCheckResult(
                component="smote_processor",
                status="unhealthy",
                timestamp=datetime.now(),
                response_time_ms=0.0,
                details={},
                error_message=str(e)
            )
    
    return health_check


# Alert handler examples
def log_alert_handler(alert: BalancingAlert) -> None:
    """Simple alert handler that logs alerts."""
    logger = logging.getLogger("balancing_alerts")
    logger.info(f"Alert: [{alert.level.value.upper()}] {alert.component} - {alert.message}")


def file_alert_handler(log_file: str) -> Callable[[BalancingAlert], None]:
    """Create alert handler that writes to file."""
    def handler(alert: BalancingAlert) -> None:
        try:
            alert_data = {
                'timestamp': alert.timestamp.isoformat(),
                'level': alert.level.value,
                'component': alert.component,
                'message': alert.message,
                'metrics': alert.metrics
            }
            
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            with open(log_file, 'a') as f:
                f.write(json.dumps(alert_data) + '\n')
                
        except Exception as e:
            logging.getLogger("balancing_monitor").error(f"Failed to write alert to file: {str(e)}")
    
    return handler