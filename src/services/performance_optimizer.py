"""
Performance optimization utilities for class balancing operations.

This module provides utilities to optimize memory usage, processing speed,
and resource management during class balancing and model training operations.
"""
import logging
import psutil
import gc
import time
from typing import Dict, Any, Optional, Tuple
import numpy as np
from dataclasses import dataclass


@dataclass
class PerformanceMetrics:
    """Performance metrics for balancing operations."""
    processing_time: float
    memory_usage_mb: float
    peak_memory_mb: float
    samples_processed: int
    samples_per_second: float
    optimization_level: str
    memory_efficiency: str


class PerformanceOptimizer:
    """
    Utility class for optimizing performance of class balancing operations.
    
    Provides memory monitoring, batch size optimization, and resource management
    for large-scale spam classification training.
    """
    
    def __init__(self):
        """Initialize the performance optimizer."""
        self.logger = logging.getLogger(__name__)
        self.start_time: Optional[float] = None
        self.start_memory: Optional[float] = None
        self.peak_memory: float = 0.0
        
        # Performance thresholds
        self.large_dataset_threshold = 50000
        self.memory_limit_mb = 4096  # 4GB default limit
        self.batch_size_min = 5000
        self.batch_size_max = 25000
        
        self.logger.info("Performance optimizer initialized")
    
    def start_monitoring(self) -> None:
        """Start performance monitoring."""
        self.start_time = time.time()
        self.start_memory = self._get_memory_usage_mb()
        self.peak_memory = self.start_memory
        
        self.logger.info(f"Performance monitoring started - Initial memory: {self.start_memory:.1f} MB")
    
    def update_peak_memory(self) -> None:
        """Update peak memory usage."""
        current_memory = self._get_memory_usage_mb()
        if current_memory > self.peak_memory:
            self.peak_memory = current_memory
    
    def get_performance_metrics(self, samples_processed: int) -> PerformanceMetrics:
        """
        Get comprehensive performance metrics.
        
        Args:
            samples_processed: Number of samples processed
            
        Returns:
            PerformanceMetrics object with detailed metrics
        """
        if self.start_time is None or self.start_memory is None:
            raise ValueError("Performance monitoring not started")
        
        processing_time = time.time() - self.start_time
        current_memory = self._get_memory_usage_mb()
        memory_usage = current_memory - self.start_memory
        
        samples_per_second = samples_processed / max(processing_time, 0.001)
        
        # Determine optimization level
        if samples_per_second > 10000:
            optimization_level = "high"
        elif samples_per_second > 1000:
            optimization_level = "medium"
        else:
            optimization_level = "low"
        
        # Determine memory efficiency
        if memory_usage < 500:  # Less than 500MB
            memory_efficiency = "excellent"
        elif memory_usage < 1000:  # Less than 1GB
            memory_efficiency = "good"
        elif memory_usage < 2000:  # Less than 2GB
            memory_efficiency = "moderate"
        else:
            memory_efficiency = "poor"
        
        return PerformanceMetrics(
            processing_time=processing_time,
            memory_usage_mb=memory_usage,
            peak_memory_mb=self.peak_memory,
            samples_processed=samples_processed,
            samples_per_second=samples_per_second,
            optimization_level=optimization_level,
            memory_efficiency=memory_efficiency
        )
    
    def calculate_optimal_batch_size(self, dataset_size: int, feature_count: int) -> int:
        """
        Calculate optimal batch size for SMOTE processing based on available memory.
        
        Args:
            dataset_size: Number of samples in dataset
            feature_count: Number of features per sample
            
        Returns:
            Optimal batch size for processing
        """
        # Get available memory
        available_memory_mb = self._get_available_memory_mb()
        
        # Estimate memory per sample (features * 8 bytes for float64 + overhead)
        memory_per_sample_mb = (feature_count * 8 * 3) / (1024 * 1024)  # 3x for SMOTE overhead
        
        # Calculate batch size based on memory constraints
        memory_based_batch_size = int(available_memory_mb * 0.3 / memory_per_sample_mb)  # Use 30% of available memory
        
        # Apply min/max constraints
        optimal_batch_size = max(self.batch_size_min, 
                               min(self.batch_size_max, memory_based_batch_size))
        
        # Don't exceed dataset size
        optimal_batch_size = min(optimal_batch_size, dataset_size)
        
        self.logger.info(f"Calculated optimal batch size: {optimal_batch_size} "
                        f"(dataset: {dataset_size}, features: {feature_count}, "
                        f"available memory: {available_memory_mb:.1f} MB)")
        
        return optimal_batch_size
    
    def should_use_batch_processing(self, dataset_size: int) -> bool:
        """
        Determine if batch processing should be used based on dataset size and available memory.
        
        Args:
            dataset_size: Number of samples in dataset
            
        Returns:
            True if batch processing is recommended
        """
        # Check dataset size threshold
        if dataset_size < self.large_dataset_threshold:
            return False
        
        # Check available memory
        available_memory_mb = self._get_available_memory_mb()
        
        # If low memory, use batch processing even for smaller datasets
        if available_memory_mb < 2000:  # Less than 2GB available
            return dataset_size > 10000
        
        return True
    
    def optimize_memory_usage(self) -> None:
        """
        Optimize memory usage by forcing garbage collection and clearing caches.
        """
        # Force garbage collection
        collected = gc.collect()
        
        # Clear any numpy caches if available
        try:
            import numpy as np
            if hasattr(np, 'clear_cache'):
                np.clear_cache()
        except:
            pass
        
        current_memory = self._get_memory_usage_mb()
        self.logger.info(f"Memory optimization completed - Current usage: {current_memory:.1f} MB, "
                        f"Objects collected: {collected}")
    
    def get_system_info(self) -> Dict[str, Any]:
        """
        Get system information relevant to performance optimization.
        
        Returns:
            Dictionary with system information
        """
        try:
            memory_info = psutil.virtual_memory()
            cpu_info = psutil.cpu_count()
            
            return {
                'total_memory_gb': memory_info.total / (1024**3),
                'available_memory_gb': memory_info.available / (1024**3),
                'memory_usage_percent': memory_info.percent,
                'cpu_cores': cpu_info,
                'cpu_usage_percent': psutil.cpu_percent(interval=1),
                'recommended_batch_processing': memory_info.available < 4 * (1024**3)  # Less than 4GB
            }
        except Exception as e:
            self.logger.error(f"Error getting system info: {str(e)}")
            return {
                'total_memory_gb': 'unknown',
                'available_memory_gb': 'unknown',
                'memory_usage_percent': 'unknown',
                'cpu_cores': 'unknown',
                'cpu_usage_percent': 'unknown',
                'recommended_batch_processing': True
            }
    
    def log_performance_summary(self, metrics: PerformanceMetrics) -> None:
        """
        Log a comprehensive performance summary.
        
        Args:
            metrics: Performance metrics to log
        """
        self.logger.info("=" * 60)
        self.logger.info("PERFORMANCE SUMMARY")
        self.logger.info("=" * 60)
        self.logger.info(f"Processing Time: {metrics.processing_time:.2f} seconds")
        self.logger.info(f"Samples Processed: {metrics.samples_processed:,}")
        self.logger.info(f"Processing Speed: {metrics.samples_per_second:.0f} samples/second")
        self.logger.info(f"Memory Usage: {metrics.memory_usage_mb:.1f} MB")
        self.logger.info(f"Peak Memory: {metrics.peak_memory_mb:.1f} MB")
        self.logger.info(f"Optimization Level: {metrics.optimization_level}")
        self.logger.info(f"Memory Efficiency: {metrics.memory_efficiency}")
        self.logger.info("=" * 60)
    
    def _get_memory_usage_mb(self) -> float:
        """Get current memory usage in MB."""
        try:
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)
        except Exception:
            return 0.0
    
    def _get_available_memory_mb(self) -> float:
        """Get available system memory in MB."""
        try:
            return psutil.virtual_memory().available / (1024 * 1024)
        except Exception:
            return 4096.0  # Default to 4GB if unable to detect


class BatchProcessor:
    """
    Utility for processing large datasets in batches to optimize memory usage.
    """
    
    def __init__(self, batch_size: int = 10000):
        """
        Initialize the batch processor.
        
        Args:
            batch_size: Size of each batch for processing
        """
        self.batch_size = batch_size
        self.logger = logging.getLogger(__name__)
    
    def process_in_batches(self, X: np.ndarray, y: np.ndarray, 
                          process_func, *args, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process data in batches using the provided function.
        
        Args:
            X: Feature matrix
            y: Target labels
            process_func: Function to apply to each batch
            *args: Additional arguments for process_func
            **kwargs: Additional keyword arguments for process_func
            
        Returns:
            Tuple of processed (X, y)
        """
        n_samples = len(X)
        n_batches = (n_samples + self.batch_size - 1) // self.batch_size
        
        self.logger.info(f"Processing {n_samples} samples in {n_batches} batches of size {self.batch_size}")
        
        processed_X_batches = []
        processed_y_batches = []
        
        for i in range(n_batches):
            start_idx = i * self.batch_size
            end_idx = min((i + 1) * self.batch_size, n_samples)
            
            self.logger.info(f"Processing batch {i+1}/{n_batches} (samples {start_idx}-{end_idx})")
            
            # Extract batch
            X_batch = X[start_idx:end_idx]
            y_batch = y[start_idx:end_idx]
            
            # Process batch
            X_processed, y_processed = process_func(X_batch, y_batch, *args, **kwargs)
            
            processed_X_batches.append(X_processed)
            processed_y_batches.append(y_processed)
            
            # Clean up batch data
            del X_batch, y_batch
            gc.collect()
        
        # Combine all batches
        X_final = np.vstack(processed_X_batches)
        y_final = np.hstack(processed_y_batches)
        
        # Clean up batch lists
        del processed_X_batches, processed_y_batches
        gc.collect()
        
        self.logger.info(f"Batch processing completed - Final size: {len(X_final)} samples")
        
        return X_final, y_final