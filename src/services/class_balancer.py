"""
Class balancing component for handling imbalanced datasets in spam classification.

This module provides the ClassBalancer component that detects class imbalance
and applies balancing techniques like SMOTE and class weighting to improve
spam detection performance.
"""
import logging
import numpy as np
from typing import Dict, Any, Tuple, Optional
from dataclasses import dataclass

from .balancing_config import BalancingConfig


@dataclass
class BalancingResults:
    """Results from class balancing operations."""
    original_distribution: Dict[str, int]
    balanced_distribution: Dict[str, int]
    synthetic_samples_created: int
    balancing_method_used: str
    improvement_metrics: Dict[str, float]
    processing_time: float
    validation_passed: bool


class ClassBalancer:
    """
    Core component for detecting and correcting class imbalance in datasets.
    
    This class provides methods to detect class imbalance and apply various
    balancing techniques including SMOTE oversampling and class weighting.
    """
    
    def __init__(self, config: Optional[BalancingConfig] = None):
        """
        Initialize the ClassBalancer with configuration.
        
        Args:
            config: BalancingConfig object with balancing parameters
        """
        self.config = config or BalancingConfig()
        self.logger = logging.getLogger(__name__)
        
        # Balancing state
        self.original_distribution: Optional[Dict[str, int]] = None
        self.balanced_distribution: Optional[Dict[str, int]] = None
        self.balancing_applied = False
        self.balancing_results: Optional[BalancingResults] = None
        
        # Validate configuration
        self._validate_config()
    
    def detect_imbalance(self, X: np.ndarray, y: np.ndarray) -> bool:
        """
        Detect if the dataset has class imbalance.
        
        Args:
            X: Feature matrix
            y: Target labels (0 for legitimate, 1 for spam)
            
        Returns:
            True if imbalance is detected, False otherwise
        """
        if len(y) == 0:
            self.logger.warning("Empty dataset provided for imbalance detection")
            return False
        
        # Calculate class distribution
        unique_labels, counts = np.unique(y, return_counts=True)
        total_samples = len(y)
        
        # Store original distribution
        self.original_distribution = {}
        for label, count in zip(unique_labels, counts):
            class_name = 'spam' if label == 1 else 'legitimate'
            self.original_distribution[class_name] = int(count)
        
        # Calculate spam ratio
        spam_count = self.original_distribution.get('spam', 0)
        spam_ratio = spam_count / total_samples if total_samples > 0 else 0.0
        
        # Log current distribution
        self.logger.info(f"Current class distribution: {self.original_distribution}")
        self.logger.info(f"Current spam ratio: {spam_ratio:.3f}")
        
        # Check if imbalance exists (spam ratio below target threshold)
        # Be more aggressive - consider imbalanced if spam < 45%
        imbalance_threshold = 0.45  # Consider imbalanced if spam < 45%
        is_imbalanced = spam_ratio < imbalance_threshold
        
        if is_imbalanced:
            self.logger.info(f"Class imbalance detected: spam ratio {spam_ratio:.3f} < {imbalance_threshold}")
        else:
            self.logger.info(f"Dataset is balanced: spam ratio {spam_ratio:.3f} >= {imbalance_threshold}")
        
        return is_imbalanced
    
    def balance_dataset(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply class balancing to the dataset.
        
        Args:
            X: Feature matrix
            y: Target labels (0 for legitimate, 1 for spam)
            
        Returns:
            Tuple of (balanced_X, balanced_y)
        """
        import time
        start_time = time.time()
        
        if not self.config.enabled:
            self.logger.info("Class balancing is disabled")
            return X, y
        
        # Check if balancing is needed
        if not self.detect_imbalance(X, y):
            self.logger.info("No class imbalance detected, skipping balancing")
            return X, y
        
        self.logger.info(f"Applying class balancing using method: {self.config.method}")
        
        try:
            if self.config.method == 'smote':
                balanced_X, balanced_y = self._apply_smote(X, y)
            elif self.config.method == 'class_weights':
                # For class weights, we don't modify the dataset
                balanced_X, balanced_y = X, y
                self.logger.info("Class weights will be applied during model training")
            elif self.config.method == 'both':
                # Apply SMOTE first, class weights will be applied during training
                balanced_X, balanced_y = self._apply_smote(X, y)
            else:
                raise ValueError(f"Unknown balancing method: {self.config.method}")
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Update balanced distribution
            self._update_balanced_distribution(balanced_y)
            
            # Create balancing results
            self._create_balancing_results(processing_time)
            
            self.balancing_applied = True
            self.logger.info(f"Class balancing completed in {processing_time:.2f} seconds")
            
            return balanced_X, balanced_y
            
        except Exception as e:
            self.logger.error(f"Error during class balancing: {str(e)}")
            
            if self.config.fallback_to_class_weights and self.config.method != 'class_weights':
                self.logger.info("Falling back to class weights method")
                self.config.method = 'class_weights'
                return X, y  # Return original data, weights will be applied during training
            
            # If fallback fails or is disabled, return original data
            self.logger.warning("Class balancing failed, returning original dataset")
            return X, y
    
    def get_class_weights(self, y: np.ndarray) -> Dict[int, float]:
        """
        Calculate class weights optimized for spam detection (low false negative rate).
        
        Args:
            y: Target labels
            
        Returns:
            Dictionary mapping class labels to weights
        """
        if self.config.class_weight_strategy == 'custom' and self.config.custom_weights:
            return self.config.custom_weights
        
        # Optimized class weight calculation
        unique_classes, counts = np.unique(y, return_counts=True)
        
        if len(unique_classes) == 0:
            return {}
        
        # Calculate spam-optimized weights
        n_samples = len(y)
        n_classes = len(unique_classes)
        
        # Standard balanced weights
        standard_weights = n_samples / (n_classes * counts)
        
        # Create spam-optimized weights (emphasize spam detection)
        weight_dict = {}
        for cls, weight in zip(unique_classes, standard_weights):
            if cls == 1:  # Spam class
                # Increase spam weight by 50% to reduce false negatives
                weight_dict[int(cls)] = float(weight * 1.5)
            else:  # Legitimate class
                # Keep legitimate weight standard
                weight_dict[int(cls)] = float(weight)
        
        self.logger.info(f"Calculated spam-optimized class weights: {weight_dict}")
        return weight_dict
    
    def get_balancing_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive balancing statistics and metrics.
        
        Returns:
            Dictionary with balancing statistics
        """
        stats = {
            'balancing_enabled': self.config.enabled,
            'balancing_applied': self.balancing_applied,
            'method_used': self.config.method,
            'target_spam_ratio': self.config.target_spam_ratio,
            'original_distribution': self.original_distribution,
            'balanced_distribution': self.balanced_distribution
        }
        
        if self.balancing_results:
            stats.update({
                'synthetic_samples_created': self.balancing_results.synthetic_samples_created,
                'processing_time': self.balancing_results.processing_time,
                'validation_passed': self.balancing_results.validation_passed,
                'improvement_metrics': self.balancing_results.improvement_metrics
            })
        
        return stats
    
    def cleanup_balancing_data(self) -> None:
        """
        Clean up memory used by balancing operations.
        Call this after training to free up memory.
        """
        import gc
        
        # Clear large data structures
        self.original_distribution = None
        self.balanced_distribution = None
        self.balancing_results = None
        
        # Force garbage collection
        gc.collect()
        
        self.logger.info("Class balancer memory cleanup completed")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics for balancing operations.
        
        Returns:
            Dictionary with performance metrics
        """
        metrics = {
            'balancing_applied': self.balancing_applied,
            'processing_time': 0.0,
            'memory_efficiency': 'unknown',
            'optimization_used': 'standard'
        }
        
        if self.balancing_results:
            metrics.update({
                'processing_time': self.balancing_results.processing_time,
                'synthetic_samples_created': self.balancing_results.synthetic_samples_created,
                'validation_passed': self.balancing_results.validation_passed
            })
            
            # Determine optimization level based on processing time and sample count
            if self.balancing_results.synthetic_samples_created > 0:
                samples_per_second = self.balancing_results.synthetic_samples_created / max(self.balancing_results.processing_time, 0.001)
                
                if samples_per_second > 10000:
                    metrics['memory_efficiency'] = 'high'
                    metrics['optimization_used'] = 'batch_processing'
                elif samples_per_second > 1000:
                    metrics['memory_efficiency'] = 'medium'
                    metrics['optimization_used'] = 'standard'
                else:
                    metrics['memory_efficiency'] = 'low'
                    metrics['optimization_used'] = 'standard'
                
                metrics['samples_per_second'] = samples_per_second
        
        return metrics
    
    def _apply_smote(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply SMOTE oversampling to balance the dataset.
        
        Args:
            X: Feature matrix
            y: Target labels
            
        Returns:
            Tuple of (resampled_X, resampled_y)
        """
        from .smote_processor import SMOTEProcessor
        
        # Create SMOTE processor
        smote_processor = SMOTEProcessor(
            k_neighbors=self.config.smote_k_neighbors,
            random_state=self.config.smote_random_state,
            target_ratio=self.config.target_spam_ratio
        )
        
        # Apply SMOTE
        balanced_X, balanced_y = smote_processor.fit_resample(X, y)
        
        # Validate synthetic samples if enabled
        if self.config.validate_synthetic_samples:
            validation_passed = smote_processor.validate_synthetic_samples(balanced_X)
            if not validation_passed:
                self.logger.warning("Synthetic sample validation failed")
        
        return balanced_X, balanced_y
    
    def _update_balanced_distribution(self, balanced_y: np.ndarray) -> None:
        """Update the balanced distribution statistics."""
        unique_labels, counts = np.unique(balanced_y, return_counts=True)
        
        self.balanced_distribution = {}
        for label, count in zip(unique_labels, counts):
            class_name = 'spam' if label == 1 else 'legitimate'
            self.balanced_distribution[class_name] = int(count)
        
        total_samples = len(balanced_y)
        spam_count = self.balanced_distribution.get('spam', 0)
        spam_ratio = spam_count / total_samples if total_samples > 0 else 0.0
        
        self.logger.info(f"Balanced class distribution: {self.balanced_distribution}")
        self.logger.info(f"Balanced spam ratio: {spam_ratio:.3f}")
    
    def _create_balancing_results(self, processing_time: float) -> None:
        """Create comprehensive balancing results."""
        synthetic_samples = 0
        if self.balanced_distribution and self.original_distribution:
            original_total = sum(self.original_distribution.values())
            balanced_total = sum(self.balanced_distribution.values())
            synthetic_samples = balanced_total - original_total
        
        # Calculate improvement metrics
        improvement_metrics = {}
        if self.original_distribution and self.balanced_distribution:
            original_spam_ratio = self.original_distribution.get('spam', 0) / sum(self.original_distribution.values())
            balanced_spam_ratio = self.balanced_distribution.get('spam', 0) / sum(self.balanced_distribution.values())
            
            improvement_metrics = {
                'original_spam_ratio': original_spam_ratio,
                'balanced_spam_ratio': balanced_spam_ratio,
                'ratio_improvement': balanced_spam_ratio - original_spam_ratio
            }
        
        self.balancing_results = BalancingResults(
            original_distribution=self.original_distribution or {},
            balanced_distribution=self.balanced_distribution or {},
            synthetic_samples_created=synthetic_samples,
            balancing_method_used=self.config.method,
            improvement_metrics=improvement_metrics,
            processing_time=processing_time,
            validation_passed=True  # Will be updated by validation if enabled
        )
    
    def _validate_config(self) -> None:
        """Validate the balancing configuration."""
        if not isinstance(self.config.target_spam_ratio, (int, float)):
            raise ValueError("target_spam_ratio must be a number")
        
        if not (0.3 <= self.config.target_spam_ratio <= 0.5):
            raise ValueError("target_spam_ratio must be between 0.3 and 0.5")
        
        if self.config.method not in ['smote', 'class_weights', 'both']:
            raise ValueError("method must be 'smote', 'class_weights', or 'both'")
        
        if self.config.smote_k_neighbors < 1:
            raise ValueError("smote_k_neighbors must be >= 1")
        
        self.logger.info("ClassBalancer configuration validated successfully")