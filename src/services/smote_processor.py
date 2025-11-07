"""
SMOTE (Synthetic Minority Oversampling Technique) processor for class balancing.

This module provides the SMOTEProcessor component that generates synthetic
samples for the minority class (spam) to balance the dataset and improve
spam detection performance.
"""
import logging
import numpy as np
from typing import Tuple, Optional, Dict, Any
from sklearn.neighbors import NearestNeighbors
from imblearn.over_sampling import SMOTE
from scipy import sparse


class SMOTEProcessor:
    """
    SMOTE processor for generating synthetic minority class samples.
    
    This class uses scikit-learn's SMOTE implementation to generate realistic
    synthetic spam samples to balance the training dataset.
    """
    
    def __init__(self, 
                 k_neighbors: int = 5,
                 random_state: int = 42,
                 target_ratio: float = 0.42):
        """
        Initialize the SMOTE processor.
        
        Args:
            k_neighbors: Number of nearest neighbors for SMOTE algorithm
            random_state: Random seed for reproducibility
            target_ratio: Target ratio for minority class (spam)
        """
        self.k_neighbors = k_neighbors
        self.random_state = random_state
        self.target_ratio = target_ratio
        self.logger = logging.getLogger(__name__)
        
        # SMOTE instance
        self.smote = None
        
        # Processing state
        self.is_fitted = False
        self.original_samples = 0
        self.synthetic_samples_created = 0
        
        # Validation parameters
        self.min_samples_for_smote = 10
        self.max_feature_variance_threshold = 100.0
        
        self._validate_parameters()
    
    def fit_resample(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply SMOTE to generate synthetic samples and balance the dataset.
        Optimized for large datasets with batch processing and memory management.
        
        Args:
            X: Feature matrix (can be sparse or dense)
            y: Target labels (0 for legitimate, 1 for spam)
            
        Returns:
            Tuple of (resampled_X, resampled_y)
        """
        if len(X) == 0 or len(y) == 0:
            raise ValueError("Input arrays cannot be empty")
        
        if len(X) != len(y):
            raise ValueError("X and y must have the same number of samples")
        
        self.logger.info(f"Starting optimized SMOTE processing with {len(X)} samples")
        
        # Check if we have enough samples for SMOTE
        unique_labels, counts = np.unique(y, return_counts=True)
        min_class_count = np.min(counts)
        
        if min_class_count < self.min_samples_for_smote:
            self.logger.warning(f"Insufficient samples for SMOTE: {min_class_count} < {self.min_samples_for_smote}")
            return X, y
        
        # Adjust k_neighbors if necessary
        effective_k_neighbors = min(self.k_neighbors, min_class_count - 1)
        if effective_k_neighbors != self.k_neighbors:
            self.logger.info(f"Adjusting k_neighbors from {self.k_neighbors} to {effective_k_neighbors}")
        
        try:
            # Calculate sampling strategy to achieve target ratio
            sampling_strategy = self._calculate_sampling_strategy(y)
            
            if sampling_strategy is None:
                self.logger.info("No resampling needed, target ratio already achieved")
                return X, y
            
            # Check dataset size for batch processing optimization
            large_dataset_threshold = 50000  # Process in batches if > 50k samples
            use_batch_processing = len(X) > large_dataset_threshold
            
            if use_batch_processing:
                self.logger.info(f"Large dataset detected ({len(X)} samples), using batch processing")
                return self._fit_resample_batched(X, y, sampling_strategy, effective_k_neighbors)
            else:
                return self._fit_resample_standard(X, y, sampling_strategy, effective_k_neighbors)
            
        except Exception as e:
            self.logger.error(f"SMOTE processing failed: {str(e)}")
            raise RuntimeError(f"SMOTE processing failed: {str(e)}")
    
    def _fit_resample_standard(self, X: np.ndarray, y: np.ndarray, 
                              sampling_strategy: Dict[int, int], 
                              effective_k_neighbors: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Standard SMOTE processing for smaller datasets.
        
        Args:
            X: Feature matrix
            y: Target labels
            sampling_strategy: SMOTE sampling strategy
            effective_k_neighbors: Adjusted k_neighbors value
            
        Returns:
            Tuple of (resampled_X, resampled_y)
        """
        # Create SMOTE instance with error handling
        self.smote = SMOTE(
            sampling_strategy=sampling_strategy,
            k_neighbors=effective_k_neighbors,
            random_state=self.random_state
        )
        
        # Handle sparse matrices
        if sparse.issparse(X):
            self.logger.info("Converting sparse matrix to dense for SMOTE processing")
            X_dense = X.toarray()
        else:
            X_dense = X
        
        # Apply SMOTE
        self.logger.info(f"Applying SMOTE with sampling strategy: {sampling_strategy}")
        X_resampled, y_resampled = self.smote.fit_resample(X_dense, y)
        
        # Calculate synthetic samples created
        self.original_samples = len(X)
        self.synthetic_samples_created = len(X_resampled) - len(X)
        self.is_fitted = True
        
        self.logger.info(f"SMOTE completed: {self.synthetic_samples_created} synthetic samples created")
        self.logger.info(f"Dataset size: {len(X)} -> {len(X_resampled)}")
        
        # Log final distribution
        unique_resampled, counts_resampled = np.unique(y_resampled, return_counts=True)
        spam_ratio = counts_resampled[1] / len(y_resampled) if len(unique_resampled) > 1 else 0.0
        self.logger.info(f"Final spam ratio: {spam_ratio:.3f}")
        
        return X_resampled, y_resampled
    
    def _fit_resample_batched(self, X: np.ndarray, y: np.ndarray, 
                             sampling_strategy: Dict[int, int], 
                             effective_k_neighbors: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Optimized batch processing for large datasets.
        
        Args:
            X: Feature matrix
            y: Target labels
            sampling_strategy: SMOTE sampling strategy
            effective_k_neighbors: Adjusted k_neighbors value
            
        Returns:
            Tuple of (resampled_X, resampled_y)
        """
        import gc
        from sklearn.model_selection import StratifiedShuffleSplit
        
        self.logger.info("Starting batch SMOTE processing for large dataset")
        
        # Handle sparse matrices
        if sparse.issparse(X):
            self.logger.info("Converting sparse matrix to dense for batch SMOTE processing")
            X_dense = X.toarray()
        else:
            X_dense = X
        
        # Calculate batch size based on available memory and dataset size
        batch_size = min(20000, len(X) // 4)  # Process in chunks of 20k or 1/4 of dataset
        self.logger.info(f"Using batch size: {batch_size}")
        
        # Separate minority and majority classes
        minority_class = min(sampling_strategy.keys())
        majority_indices = np.where(y != minority_class)[0]
        minority_indices = np.where(y == minority_class)[0]
        
        # Keep all minority samples
        X_minority = X_dense[minority_indices]
        y_minority = y[minority_indices]
        
        # Sample majority class to create manageable batches
        majority_sample_size = min(len(majority_indices), batch_size * 2)
        if len(majority_indices) > majority_sample_size:
            # Randomly sample majority class for efficiency
            np.random.seed(self.random_state)
            sampled_majority_indices = np.random.choice(
                majority_indices, size=majority_sample_size, replace=False
            )
        else:
            sampled_majority_indices = majority_indices
        
        X_majority_sampled = X_dense[sampled_majority_indices]
        y_majority_sampled = y[sampled_majority_indices]
        
        # Combine for SMOTE processing
        X_combined = np.vstack([X_minority, X_majority_sampled])
        y_combined = np.hstack([y_minority, y_majority_sampled])
        
        # Apply SMOTE to the combined smaller dataset
        self.smote = SMOTE(
            sampling_strategy=sampling_strategy,
            k_neighbors=effective_k_neighbors,
            random_state=self.random_state
        )
        
        self.logger.info(f"Applying batch SMOTE to {len(X_combined)} samples")
        X_resampled_batch, y_resampled_batch = self.smote.fit_resample(X_combined, y_combined)
        
        # Combine with remaining majority samples
        remaining_majority_indices = np.setdiff1d(majority_indices, sampled_majority_indices)
        if len(remaining_majority_indices) > 0:
            X_remaining = X_dense[remaining_majority_indices]
            y_remaining = y[remaining_majority_indices]
            
            X_final = np.vstack([X_resampled_batch, X_remaining])
            y_final = np.hstack([y_resampled_batch, y_remaining])
        else:
            X_final = X_resampled_batch
            y_final = y_resampled_batch
        
        # Clean up memory
        del X_dense, X_combined, X_resampled_batch
        gc.collect()
        
        # Calculate synthetic samples created
        self.original_samples = len(X)
        self.synthetic_samples_created = len(X_final) - len(X)
        self.is_fitted = True
        
        self.logger.info(f"Batch SMOTE completed: {self.synthetic_samples_created} synthetic samples created")
        self.logger.info(f"Dataset size: {len(X)} -> {len(X_final)}")
        
        # Log final distribution
        unique_resampled, counts_resampled = np.unique(y_final, return_counts=True)
        spam_ratio = counts_resampled[1] / len(y_final) if len(unique_resampled) > 1 else 0.0
        self.logger.info(f"Final spam ratio: {spam_ratio:.3f}")
        
        return X_final, y_final
    
    def validate_synthetic_samples(self, X_synthetic: np.ndarray) -> bool:
        """
        Validate the quality of synthetic samples generated by SMOTE.
        
        Args:
            X_synthetic: The complete dataset including synthetic samples
            
        Returns:
            True if validation passes, False otherwise
        """
        if not self.is_fitted:
            self.logger.warning("SMOTE processor not fitted, skipping validation")
            return True
        
        try:
            self.logger.info("Validating synthetic samples quality...")
            
            # Check for NaN or infinite values
            if np.any(np.isnan(X_synthetic)) or np.any(np.isinf(X_synthetic)):
                self.logger.error("Synthetic samples contain NaN or infinite values")
                return False
            
            # Check feature variance (synthetic samples shouldn't be too uniform)
            if X_synthetic.shape[1] > 0:
                feature_variances = np.var(X_synthetic, axis=0)
                
                # Check if any features have extremely high variance (potential outliers)
                max_variance = np.max(feature_variances)
                if max_variance > self.max_feature_variance_threshold:
                    self.logger.warning(f"High feature variance detected: {max_variance:.2f}")
                
                # Check if features are too uniform (low variance)
                min_variance = np.min(feature_variances)
                if min_variance < 1e-10:
                    self.logger.warning("Some features have very low variance in synthetic samples")
            
            # Check sample distribution (synthetic samples should be within reasonable bounds)
            original_samples = X_synthetic[:self.original_samples]
            synthetic_samples = X_synthetic[self.original_samples:]
            
            if len(synthetic_samples) > 0:
                # Compare feature ranges
                original_min = np.min(original_samples, axis=0)
                original_max = np.max(original_samples, axis=0)
                synthetic_min = np.min(synthetic_samples, axis=0)
                synthetic_max = np.max(synthetic_samples, axis=0)
                
                # Check if synthetic samples are within reasonable bounds
                range_expansion_factor = 1.5  # Allow 50% expansion of original range
                
                for i in range(X_synthetic.shape[1]):
                    original_range = original_max[i] - original_min[i]
                    if original_range > 0:  # Avoid division by zero
                        lower_bound = original_min[i] - (original_range * range_expansion_factor)
                        upper_bound = original_max[i] + (original_range * range_expansion_factor)
                        
                        if (synthetic_min[i] < lower_bound or synthetic_max[i] > upper_bound):
                            self.logger.warning(f"Feature {i}: synthetic samples outside reasonable bounds")
            
            self.logger.info("Synthetic sample validation completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Synthetic sample validation failed: {str(e)}")
            return False
    
    def get_smote_info(self) -> Dict[str, Any]:
        """
        Get information about the SMOTE processing.
        
        Returns:
            Dictionary with SMOTE processing information
        """
        return {
            'is_fitted': self.is_fitted,
            'k_neighbors': self.k_neighbors,
            'random_state': self.random_state,
            'target_ratio': self.target_ratio,
            'original_samples': self.original_samples,
            'synthetic_samples_created': self.synthetic_samples_created,
            'total_samples': self.original_samples + self.synthetic_samples_created
        }
    
    def cleanup_memory(self) -> None:
        """
        Clean up memory used by SMOTE processing.
        Call this after training to free up memory from synthetic samples.
        """
        import gc
        
        if hasattr(self, 'smote') and self.smote is not None:
            # Clear SMOTE instance
            self.smote = None
            
        # Force garbage collection
        gc.collect()
        
        self.logger.info("SMOTE memory cleanup completed")
    
    def get_memory_usage_estimate(self, X_shape: Tuple[int, int]) -> Dict[str, float]:
        """
        Estimate memory usage for SMOTE processing.
        
        Args:
            X_shape: Shape of input feature matrix (samples, features)
            
        Returns:
            Dictionary with memory usage estimates in MB
        """
        samples, features = X_shape
        
        # Estimate memory for different components (in bytes)
        # Assuming float64 (8 bytes per element)
        bytes_per_element = 8
        
        original_data_mb = (samples * features * bytes_per_element) / (1024 * 1024)
        
        # Estimate synthetic samples based on target ratio
        current_minority_ratio = 0.3  # Conservative estimate
        if current_minority_ratio < self.target_ratio:
            synthetic_samples_estimate = int(
                (self.target_ratio * samples) / (1 - self.target_ratio) - 
                (current_minority_ratio * samples) / (1 - current_minority_ratio)
            )
        else:
            synthetic_samples_estimate = 0
        
        synthetic_data_mb = (synthetic_samples_estimate * features * bytes_per_element) / (1024 * 1024)
        
        # SMOTE algorithm overhead (k-neighbors, distance calculations)
        smote_overhead_mb = original_data_mb * 0.5  # Estimate 50% overhead
        
        total_peak_mb = original_data_mb + synthetic_data_mb + smote_overhead_mb
        
        return {
            'original_data_mb': original_data_mb,
            'estimated_synthetic_data_mb': synthetic_data_mb,
            'smote_overhead_mb': smote_overhead_mb,
            'total_peak_usage_mb': total_peak_mb,
            'estimated_synthetic_samples': synthetic_samples_estimate
        }
    
    def _calculate_sampling_strategy(self, y: np.ndarray) -> Optional[Dict[int, int]]:
        """
        Calculate aggressive sampling strategy optimized for spam detection.
        
        Args:
            y: Target labels
            
        Returns:
            Dictionary with sampling strategy or None if no resampling needed
        """
        unique_labels, counts = np.unique(y, return_counts=True)
        
        if len(unique_labels) != 2:
            raise ValueError("SMOTE requires exactly 2 classes")
        
        # Identify majority and minority classes
        majority_class = unique_labels[np.argmax(counts)]
        minority_class = unique_labels[np.argmin(counts)]
        
        majority_count = counts[np.argmax(counts)]
        minority_count = counts[np.argmin(counts)]
        
        self.logger.info(f"Class distribution - Majority ({majority_class}): {majority_count}, "
                        f"Minority ({minority_class}): {minority_count}")
        
        # Calculate current ratio
        total_samples = len(y)
        current_minority_ratio = minority_count / total_samples
        
        self.logger.info(f"Current minority ratio: {current_minority_ratio:.3f}, "
                        f"Target ratio: {self.target_ratio:.3f}")
        
        # Be more aggressive - always apply SMOTE if ratio is below target
        # Even if close to target, still apply some oversampling
        min_improvement_threshold = 0.02  # Always improve by at least 2%
        effective_target = max(self.target_ratio, current_minority_ratio + min_improvement_threshold)
        
        if current_minority_ratio >= effective_target:
            return None
        
        # Calculate target minority count to achieve desired ratio
        # target_ratio = target_minority_count / (majority_count + target_minority_count)
        # Solving for target_minority_count:
        target_minority_count = int((effective_target * majority_count) / (1 - effective_target))
        
        # Ensure we don't reduce the minority class
        target_minority_count = max(target_minority_count, minority_count)
        
        # Add extra samples for better spam detection (10% buffer)
        target_minority_count = int(target_minority_count * 1.1)
        
        sampling_strategy = {int(minority_class): target_minority_count}
        
        self.logger.info(f"Calculated aggressive sampling strategy: {sampling_strategy}")
        return sampling_strategy
    
    def _validate_parameters(self) -> None:
        """Validate SMOTE processor parameters."""
        if self.k_neighbors < 1:
            raise ValueError("k_neighbors must be >= 1")
        
        if not (0.1 <= self.target_ratio <= 0.9):
            raise ValueError("target_ratio must be between 0.1 and 0.9")
        
        if self.random_state is not None and self.random_state < 0:
            raise ValueError("random_state must be non-negative")
        
        self.logger.info("SMOTEProcessor parameters validated successfully")