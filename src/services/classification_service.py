"""
Core classification service for email spam detection.

This module provides the ClassificationService class that integrates
preprocessing pipeline with model prediction and handles single email
classification with confidence scoring.
"""

import time
import logging
from datetime import datetime, date
from typing import List, Optional, Dict
from collections import defaultdict, deque
import numpy as np

from src.models.data_models import ClassificationResult, ClassificationServiceInterface
from src.models.model_manager import ModelManager
from src.preprocessing import PreprocessingPipeline
from src.data_loader import EmailValidator


class ClassificationService(ClassificationServiceInterface):
    """
    Core service for email spam classification.
    
    Integrates preprocessing pipeline with ML models to provide
    single email classification with confidence scores.
    """
    
    def __init__(self, 
                 model_manager: Optional[ModelManager] = None,
                 preprocessing_pipeline: Optional[PreprocessingPipeline] = None):
        """
        Initialize the classification service.
        
        Args:
            model_manager: ModelManager instance for ML models
            preprocessing_pipeline: PreprocessingPipeline for text processing
        """
        self.model_manager = model_manager or ModelManager()
        self.preprocessing_pipeline = preprocessing_pipeline or PreprocessingPipeline()
        self.validator = EmailValidator()
        self.logger = logging.getLogger(__name__)
        
        # Processing statistics tracking
        self._daily_stats = defaultdict(lambda: {'count': 0, 'total_time': 0.0})
        self._response_times = deque(maxlen=100)  # Keep last 100 response times
        self._total_processed = 0
        
        # Classification result tracking
        self._spam_count = 0
        self._legitimate_count = 0
        
        # Load existing models if available
        self._initialize_service()
    
    def _initialize_service(self) -> None:
        """Initialize the service by loading models and preprocessing pipeline."""
        try:
            # Try to load existing models
            loaded_models = self.model_manager.load_models()
            if loaded_models:
                self.logger.info(f"Loaded {len(loaded_models)} trained models")
                
                # Try to load preprocessing pipeline if models exist
                self._try_load_preprocessing_pipeline()
            else:
                self.logger.warning("No trained models found. Service will need training before use.")
            
        except Exception as e:
            self.logger.error(f"Error initializing classification service: {str(e)}")
    
    def _try_load_preprocessing_pipeline(self) -> None:
        """Try to load existing preprocessing pipeline."""
        try:
            import pickle
            import os
            
            # Try to load preprocessing pipeline from models directory
            pipeline_path = os.path.join(self.model_manager.models_dir, "preprocessing_pipeline.pkl")
            self.logger.debug(f"Looking for preprocessing pipeline at: {pipeline_path}")
            
            if os.path.exists(pipeline_path):
                self.logger.info(f"Found preprocessing pipeline file: {pipeline_path}")
                with open(pipeline_path, 'rb') as f:
                    self.preprocessing_pipeline = pickle.load(f)
                self.logger.info("Successfully loaded existing preprocessing pipeline")
                
                # Verify pipeline is fitted
                if hasattr(self.preprocessing_pipeline, '_is_fitted') and self.preprocessing_pipeline._is_fitted:
                    self.logger.info("Preprocessing pipeline is fitted and ready")
                else:
                    self.logger.warning("Preprocessing pipeline may not be fitted")
            else:
                self.logger.warning(f"No preprocessing pipeline found at: {pipeline_path}")
                
        except Exception as e:
            self.logger.error(f"Error loading preprocessing pipeline: {str(e)}", exc_info=True)
    
    def classify_email(self, email_text: str) -> ClassificationResult:
        """
        Classify a single email as spam or legitimate.
        
        Args:
            email_text: The email text to classify
            
        Returns:
            ClassificationResult with prediction, confidence, and metadata
            
        Raises:
            ValueError: If email text is invalid
            RuntimeError: If no trained models are available
        """
        start_time = time.time()
        
        # Handle None input early
        if email_text is None:
            raise ValueError("Invalid email text: Email text cannot be None")
        
        self.logger.debug(f"Starting email classification for text length: {len(email_text)}")
        
        # Validate input
        validation_result = self.validator.validate_email_text(email_text)
        if not validation_result.is_valid:
            self.logger.warning(f"Invalid email text: {'; '.join(validation_result.errors)}")
            raise ValueError(f"Invalid email text: {'; '.join(validation_result.errors)}")
        
        # Get the best model
        self.logger.debug("Getting best model for classification")
        model_name, model = self.model_manager.get_best_model()
        self.logger.debug(f"Using model: {model_name}")
        
        if not hasattr(model, 'is_trained') or not model.is_trained:
            self.logger.error("No trained models available")
            raise RuntimeError("No trained models available. Please train models first.")
        
        try:
            # Preprocess the email text
            self.logger.debug("Checking preprocessing pipeline status")
            if not hasattr(self.preprocessing_pipeline, '_is_fitted') or not self.preprocessing_pipeline._is_fitted:
                self.logger.error("Preprocessing pipeline not fitted")
                raise RuntimeError("Preprocessing pipeline not fitted. Please train the system first.")
            
            self.logger.debug("Transforming email text to features")
            features = self.preprocessing_pipeline.transform([email_text])
            self.logger.debug(f"Feature shape: {features.shape}")
            
            # Make prediction
            self.logger.debug("Making prediction with model")
            prediction = model.predict(features)[0]
            
            # Get prediction probabilities for confidence score
            self.logger.debug("Getting prediction probabilities")
            probabilities = model.predict_proba(features)[0]
            
            # Calculate confidence score (max probability)
            confidence = float(np.max(probabilities))
            self.logger.debug(f"Prediction: {prediction}, Confidence: {confidence}")
            
            # Convert prediction to readable format
            prediction_label = self._format_prediction(prediction)
            
            processing_time = time.time() - start_time
            
            # Update processing statistics
            self._update_processing_stats(processing_time, prediction_label)
            
            # Check if the model used was balanced
            model_metrics = self.model_manager.get_model_metrics(model_name)
            is_balanced = model_metrics.class_balancing_enabled if model_metrics else False
            
            result = ClassificationResult(
                prediction=prediction_label,
                confidence=confidence,
                model_used=model_name,
                processing_time=processing_time,
                timestamp=datetime.now()
            )
            
            # Add balancing information to result (extend the result object)
            result.model_balanced = is_balanced
            if is_balanced and model_metrics:
                result.balancing_method = model_metrics.balancing_method
                result.false_negative_rate = model_metrics.false_negative_rate
            
            self.logger.info(f"Classified email as {prediction_label} with confidence {confidence:.3f}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error during email classification: {str(e)}")
            raise RuntimeError(f"Classification failed: {str(e)}")
    
    def classify_batch(self, emails: List[str]) -> List[ClassificationResult]:
        """
        Classify multiple emails.
        
        Args:
            emails: List of email texts to classify
            
        Returns:
            List of ClassificationResult objects
            
        Raises:
            ValueError: If email batch is invalid
            RuntimeError: If no trained models are available
        """
        if not emails:
            raise ValueError("Email list cannot be empty")
        
        # Validate batch
        batch_validation, invalid_indices = self.validator.validate_batch_emails(emails)
        if not batch_validation.is_valid:
            raise ValueError(f"Invalid email batch: {'; '.join(batch_validation.errors)}")
        
        results = []
        
        for i, email in enumerate(emails):
            try:
                result = self.classify_email(email)
                results.append(result)
            except Exception as e:
                # Create error result for failed classification
                error_result = ClassificationResult(
                    prediction="Error",
                    confidence=0.0,
                    model_used="None",
                    processing_time=0.0,
                    timestamp=datetime.now()
                )
                results.append(error_result)
                self.logger.error(f"Failed to classify email {i+1}: {str(e)}")
        
        return results
    
    def get_active_model(self) -> str:
        """
        Get the name of the currently active (best) model.
        
        Returns:
            Name of the active model
        """
        model_name, _ = self.model_manager.get_best_model()
        return model_name
    
    def get_model_metrics(self) -> dict:
        """
        Get performance metrics for all models.
        
        Returns:
            Dictionary of model names and their metrics
        """
        metrics = {}
        for model_name, model_metrics in self.model_manager.get_all_metrics().items():
            metrics[model_name] = {
                'accuracy': model_metrics.accuracy,
                'precision': model_metrics.precision,
                'recall': model_metrics.recall,
                'f1_score': model_metrics.f1_score,
                'training_date': model_metrics.training_date.isoformat(),
                'test_samples': model_metrics.test_samples,
                'class_balancing_enabled': model_metrics.class_balancing_enabled,
                'original_spam_ratio': model_metrics.original_spam_ratio,
                'balanced_spam_ratio': model_metrics.balanced_spam_ratio,
                'false_negative_rate': model_metrics.false_negative_rate,
                'synthetic_samples_used': model_metrics.synthetic_samples_used,
                'balancing_method': model_metrics.balancing_method
            }
        return metrics
    
    def is_ready(self) -> bool:
        """
        Check if the service is ready for classification.
        
        Returns:
            True if service has trained models and fitted preprocessing pipeline
        """
        try:
            self.logger.debug("Checking service readiness...")
            
            # Check if we have a trained model
            model_name, model = self.model_manager.get_best_model()
            has_trained_model = hasattr(model, 'is_trained') and model.is_trained
            self.logger.debug(f"Has trained model ({model_name}): {has_trained_model}")
            
            # Check if preprocessing pipeline is fitted
            has_fitted_pipeline = (hasattr(self.preprocessing_pipeline, '_is_fitted') and 
                                 self.preprocessing_pipeline._is_fitted)
            self.logger.debug(f"Has fitted preprocessing pipeline: {has_fitted_pipeline}")
            
            is_ready = has_trained_model and has_fitted_pipeline
            self.logger.info(f"Service readiness check: {is_ready}")
            
            return is_ready
        except Exception as e:
            self.logger.error(f"Error checking service readiness: {str(e)}", exc_info=True)
            return False
    
    def train_system(self, training_texts: List[str], training_labels: List[str]) -> Dict[str, float]:
        """
        Train the entire system including preprocessing pipeline and models.
        
        Args:
            training_texts: List of email texts for training
            training_labels: List of corresponding labels
            
        Returns:
            Dictionary of model training accuracies
        """
        try:
            # Fit the preprocessing pipeline
            self.preprocessing_pipeline.fit(training_texts, training_labels)
            
            # Transform training data
            X_train = self.preprocessing_pipeline.transform(training_texts)
            
            # Create training DataFrame for model manager
            import pandas as pd
            training_df = pd.DataFrame({
                'text': X_train,  # Use transformed features
                'label': training_labels
            })
            
            # Train models
            training_results = self.model_manager.train_models_with_features(X_train, training_labels)
            
            # Save preprocessing pipeline
            self._save_preprocessing_pipeline()
            
            self.logger.info("System training completed successfully")
            return training_results
            
        except Exception as e:
            self.logger.error(f"Error training system: {str(e)}")
            raise RuntimeError(f"System training failed: {str(e)}")
    
    def _save_preprocessing_pipeline(self) -> None:
        """Save the preprocessing pipeline to disk."""
        try:
            import pickle
            import os
            
            pipeline_path = os.path.join(self.model_manager.models_dir, "preprocessing_pipeline.pkl")
            with open(pipeline_path, 'wb') as f:
                pickle.dump(self.preprocessing_pipeline, f)
            
            self.logger.info("Saved preprocessing pipeline")
            
        except Exception as e:
            self.logger.error(f"Error saving preprocessing pipeline: {str(e)}")
    
    def _format_prediction(self, prediction) -> str:
        """
        Format model prediction to standard output format.
        
        Args:
            prediction: Raw model prediction
            
        Returns:
            Formatted prediction string
        """
        if isinstance(prediction, (int, float)):
            return "Spam" if prediction == 1 else "Legitimate"
        
        prediction_str = str(prediction).lower()
        if prediction_str in ['spam', '1']:
            return "Spam"
        elif prediction_str in ['legitimate', 'ham', '0']:
            return "Legitimate"
        else:
            return "Unknown"
    
    def _update_processing_stats(self, processing_time: float, prediction_label: str = None) -> None:
        """
        Update processing statistics with new classification.
        
        Args:
            processing_time: Time taken for the classification in seconds
            prediction_label: The classification result (Spam or Legitimate)
        """
        try:
            # Update daily stats
            today = date.today().isoformat()
            self._daily_stats[today]['count'] += 1
            self._daily_stats[today]['total_time'] += processing_time
            
            # Update response times deque
            self._response_times.append(processing_time)
            
            # Update total processed count
            self._total_processed += 1
            
            # Update classification result counters
            if prediction_label:
                if prediction_label.lower() == "spam":
                    self._spam_count += 1
                elif prediction_label.lower() == "legitimate":
                    self._legitimate_count += 1
            
            self.logger.debug(f"Updated processing stats: daily count={self._daily_stats[today]['count']}, "
                            f"processing_time={processing_time:.3f}s, spam_count={self._spam_count}, "
                            f"legitimate_count={self._legitimate_count}")
            
        except Exception as e:
            self.logger.warning(f"Error updating processing stats: {str(e)}")
    
    def get_processing_stats(self) -> Dict[str, any]:
        """
        Get current processing statistics.
        
        Returns:
            Dictionary containing processing statistics
        """
        try:
            today = date.today().isoformat()
            today_stats = self._daily_stats[today]
            
            # Calculate average response time
            avg_response_time = 0.0
            if self._response_times:
                avg_response_time = sum(self._response_times) / len(self._response_times)
            
            # Calculate today's average response time
            today_avg_response_time = 0.0
            if today_stats['count'] > 0:
                today_avg_response_time = today_stats['total_time'] / today_stats['count']
            
            stats = {
                'processed_today': today_stats['count'],
                'avg_response_time': avg_response_time,
                'today_avg_response_time': today_avg_response_time,
                'total_processed': self._total_processed,
                'recent_response_times_count': len(self._response_times),
                'last_updated': datetime.now().isoformat()
            }
            
            self.logger.debug(f"Retrieved processing stats: {stats}")
            return stats
            
        except Exception as e:
            self.logger.error(f"Error getting processing stats: {str(e)}")
            return {
                'processed_today': 0,
                'avg_response_time': 0.0,
                'today_avg_response_time': 0.0,
                'total_processed': 0,
                'recent_response_times_count': 0,
                'last_updated': datetime.now().isoformat()
            }
    
    def get_classification_distribution(self) -> Dict[str, int]:
        """
        Get distribution of classification results.
        
        Returns:
            Dictionary containing spam count, legitimate count, total count, and percentages
        """
        try:
            total_count = self._spam_count + self._legitimate_count
            
            # Calculate percentages
            spam_percentage = (self._spam_count / total_count * 100) if total_count > 0 else 0.0
            legitimate_percentage = (self._legitimate_count / total_count * 100) if total_count > 0 else 0.0
            
            distribution = {
                'spam_count': self._spam_count,
                'legitimate_count': self._legitimate_count,
                'total_count': total_count,
                'spam_percentage': round(spam_percentage, 1),
                'legitimate_percentage': round(legitimate_percentage, 1)
            }
            
            self.logger.debug(f"Retrieved classification distribution: {distribution}")
            return distribution
            
        except Exception as e:
            self.logger.error(f"Error getting classification distribution: {str(e)}")
            return {
                'spam_count': 0,
                'legitimate_count': 0,
                'total_count': 0,
                'spam_percentage': 0.0,
                'legitimate_percentage': 0.0
            }
    
    def reset_daily_stats(self) -> None:
        """Reset daily statistics (typically called at midnight)."""
        try:
            today = date.today().isoformat()
            if today in self._daily_stats:
                del self._daily_stats[today]
            self.logger.info("Daily statistics reset")
        except Exception as e:
            self.logger.warning(f"Error resetting daily stats: {str(e)}")
    
    def get_stats_summary(self) -> Dict[str, any]:
        """
        Get a summary of all processing statistics.
        
        Returns:
            Dictionary containing comprehensive statistics summary
        """
        try:
            processing_stats = self.get_processing_stats()
            
            # Get model information
            active_model = self.get_active_model()
            model_metrics = self.get_model_metrics()
            
            summary = {
                'service_ready': self.is_ready(),
                'active_model': active_model,
                'processing_stats': processing_stats,
                'model_count': len(model_metrics),
                'models_available': list(model_metrics.keys()) if model_metrics else [],
                'last_updated': datetime.now().isoformat()
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error getting stats summary: {str(e)}")
            return {
                'service_ready': False,
                'active_model': 'Error',
                'processing_stats': self.get_processing_stats(),
                'model_count': 0,
                'models_available': [],
                'last_updated': datetime.now().isoformat()
            }