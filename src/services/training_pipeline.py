"""
Training and evaluation pipeline for email spam classification models.

This module provides comprehensive training workflows, evaluation metrics calculation,
and model comparison functionality for the email spam classifier system.
"""
import os
import logging
import time
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix

from src.models.model_manager import ModelManager
from src.models.data_models import ModelMetrics
from src.preprocessing import PreprocessingPipeline
from src.services.class_balancer import ClassBalancer
from src.services.balancing_config import BalancingConfig
from src.services.performance_optimizer import PerformanceOptimizer
from src.services.balancing_validator import BalancingValidator
from src.services.balancing_monitor import BalancingMonitor, create_class_balancer_health_check, create_smote_processor_health_check


class TrainingPipeline:
    """
    Complete training and evaluation pipeline for spam classification models.
    
    Handles data preprocessing, model training, evaluation, and comparison.
    """
    
    def __init__(self, 
                 models_dir: str = "models/trained",
                 preprocessing_params: Optional[Dict] = None,
                 balancing_config: Optional[BalancingConfig] = None):
        """
        Initialize the training pipeline.
        
        Args:
            models_dir: Directory to save trained models
            preprocessing_params: Parameters for preprocessing pipeline
            balancing_config: Configuration for class balancing operations
        """
        self.models_dir = models_dir
        self.model_manager = ModelManager(models_dir)
        self.preprocessing_pipeline = PreprocessingPipeline(preprocessing_params)
        
        # Initialize class balancer
        self.balancing_config = balancing_config or BalancingConfig()
        self.class_balancer = ClassBalancer(self.balancing_config)
        
        # Initialize performance optimizer
        self.performance_optimizer = PerformanceOptimizer()
        
        # Initialize balancing validator
        self.balancing_validator = BalancingValidator()
        
        # Setup logging first
        self.logger = logging.getLogger(__name__)
        
        # Initialize production monitor
        self.balancing_monitor = BalancingMonitor()
        self._setup_monitoring()
        
        # Training state
        self.is_fitted = False
        self.training_history: List[Dict] = []
        
        # Ensure models directory exists
        os.makedirs(self.models_dir, exist_ok=True)
    
    def prepare_data(self, 
                    data: pd.DataFrame,
                    text_column: str = 'text',
                    label_column: str = 'label',
                    test_size: float = 0.2,
                    validation_size: float = 0.1,
                    random_state: int = 42,
                    enable_balancing: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Prepare and split data for training, validation, and testing.
        Applies class balancing after data splitting if enabled.
        
        Args:
            data: Input DataFrame with email text and labels
            text_column: Name of the text column
            label_column: Name of the label column
            test_size: Proportion of data for testing
            validation_size: Proportion of training data for validation
            random_state: Random seed for reproducibility
            enable_balancing: Whether to apply class balancing to training data
            
        Returns:
            Tuple of (train_data, validation_data, test_data)
        """
        if text_column not in data.columns or label_column not in data.columns:
            raise ValueError(f"Data must contain '{text_column}' and '{label_column}' columns")
        
        # Ensure we have the right column names
        if text_column != 'text' or label_column != 'label':
            data = data.rename(columns={text_column: 'text', label_column: 'label'})
        
        # Remove any rows with missing values
        data = data.dropna(subset=['text', 'label'])
        
        # Convert labels to standard format (0 for legitimate, 1 for spam)
        label_mapping = {'ham': 0, 'legitimate': 0, 'spam': 1}
        data['label'] = data['label'].str.lower().map(label_mapping)
        
        # Remove any unmapped labels
        data = data.dropna(subset=['label'])
        data['label'] = data['label'].astype(int)
        
        self.logger.info(f"Dataset prepared: {len(data)} samples")
        self.logger.info(f"Label distribution: {data['label'].value_counts().to_dict()}")
        
        # First split: separate test set
        train_val_data, test_data = train_test_split(
            data, 
            test_size=test_size, 
            random_state=random_state,
            stratify=data['label']
        )
        
        # Second split: separate validation from training
        if validation_size > 0:
            train_data, val_data = train_test_split(
                train_val_data,
                test_size=validation_size,
                random_state=random_state,
                stratify=train_val_data['label']
            )
        else:
            train_data = train_val_data
            val_data = pd.DataFrame(columns=data.columns)
        
        self.logger.info(f"Data split - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
        
        # Apply class balancing to training data if enabled
        if enable_balancing and self.balancing_config.enabled:
            try:
                self.logger.info("Applying optimized class balancing to training data...")
                
                # Start performance monitoring
                self.performance_optimizer.start_monitoring()
                
                # Convert training data to features for balancing
                X_train_text = train_data['text'].tolist()
                y_train = train_data['label'].values
                
                # Fit preprocessing pipeline to get features
                X_train_features = self.preprocessing_pipeline.fit_transform(X_train_text)
                
                # Log system info for optimization
                system_info = self.performance_optimizer.get_system_info()
                self.logger.info(f"System info - Available memory: {system_info.get('available_memory_gb', 'unknown'):.1f} GB, "
                               f"CPU cores: {system_info.get('cpu_cores', 'unknown')}")
                
                # Apply class balancing with performance monitoring
                self.performance_optimizer.update_peak_memory()
                
                # Record balancing operation start
                balancing_start_time = time.time()
                
                try:
                    X_balanced, y_balanced = self.class_balancer.balance_dataset(X_train_features, y_train)
                    
                    # Record successful balancing operation
                    balancing_duration = (time.time() - balancing_start_time) * 1000
                    self.balancing_monitor.record_balancing_operation(
                        operation_type="class_balancing",
                        success=True,
                        duration_ms=balancing_duration,
                        details={
                            'method': self.balancing_config.method,
                            'original_samples': len(y_train),
                            'balanced_samples': len(y_balanced),
                            'synthetic_created': len(y_balanced) - len(y_train)
                        }
                    )
                    
                except Exception as balancing_error:
                    # Record failed balancing operation
                    balancing_duration = (time.time() - balancing_start_time) * 1000
                    self.balancing_monitor.record_balancing_operation(
                        operation_type="class_balancing",
                        success=False,
                        duration_ms=balancing_duration,
                        details={
                            'error': str(balancing_error),
                            'method': self.balancing_config.method
                        }
                    )
                    raise balancing_error
                
                self.performance_optimizer.update_peak_memory()
                
                # If balancing was applied (dataset was modified)
                if not np.array_equal(X_train_features, X_balanced) or not np.array_equal(y_train, y_balanced):
                    self.logger.info("Class balancing applied - reconstructing training DataFrame")
                    
                    # For SMOTE, we need to handle synthetic samples
                    # Since we can't easily convert features back to text, we'll store the balanced features
                    # and use them directly in training
                    self._balanced_features = X_balanced
                    self._balanced_labels = y_balanced
                    self._balancing_applied = True
                    
                    # Update training data with balanced labels (for logging purposes)
                    # Note: The actual features will be used from _balanced_features
                    balanced_df_data = []
                    for i in range(len(y_balanced)):
                        if i < len(train_data):
                            # Original sample
                            balanced_df_data.append({
                                'text': train_data.iloc[i]['text'],
                                'label': int(y_balanced[i])
                            })
                        else:
                            # Synthetic sample - use placeholder text
                            balanced_df_data.append({
                                'text': f"[SYNTHETIC_SAMPLE_{i}]",
                                'label': int(y_balanced[i])
                            })
                    
                    train_data = pd.DataFrame(balanced_df_data)
                    
                    self.logger.info(f"Balanced training data: {len(train_data)} samples")
                    self.logger.info(f"Balanced label distribution: {pd.Series(y_balanced).value_counts().to_dict()}")
                    
                    # Log performance metrics
                    perf_metrics = self.performance_optimizer.get_performance_metrics(len(y_balanced))
                    self.performance_optimizer.log_performance_summary(perf_metrics)
                    
                    # Record performance metrics for monitoring
                    self.balancing_monitor.record_performance_metrics(
                        processing_time_ms=perf_metrics.processing_time,
                        memory_usage_mb=perf_metrics.memory_usage_mb,
                        samples_processed=perf_metrics.samples_processed,
                        false_negative_rate=0.0,  # Will be updated after evaluation
                        accuracy=0.0,  # Will be updated after evaluation
                        synthetic_samples_created=len(y_balanced) - len(y_train),
                        balancing_method=self.balancing_config.method,
                        validation_passed=True
                    )
                else:
                    self.logger.info("No balancing applied - using original training data")
                    self._balanced_features = None
                    self._balanced_labels = None
                    self._balancing_applied = False
                
                # Clean up memory after balancing
                self.performance_optimizer.optimize_memory_usage()
                
            except Exception as e:
                self.logger.error(f"Error during class balancing: {str(e)}")
                self.logger.info("Falling back to unbalanced training data")
                self._balanced_features = None
                self._balanced_labels = None
                self._balancing_applied = False
        else:
            self.logger.info("Class balancing disabled - using original training data")
            self._balanced_features = None
            self._balanced_labels = None
            self._balancing_applied = False
        
        return train_data, val_data, test_data
    
    def train_all_models(self, 
                        training_data: pd.DataFrame,
                        validation_data: Optional[pd.DataFrame] = None,
                        use_class_weights: bool = True) -> Dict[str, float]:
        """
        Train all models with comprehensive evaluation and class balancing support.
        
        Args:
            training_data: Training dataset with 'text' and 'label' columns
            validation_data: Optional validation dataset
            use_class_weights: Whether to apply class weights to sklearn models
            
        Returns:
            Dictionary of model names and their training accuracies
        """
        self.logger.info("Starting model training pipeline...")
        
        # Use balanced features if available, otherwise prepare features normally
        if hasattr(self, '_balanced_features') and self._balanced_features is not None:
            self.logger.info("Using balanced features from class balancing")
            X_train_features = self._balanced_features
            y_train = self._balanced_labels.tolist()
            self.is_fitted = True  # Preprocessing was already fitted during balancing
        else:
            # Prepare features normally
            X_train = training_data['text'].tolist()
            y_train = training_data['label'].tolist()
            
            # Fit preprocessing pipeline
            self.logger.info("Fitting preprocessing pipeline...")
            X_train_features = self.preprocessing_pipeline.fit_transform(X_train)
            self.is_fitted = True
        
        # Prepare validation data if provided
        X_val_features = None
        y_val = None
        if validation_data is not None and len(validation_data) > 0:
            X_val = validation_data['text'].tolist()
            y_val = validation_data['label'].tolist()
            X_val_features = self.preprocessing_pipeline.transform(X_val)
        
        # Calculate class weights if enabled
        class_weights = None
        if use_class_weights and (self.balancing_config.method in ['class_weights', 'both']):
            try:
                class_weights = self.class_balancer.get_class_weights(np.array(y_train))
                self.logger.info(f"Calculated class weights: {class_weights}")
            except Exception as e:
                self.logger.error(f"Error calculating class weights: {str(e)}")
                class_weights = None
        
        # Train models using ModelManager
        training_results = {}
        
        for model_name in self.model_manager.models.keys():
            try:
                self.logger.info(f"Training {model_name}...")
                
                # Get the model
                model = self.model_manager.models[model_name]
                
                # Apply class weights to sklearn models if available
                if class_weights is not None and hasattr(model, 'model'):
                    self._apply_class_weights_to_model(model.model, class_weights, model_name)
                
                # Train the model
                model.train(X_train_features, y_train)
                
                # Calculate training accuracy
                train_predictions = model.predict(X_train_features)
                train_accuracy = np.mean(train_predictions == y_train)
                training_results[model_name] = train_accuracy
                
                # Evaluate on validation set if available
                if X_val_features is not None and y_val is not None:
                    val_predictions = model.predict(X_val_features)
                    val_metrics = self._calculate_detailed_metrics(
                        y_val, val_predictions, model_name, len(y_val)
                    )
                    self.model_manager.model_metrics[model_name] = val_metrics
                    
                    self.logger.info(f"{model_name} - Train Acc: {train_accuracy:.4f}, Val Acc: {val_metrics.accuracy:.4f}")
                else:
                    self.logger.info(f"{model_name} - Train Acc: {train_accuracy:.4f}")
                
                # Perform cross-validation for more robust evaluation
                cv_scores = cross_val_score(
                    model.model, X_train_features, y_train, cv=5, scoring='accuracy'
                )
                self.logger.info(f"{model_name} - CV Acc: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
                
            except Exception as e:
                self.logger.error(f"Error training {model_name}: {str(e)}")
                training_results[model_name] = 0.0
        
        # Update best model using composite scoring
        self.model_manager.force_best_model_update()
        
        # Save training history
        training_record = {
            'timestamp': pd.Timestamp.now(),
            'training_samples': len(training_data),
            'validation_samples': len(validation_data) if validation_data is not None else 0,
            'results': training_results.copy()
        }
        self.training_history.append(training_record)
        
        # Save the fitted preprocessing pipeline
        self._save_preprocessing_pipeline()
        
        self.logger.info("Model training completed successfully")
        return training_results
    
    def evaluate_all_models(self, test_data: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """
        Comprehensive evaluation of all trained models.
        
        Args:
            test_data: Test dataset with 'text' and 'label' columns
            
        Returns:
            Dictionary with detailed evaluation results for each model
        """
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before evaluation")
        
        self.logger.info("Starting comprehensive model evaluation...")
        
        # Prepare test features
        X_test = test_data['text'].tolist()
        y_test = test_data['label'].tolist()
        X_test_features = self.preprocessing_pipeline.transform(X_test)
        
        evaluation_results = {}
        
        for model_name, model in self.model_manager.models.items():
            try:
                if not hasattr(model, 'is_trained') or not model.is_trained:
                    self.logger.warning(f"Model {model_name} is not trained, skipping evaluation")
                    continue
                
                self.logger.info(f"Evaluating {model_name}...")
                
                # Make predictions
                predictions = model.predict(X_test_features)
                probabilities = model.predict_proba(X_test_features)
                
                # Calculate detailed metrics
                metrics = self._calculate_detailed_metrics(
                    y_test, predictions, model_name, len(y_test)
                )
                
                # Generate classification report
                class_report = classification_report(
                    y_test, predictions, 
                    target_names=['Legitimate', 'Spam'],
                    output_dict=True
                )
                
                # Generate confusion matrix
                conf_matrix = confusion_matrix(y_test, predictions)
                
                # Store results
                evaluation_results[model_name] = {
                    'metrics': {
                        'accuracy': metrics.accuracy,
                        'precision': metrics.precision,
                        'recall': metrics.recall,
                        'f1_score': metrics.f1_score
                    },
                    'classification_report': class_report,
                    'confusion_matrix': conf_matrix.tolist(),
                    'predictions': predictions.tolist(),
                    'probabilities': probabilities.tolist() if probabilities is not None else None
                }
                
                # Update stored metrics
                self.model_manager.model_metrics[model_name] = metrics
                
                self.logger.info(f"{model_name} evaluation completed - Accuracy: {metrics.accuracy:.4f}")
                
            except Exception as e:
                self.logger.error(f"Error evaluating {model_name}: {str(e)}")
        
        # Force best model recalculation with composite scoring after all evaluations
        self.logger.info("Recalculating best model selection using composite scoring...")
        self.model_manager.force_best_model_update()
        
        # Save all trained models with final test metrics
        self.model_manager._save_all_models()
        
        return evaluation_results
    
    def compare_models(self) -> pd.DataFrame:
        """
        Generate a comparison table of all model performances.
        
        Returns:
            DataFrame with model comparison metrics
        """
        if not self.model_manager.model_metrics:
            return pd.DataFrame()
        
        comparison_data = []
        
        for model_name, metrics in self.model_manager.model_metrics.items():
            comparison_data.append({
                'Model': model_name,
                'Accuracy': metrics.accuracy,
                'Precision': metrics.precision,
                'Recall': metrics.recall,
                'F1-Score': metrics.f1_score,
                'Test Samples': metrics.test_samples,
                'Training Date': metrics.training_date.strftime('%Y-%m-%d %H:%M')
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Sort by accuracy (descending)
        comparison_df = comparison_df.sort_values('Accuracy', ascending=False)
        
        return comparison_df
    
    def get_best_model_info(self) -> Dict[str, Any]:
        """
        Get information about the best performing model.
        
        Returns:
            Dictionary with best model information
        """
        best_model_name, best_model = self.model_manager.get_best_model()
        
        if best_model_name in self.model_manager.model_metrics:
            metrics = self.model_manager.model_metrics[best_model_name]
            return {
                'name': best_model_name,
                'display_name': best_model.get_model_name(),
                'accuracy': metrics.accuracy,
                'precision': metrics.precision,
                'recall': metrics.recall,
                'f1_score': metrics.f1_score,
                'training_date': metrics.training_date,
                'test_samples': metrics.test_samples
            }
        
        return {
            'name': best_model_name,
            'display_name': best_model.get_model_name() if best_model else 'Unknown',
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'training_date': None,
            'test_samples': 0
        }
    
    def save_evaluation_report(self, 
                              evaluation_results: Dict[str, Dict[str, Any]], 
                              output_path: str = "models/evaluation_report.txt",
                              include_balancing_analysis: bool = True) -> None:
        """
        Save a detailed evaluation report to file with optional class balancing analysis.
        
        Args:
            evaluation_results: Results from evaluate_all_models
            output_path: Path to save the report
            include_balancing_analysis: Whether to include class balancing analysis
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            f.write("Email Spam Classification - Model Evaluation Report\n")
            f.write("=" * 60 + "\n\n")
            
            # Class balancing analysis section
            if include_balancing_analysis and hasattr(self, 'balancing_metrics') and self.balancing_metrics:
                f.write("CLASS BALANCING ANALYSIS\n")
                f.write("=" * 60 + "\n\n")
                
                # Get balancing summary
                balancing_summary = self.balancing_metrics.get_balancing_summary()
                
                if balancing_summary.get('balancing_applied', False):
                    f.write("Balancing Status: ENABLED\n")
                    f.write(f"Method Used: {balancing_summary.get('method_used', 'Unknown').upper()}\n")
                    f.write(f"Target Spam Ratio: {balancing_summary.get('target_spam_ratio', 0.0):.1%}\n")
                    f.write(f"Processing Time: {balancing_summary.get('processing_time', 0.0):.2f} seconds\n")
                    f.write(f"Validation: {'PASSED' if balancing_summary.get('validation_passed', False) else 'FAILED'}\n\n")
                    
                    # Distribution comparison
                    original_dist = balancing_summary.get('original_distribution', {})
                    balanced_dist = balancing_summary.get('balanced_distribution', {})
                    
                    if original_dist and balanced_dist:
                        f.write("Distribution Comparison:\n")
                        f.write("-" * 40 + "\n")
                        f.write(f"{'Metric':<20} {'Original':<12} {'Balanced':<12} {'Change':<10}\n")
                        f.write("-" * 54 + "\n")
                        
                        # Total samples
                        orig_total = original_dist.get('total_samples', 0)
                        bal_total = balanced_dist.get('total_samples', 0)
                        f.write(f"{'Total Samples':<20} {orig_total:<12,} {bal_total:<12,} {bal_total - orig_total:<10,}\n")
                        
                        # Spam samples
                        orig_spam = original_dist.get('spam_count', 0)
                        bal_spam = balanced_dist.get('spam_count', 0)
                        f.write(f"{'Spam Samples':<20} {orig_spam:<12,} {bal_spam:<12,} {bal_spam - orig_spam:<10,}\n")
                        
                        # Spam ratio
                        orig_ratio = original_dist.get('spam_ratio', 0.0)
                        bal_ratio = balanced_dist.get('spam_ratio', 0.0)
                        f.write(f"{'Spam Ratio':<20} {orig_ratio:<12.1%} {bal_ratio:<12.1%} {bal_ratio - orig_ratio:<10.1%}\n")
                        
                        # Synthetic samples
                        synthetic_count = balancing_summary.get('synthetic_samples_created', 0)
                        f.write(f"{'Synthetic Added':<20} {0:<12,} {synthetic_count:<12,} {synthetic_count:<10,}\n\n")
                    
                    # Performance comparison if available
                    perf_comparison = balancing_summary.get('performance_comparison', {})
                    if perf_comparison:
                        improvements = perf_comparison.get('improvements', {})
                        f.write("Performance Impact Analysis:\n")
                        f.write("-" * 40 + "\n")
                        f.write(f"{'Metric':<20} {'Improvement':<12} {'Status':<10}\n")
                        f.write("-" * 42 + "\n")
                        
                        # Key improvements
                        accuracy_imp = improvements.get('accuracy_improvement', 0.0)
                        recall_imp = improvements.get('recall_improvement', 0.0)
                        fnr_reduction = improvements.get('fnr_reduction', 0.0)
                        fpr_change = improvements.get('fpr_change', 0.0)
                        
                        f.write(f"{'Accuracy':<20} {accuracy_imp:<12.3f} {'↑' if accuracy_imp > 0 else '↓' if accuracy_imp < 0 else '=':<10}\n")
                        f.write(f"{'Recall (Spam Det.)':<20} {recall_imp:<12.3f} {'↑' if recall_imp > 0 else '↓' if recall_imp < 0 else '=':<10}\n")
                        f.write(f"{'FNR Reduction':<20} {fnr_reduction:<12.3f} {'✓' if fnr_reduction > 0 else '✗':<10}\n")
                        f.write(f"{'FPR Change':<20} {fpr_change:<12.3f} {'⚠' if fpr_change > 0.02 else '✓':<10}\n\n")
                        
                        # Recommendations
                        if hasattr(self.balancing_metrics, 'balancing_history') and self.balancing_metrics.balancing_history:
                            latest_report = self.balancing_metrics.balancing_history[-1]
                            if latest_report.recommendations:
                                f.write("Recommendations:\n")
                                f.write("-" * 40 + "\n")
                                for i, rec in enumerate(latest_report.recommendations, 1):
                                    f.write(f"{i}. {rec}\n")
                                f.write("\n")
                else:
                    f.write("Balancing Status: DISABLED\n")
                    f.write("No class balancing was applied during training.\n\n")
                
                f.write("=" * 60 + "\n\n")
            
            # Model comparison table
            comparison_df = self.compare_models()
            if not comparison_df.empty:
                f.write("Model Performance Comparison:\n")
                f.write("-" * 40 + "\n")
                f.write(comparison_df.to_string(index=False))
                f.write("\n\n")
            
            # Detailed results for each model
            for model_name, results in evaluation_results.items():
                f.write(f"Detailed Results for {model_name}:\n")
                f.write("-" * 40 + "\n")
                
                # Metrics
                metrics = results['metrics']
                f.write(f"Accuracy:  {metrics['accuracy']:.4f}\n")
                f.write(f"Precision: {metrics['precision']:.4f}\n")
                f.write(f"Recall:    {metrics['recall']:.4f}\n")
                f.write(f"F1-Score:  {metrics['f1_score']:.4f}\n")
                
                # Add false negative rate if available
                if 'false_negative_rate' in metrics:
                    f.write(f"False Negative Rate: {metrics['false_negative_rate']:.4f}\n")
                
                # Add balancing information if available
                model_metrics = self.model_manager.get_model_metrics(model_name)
                if model_metrics and model_metrics.class_balancing_enabled:
                    f.write(f"\nClass Balancing Information:\n")
                    f.write(f"  Method: {model_metrics.balancing_method}\n")
                    f.write(f"  Original Spam Ratio: {model_metrics.original_spam_ratio:.1%}\n")
                    f.write(f"  Balanced Spam Ratio: {model_metrics.balanced_spam_ratio:.1%}\n")
                    f.write(f"  Synthetic Samples: {model_metrics.synthetic_samples_used:,}\n")
                
                f.write("\n")
                
                # Confusion Matrix
                f.write("Confusion Matrix:\n")
                conf_matrix = np.array(results['confusion_matrix'])
                f.write(f"                Predicted\n")
                f.write(f"Actual    Legit  Spam\n")
                f.write(f"Legit     {conf_matrix[0][0]:5d}  {conf_matrix[0][1]:4d}\n")
                f.write(f"Spam      {conf_matrix[1][0]:5d}  {conf_matrix[1][1]:4d}\n\n")
                
                f.write("-" * 60 + "\n\n")
            
            # Best model summary
            best_model_info = self.get_best_model_info()
            f.write("Best Model Summary:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Model: {best_model_info['display_name']}\n")
            f.write(f"Accuracy: {best_model_info['accuracy']:.4f}\n")
            f.write(f"F1-Score: {best_model_info['f1_score']:.4f}\n")
            
            # Add balancing status to best model summary
            best_model_name = best_model_info.get('model_name', '')
            if best_model_name:
                best_metrics = self.model_manager.get_model_metrics(best_model_name)
                if best_metrics and best_metrics.class_balancing_enabled:
                    f.write(f"Class Balancing: ENABLED ({best_metrics.balancing_method})\n")
                    f.write(f"False Negative Rate: {best_metrics.false_negative_rate:.4f}\n")
                else:
                    f.write(f"Class Balancing: DISABLED\n")
            
        self.logger.info(f"Evaluation report saved to {output_path}")
    
    def generate_balancing_impact_report(self, output_path: str = "models/balancing_impact_report.txt") -> None:
        """
        Generate a dedicated class balancing impact analysis report.
        
        Args:
            output_path: Path to save the balancing report
        """
        if not hasattr(self, 'balancing_metrics') or not self.balancing_metrics:
            self.logger.warning("No balancing metrics available for report generation")
            return
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Generate the detailed balancing report
        balancing_report_text = self.balancing_metrics.generate_balancing_report()
        
        with open(output_path, 'w') as f:
            f.write(balancing_report_text)
            
            # Add model-specific balancing information
            f.write("\nMODEL-SPECIFIC BALANCING INFORMATION\n")
            f.write("=" * 60 + "\n\n")
            
            all_metrics = self.model_manager.get_all_metrics()
            for model_name, metrics in all_metrics.items():
                if metrics.class_balancing_enabled:
                    f.write(f"{model_name.upper()} MODEL:\n")
                    f.write("-" * 30 + "\n")
                    f.write(f"Balancing Method: {metrics.balancing_method}\n")
                    f.write(f"Original Spam Ratio: {metrics.original_spam_ratio:.1%}\n")
                    f.write(f"Balanced Spam Ratio: {metrics.balanced_spam_ratio:.1%}\n")
                    f.write(f"Synthetic Samples Used: {metrics.synthetic_samples_used:,}\n")
                    f.write(f"False Negative Rate: {metrics.false_negative_rate:.4f}\n")
                    f.write(f"Training Date: {metrics.training_date.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Add comparison table if multiple models
            if len([m for m in all_metrics.values() if m.class_balancing_enabled]) > 1:
                f.write("BALANCED MODELS COMPARISON\n")
                f.write("-" * 40 + "\n")
                f.write(f"{'Model':<15} {'Accuracy':<10} {'Recall':<10} {'FNR':<10} {'Method':<10}\n")
                f.write("-" * 55 + "\n")
                
                for model_name, metrics in all_metrics.items():
                    if metrics.class_balancing_enabled:
                        f.write(f"{model_name:<15} {metrics.accuracy:<10.3f} {metrics.recall:<10.3f} "
                               f"{metrics.false_negative_rate:<10.3f} {metrics.balancing_method:<10}\n")
                f.write("\n")
        
        self.logger.info(f"Balancing impact report saved to {output_path}")
    
    def save_before_after_comparison(self, 
                                   original_results: Dict[str, Dict[str, Any]], 
                                   balanced_results: Dict[str, Dict[str, Any]], 
                                   output_path: str = "models/before_after_comparison.txt") -> None:
        """
        Save a before/after comparison report showing performance improvements.
        
        Args:
            original_results: Results from unbalanced models
            balanced_results: Results from balanced models
            output_path: Path to save the comparison report
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            f.write("BEFORE/AFTER CLASS BALANCING COMPARISON\n")
            f.write("=" * 60 + "\n\n")
            
            # Summary table
            f.write("PERFORMANCE SUMMARY\n")
            f.write("-" * 40 + "\n")
            f.write(f"{'Model':<15} {'Metric':<12} {'Before':<10} {'After':<10} {'Change':<10}\n")
            f.write("-" * 57 + "\n")
            
            for model_name in original_results.keys():
                if model_name in balanced_results:
                    orig_metrics = original_results[model_name]['metrics']
                    bal_metrics = balanced_results[model_name]['metrics']
                    
                    # Accuracy
                    acc_change = bal_metrics['accuracy'] - orig_metrics['accuracy']
                    f.write(f"{model_name:<15} {'Accuracy':<12} {orig_metrics['accuracy']:<10.3f} "
                           f"{bal_metrics['accuracy']:<10.3f} {acc_change:<+10.3f}\n")
                    
                    # Recall
                    rec_change = bal_metrics['recall'] - orig_metrics['recall']
                    f.write(f"{'':15} {'Recall':<12} {orig_metrics['recall']:<10.3f} "
                           f"{bal_metrics['recall']:<10.3f} {rec_change:<+10.3f}\n")
                    
                    # False Negative Rate (if available)
                    if 'false_negative_rate' in orig_metrics and 'false_negative_rate' in bal_metrics:
                        fnr_change = bal_metrics['false_negative_rate'] - orig_metrics['false_negative_rate']
                        f.write(f"{'':15} {'FNR':<12} {orig_metrics['false_negative_rate']:<10.3f} "
                               f"{bal_metrics['false_negative_rate']:<10.3f} {fnr_change:<+10.3f}\n")
                    
                    f.write("\n")
            
            # Detailed analysis
            f.write("\nDETAILED ANALYSIS\n")
            f.write("-" * 40 + "\n")
            
            for model_name in original_results.keys():
                if model_name in balanced_results:
                    f.write(f"\n{model_name.upper()} MODEL ANALYSIS:\n")
                    f.write("-" * 30 + "\n")
                    
                    orig_metrics = original_results[model_name]['metrics']
                    bal_metrics = balanced_results[model_name]['metrics']
                    
                    # Calculate improvements
                    improvements = {
                        'accuracy': bal_metrics['accuracy'] - orig_metrics['accuracy'],
                        'precision': bal_metrics['precision'] - orig_metrics['precision'],
                        'recall': bal_metrics['recall'] - orig_metrics['recall'],
                        'f1_score': bal_metrics['f1_score'] - orig_metrics['f1_score']
                    }
                    
                    # Analysis
                    f.write(f"Accuracy Change: {improvements['accuracy']:+.3f} ")
                    f.write(f"({'Improved' if improvements['accuracy'] > 0 else 'Decreased' if improvements['accuracy'] < 0 else 'No Change'})\n")
                    
                    f.write(f"Recall Change: {improvements['recall']:+.3f} ")
                    f.write(f"({'Better Spam Detection' if improvements['recall'] > 0 else 'Worse Spam Detection' if improvements['recall'] < 0 else 'No Change'})\n")
                    
                    # Recommendation
                    if improvements['recall'] > 0.01 and improvements['accuracy'] > -0.02:
                        f.write("Recommendation: Use balanced model - improved spam detection with minimal accuracy loss\n")
                    elif improvements['recall'] > 0.02:
                        f.write("Recommendation: Consider balanced model - significant spam detection improvement\n")
                    elif improvements['accuracy'] < -0.05:
                        f.write("Recommendation: Use original model - balancing caused significant accuracy loss\n")
                    else:
                        f.write("Recommendation: Evaluate based on business requirements\n")
        
        self.logger.info(f"Before/after comparison saved to {output_path}")
    
    def _calculate_detailed_metrics(self, y_true, y_pred, model_name: str, test_samples: int) -> ModelMetrics:
        """Calculate detailed evaluation metrics with class balancing information."""
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
        from datetime import datetime
        
        # Calculate basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # Calculate false negative rate
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        
        # Get balancing information
        balancing_enabled = getattr(self, '_balancing_applied', False) or self.balancing_config.enabled
        balancing_method = self.balancing_config.method if balancing_enabled else 'none'
        
        # Get original and balanced spam ratios
        original_spam_ratio = 0.0
        balanced_spam_ratio = 0.0
        synthetic_samples_used = 0
        
        if hasattr(self, 'class_balancer') and self.class_balancer.original_distribution:
            original_total = sum(self.class_balancer.original_distribution.values())
            original_spam_count = self.class_balancer.original_distribution.get('spam', 0)
            original_spam_ratio = original_spam_count / original_total if original_total > 0 else 0.0
            
            if self.class_balancer.balanced_distribution:
                balanced_total = sum(self.class_balancer.balanced_distribution.values())
                balanced_spam_count = self.class_balancer.balanced_distribution.get('spam', 0)
                balanced_spam_ratio = balanced_spam_count / balanced_total if balanced_total > 0 else 0.0
                synthetic_samples_used = balanced_total - original_total
        
        return ModelMetrics(
            model_name=model_name,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            training_date=datetime.now(),
            test_samples=test_samples,
            # Class balancing fields
            class_balancing_enabled=balancing_enabled,
            original_spam_ratio=original_spam_ratio,
            balanced_spam_ratio=balanced_spam_ratio,
            false_negative_rate=false_negative_rate,
            synthetic_samples_used=synthetic_samples_used,
            balancing_method=balancing_method
        )
    
    def _save_preprocessing_pipeline(self) -> None:
        """Save the fitted preprocessing pipeline to disk."""
        import pickle
        
        if not self.is_fitted:
            self.logger.warning("Preprocessing pipeline not fitted, skipping save")
            return
        
        try:
            pipeline_path = os.path.join(self.models_dir, "preprocessing_pipeline.pkl")
            with open(pipeline_path, 'wb') as f:
                pickle.dump(self.preprocessing_pipeline, f)
            
            self.logger.info(f"Preprocessing pipeline saved to {pipeline_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving preprocessing pipeline: {str(e)}")
    
    def _apply_class_weights_to_model(self, sklearn_model, class_weights: Dict[int, float], model_name: str) -> None:
        """
        Apply optimized class weights to sklearn models for better spam detection.
        
        Args:
            sklearn_model: The sklearn model instance
            class_weights: Dictionary mapping class labels to weights
            model_name: Name of the model for logging
        """
        try:
            # Model-specific optimizations for spam detection
            optimized_weights = class_weights.copy()
            
            # Apply model-specific weight adjustments
            if 'random_forest' in model_name.lower():
                # Random Forest needs more aggressive spam weighting
                if 1 in optimized_weights:
                    optimized_weights[1] *= 2.0  # Double spam weight for RF
                self.logger.info(f"Applied Random Forest spam optimization to {model_name}")
            
            elif 'decision_tree' in model_name.lower():
                # Decision Tree also benefits from aggressive spam weighting
                if 1 in optimized_weights:
                    optimized_weights[1] *= 1.8  # 80% increase for DT
                self.logger.info(f"Applied Decision Tree spam optimization to {model_name}")
            
            # Check if model supports class_weight parameter
            if hasattr(sklearn_model, 'class_weight'):
                sklearn_model.class_weight = optimized_weights
                self.logger.info(f"Applied optimized class weights to {model_name}: {optimized_weights}")
            
            # Special handling for specific model types
            elif 'RandomForest' in str(type(sklearn_model)):
                if hasattr(sklearn_model, 'set_params'):
                    sklearn_model.set_params(class_weight=optimized_weights)
                    self.logger.info(f"Applied class weights to RandomForest {model_name}: {optimized_weights}")
            
            elif 'DecisionTree' in str(type(sklearn_model)):
                if hasattr(sklearn_model, 'set_params'):
                    sklearn_model.set_params(class_weight=optimized_weights)
                    self.logger.info(f"Applied class weights to DecisionTree {model_name}: {optimized_weights}")
            
            # For Naive Bayes models, class weights are typically handled differently
            elif 'Naive' in str(type(sklearn_model)) or 'NB' in str(type(sklearn_model)):
                # Naive Bayes doesn't directly support class_weight, but we can handle this
                # by adjusting sample weights during training if the wrapper supports it
                self.logger.info(f"Naive Bayes {model_name} - class weights will be handled via sample weighting if supported")
            
            else:
                self.logger.warning(f"Model {model_name} ({type(sklearn_model)}) may not support class weights")
                
        except Exception as e:
            self.logger.error(f"Error applying class weights to {model_name}: {str(e)}")
    
    def _calculate_sample_weights(self, y_train: np.ndarray, class_weights: Dict[int, float]) -> np.ndarray:
        """
        Calculate sample weights based on class weights.
        
        Args:
            y_train: Training labels
            class_weights: Class weight mapping
            
        Returns:
            Array of sample weights
        """
        sample_weights = np.ones(len(y_train))
        
        for i, label in enumerate(y_train):
            if label in class_weights:
                sample_weights[i] = class_weights[label]
        
        return sample_weights
    
    def evaluate_balanced_performance(self, test_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Compare balanced vs unbalanced model performance with focus on false negative rates.
        
        Args:
            test_data: Test dataset with 'text' and 'label' columns
            
        Returns:
            Dictionary with comparative performance analysis
        """
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before evaluation")
        
        self.logger.info("Starting balanced vs unbalanced performance comparison...")
        
        # Prepare test features
        X_test = test_data['text'].tolist()
        y_test = test_data['label'].tolist()
        X_test_features = self.preprocessing_pipeline.transform(X_test)
        
        comparison_results = {
            'test_samples': len(y_test),
            'test_distribution': pd.Series(y_test).value_counts().to_dict(),
            'balancing_applied': getattr(self, '_balancing_applied', False) or self.balancing_config.enabled,
            'balancing_stats': self.class_balancer.get_balancing_stats() if hasattr(self, 'class_balancer') else {},
            'model_comparisons': {},
            'best_model_selection': {},
            'performance_summary': {}
        }
        
        # Evaluate each model
        for model_name, model in self.model_manager.models.items():
            try:
                if not hasattr(model, 'is_trained') or not model.is_trained:
                    self.logger.warning(f"Model {model_name} is not trained, skipping evaluation")
                    continue
                
                self.logger.info(f"Evaluating {model_name} performance...")
                
                # Make predictions
                predictions = model.predict(X_test_features)
                probabilities = model.predict_proba(X_test_features)
                
                # Calculate detailed metrics with focus on false negatives
                model_metrics = self._calculate_balanced_metrics(y_test, predictions, model_name)
                
                # Store model comparison results
                comparison_results['model_comparisons'][model_name] = {
                    'accuracy': model_metrics['accuracy'],
                    'precision': model_metrics['precision'],
                    'recall': model_metrics['recall'],
                    'f1_score': model_metrics['f1_score'],
                    'false_negative_rate': model_metrics['false_negative_rate'],
                    'false_positive_rate': model_metrics['false_positive_rate'],
                    'true_negatives': model_metrics['true_negatives'],
                    'false_positives': model_metrics['false_positives'],
                    'false_negatives': model_metrics['false_negatives'],
                    'true_positives': model_metrics['true_positives'],
                    'spam_detection_rate': model_metrics['spam_detection_rate'],
                    'confusion_matrix': model_metrics['confusion_matrix']
                }
                
                self.logger.info(f"{model_name} - Accuracy: {model_metrics['accuracy']:.4f}, "
                               f"FNR: {model_metrics['false_negative_rate']:.4f}, "
                               f"Spam Detection: {model_metrics['spam_detection_rate']:.4f}")
                
            except Exception as e:
                self.logger.error(f"Error evaluating {model_name}: {str(e)}")
        
        # Determine best performing model based on balanced criteria
        best_model_info = self._select_best_balanced_model(comparison_results['model_comparisons'])
        comparison_results['best_model_selection'] = best_model_info
        
        # Generate performance summary
        comparison_results['performance_summary'] = self._generate_performance_summary(
            comparison_results['model_comparisons'], 
            comparison_results['balancing_stats']
        )
        
        self.logger.info("Balanced performance evaluation completed")
        return comparison_results
    
    def _calculate_balanced_metrics(self, y_true: List[int], y_pred: List[int], model_name: str) -> Dict[str, Any]:
        """
        Calculate comprehensive metrics with focus on class balancing effectiveness.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            model_name: Name of the model
            
        Returns:
            Dictionary with detailed metrics
        """
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
        
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # Confusion matrix components
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # Calculate rates focusing on spam detection
        false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        spam_detection_rate = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # Same as recall for spam class
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'false_negative_rate': false_negative_rate,
            'false_positive_rate': false_positive_rate,
            'spam_detection_rate': spam_detection_rate,
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp),
            'confusion_matrix': [[int(tn), int(fp)], [int(fn), int(tp)]]
        }
    
    def _select_best_balanced_model(self, model_comparisons: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Select the best performing model based on balanced criteria.
        Prioritizes low false negative rate while maintaining good overall accuracy.
        
        Args:
            model_comparisons: Dictionary of model performance metrics
            
        Returns:
            Dictionary with best model selection information
        """
        if not model_comparisons:
            return {'best_model': None, 'selection_criteria': 'No models available'}
        
        best_model = None
        best_score = -1
        selection_criteria = []
        
        # Define scoring criteria (weighted combination)
        # Priority: Low FNR (40%), High Accuracy (30%), High F1 (20%), High Spam Detection (10%)
        for model_name, metrics in model_comparisons.items():
            # Calculate composite score (higher is better)
            fnr_score = 1.0 - metrics['false_negative_rate']  # Lower FNR is better
            accuracy_score = metrics['accuracy']
            f1_score = metrics['f1_score']
            spam_detection_score = metrics['spam_detection_rate']
            
            composite_score = (
                0.4 * fnr_score +           # 40% weight on low false negative rate
                0.3 * accuracy_score +      # 30% weight on accuracy
                0.2 * f1_score +           # 20% weight on F1 score
                0.1 * spam_detection_score  # 10% weight on spam detection rate
            )
            
            if composite_score > best_score:
                best_score = composite_score
                best_model = model_name
        
        # Generate selection criteria explanation
        if best_model:
            best_metrics = model_comparisons[best_model]
            selection_criteria = [
                f"Lowest false negative rate: {best_metrics['false_negative_rate']:.4f}",
                f"High accuracy: {best_metrics['accuracy']:.4f}",
                f"Good F1 score: {best_metrics['f1_score']:.4f}",
                f"Strong spam detection: {best_metrics['spam_detection_rate']:.4f}"
            ]
        
        return {
            'best_model': best_model,
            'composite_score': best_score,
            'selection_criteria': selection_criteria,
            'metrics': model_comparisons.get(best_model, {}) if best_model else {}
        }
    
    def _generate_performance_summary(self, model_comparisons: Dict[str, Dict[str, Any]], 
                                    balancing_stats: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a comprehensive performance summary.
        
        Args:
            model_comparisons: Model performance metrics
            balancing_stats: Class balancing statistics
            
        Returns:
            Dictionary with performance summary
        """
        if not model_comparisons:
            return {'summary': 'No models evaluated'}
        
        # Calculate aggregate statistics
        all_fnr = [metrics['false_negative_rate'] for metrics in model_comparisons.values()]
        all_accuracy = [metrics['accuracy'] for metrics in model_comparisons.values()]
        all_spam_detection = [metrics['spam_detection_rate'] for metrics in model_comparisons.values()]
        
        summary = {
            'total_models_evaluated': len(model_comparisons),
            'average_false_negative_rate': np.mean(all_fnr),
            'best_false_negative_rate': min(all_fnr),
            'average_accuracy': np.mean(all_accuracy),
            'best_accuracy': max(all_accuracy),
            'average_spam_detection_rate': np.mean(all_spam_detection),
            'best_spam_detection_rate': max(all_spam_detection),
            'target_fnr_achieved': min(all_fnr) < 0.05,  # Target: < 5% FNR
            'balancing_effectiveness': {}
        }
        
        # Add balancing effectiveness analysis
        if balancing_stats.get('balancing_applied', False):
            original_dist = balancing_stats.get('original_distribution', {})
            balanced_dist = balancing_stats.get('balanced_distribution', {})
            
            if original_dist and balanced_dist:
                original_spam_ratio = original_dist.get('spam', 0) / sum(original_dist.values())
                balanced_spam_ratio = balanced_dist.get('spam', 0) / sum(balanced_dist.values())
                
                summary['balancing_effectiveness'] = {
                    'original_spam_ratio': original_spam_ratio,
                    'balanced_spam_ratio': balanced_spam_ratio,
                    'ratio_improvement': balanced_spam_ratio - original_spam_ratio,
                    'synthetic_samples_created': balancing_stats.get('synthetic_samples_created', 0),
                    'method_used': balancing_stats.get('method_used', 'unknown')
                }
        
        return summary
    
    def validate_balancing_effectiveness(self, test_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate the effectiveness of class balancing with comprehensive testing.
        
        Args:
            test_data: Test dataset for validation
            
        Returns:
            Dictionary with comprehensive validation results
        """
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before validation")
        
        self.logger.info("Starting comprehensive balancing effectiveness validation...")
        
        # Prepare test data
        X_test = test_data['text'].tolist()
        y_test = test_data['label'].tolist()
        X_test_features = self.preprocessing_pipeline.transform(X_test)
        
        validation_results = {
            'test_samples': len(y_test),
            'test_distribution': pd.Series(y_test).value_counts().to_dict(),
            'model_validations': {},
            'overall_validation': {},
            'recommendations': []
        }
        
        # Get original (unbalanced) and balanced model results for comparison
        original_results = {}
        balanced_results = {}
        
        for model_name, model in self.model_manager.models.items():
            if not hasattr(model, 'is_trained') or not model.is_trained:
                continue
            
            try:
                # Get predictions
                predictions = model.predict(X_test_features)
                
                # Calculate comprehensive metrics
                model_metrics = self.balancing_validator._calculate_performance_metrics(y_test, predictions)
                
                # Store results based on whether model used balancing
                model_info = self.model_manager.get_model_metrics(model_name)
                if model_info and model_info.class_balancing_enabled:
                    balanced_results[model_name] = model_metrics
                else:
                    original_results[model_name] = model_metrics
                
                self.logger.info(f"Validation metrics for {model_name}: "
                               f"Accuracy: {model_metrics['accuracy']:.4f}, "
                               f"FNR: {model_metrics['fnr']:.4f}")
                
            except Exception as e:
                self.logger.error(f"Error validating {model_name}: {str(e)}")
        
        # Perform comparative validation if we have both types
        if original_results and balanced_results:
            # Use best performing models from each category
            best_original = max(original_results.items(), key=lambda x: x[1]['accuracy'])
            best_balanced = max(balanced_results.items(), key=lambda x: x[1]['accuracy'])
            
            self.logger.info(f"Comparing best original model ({best_original[0]}) "
                           f"vs best balanced model ({best_balanced[0]})")
            
            # Validate effectiveness
            effectiveness_validation = self.balancing_validator.validate_balancing_effectiveness(
                original_results=best_original[1],
                balanced_results=best_balanced[1]
            )
            
            validation_results['overall_validation'] = {
                'validation_passed': effectiveness_validation.overall_validation_passed,
                'fnr_target_met': effectiveness_validation.fnr_target_met,
                'fnr_improvement': effectiveness_validation.fnr_improvement,
                'accuracy_maintained': effectiveness_validation.accuracy_maintained,
                'accuracy_change': effectiveness_validation.accuracy_change,
                'synthetic_quality_score': effectiveness_validation.synthetic_quality_score,
                'recommendations': effectiveness_validation.recommendations
            }
            
            # Validate specific performance improvements
            y_pred_original = self.model_manager.models[best_original[0]].predict(X_test_features)
            y_pred_balanced = self.model_manager.models[best_balanced[0]].predict(X_test_features)
            
            improvement_validation = self.balancing_validator.validate_model_performance_improvement(
                np.array(y_test), y_pred_original, y_pred_balanced
            )
            
            validation_results['performance_improvement'] = improvement_validation
            
            # Check if FNR target is achieved
            fnr_target_achieved = effectiveness_validation.fnr_balanced <= 0.05
            validation_results['fnr_target_achieved'] = fnr_target_achieved
            
            if fnr_target_achieved:
                self.logger.info(f"✓ FNR target achieved: {effectiveness_validation.fnr_balanced:.4f} ≤ 0.05")
            else:
                self.logger.warning(f"✗ FNR target not met: {effectiveness_validation.fnr_balanced:.4f} > 0.05")
            
            # Validate synthetic sample quality if available
            if hasattr(self, '_balanced_features') and self._balanced_features is not None:
                original_features = X_test_features[:len(X_test_features)//2]  # Approximate original samples
                synthetic_features = self._balanced_features[len(y_test):]  # Synthetic samples
                
                if len(synthetic_features) > 0:
                    synthetic_validation = self.balancing_validator.validate_synthetic_sample_realism(
                        synthetic_features, original_features
                    )
                    validation_results['synthetic_validation'] = synthetic_validation
                    
                    if synthetic_validation['validation_passed']:
                        self.logger.info(f"✓ Synthetic sample quality validated: {synthetic_validation['quality_score']:.3f}")
                    else:
                        self.logger.warning(f"✗ Synthetic sample quality issues detected: {synthetic_validation['quality_score']:.3f}")
        
        else:
            self.logger.warning("Cannot perform comparative validation - missing original or balanced model results")
            validation_results['overall_validation'] = {
                'validation_passed': False,
                'error': 'Insufficient models for comparison'
            }
        
        # Generate final recommendations
        validation_results['recommendations'] = self._generate_final_validation_recommendations(validation_results)
        
        self.logger.info("Balancing effectiveness validation completed")
        return validation_results
    
    def _generate_final_validation_recommendations(self, validation_results: Dict[str, Any]) -> List[str]:
        """Generate final recommendations based on validation results."""
        recommendations = []
        
        overall_validation = validation_results.get('overall_validation', {})
        
        if overall_validation.get('validation_passed', False):
            recommendations.append("✓ Class balancing validation PASSED - recommend using balanced models in production")
            
            if validation_results.get('fnr_target_achieved', False):
                recommendations.append("✓ False negative rate target achieved - excellent spam detection performance")
            
            if overall_validation.get('accuracy_maintained', False):
                recommendations.append("✓ Accuracy adequately maintained - good balance between spam detection and overall performance")
        
        else:
            recommendations.append("✗ Class balancing validation FAILED - consider adjustments before production deployment")
            
            if not overall_validation.get('fnr_target_met', True):
                fnr_current = overall_validation.get('fnr_improvement', 0.0)
                recommendations.append(f"• FNR target not met - consider increasing target spam ratio or adjusting SMOTE parameters")
            
            if not overall_validation.get('accuracy_maintained', True):
                acc_change = overall_validation.get('accuracy_change', 0.0)
                recommendations.append(f"• Accuracy loss too significant ({acc_change:+.3f}) - consider reducing synthetic sample ratio")
        
        # Add synthetic sample recommendations if available
        synthetic_validation = validation_results.get('synthetic_validation', {})
        if synthetic_validation and not synthetic_validation.get('validation_passed', True):
            recommendations.append("• Synthetic sample quality issues detected - consider adjusting k-neighbors or preprocessing")
        
        # Add performance improvement recommendations
        perf_improvement = validation_results.get('performance_improvement', {})
        if perf_improvement:
            improvement_validation = perf_improvement.get('validation', {})
            if improvement_validation.get('overall_improvement', False):
                recommendations.append("✓ Overall performance improvement confirmed")
            else:
                recommendations.append("• Mixed performance results - evaluate based on business priorities")
        
        return recommendations
    
    def _setup_monitoring(self) -> None:
        """Setup production monitoring for balancing components."""
        try:
            # Add health checks for balancing components
            self.balancing_monitor.add_health_check(
                "class_balancer",
                create_class_balancer_health_check(self.class_balancer)
            )
            
            # Add health check for SMOTE processor (will be available after balancing)
            def smote_health_check():
                from src.services.balancing_monitor import HealthCheckResult
                from datetime import datetime
                
                if hasattr(self.class_balancer, '_smote_processor'):
                    return create_smote_processor_health_check(self.class_balancer._smote_processor)()
                else:
                    return HealthCheckResult(
                        component="smote_processor",
                        status="healthy",
                        timestamp=datetime.now(),
                        response_time_ms=0.0,
                        details={'status': 'not_initialized'}
                    )
            
            self.balancing_monitor.add_health_check("smote_processor", smote_health_check)
            
            # Add alert handlers
            from src.services.balancing_monitor import log_alert_handler, file_alert_handler
            
            self.balancing_monitor.add_alert_handler(log_alert_handler)
            
            # Add file alert handler if logs directory exists
            if os.path.exists("logs"):
                alert_file_handler = file_alert_handler("logs/balancing_alerts.jsonl")
                self.balancing_monitor.add_alert_handler(alert_file_handler)
            
            self.logger.info("Production monitoring setup completed")
            
        except Exception as e:
            self.logger.error(f"Error setting up monitoring: {str(e)}")
    
    def start_production_monitoring(self) -> None:
        """Start production monitoring for balancing operations."""
        try:
            self.balancing_monitor.start_monitoring()
            self.logger.info("Production monitoring started")
        except Exception as e:
            self.logger.error(f"Failed to start production monitoring: {str(e)}")
    
    def stop_production_monitoring(self) -> None:
        """Stop production monitoring."""
        try:
            self.balancing_monitor.stop_monitoring()
            self.logger.info("Production monitoring stopped")
        except Exception as e:
            self.logger.error(f"Failed to stop production monitoring: {str(e)}")
    
    def get_monitoring_status(self) -> Dict[str, Any]:
        """
        Get current monitoring status and metrics.
        
        Returns:
            Dictionary with monitoring status and performance metrics
        """
        try:
            system_status = self.balancing_monitor.get_system_status()
            performance_trends = self.balancing_monitor.get_performance_trends(hours=24)
            
            return {
                'monitoring_active': system_status.get('monitoring_active', False),
                'overall_health': system_status.get('overall_health', 'unknown'),
                'component_status': system_status.get('component_status', {}),
                'active_alerts': system_status.get('active_alerts_count', 0),
                'recent_performance': system_status.get('recent_performance', {}),
                'performance_trends': performance_trends,
                'balancing_config': {
                    'enabled': self.balancing_config.enabled,
                    'method': self.balancing_config.method,
                    'target_spam_ratio': self.balancing_config.target_spam_ratio
                }
            }
        except Exception as e:
            self.logger.error(f"Error getting monitoring status: {str(e)}")
            return {'error': str(e)}
    
    def export_monitoring_data(self, file_path: str, hours: int = 24) -> None:
        """
        Export monitoring data to file.
        
        Args:
            file_path: Path to save monitoring data
            hours: Number of hours of data to export
        """
        try:
            self.balancing_monitor.export_metrics(file_path, hours)
            self.logger.info(f"Monitoring data exported to {file_path}")
        except Exception as e:
            self.logger.error(f"Failed to export monitoring data: {str(e)}")