"""
ML Model Manager for Email Spam Classification System.

This module provides the ModelManager class that handles multiple machine learning
algorithms, model training, evaluation, and persistence.
"""
import os
import pickle
import logging
from typing import Dict, Any, Tuple, List, Optional
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

from src.models.data_models import ModelMetrics, MLModelInterface


class SklearnModelWrapper(MLModelInterface):
    """Wrapper class for sklearn models to implement MLModelInterface."""
    
    def __init__(self, model, model_name: str):
        self.model = model
        self.model_name = model_name
        self.is_trained = False
    
    def train(self, X_train, y_train) -> None:
        """Train the wrapped sklearn model."""
        self.model.fit(X_train, y_train)
        self.is_trained = True
    
    def predict(self, X) -> np.ndarray:
        """Make predictions using the trained model."""
        if not self.is_trained:
            raise ValueError(f"Model {self.model_name} must be trained before making predictions")
        return self.model.predict(X)
    
    def predict_proba(self, X) -> np.ndarray:
        """Get prediction probabilities."""
        if not self.is_trained:
            raise ValueError(f"Model {self.model_name} must be trained before making predictions")
        return self.model.predict_proba(X)
    
    def get_model_name(self) -> str:
        """Return the model name."""
        return self.model_name


class ModelManager:
    """
    Manages multiple ML models for email spam classification.
    
    Supports Naive Bayes, Random Forest, and Decision Tree algorithms
    with training, evaluation, and persistence capabilities.
    """
    
    def __init__(self, models_dir: str = "models/trained"):
        """
        Initialize ModelManager with models directory.
        
        Args:
            models_dir: Directory to save/load trained models
        """
        self.models_dir = models_dir
        self.models: Dict[str, MLModelInterface] = {}
        self.model_metrics: Dict[str, ModelMetrics] = {}
        self.best_model_name: Optional[str] = None
        
        # Ensure models directory exists
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        
        # Initialize default models
        self._initialize_models()
    
    def _initialize_models(self) -> None:
        """Initialize the three ML models with default parameters."""
        # Naive Bayes - good baseline for text classification
        nb_model = MultinomialNB(alpha=1.0)
        self.models["naive_bayes"] = SklearnModelWrapper(nb_model, "Naive Bayes")
        
        # Random Forest - ensemble method for better accuracy
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        self.models["random_forest"] = SklearnModelWrapper(rf_model, "Random Forest")
        
        # Decision Tree - interpretable model
        dt_model = DecisionTreeClassifier(
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        self.models["decision_tree"] = SklearnModelWrapper(dt_model, "Decision Tree")
    
    def load_models(self) -> Dict[str, Any]:
        """
        Load trained models from disk, handling both old and new formats.
        
        Returns:
            Dictionary of loaded models with their names as keys
        """
        loaded_models = {}
        
        for model_name in self.models.keys():
            model_path = os.path.join(self.models_dir, f"{model_name}.pkl")
            metrics_path = os.path.join(self.models_dir, f"{model_name}_metrics.pkl")
            
            try:
                if os.path.exists(model_path):
                    with open(model_path, 'rb') as f:
                        model_data = pickle.load(f)
                    
                    # Handle different model formats for backward compatibility
                    if isinstance(model_data, dict) and 'version' in model_data:
                        # New format (version 2.0+) with metadata
                        self.models[model_name] = model_data['model']
                        loaded_models[model_name] = model_data['model']
                        self.logger.info(f"Loaded model: {model_name} (version {model_data.get('version', 'unknown')})")
                    else:
                        # Legacy format (version 1.0) - direct model object
                        self.models[model_name] = model_data
                        loaded_models[model_name] = model_data
                        self.logger.info(f"Loaded legacy model: {model_name} (migrated to version 2.0)")
                        
                        # Migrate legacy model by re-saving in new format
                        self._migrate_legacy_model(model_name, model_data)
                    
                    # Load metrics if available
                    if os.path.exists(metrics_path):
                        with open(metrics_path, 'rb') as f:
                            metrics = pickle.load(f)
                            
                            # Migrate legacy metrics if needed
                            if not hasattr(metrics, 'class_balancing_enabled'):
                                metrics = self._migrate_legacy_metrics(metrics)
                            
                            self.model_metrics[model_name] = metrics
                    
                else:
                    self.logger.warning(f"Model file not found: {model_path}")
            
            except Exception as e:
                self.logger.error(f"Error loading model {model_name}: {str(e)}")
        
        # Determine best model if metrics are available
        if self.model_metrics:
            self._update_best_model()
        
        return loaded_models
    
    def train_models(self, training_data: pd.DataFrame) -> Dict[str, float]:
        """
        Train all models with the provided training data.
        
        Args:
            training_data: DataFrame with 'text' and 'label' columns
            
        Returns:
            Dictionary of model names and their training accuracies
        """
        if 'text' not in training_data.columns or 'label' not in training_data.columns:
            raise ValueError("Training data must have 'text' and 'label' columns")
        
        # Split data for training and validation
        X = training_data['text']
        y = training_data['label']
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        training_accuracies = {}
        
        for model_name, model in self.models.items():
            try:
                self.logger.info(f"Training {model_name}...")
                
                # Train the model
                model.train(X_train, y_train)
                
                # Calculate training accuracy
                train_predictions = model.predict(X_train)
                training_accuracy = accuracy_score(y_train, train_predictions)
                training_accuracies[model_name] = training_accuracy
                
                # Evaluate on validation set
                val_predictions = model.predict(X_val)
                metrics = self._calculate_metrics(
                    y_val, val_predictions, model_name, len(y_val)
                )
                self.model_metrics[model_name] = metrics
                
                self.logger.info(f"{model_name} training completed. Accuracy: {training_accuracy:.4f}")
                
            except Exception as e:
                self.logger.error(f"Error training {model_name}: {str(e)}")
                training_accuracies[model_name] = 0.0
        
        # Update best model
        self._update_best_model()
        
        # Save trained models
        self._save_all_models()
        
        return training_accuracies
    
    def evaluate_models(self, test_data: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """
        Evaluate all trained models on test data.
        
        Args:
            test_data: DataFrame with 'text' and 'label' columns
            
        Returns:
            Dictionary of model names and their evaluation metrics
        """
        if 'text' not in test_data.columns or 'label' not in test_data.columns:
            raise ValueError("Test data must have 'text' and 'label' columns")
        
        X_test = test_data['text']
        y_test = test_data['label']
        
        evaluation_results = {}
        
        for model_name, model in self.models.items():
            try:
                if not hasattr(model, 'is_trained') or not model.is_trained:
                    self.logger.warning(f"Model {model_name} is not trained, skipping evaluation")
                    continue
                
                # Make predictions
                predictions = model.predict(X_test)
                
                # Calculate metrics
                metrics = self._calculate_metrics(
                    y_test, predictions, model_name, len(y_test)
                )
                
                # Update stored metrics
                self.model_metrics[model_name] = metrics
                
                # Convert to dictionary for return
                evaluation_results[model_name] = {
                    'accuracy': metrics.accuracy,
                    'precision': metrics.precision,
                    'recall': metrics.recall,
                    'f1_score': metrics.f1_score
                }
                
                self.logger.info(f"{model_name} evaluation completed. Accuracy: {metrics.accuracy:.4f}")
                
            except Exception as e:
                self.logger.error(f"Error evaluating {model_name}: {str(e)}")
        
        # Update best model after evaluation
        self._update_best_model()
        
        return evaluation_results
    
    def get_best_model(self) -> Tuple[str, MLModelInterface]:
        """
        Get the best performing model based on accuracy.
        
        Returns:
            Tuple of (model_name, model_instance)
        """
        if not self.best_model_name or self.best_model_name not in self.models:
            # Intelligent fallback: try to select best available model
            if self.models:
                # If we have models but no metrics, trigger selection
                if self.model_metrics:
                    self._update_best_model()
                else:
                    # Fallback order: decision_tree > random_forest > naive_bayes
                    fallback_order = ["decision_tree", "random_forest", "naive_bayes"]
                    for model_name in fallback_order:
                        if model_name in self.models:
                            self.best_model_name = model_name
                            self.logger.info(f"Fallback model selected: {model_name}")
                            break
            else:
                self.logger.warning("No models available for selection")
        
        return self.best_model_name, self.models[self.best_model_name]
    
    def get_model(self, model_name: str) -> Optional[MLModelInterface]:
        """
        Get a specific model by name.
        
        Args:
            model_name: Name of the model to retrieve
            
        Returns:
            Model instance or None if not found
        """
        return self.models.get(model_name)
    
    def get_model_metrics(self, model_name: str) -> Optional[ModelMetrics]:
        """
        Get metrics for a specific model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            ModelMetrics instance or None if not found
        """
        return self.model_metrics.get(model_name)
    
    def get_all_metrics(self) -> Dict[str, ModelMetrics]:
        """Get metrics for all models."""
        return self.model_metrics.copy()
    
    def force_best_model_update(self) -> None:
        """Force recalculation of the best model using composite scoring."""
        self.logger.info("Forcing best model recalculation with composite scoring...")
        self._update_best_model()
    
    def is_model_balanced(self, model_name: str) -> bool:
        """
        Check if a specific model was trained with class balancing.
        
        Args:
            model_name: Name of the model to check
            
        Returns:
            True if model was trained with balancing, False otherwise
        """
        metrics = self.get_model_metrics(model_name)
        return metrics.class_balancing_enabled if metrics else False
    
    def get_model_balancing_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Get class balancing information for a specific model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dictionary with balancing information or None if not found
        """
        metrics = self.get_model_metrics(model_name)
        if not metrics:
            return None
        
        return {
            'class_balancing_enabled': metrics.class_balancing_enabled,
            'original_spam_ratio': metrics.original_spam_ratio,
            'balanced_spam_ratio': metrics.balanced_spam_ratio,
            'false_negative_rate': metrics.false_negative_rate,
            'synthetic_samples_used': metrics.synthetic_samples_used,
            'balancing_method': metrics.balancing_method
        }
    
    def get_balanced_models(self) -> Dict[str, ModelMetrics]:
        """
        Get all models that were trained with class balancing.
        
        Returns:
            Dictionary of balanced model names and their metrics
        """
        balanced_models = {}
        for model_name, metrics in self.model_metrics.items():
            if metrics.class_balancing_enabled:
                balanced_models[model_name] = metrics
        return balanced_models
    
    def get_unbalanced_models(self) -> Dict[str, ModelMetrics]:
        """
        Get all models that were trained without class balancing.
        
        Returns:
            Dictionary of unbalanced model names and their metrics
        """
        unbalanced_models = {}
        for model_name, metrics in self.model_metrics.items():
            if not metrics.class_balancing_enabled:
                unbalanced_models[model_name] = metrics
        return unbalanced_models
    
    def update_model_metrics_with_balancing(self, model_name: str, 
                                          balancing_metadata: Dict[str, Any]) -> None:
        """
        Update existing model metrics with class balancing information.
        
        Args:
            model_name: Name of the model to update
            balancing_metadata: Dictionary containing balancing information
        """
        if model_name not in self.model_metrics:
            self.logger.warning(f"No metrics found for model {model_name}")
            return
        
        current_metrics = self.model_metrics[model_name]
        
        # Create updated metrics with balancing information
        updated_metrics = ModelMetrics(
            model_name=current_metrics.model_name,
            accuracy=current_metrics.accuracy,
            precision=current_metrics.precision,
            recall=current_metrics.recall,
            f1_score=current_metrics.f1_score,
            training_date=current_metrics.training_date,
            test_samples=current_metrics.test_samples,
            class_balancing_enabled=balancing_metadata.get('class_balancing_enabled', False),
            original_spam_ratio=balancing_metadata.get('original_spam_ratio', 0.0),
            balanced_spam_ratio=balancing_metadata.get('balanced_spam_ratio', 0.0),
            false_negative_rate=balancing_metadata.get('false_negative_rate', current_metrics.false_negative_rate),
            synthetic_samples_used=balancing_metadata.get('synthetic_samples_used', 0),
            balancing_method=balancing_metadata.get('balancing_method', 'none')
        )
        
        self.model_metrics[model_name] = updated_metrics
        self.logger.info(f"Updated {model_name} metrics with balancing information")
    
    def calculate_metrics_with_balancing(self, y_true, y_pred, model_name: str, 
                                       test_samples: int, 
                                       balancing_metadata: Optional[Dict[str, Any]] = None) -> ModelMetrics:
        """
        Calculate evaluation metrics for a model with optional balancing metadata.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            model_name: Name of the model
            test_samples: Number of test samples
            balancing_metadata: Optional balancing information
            
        Returns:
            ModelMetrics with balancing information
        """
        return self._calculate_metrics(y_true, y_pred, model_name, test_samples, balancing_metadata)
    
    def _calculate_metrics(self, y_true, y_pred, model_name: str, test_samples: int, 
                         balancing_metadata: Optional[Dict[str, Any]] = None) -> ModelMetrics:
        """Calculate evaluation metrics for a model."""
        # Calculate confusion matrix components for false negative rate
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Calculate false negative rate (spam classified as legitimate)
        # Assuming label 1 is spam, label 0 is legitimate
        false_negative_rate = 0.0
        if len(cm) > 1 and cm[1, 1] + cm[1, 0] > 0:  # If there are spam samples
            false_negative_rate = cm[1, 0] / (cm[1, 1] + cm[1, 0])
        
        # Extract balancing metadata if provided
        balancing_enabled = False
        original_spam_ratio = 0.0
        balanced_spam_ratio = 0.0
        synthetic_samples_used = 0
        balancing_method = 'none'
        
        if balancing_metadata:
            balancing_enabled = balancing_metadata.get('class_balancing_enabled', False)
            original_spam_ratio = balancing_metadata.get('original_spam_ratio', 0.0)
            balanced_spam_ratio = balancing_metadata.get('balanced_spam_ratio', 0.0)
            synthetic_samples_used = balancing_metadata.get('synthetic_samples_used', 0)
            balancing_method = balancing_metadata.get('balancing_method', 'none')
        
        return ModelMetrics(
            model_name=model_name,
            accuracy=accuracy_score(y_true, y_pred),
            precision=precision_score(y_true, y_pred, average='weighted', zero_division=0),
            recall=recall_score(y_true, y_pred, average='weighted', zero_division=0),
            f1_score=f1_score(y_true, y_pred, average='weighted', zero_division=0),
            training_date=datetime.now(),
            test_samples=test_samples,
            class_balancing_enabled=balancing_enabled,
            original_spam_ratio=original_spam_ratio,
            balanced_spam_ratio=balanced_spam_ratio,
            false_negative_rate=false_negative_rate,
            synthetic_samples_used=synthetic_samples_used,
            balancing_method=balancing_method
        )
    
    def _update_best_model(self) -> None:
        """Update the best model based on weighted composite scoring that prioritizes spam detection."""
        if not self.model_metrics:
            return
        
        best_score = -1.0
        best_model = None
        best_accuracy = 0.0
        
        self.logger.info("Evaluating models using composite scoring (FNR: 40%, Accuracy: 30%, F1: 20%, Recall: 10%)")
        
        for model_name, metrics in self.model_metrics.items():
            # Calculate false negative rate (lower is better)
            false_negative_rate = 1.0 - metrics.recall if metrics.recall > 0 else 1.0
            
            # Calculate composite score (higher is better)
            fnr_score = 1.0 - false_negative_rate  # Convert to positive score
            accuracy_score = metrics.accuracy
            f1_score = metrics.f1_score
            recall_score = metrics.recall  # Spam detection rate
            
            composite_score = (
                0.4 * fnr_score +      # 40% weight on low false negative rate (critical for spam detection)
                0.3 * accuracy_score + # 30% weight on overall accuracy
                0.2 * f1_score +       # 20% weight on F1 score (balance of precision/recall)
                0.1 * recall_score     # 10% weight on recall (spam detection rate)
            )
            
            self.logger.info(f"{model_name}: FNR={false_negative_rate:.4f}, Acc={accuracy_score:.4f}, "
                           f"F1={f1_score:.4f}, Recall={recall_score:.4f}, Composite={composite_score:.4f}")
            
            if composite_score > best_score:
                best_score = composite_score
                best_model = model_name
                best_accuracy = metrics.accuracy
        
        self.best_model_name = best_model
        if best_model:
            best_metrics = self.model_metrics[best_model]
            fnr = 1.0 - best_metrics.recall if best_metrics.recall > 0 else 1.0
            self.logger.info(f"Best model selected: {best_model}")
            self.logger.info(f"  Composite Score: {best_score:.4f}")
            self.logger.info(f"  Accuracy: {best_accuracy:.4f}")
            self.logger.info(f"  False Negative Rate: {fnr:.4f}")
            self.logger.info(f"  F1 Score: {best_metrics.f1_score:.4f}")
            self.logger.info(f"  Recall (Spam Detection): {best_metrics.recall:.4f}")
    
    def _save_all_models(self) -> None:
        """Save all trained models and their metrics to disk."""
        for model_name, model in self.models.items():
            if hasattr(model, 'is_trained') and model.is_trained:
                self._save_model(model_name, model)
    
    def train_models_with_features(self, X_features: np.ndarray, y_labels: List[str]) -> Dict[str, float]:
        """
        Train models with pre-extracted features.
        
        Args:
            X_features: Pre-extracted feature matrix
            y_labels: Corresponding labels
            
        Returns:
            Dictionary of model names and their training accuracies
        """
        # Split data for training and validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_features, y_labels, test_size=0.2, random_state=42, stratify=y_labels
        )
        
        training_accuracies = {}
        
        for model_name, model in self.models.items():
            try:
                self.logger.info(f"Training {model_name}...")
                
                # Train the model
                model.train(X_train, y_train)
                
                # Calculate training accuracy
                train_predictions = model.predict(X_train)
                training_accuracy = accuracy_score(y_train, train_predictions)
                training_accuracies[model_name] = training_accuracy
                
                # Evaluate on validation set
                val_predictions = model.predict(X_val)
                metrics = self._calculate_metrics(
                    y_val, val_predictions, model_name, len(y_val)
                )
                self.model_metrics[model_name] = metrics
                
                self.logger.info(f"{model_name} training completed. Accuracy: {training_accuracy:.4f}")
                
            except Exception as e:
                self.logger.error(f"Error training {model_name}: {str(e)}")
                training_accuracies[model_name] = 0.0
        
        # Update best model
        self._update_best_model()
        
        # Save trained models
        self._save_all_models()
        
        return training_accuracies
    
    def _save_model(self, model_name: str, model: MLModelInterface) -> None:
        """Save a single model and its metrics to disk with version information."""
        try:
            # Create model data package with version info for backward compatibility
            model_data = {
                'model': model,
                'version': '2.0',  # Version 2.0 supports balancing metadata
                'saved_date': datetime.now(),
                'model_name': model_name
            }
            
            # Save model with version information
            model_path = os.path.join(self.models_dir, f"{model_name}.pkl")
            with open(model_path, 'wb') as f:
                pickle.dump(model_data, f)
            
            # Save metrics if available
            if model_name in self.model_metrics:
                metrics_path = os.path.join(self.models_dir, f"{model_name}_metrics.pkl")
                with open(metrics_path, 'wb') as f:
                    pickle.dump(self.model_metrics[model_name], f)
            
            self.logger.info(f"Saved model: {model_name} (version 2.0 with balancing support)")
            
        except Exception as e:
            self.logger.error(f"Error saving model {model_name}: {str(e)}")
    
    def _migrate_legacy_model(self, model_name: str, model: MLModelInterface) -> None:
        """
        Migrate a legacy model to the new format with version information.
        
        Args:
            model_name: Name of the model to migrate
            model: The legacy model object
        """
        try:
            self.logger.info(f"Migrating legacy model {model_name} to version 2.0")
            self._save_model(model_name, model)
        except Exception as e:
            self.logger.error(f"Error migrating legacy model {model_name}: {str(e)}")
    
    def _migrate_legacy_metrics(self, legacy_metrics) -> ModelMetrics:
        """
        Migrate legacy metrics to include balancing fields.
        
        Args:
            legacy_metrics: Legacy ModelMetrics object without balancing fields
            
        Returns:
            Updated ModelMetrics with balancing fields
        """
        # Create new metrics with balancing fields set to defaults
        return ModelMetrics(
            model_name=getattr(legacy_metrics, 'model_name', 'unknown'),
            accuracy=getattr(legacy_metrics, 'accuracy', 0.0),
            precision=getattr(legacy_metrics, 'precision', 0.0),
            recall=getattr(legacy_metrics, 'recall', 0.0),
            f1_score=getattr(legacy_metrics, 'f1_score', 0.0),
            training_date=getattr(legacy_metrics, 'training_date', datetime.now()),
            test_samples=getattr(legacy_metrics, 'test_samples', 0),
            class_balancing_enabled=False,  # Legacy models didn't have balancing
            original_spam_ratio=0.0,
            balanced_spam_ratio=0.0,
            false_negative_rate=getattr(legacy_metrics, 'false_negative_rate', 0.0),
            synthetic_samples_used=0,
            balancing_method='none'
        )
    
    def check_model_compatibility(self) -> Dict[str, str]:
        """
        Check compatibility status of all saved models.
        
        Returns:
            Dictionary mapping model names to their version status
        """
        compatibility_status = {}
        
        for model_name in self.models.keys():
            model_path = os.path.join(self.models_dir, f"{model_name}.pkl")
            
            if not os.path.exists(model_path):
                compatibility_status[model_name] = "not_found"
                continue
            
            try:
                with open(model_path, 'rb') as f:
                    model_data = pickle.load(f)
                
                if isinstance(model_data, dict) and 'version' in model_data:
                    compatibility_status[model_name] = f"version_{model_data['version']}"
                else:
                    compatibility_status[model_name] = "legacy_v1.0"
                    
            except Exception as e:
                compatibility_status[model_name] = f"error: {str(e)}"
        
        return compatibility_status
    
    def upgrade_legacy_models(self) -> Dict[str, bool]:
        """
        Upgrade all legacy models to the current format.
        
        Returns:
            Dictionary mapping model names to upgrade success status
        """
        upgrade_results = {}
        compatibility_status = self.check_model_compatibility()
        
        for model_name, status in compatibility_status.items():
            if status == "legacy_v1.0":
                try:
                    # Model is already loaded in memory, just re-save in new format
                    if model_name in self.models:
                        self._save_model(model_name, self.models[model_name])
                        upgrade_results[model_name] = True
                        self.logger.info(f"Successfully upgraded {model_name} to version 2.0")
                    else:
                        upgrade_results[model_name] = False
                        self.logger.warning(f"Model {model_name} not loaded in memory, skipping upgrade")
                except Exception as e:
                    upgrade_results[model_name] = False
                    self.logger.error(f"Failed to upgrade {model_name}: {str(e)}")
            else:
                upgrade_results[model_name] = True  # Already compatible or not applicable
        
        return upgrade_results
    
    def get_model_version_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Get version information for a specific model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dictionary with version information or None if not found
        """
        model_path = os.path.join(self.models_dir, f"{model_name}.pkl")
        
        if not os.path.exists(model_path):
            return None
        
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            if isinstance(model_data, dict) and 'version' in model_data:
                return {
                    'version': model_data.get('version', 'unknown'),
                    'saved_date': model_data.get('saved_date'),
                    'model_name': model_data.get('model_name', model_name),
                    'format': 'new'
                }
            else:
                return {
                    'version': '1.0',
                    'saved_date': None,
                    'model_name': model_name,
                    'format': 'legacy'
                }
                
        except Exception as e:
            self.logger.error(f"Error reading version info for {model_name}: {str(e)}")
            return None