#!/usr/bin/env python3
"""
Training script for Email Spam Classification models.

This script loads the spam dataset, trains all three ML models (Naive Bayes, 
Random Forest, Decision Tree), evaluates their performance, and saves the 
trained models to disk.
"""

import os
import sys
import logging
import pandas as pd
from pathlib import Path

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.services.training_pipeline import TrainingPipeline
from src.models.model_manager import ModelManager
from src.preprocessing import PreprocessingPipeline
from config import config


def setup_logging():
    """Set up logging for the training process."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('logs/training.log', mode='a')
        ]
    )
    return logging.getLogger(__name__)


def load_spam_dataset(dataset_path: str) -> pd.DataFrame:
    """
    Load and prepare the spam dataset.
    
    Args:
        dataset_path: Path to the spam.csv file
        
    Returns:
        Cleaned DataFrame with 'text' and 'label' columns
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Load the dataset
        logger.info(f"Loading dataset from {dataset_path}")
        df = pd.read_csv(dataset_path, encoding='ISO-8859-1')
        
        # The spam.csv format has columns: v1 (label), v2 (text), and empty columns
        # Rename columns to standard format
        df = df.rename(columns={'v1': 'label', 'v2': 'text'})
        
        # Keep only the relevant columns
        df = df[['label', 'text']].copy()
        
        # Remove any rows with missing values
        df = df.dropna()
        
        # Convert labels to standard format
        df['label'] = df['label'].str.lower()
        
        logger.info(f"Dataset loaded successfully: {len(df)} samples")
        logger.info(f"Label distribution: {df['label'].value_counts().to_dict()}")
        
        return df
        
    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")
        raise


def main():
    """Main training function."""
    logger = setup_logging()
    logger.info("Starting Email Spam Classifier training pipeline")
    
    try:
        # Initialize configuration
        app_config = config['default']()
        
        # Create necessary directories
        os.makedirs('logs', exist_ok=True)
        os.makedirs(app_config.TRAINED_MODELS_DIR, exist_ok=True)
        
        # Find and load the spam dataset
        dataset_paths = [
            'data/mega_combined_spam.csv',  # Mega combined dataset (BEST)
            'data/combined_spam.csv',       # Combined dataset (good)
            'data/spam.csv',                # Original dataset (fallback)
            'oibsip/spam.csv',
            'niladri/spam.csv'
        ]
        
        dataset_path = None
        for path in dataset_paths:
            if os.path.exists(path):
                dataset_path = path
                break
        
        if not dataset_path:
            logger.error("No spam dataset found. Please ensure spam.csv exists in one of the expected locations.")
            return False
        
        # Load the dataset
        df = load_spam_dataset(dataset_path)
        
        if len(df) < 100:
            logger.error("Dataset too small for training. Need at least 100 samples.")
            return False
        
        # Initialize training pipeline
        logger.info("Initializing training pipeline...")
        training_pipeline = TrainingPipeline(
            models_dir=app_config.TRAINED_MODELS_DIR,
            preprocessing_params={
                'max_features': 5000,
                'min_df': 2,
                'max_df': 0.95,
                'ngram_range': (1, 2)
            }
        )
        
        # Prepare data splits with class balancing enabled
        logger.info("Preparing data splits...")
        train_data, val_data, test_data = training_pipeline.prepare_data(
            df, 
            text_column='text',
            label_column='label',
            test_size=0.2,
            validation_size=0.1,
            enable_balancing=True  # Enable class balancing for improved spam detection
        )
        
        # Train all models with class balancing
        logger.info("Training all models...")
        training_results = training_pipeline.train_all_models(
            train_data, 
            val_data, 
            use_class_weights=True  # Apply class weights to sklearn models
        )
        
        logger.info("Training Results:")
        for model_name, accuracy in training_results.items():
            logger.info(f"  {model_name}: {accuracy:.4f}")
        
        # Evaluate models on test set
        logger.info("Evaluating models on test set...")
        evaluation_results = training_pipeline.evaluate_all_models(test_data)
        
        logger.info("Evaluation Results:")
        for model_name, results in evaluation_results.items():
            metrics = results['metrics']
            logger.info(f"  {model_name}:")
            logger.info(f"    Accuracy:  {metrics['accuracy']:.4f}")
            logger.info(f"    Precision: {metrics['precision']:.4f}")
            logger.info(f"    Recall:    {metrics['recall']:.4f}")
            logger.info(f"    F1-Score:  {metrics['f1_score']:.4f}")
        
        # Generate model comparison
        comparison_df = training_pipeline.compare_models()
        logger.info("\nModel Comparison:")
        logger.info(comparison_df.to_string(index=False))
        
        # Get best model info
        best_model_info = training_pipeline.get_best_model_info()
        logger.info(f"\nBest Model: {best_model_info['display_name']}")
        logger.info(f"Best Model Accuracy: {best_model_info['accuracy']:.4f}")
        
        # Save evaluation report
        report_path = os.path.join(app_config.TRAINED_MODELS_DIR, 'evaluation_report.txt')
        training_pipeline.save_evaluation_report(evaluation_results, report_path)
        logger.info(f"Evaluation report saved to: {report_path}")
        
        # Verify models were saved
        model_files = []
        for model_name in ['naive_bayes', 'random_forest', 'decision_tree']:
            model_file = os.path.join(app_config.TRAINED_MODELS_DIR, f"{model_name}.pkl")
            if os.path.exists(model_file):
                model_files.append(model_file)
        
        logger.info(f"Saved {len(model_files)} trained models:")
        for model_file in model_files:
            logger.info(f"  {model_file}")
        
        # Check if minimum accuracy requirement is met
        best_accuracy = best_model_info['accuracy']
        min_required_accuracy = 0.95  # 95% as per requirements
        
        if best_accuracy >= min_required_accuracy:
            logger.info(f"SUCCESS: Training successful! Best model accuracy ({best_accuracy:.4f}) meets requirement (>={min_required_accuracy})")
        else:
            logger.warning(f"WARNING: Best model accuracy ({best_accuracy:.4f}) is below requirement (>={min_required_accuracy})")
        
        logger.info("Training pipeline completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)