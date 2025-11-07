"""
Data models and interfaces for the Email Spam Classifier System.
"""
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, Any
from abc import ABC, abstractmethod


@dataclass
class ClassificationResult:
    """Result of email classification with confidence and metadata."""
    prediction: str  # "Spam" or "Legitimate"
    confidence: float  # 0.0 to 1.0
    model_used: str
    processing_time: float
    timestamp: datetime


@dataclass
class ModelMetrics:
    """Performance metrics for ML models."""
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    training_date: datetime
    test_samples: int
    
    # Class balancing fields
    class_balancing_enabled: bool = False
    original_spam_ratio: float = 0.0
    balanced_spam_ratio: float = 0.0
    false_negative_rate: float = 0.0
    synthetic_samples_used: int = 0
    balancing_method: str = 'none'


@dataclass
class BatchJob:
    """Batch processing job status and metadata."""
    job_id: str
    status: str  # "pending", "processing", "completed", "failed"
    total_emails: int
    processed_emails: int
    created_at: datetime
    completed_at: Optional[datetime] = None
    result_file_path: Optional[str] = None


class MLModelInterface(ABC):
    """Abstract interface for machine learning models."""
    
    @abstractmethod
    def train(self, X_train, y_train) -> None:
        """Train the model with training data."""
        pass
    
    @abstractmethod
    def predict(self, X) -> Any:
        """Make predictions on input data."""
        pass
    
    @abstractmethod
    def predict_proba(self, X) -> Any:
        """Get prediction probabilities."""
        pass
    
    @abstractmethod
    def get_model_name(self) -> str:
        """Return the model name."""
        pass


class PreprocessorInterface(ABC):
    """Abstract interface for text preprocessing."""
    
    @abstractmethod
    def clean_text(self, text: str) -> str:
        """Clean and normalize input text."""
        pass
    
    @abstractmethod
    def extract_features(self, text: str):
        """Extract features from text."""
        pass
    
    @abstractmethod
    def fit_vectorizer(self, texts) -> None:
        """Fit the vectorizer on training texts."""
        pass


class ClassificationServiceInterface(ABC):
    """Abstract interface for classification service."""
    
    @abstractmethod
    def classify_email(self, email_text: str) -> ClassificationResult:
        """Classify a single email."""
        pass
    
    @abstractmethod
    def classify_batch(self, emails) -> list:
        """Classify multiple emails."""
        pass
    
    @abstractmethod
    def get_active_model(self) -> str:
        """Get the currently active model name."""
        pass