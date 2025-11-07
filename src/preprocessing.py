"""
Text preprocessing utilities for email spam classification.

This module provides text cleaning, normalization, and feature extraction
functionality for the email spam classifier system.
"""

import re
import string
from typing import List, Optional
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin


class TextCleaner:
    """Handles text cleaning and normalization operations."""
    
    def __init__(self, remove_stopwords: bool = True):
        """
        Initialize the text cleaner.
        
        Args:
            remove_stopwords: Whether to remove common English stop words
        """
        self.remove_stopwords = remove_stopwords
        # Common English stop words
        self.stopwords = {
            'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 
            'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 
            'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 
            'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 
            'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 
            'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 
            'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 
            'while', 'of', 'at', 'by', 'for', 'with', 'through', 'during', 'before', 
            'after', 'above', 'below', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 
            'under', 'again', 'further', 'then', 'once'
        }
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize email text.
        
        Args:
            text: Raw email text to clean
            
        Returns:
            Cleaned and normalized text
        """
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+', '', text)
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove extra whitespace and newlines
        text = re.sub(r'\s+', ' ', text)
        
        # Remove punctuation except for basic sentence structure
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Remove numbers
        text = re.sub(r'\d+', '', text)
        
        # Remove extra spaces
        text = ' '.join(text.split())
        
        return text.strip()
    
    def remove_stop_words(self, text: str) -> str:
        """
        Remove stop words from text.
        
        Args:
            text: Text to process
            
        Returns:
            Text with stop words removed
        """
        if not self.remove_stopwords:
            return text
            
        words = text.split()
        filtered_words = [word for word in words if word not in self.stopwords and len(word) > 2]
        return ' '.join(filtered_words)
    
    def preprocess(self, text: str) -> str:
        """
        Apply complete preprocessing pipeline to text.
        
        Args:
            text: Raw text to preprocess
            
        Returns:
            Fully preprocessed text
        """
        cleaned = self.clean_text(text)
        return self.remove_stop_words(cleaned)


class EmailVectorizer(BaseEstimator, TransformerMixin):
    """TF-IDF vectorizer specifically configured for email text."""
    
    def __init__(self, 
                 max_features: int = 5000,
                 min_df: int = 2,
                 max_df: float = 0.95,
                 ngram_range: tuple = (1, 2),
                 use_idf: bool = True,
                 sublinear_tf: bool = True):
        """
        Initialize the email vectorizer.
        
        Args:
            max_features: Maximum number of features to extract
            min_df: Minimum document frequency for terms
            max_df: Maximum document frequency for terms
            ngram_range: Range of n-grams to extract
            use_idf: Whether to use inverse document frequency
            sublinear_tf: Whether to apply sublinear tf scaling
        """
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.ngram_range = ngram_range
        self.use_idf = use_idf
        self.sublinear_tf = sublinear_tf
        
        self.text_cleaner = TextCleaner()
        self.vectorizer = None
        self._is_fitted = False
    
    def fit(self, X, y=None):
        """
        Fit the vectorizer to the training data.
        
        Args:
            X: Training text data
            y: Target labels (unused)
            
        Returns:
            self
        """
        # Clean the text data
        cleaned_texts = [self.text_cleaner.preprocess(text) for text in X]
        
        # Initialize and fit the TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            min_df=self.min_df,
            max_df=self.max_df,
            ngram_range=self.ngram_range,
            use_idf=self.use_idf,
            sublinear_tf=self.sublinear_tf,
            stop_words=None  # We handle stop words in preprocessing
        )
        
        self.vectorizer.fit(cleaned_texts)
        self._is_fitted = True
        
        return self
    
    def transform(self, X):
        """
        Transform text data to TF-IDF features.
        
        Args:
            X: Text data to transform
            
        Returns:
            TF-IDF feature matrix as numpy array
        """
        if not self._is_fitted:
            raise ValueError("Vectorizer must be fitted before transform")
        
        # Clean the text data
        cleaned_texts = [self.text_cleaner.preprocess(text) for text in X]
        
        # Transform to TF-IDF features and convert to dense array
        return self.vectorizer.transform(cleaned_texts).toarray()
    
    def fit_transform(self, X, y=None):
        """
        Fit the vectorizer and transform the data.
        
        Args:
            X: Training text data
            y: Target labels (unused)
            
        Returns:
            TF-IDF feature matrix
        """
        return self.fit(X, y).transform(X)
    
    def get_feature_names(self) -> List[str]:
        """
        Get the feature names from the fitted vectorizer.
        
        Returns:
            List of feature names
        """
        if not self._is_fitted:
            raise ValueError("Vectorizer must be fitted before getting feature names")
        
        return self.vectorizer.get_feature_names_out().tolist()


class PreprocessingPipeline:
    """Complete preprocessing pipeline for email spam classification."""
    
    def __init__(self, vectorizer_params: Optional[dict] = None):
        """
        Initialize the preprocessing pipeline.
        
        Args:
            vectorizer_params: Parameters for the TF-IDF vectorizer
        """
        self.vectorizer_params = vectorizer_params or {}
        self.vectorizer = EmailVectorizer(**self.vectorizer_params)
        self._is_fitted = False
    
    def fit(self, texts: List[str], labels: Optional[List[str]] = None):
        """
        Fit the preprocessing pipeline to training data.
        
        Args:
            texts: List of email texts
            labels: List of labels (unused but kept for compatibility)
        """
        self.vectorizer.fit(texts)
        self._is_fitted = True
        return self
    
    def transform(self, texts: List[str]) -> np.ndarray:
        """
        Transform texts to feature vectors.
        
        Args:
            texts: List of email texts to transform
            
        Returns:
            Feature matrix as numpy array
        """
        if not self._is_fitted:
            raise ValueError("Pipeline must be fitted before transform")
        
        result = self.vectorizer.transform(texts)
        # Check if result is already a numpy array or sparse matrix
        if hasattr(result, 'toarray'):
            return result.toarray()
        return result
    
    def fit_transform(self, texts: List[str], labels: Optional[List[str]] = None) -> np.ndarray:
        """
        Fit the pipeline and transform the data.
        
        Args:
            texts: List of email texts
            labels: List of labels (unused but kept for compatibility)
            
        Returns:
            Feature matrix as numpy array
        """
        return self.fit(texts, labels).transform(texts)
    
    def preprocess_single(self, text: str) -> np.ndarray:
        """
        Preprocess a single email text.
        
        Args:
            text: Single email text
            
        Returns:
            Feature vector as numpy array
        """
        return self.transform([text])[0]
    
    def get_feature_names(self) -> List[str]:
        """
        Get feature names from the fitted pipeline.
        
        Returns:
            List of feature names
        """
        return self.vectorizer.get_feature_names()