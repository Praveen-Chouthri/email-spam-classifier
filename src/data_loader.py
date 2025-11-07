"""
Data loading and validation utilities for email spam classification.

This module provides functionality for loading training data from CSV files
and validating email text and batch file formats.
"""

import os
import csv
import pandas as pd
from typing import List, Tuple, Dict, Optional, Union
from dataclasses import dataclass
import logging


@dataclass
class ValidationResult:
    """Result of data validation operations."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    
    def add_error(self, error: str):
        """Add an error message."""
        self.errors.append(error)
        self.is_valid = False
    
    def add_warning(self, warning: str):
        """Add a warning message."""
        self.warnings.append(warning)


class EmailValidator:
    """Validates email text and format requirements."""
    
    def __init__(self, 
                 min_length: int = 10,
                 max_length: int = 10000,
                 required_encoding: str = 'utf-8'):
        """
        Initialize the email validator.
        
        Args:
            min_length: Minimum acceptable email length
            max_length: Maximum acceptable email length
            required_encoding: Required text encoding
        """
        self.min_length = min_length
        self.max_length = max_length
        self.required_encoding = required_encoding
    
    def validate_email_text(self, text: str) -> ValidationResult:
        """
        Validate a single email text.
        
        Args:
            text: Email text to validate
            
        Returns:
            ValidationResult with validation status and messages
        """
        result = ValidationResult(is_valid=True, errors=[], warnings=[])
        
        # Check if text is None or empty
        if text is None:
            result.add_error("Email text cannot be None")
            return result
        
        if not isinstance(text, str):
            result.add_error(f"Email text must be string, got {type(text)}")
            return result
        
        # Check for empty text first
        stripped_text = text.strip()
        if stripped_text == "":
            result.add_error("Email text cannot be empty or only whitespace")
            return result
        
        # Check text length
        if len(stripped_text) < self.min_length:
            result.add_error(f"Email text too short (minimum {self.min_length} characters)")
        
        if len(text) > self.max_length:
            result.add_error(f"Email text too long (maximum {self.max_length} characters)")
        
        # Warning for very short emails
        if self.min_length <= len(text.strip()) < 50:
            result.add_warning("Email text is very short, classification may be less accurate")
        
        # Check encoding
        try:
            text.encode(self.required_encoding)
        except UnicodeEncodeError:
            result.add_error(f"Email text contains characters not compatible with {self.required_encoding}")
        
        return result
    
    def validate_batch_emails(self, emails: List[str]) -> Tuple[ValidationResult, List[int]]:
        """
        Validate a batch of email texts.
        
        Args:
            emails: List of email texts to validate
            
        Returns:
            Tuple of (overall ValidationResult, list of invalid email indices)
        """
        overall_result = ValidationResult(is_valid=True, errors=[], warnings=[])
        invalid_indices = []
        
        if not emails:
            overall_result.add_error("Email list cannot be empty")
            return overall_result, invalid_indices
        
        if len(emails) > 1000:
            overall_result.add_error("Batch size exceeds maximum limit of 1000 emails")
        
        for i, email in enumerate(emails):
            email_result = self.validate_email_text(email)
            if not email_result.is_valid:
                invalid_indices.append(i)
                for error in email_result.errors:
                    overall_result.add_error(f"Email {i+1}: {error}")
            
            # Collect warnings
            for warning in email_result.warnings:
                overall_result.add_warning(f"Email {i+1}: {warning}")
        
        return overall_result, invalid_indices


class CSVDataLoader:
    """Loads and validates email data from CSV files."""
    
    def __init__(self, 
                 text_column: str = 'text',
                 label_column: str = 'label',
                 encoding: str = 'utf-8'):
        """
        Initialize the CSV data loader.
        
        Args:
            text_column: Name of the column containing email text
            label_column: Name of the column containing labels
            encoding: File encoding to use
        """
        self.text_column = text_column
        self.label_column = label_column
        self.encoding = encoding
        self.validator = EmailValidator()
    
    def validate_csv_file(self, file_path: str) -> ValidationResult:
        """
        Validate CSV file format and structure.
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            ValidationResult with validation status
        """
        result = ValidationResult(is_valid=True, errors=[], warnings=[])
        
        # Check if file exists
        if not os.path.exists(file_path):
            result.add_error(f"File not found: {file_path}")
            return result
        
        # Check file extension
        if not file_path.lower().endswith('.csv'):
            result.add_error("File must have .csv extension")
        
        # Check file size (warn if very large)
        try:
            file_size = os.path.getsize(file_path)
            if file_size > 100 * 1024 * 1024:  # 100MB
                result.add_warning("File is very large (>100MB), processing may be slow")
            elif file_size == 0:
                result.add_error("File is empty")
        except OSError as e:
            result.add_error(f"Cannot access file: {e}")
            return result
        
        # Try to read and validate CSV structure
        try:
            # Read just the header first
            with open(file_path, 'r', encoding=self.encoding, newline='') as f:
                reader = csv.reader(f)
                header = next(reader, None)
                
                if header is None:
                    result.add_error("CSV file has no header row")
                    return result
                
                # Check for required columns
                if self.text_column not in header:
                    result.add_error(f"Required column '{self.text_column}' not found in CSV")
                
                if self.label_column not in header:
                    result.add_error(f"Required column '{self.label_column}' not found in CSV")
                
                # Count rows (sample first few to check format)
                row_count = 0
                for i, row in enumerate(reader):
                    row_count += 1
                    if i < 5:  # Check first 5 rows for format issues
                        if len(row) != len(header):
                            result.add_warning(f"Row {i+2} has {len(row)} columns, expected {len(header)}")
                    
                    if row_count > 50000:  # Stop counting after 50k for performance
                        result.add_warning("File has more than 50,000 rows, only counted first 50,000")
                        break
                
                if row_count == 0:
                    result.add_error("CSV file has no data rows")
                
        except UnicodeDecodeError:
            result.add_error(f"Cannot decode file with {self.encoding} encoding")
        except csv.Error as e:
            result.add_error(f"CSV format error: {e}")
        except Exception as e:
            result.add_error(f"Unexpected error reading file: {e}")
        
        return result
    
    def load_training_data(self, file_path: str) -> Tuple[pd.DataFrame, ValidationResult]:
        """
        Load training data from CSV file.
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            Tuple of (DataFrame with loaded data, ValidationResult)
        """
        # First validate the file
        validation_result = self.validate_csv_file(file_path)
        
        if not validation_result.is_valid:
            return pd.DataFrame(), validation_result
        
        try:
            # Load the CSV file
            df = pd.read_csv(file_path, encoding=self.encoding)
            
            # Validate data content
            self._validate_dataframe_content(df, validation_result)
            
            # Clean up the data
            df = self._clean_dataframe(df)
            
            return df, validation_result
            
        except Exception as e:
            validation_result.add_error(f"Error loading CSV file: {e}")
            return pd.DataFrame(), validation_result
    
    def load_batch_file(self, file_path: str) -> Tuple[List[str], ValidationResult]:
        """
        Load emails from CSV file for batch processing.
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            Tuple of (list of email texts, ValidationResult)
        """
        df, validation_result = self.load_training_data(file_path)
        
        if not validation_result.is_valid or df.empty:
            return [], validation_result
        
        # Extract email texts
        emails = df[self.text_column].tolist()
        
        # Validate the email batch
        batch_validation, invalid_indices = self.validator.validate_batch_emails(emails)
        
        # Merge validation results
        validation_result.errors.extend(batch_validation.errors)
        validation_result.warnings.extend(batch_validation.warnings)
        if not batch_validation.is_valid:
            validation_result.is_valid = False
        
        # Remove invalid emails if any
        if invalid_indices:
            valid_emails = [email for i, email in enumerate(emails) if i not in invalid_indices]
            validation_result.add_warning(f"Removed {len(invalid_indices)} invalid emails from batch")
            return valid_emails, validation_result
        
        return emails, validation_result
    
    def _validate_dataframe_content(self, df: pd.DataFrame, result: ValidationResult):
        """Validate the content of the loaded DataFrame."""
        # Check for missing values
        text_missing = df[self.text_column].isnull().sum()
        label_missing = df[self.label_column].isnull().sum()
        
        if text_missing > 0:
            result.add_warning(f"{text_missing} rows have missing email text")
        
        if label_missing > 0:
            result.add_warning(f"{label_missing} rows have missing labels")
        
        # Check label values
        unique_labels = df[self.label_column].dropna().unique()
        expected_labels = {'spam', 'ham', 'legitimate', 0, 1, '0', '1'}
        
        unexpected_labels = set(unique_labels) - expected_labels
        if unexpected_labels:
            result.add_warning(f"Unexpected label values found: {unexpected_labels}")
        
        # Check for duplicate emails
        duplicates = df.duplicated(subset=[self.text_column]).sum()
        if duplicates > 0:
            result.add_warning(f"Found {duplicates} duplicate email texts")
    
    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean the DataFrame by removing invalid rows."""
        # Remove rows with missing text or labels
        df = df.dropna(subset=[self.text_column, self.label_column])
        
        # Remove duplicate emails
        df = df.drop_duplicates(subset=[self.text_column])
        
        # Normalize labels
        df[self.label_column] = df[self.label_column].apply(self._normalize_label)
        
        return df.reset_index(drop=True)
    
    def _normalize_label(self, label) -> str:
        """Normalize label values to standard format."""
        if pd.isna(label):
            return 'unknown'
        
        label_str = str(label).lower().strip()
        
        if label_str in ['spam', '1', 1]:
            return 'spam'
        elif label_str in ['ham', 'legitimate', '0', 0]:
            return 'legitimate'
        else:
            return 'unknown'


class DataLoader:
    """Main data loading interface for the email spam classifier."""
    
    def __init__(self, 
                 text_column: str = 'text',
                 label_column: str = 'label',
                 encoding: str = 'utf-8'):
        """
        Initialize the data loader.
        
        Args:
            text_column: Name of the column containing email text
            label_column: Name of the column containing labels  
            encoding: File encoding to use
        """
        self.csv_loader = CSVDataLoader(text_column, label_column, encoding)
        self.validator = EmailValidator()
    
    def load_training_data(self, file_path: str) -> Tuple[List[str], List[str], ValidationResult]:
        """
        Load training data and return texts and labels separately.
        
        Args:
            file_path: Path to the training data CSV file
            
        Returns:
            Tuple of (email texts, labels, ValidationResult)
        """
        df, validation_result = self.csv_loader.load_training_data(file_path)
        
        if df.empty:
            return [], [], validation_result
        
        texts = df[self.csv_loader.text_column].tolist()
        labels = df[self.csv_loader.label_column].tolist()
        
        return texts, labels, validation_result
    
    def load_batch_emails(self, file_path: str) -> Tuple[List[str], ValidationResult]:
        """
        Load emails for batch processing.
        
        Args:
            file_path: Path to the batch emails CSV file
            
        Returns:
            Tuple of (email texts, ValidationResult)
        """
        return self.csv_loader.load_batch_file(file_path)
    
    def validate_single_email(self, email_text: str) -> ValidationResult:
        """
        Validate a single email text.
        
        Args:
            email_text: Email text to validate
            
        Returns:
            ValidationResult with validation status
        """
        return self.validator.validate_email_text(email_text)