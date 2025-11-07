"""
Flexible CSV processor for handling multiple CSV formats in batch processing.

This module provides functionality for automatically detecting and processing
CSV files with different column structures and naming conventions.
"""

import os
import csv
import pandas as pd
from typing import List, Tuple, Dict, Optional, Set
from dataclasses import dataclass
import logging
import re


@dataclass
class CSVFormat:
    """Information about detected CSV format."""
    email_column: Optional[str]
    label_column: Optional[str]
    column_count: int
    has_header: bool
    detected_encoding: str
    format_type: str  # "batch_only", "training_data", "unknown"
    confidence_score: float


@dataclass
class ValidationResult:
    """Result of CSV validation operations."""
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


class FlexibleCSVProcessor:
    """
    Flexible CSV processor that can handle multiple CSV formats for batch processing.
    
    Supports:
    - Single column CSV files with email content
    - Multi-column CSV files with various naming conventions
    - Automatic column detection and format recognition
    """
    
    # Common column names for email content (expanded for better detection)
    EMAIL_COLUMN_NAMES = {
        'text', 'email_content', 'message', 'content', 'email', 
        'email_text', 'body', 'mail_content', 'text_content',
        'mail_body', 'email_body', 'msg', 'msg_text', 'mail_text',
        'description', 'comment', 'review', 'feedback', 'note',
        'v1', 'v2', 'col1', 'column1', 'data', 'input'  # Legacy and generic formats
    }
    
    # Common column names for labels (expanded for better detection)
    LABEL_COLUMN_NAMES = {
        'label', 'class', 'category', 'spam', 'classification',
        'target', 'y', 'labels', 'spam_label', 'class_label',
        'type', 'result', 'output', 'prediction', 'ground_truth',
        'is_spam', 'spam_flag', 'ham_spam', 'verdict'
    }
    
    def __init__(self, encoding: str = 'utf-8'):
        """
        Initialize the flexible CSV processor.
        
        Args:
            encoding: Default file encoding to use
        """
        self.encoding = encoding
        self.logger = logging.getLogger(__name__)
    
    def detect_csv_format(self, file_path: str) -> CSVFormat:
        """
        Detect the format of a CSV file.
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            CSVFormat object with detected format information
        """
        format_info = CSVFormat(
            email_column=None,
            label_column=None,
            column_count=0,
            has_header=True,
            detected_encoding=self.encoding,
            format_type="unknown",
            confidence_score=0.0
        )
        
        if not os.path.exists(file_path):
            return format_info
        
        try:
            # Try to detect encoding
            format_info.detected_encoding = self._detect_encoding(file_path)
            
            # Read the CSV file
            with open(file_path, 'r', encoding=format_info.detected_encoding, newline='') as f:
                # Peek at first few lines to determine structure
                sample_lines = []
                for i, line in enumerate(f):
                    sample_lines.append(line.strip())
                    if i >= 10:  # Read first 10 lines for analysis
                        break
                
                if not sample_lines:
                    return format_info
                
                # Reset file pointer
                f.seek(0)
                reader = csv.reader(f)
                
                # Get header row
                try:
                    header = next(reader)
                    format_info.column_count = len(header)
                    
                    # Check if first row looks like a header
                    format_info.has_header = self._is_header_row(header, sample_lines)
                    
                    if format_info.has_header:
                        # Detect columns based on header names
                        format_info.email_column = self._detect_email_column(header)
                        format_info.label_column = self._detect_label_column(header)
                    else:
                        # No header, assume first column is email content
                        if format_info.column_count >= 1:
                            format_info.email_column = 0  # Use index for headerless files
                        if format_info.column_count >= 2:
                            format_info.label_column = 1
                    
                    # Determine format type and confidence
                    format_info.format_type, format_info.confidence_score = self._determine_format_type(
                        format_info, sample_lines
                    )
                    
                except StopIteration:
                    # Empty file
                    pass
                    
        except Exception as e:
            self.logger.error(f"Error detecting CSV format: {e}")
        
        return format_info
    
    def load_emails_from_csv(self, file_path: str) -> Tuple[List[str], ValidationResult]:
        """
        Load emails from a CSV file with automatic format detection.
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            Tuple of (list of email texts, ValidationResult)
        """
        validation_result = ValidationResult(is_valid=True, errors=[], warnings=[])
        
        # Detect format first
        csv_format = self.detect_csv_format(file_path)
        
        if csv_format.format_type == "unknown" or csv_format.confidence_score < 0.5:
            validation_result.add_error(
                f"Cannot automatically detect CSV format. "
                f"Confidence: {csv_format.confidence_score:.2f}"
            )
            return [], validation_result
        
        if not csv_format.email_column:
            validation_result.add_error("No email content column detected")
            return [], validation_result
        
        try:
            # Load the CSV file with enhanced error handling for inconsistent columns
            try:
                if csv_format.has_header:
                    df = pd.read_csv(file_path, encoding=csv_format.detected_encoding)
                    email_column = csv_format.email_column
                    
                    # Remove empty columns automatically
                    original_columns = list(df.columns)
                    empty_columns = [col for col in df.columns if not col.strip()]
                    if empty_columns:
                        df = df.drop(columns=empty_columns)
                        validation_result.add_warning(f"Automatically removed {len(empty_columns)} empty columns")
                    
                    # Handle duplicate column names by renaming them
                    if len(df.columns) != len(set(df.columns)):
                        # Rename duplicate columns
                        new_columns = []
                        seen = {}
                        for col in df.columns:
                            if col in seen:
                                seen[col] += 1
                                new_columns.append(f"{col}_{seen[col]}")
                            else:
                                seen[col] = 0
                                new_columns.append(col)
                        df.columns = new_columns
                        validation_result.add_warning("Automatically renamed duplicate column names")
                    
                    # Handle BOM (Byte Order Mark) in column names
                    if email_column not in df.columns:
                        # Remove BOM from email_column if it exists
                        if email_column.startswith('\ufeff'):
                            clean_email_column = email_column.replace('\ufeff', '')
                            if clean_email_column in df.columns:
                                email_column = clean_email_column
                                validation_result.add_warning("CSV file contains BOM (Byte Order Mark), handled automatically")
                            else:
                                # Try to find column by partial match (case insensitive)
                                matching_cols = [col for col in df.columns if clean_email_column.lower() in col.lower()]
                                if matching_cols:
                                    email_column = matching_cols[0]
                                    validation_result.add_warning(f"Using column '{email_column}' as closest match")
                        else:
                            # Try to find column by partial match (case insensitive)
                            matching_cols = [col for col in df.columns if email_column.lower() in col.lower()]
                            if matching_cols:
                                email_column = matching_cols[0]
                                validation_result.add_warning(f"Using column '{email_column}' as closest match")
                else:
                    df = pd.read_csv(file_path, encoding=csv_format.detected_encoding, header=None)
                    email_column = csv_format.email_column  # This will be an index
                    
                    # For headerless files, remove empty columns (columns that are all empty)
                    empty_cols = [col for col in df.columns if df[col].isna().all() or (df[col] == '').all()]
                    if empty_cols:
                        df = df.drop(columns=empty_cols)
                        validation_result.add_warning(f"Automatically removed {len(empty_cols)} empty columns from headerless CSV")
            except pd.errors.ParserError as pe:
                # Try to handle inconsistent column counts by using manual CSV parsing
                self.logger.warning(f"Pandas failed to parse CSV, trying manual parsing: {pe}")
                return self._load_emails_manual_parsing(file_path, csv_format, validation_result)
            
            # Extract emails
            if email_column in df.columns:
                emails = df[email_column].dropna().astype(str).tolist()
            else:
                validation_result.add_error(f"Column '{email_column}' not found in CSV")
                return [], validation_result
            
            # Validate emails
            emails = self._validate_and_clean_emails(emails, validation_result)
            
            validation_result.add_warning(
                f"Detected format: {csv_format.format_type} "
                f"(confidence: {csv_format.confidence_score:.2f})"
            )
            
            return emails, validation_result
            
        except Exception as e:
            validation_result.add_error(f"Error loading CSV file: {e}")
            return [], validation_result
    
    def validate_csv_structure(self, file_path: str) -> ValidationResult:
        """
        Validate CSV file structure without loading all data.
        
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
        
        # Check file size
        try:
            file_size = os.path.getsize(file_path)
            if file_size > 100 * 1024 * 1024:  # 100MB
                result.add_warning("File is very large (>100MB), processing may be slow")
            elif file_size == 0:
                result.add_error("File is empty")
                return result
        except OSError as e:
            result.add_error(f"Cannot access file: {e}")
            return result
        
        # Detect format with enhanced validation
        csv_format = self.detect_csv_format(file_path)
        
        # Enhanced format validation with detailed error messages
        if csv_format.format_type == "unknown":
            result.add_error("Cannot automatically detect CSV format")
            # Add specific suggestions based on what we found
            suggestion = self.suggest_format_fix(csv_format)
            result.add_error(f"Format suggestions: {suggestion}")
        elif csv_format.confidence_score < 0.3:
            result.add_error(f"Very low confidence in format detection ({csv_format.confidence_score:.2f})")
            suggestion = self.suggest_format_fix(csv_format)
            result.add_error(f"Format suggestions: {suggestion}")
        elif csv_format.confidence_score < 0.7:
            result.add_warning(f"Moderate confidence in format detection ({csv_format.confidence_score:.2f})")
            suggestion = self.suggest_format_fix(csv_format)
            result.add_warning(f"Consider: {suggestion}")
        
        if not csv_format.email_column:
            result.add_error("No email content column detected in CSV file")
            result.add_error("Ensure your CSV has a column with email text using one of these names: " + 
                           ", ".join(sorted(list(self.EMAIL_COLUMN_NAMES)[:8])) + "...")
        
        # Additional structural validation
        self._validate_csv_content_structure(file_path, csv_format, result)
        
        return result
    
    def suggest_format_fix(self, detected_format: CSVFormat) -> str:
        """
        Suggest how to fix CSV format issues with detailed recommendations.
        
        Args:
            detected_format: The detected CSV format
            
        Returns:
            String with format suggestions
        """
        suggestions = []
        
        if detected_format.format_type == "unknown":
            suggestions.append("Ensure your CSV has a clear header row with column names")
            suggestions.append(f"Use one of these column names for email content: {', '.join(sorted(list(self.EMAIL_COLUMN_NAMES)[:6]))}")
            if detected_format.column_count == 0:
                suggestions.append("File appears to be empty or corrupted")
            elif detected_format.column_count == 1:
                suggestions.append("Single column detected - name it 'email_content' or 'text'")
        
        if not detected_format.email_column:
            if detected_format.column_count > 1:
                suggestions.append("Multiple columns detected but none recognized as email content")
                suggestions.append("Rename your email column to: 'email_content', 'text', 'message', or 'content'")
            else:
                suggestions.append("Make sure you have a column containing email text")
                suggestions.append("For single column files, add a header row with 'email_content'")
        
        if detected_format.confidence_score < 0.3:
            suggestions.append("Very low format confidence - check file structure")
            if not detected_format.has_header:
                suggestions.append("Add a header row with column names")
            suggestions.append("Ensure data rows contain actual email text (not just numbers or short codes)")
        elif detected_format.confidence_score < 0.7:
            suggestions.append("Consider using standard column names for better detection")
            suggestions.append("Recommended: 'email_content' for text, 'label' for classification")
        
        # Encoding suggestions
        if detected_format.detected_encoding != 'utf-8':
            suggestions.append(f"File encoding detected as {detected_format.detected_encoding} - consider saving as UTF-8")
        
        # Format-specific suggestions
        if detected_format.format_type == "batch_only":
            suggestions.append("Format detected as batch processing (email content only)")
        elif detected_format.format_type == "training_data":
            suggestions.append("Format detected as training data (email content + labels)")
        
        if not suggestions:
            suggestions.append("CSV format looks good!")
        
        return " | ".join(suggestions)
    
    def _detect_encoding(self, file_path: str) -> str:
        """Detect file encoding."""
        encodings_to_try = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']
        
        for encoding in encodings_to_try:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    f.read(1024)  # Try to read first 1KB
                return encoding
            except UnicodeDecodeError:
                continue
        
        return 'utf-8'  # Default fallback
    
    def _is_header_row(self, first_row: List[str], sample_lines: List[str]) -> bool:
        """
        Determine if the first row is a header row.
        
        Args:
            first_row: First row of the CSV
            sample_lines: Sample lines from the file
            
        Returns:
            True if first row appears to be a header
        """
        if not first_row:
            return False
        
        # Check if any column names match known patterns
        header_score = 0
        for col in first_row:
            col_lower = col.lower().strip()
            if col_lower in self.EMAIL_COLUMN_NAMES or col_lower in self.LABEL_COLUMN_NAMES:
                header_score += 2
            elif any(keyword in col_lower for keyword in ['text', 'email', 'message', 'content']):
                header_score += 1
        
        # Check if first row looks like data (contains long text)
        data_score = 0
        for col in first_row:
            if len(col.strip()) > 50:  # Long text suggests data, not header
                data_score += 1
        
        return header_score > data_score
    
    def _detect_email_column(self, header: List[str]) -> Optional[str]:
        """
        Detect which column contains email content with enhanced logic.
        
        Args:
            header: List of column names
            
        Returns:
            Name of the email column or None
        """
        if not header:
            return None
        
        # Score each column for likelihood of containing email content
        column_scores = {}
        
        for col in header:
            col_lower = col.lower().strip()
            score = 0
            
            # Direct match with known column names (highest score)
            if col_lower in self.EMAIL_COLUMN_NAMES:
                score += 10
            
            # Partial matches with email-related keywords
            email_keywords = ['text', 'email', 'message', 'content', 'mail', 'body']
            for keyword in email_keywords:
                if keyword in col_lower:
                    score += 5
                    break
            
            # Additional keywords that might indicate email content
            content_keywords = ['msg', 'description', 'comment', 'review', 'feedback', 'note']
            for keyword in content_keywords:
                if keyword in col_lower:
                    score += 3
                    break
            
            # Generic data column names (lower score)
            generic_keywords = ['data', 'input', 'col', 'column', 'field']
            for keyword in generic_keywords:
                if keyword in col_lower:
                    score += 1
                    break
            
            # Penalize columns that look like labels or metadata
            label_indicators = ['label', 'class', 'spam', 'target', 'id', 'index', 'num']
            for indicator in label_indicators:
                if indicator in col_lower:
                    score -= 5
                    break
            
            column_scores[col] = score
        
        # Find the column with the highest score
        if column_scores:
            best_column = max(column_scores, key=column_scores.get)
            if column_scores[best_column] > 0:
                return best_column
        
        # If no column scored positively, use fallback logic
        # If only one column, assume it's the email column
        if len(header) == 1:
            return header[0]
        
        # Look for the first column that doesn't look like an ID or index
        for col in header:
            col_lower = col.lower().strip()
            if not any(indicator in col_lower for indicator in ['id', 'index', 'num', 'count']):
                return col
        
        # Last resort: return first column
        return header[0]
    
    def _detect_label_column(self, header: List[str]) -> Optional[str]:
        """
        Detect which column contains labels with enhanced logic.
        
        Args:
            header: List of column names
            
        Returns:
            Name of the label column or None
        """
        if not header:
            return None
        
        # Score each column for likelihood of containing labels
        column_scores = {}
        
        for col in header:
            col_lower = col.lower().strip()
            score = 0
            
            # Direct match with known label column names (highest score)
            if col_lower in self.LABEL_COLUMN_NAMES:
                score += 10
            
            # Partial matches with label-related keywords
            label_keywords = ['label', 'class', 'spam', 'category', 'target', 'type']
            for keyword in label_keywords:
                if keyword in col_lower:
                    score += 5
                    break
            
            # Additional keywords that might indicate labels
            classification_keywords = ['result', 'output', 'prediction', 'verdict', 'flag']
            for keyword in classification_keywords:
                if keyword in col_lower:
                    score += 3
                    break
            
            # Penalize columns that look like content or metadata
            content_indicators = ['text', 'email', 'message', 'content', 'body', 'id', 'index']
            for indicator in content_indicators:
                if indicator in col_lower:
                    score -= 5
                    break
            
            column_scores[col] = score
        
        # Find the column with the highest score
        if column_scores:
            best_column = max(column_scores, key=column_scores.get)
            if column_scores[best_column] > 0:
                return best_column
        
        return None
    
    def get_detailed_format_analysis(self, file_path: str) -> Dict[str, any]:
        """
        Provide detailed analysis of CSV format for debugging and user feedback.
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            Dictionary with detailed format analysis
        """
        analysis = {
            'file_exists': os.path.exists(file_path),
            'file_size': 0,
            'encoding': 'unknown',
            'has_header': False,
            'column_count': 0,
            'columns': [],
            'sample_rows': [],
            'detected_format': None,
            'confidence': 0.0,
            'issues': [],
            'recommendations': []
        }
        
        if not analysis['file_exists']:
            analysis['issues'].append('File does not exist')
            return analysis
        
        try:
            analysis['file_size'] = os.path.getsize(file_path)
            analysis['encoding'] = self._detect_encoding(file_path)
            
            # Read sample data
            with open(file_path, 'r', encoding=analysis['encoding'], newline='') as f:
                reader = csv.reader(f)
                rows = []
                for i, row in enumerate(reader):
                    rows.append(row)
                    if i >= 5:  # Read first 6 rows
                        break
                
                if rows:
                    analysis['columns'] = rows[0]
                    analysis['column_count'] = len(rows[0])
                    analysis['has_header'] = self._is_header_row(rows[0], [','.join(row) for row in rows])
                    analysis['sample_rows'] = rows[1:] if analysis['has_header'] else rows
            
            # Get format detection results
            csv_format = self.detect_csv_format(file_path)
            analysis['detected_format'] = {
                'email_column': csv_format.email_column,
                'label_column': csv_format.label_column,
                'format_type': csv_format.format_type,
                'confidence_score': csv_format.confidence_score
            }
            analysis['confidence'] = csv_format.confidence_score
            
            # Analyze issues and provide recommendations
            if csv_format.format_type == "unknown":
                analysis['issues'].append('Cannot automatically detect CSV format')
            if not csv_format.email_column:
                analysis['issues'].append('No email content column detected')
            if csv_format.confidence_score < 0.5:
                analysis['issues'].append(f'Low confidence in format detection ({csv_format.confidence_score:.2f})')
            
            analysis['recommendations'] = self.suggest_format_fix(csv_format).split(' | ')
            
        except Exception as e:
            analysis['issues'].append(f'Error analyzing file: {e}')
        
        return analysis
    
    def _determine_format_type(self, format_info: CSVFormat, sample_lines: List[str]) -> Tuple[str, float]:
        """
        Determine the format type and confidence score.
        
        Args:
            format_info: Current format information
            sample_lines: Sample lines from the file
            
        Returns:
            Tuple of (format_type, confidence_score)
        """
        confidence = 0.0
        format_type = "unknown"
        
        # Check if we have email column
        if format_info.email_column:
            confidence += 0.4
            
            # Check if we have label column
            if format_info.label_column:
                format_type = "training_data"
                confidence += 0.3
            else:
                format_type = "batch_only"
                confidence += 0.2
        
        # Boost confidence if header looks good
        if format_info.has_header and format_info.email_column:
            if format_info.email_column.lower() in self.EMAIL_COLUMN_NAMES:
                confidence += 0.3
        
        # Check sample content
        if len(sample_lines) > 1:
            # Look at second line (first data line if header exists)
            data_line_idx = 1 if format_info.has_header else 0
            if data_line_idx < len(sample_lines):
                try:
                    reader = csv.reader([sample_lines[data_line_idx]])
                    row = next(reader)
                    if row and len(row[0].strip()) > 20:  # Reasonable email length
                        confidence += 0.2
                except:
                    pass
        
        return format_type, min(confidence, 1.0)
    
    def _validate_and_clean_emails(self, emails: List[str], validation_result: ValidationResult) -> List[str]:
        """
        Validate and clean email texts.
        
        Args:
            emails: List of email texts
            validation_result: ValidationResult to update
            
        Returns:
            List of cleaned email texts
        """
        if not emails:
            validation_result.add_error("No emails found in CSV file")
            return []
        
        cleaned_emails = []
        invalid_count = 0
        
        for i, email in enumerate(emails):
            # Basic validation
            if not email or not isinstance(email, str):
                invalid_count += 1
                continue
            
            email = email.strip()
            if len(email) < 10:  # Too short to be meaningful
                invalid_count += 1
                continue
            
            if len(email) > 10000:  # Too long
                validation_result.add_warning(f"Email {i+1} is very long ({len(email)} chars), truncating")
                email = email[:10000]
            
            cleaned_emails.append(email)
        
        if invalid_count > 0:
            validation_result.add_warning(f"Removed {invalid_count} invalid emails")
        
        if len(cleaned_emails) == 0:
            validation_result.add_error("No valid emails found after cleaning")
        
        return cleaned_emails
    
    def _validate_csv_content_structure(self, file_path: str, csv_format: CSVFormat, result: ValidationResult) -> None:
        """
        Validate the actual content structure of the CSV file.
        
        Args:
            file_path: Path to the CSV file
            csv_format: Detected CSV format
            result: ValidationResult to update with findings
        """
        try:
            # Read a sample of rows to validate content
            sample_size = min(10, 100)  # Read up to 10 rows for validation
            
            with open(file_path, 'r', encoding=csv_format.detected_encoding, newline='') as f:
                reader = csv.reader(f)
                
                # Skip header if present
                if csv_format.has_header:
                    try:
                        header = next(reader)
                        # Validate header structure
                        if len(header) != csv_format.column_count:
                            result.add_warning("Inconsistent column count in header")
                        
                        # Check for duplicate column names (warn instead of error since we can handle it)
                        if len(header) != len(set(header)):
                            result.add_warning("Duplicate column names detected in header (will be automatically renamed)")
                        
                        # Check for empty column names (warn instead of error since we can handle it)
                        empty_cols = [col for col in header if not col.strip()]
                        if empty_cols:
                            result.add_warning(f"Empty column names found in header (will be automatically removed): {len(empty_cols)} columns")
                            
                    except StopIteration:
                        result.add_error("File contains only header row, no data")
                        return
                
                # Validate data rows
                row_count = 0
                empty_rows = 0
                inconsistent_columns = 0
                email_column_idx = self._get_email_column_index(csv_format, header if csv_format.has_header else None)
                
                for i, row in enumerate(reader):
                    if i >= sample_size:
                        break
                    
                    row_count += 1
                    
                    # Check for empty rows
                    if not any(cell.strip() for cell in row):
                        empty_rows += 1
                        continue
                    
                    # Check column count consistency (be more tolerant of trailing empty columns)
                    actual_columns = len([cell for cell in row if cell.strip()])  # Count non-empty columns
                    expected_columns = len([cell for cell in (header if csv_format.has_header else ['email_content']) if cell.strip()])
                    
                    if len(row) != csv_format.column_count:
                        # Check if it's just trailing empty columns
                        if not all(cell.strip() == '' for cell in row[expected_columns:]):
                            inconsistent_columns += 1
                    
                    # Validate email content if we can identify the column
                    if email_column_idx is not None and email_column_idx < len(row):
                        email_content = row[email_column_idx].strip()
                        if len(email_content) < 5:
                            result.add_warning(f"Row {i+1}: Email content is very short ({len(email_content)} chars)")
                        elif len(email_content) > 5000:
                            result.add_warning(f"Row {i+1}: Email content is very long ({len(email_content)} chars)")
                
                # Report findings
                if row_count == 0:
                    result.add_error("No data rows found in CSV file")
                elif empty_rows > row_count * 0.5:
                    result.add_warning(f"High number of empty rows detected ({empty_rows}/{row_count})")
                
                if inconsistent_columns > 0:
                    result.add_error(f"Inconsistent column count in {inconsistent_columns} rows")
                
                # Check if we have reasonable amount of data
                if row_count < 5:
                    result.add_warning(f"Very few data rows ({row_count}) - ensure file contains sufficient data")
                
        except Exception as e:
            result.add_warning(f"Could not fully validate CSV content: {e}")
    
    def _get_email_column_index(self, csv_format: CSVFormat, header: Optional[List[str]]) -> Optional[int]:
        """
        Get the index of the email column for validation.
        
        Args:
            csv_format: Detected CSV format
            header: Header row if available
            
        Returns:
            Index of email column or None if not found
        """
        if not csv_format.email_column:
            return None
        
        if isinstance(csv_format.email_column, int):
            return csv_format.email_column
        
        if header and csv_format.email_column in header:
            return header.index(csv_format.email_column)
        
        return None
    
    def _load_emails_manual_parsing(self, file_path: str, csv_format: CSVFormat, validation_result: ValidationResult) -> Tuple[List[str], ValidationResult]:
        """
        Manually parse CSV file when pandas fails due to inconsistent column counts.
        
        Args:
            file_path: Path to the CSV file
            csv_format: Detected CSV format
            validation_result: ValidationResult to update
            
        Returns:
            Tuple of (list of email texts, ValidationResult)
        """
        emails = []
        
        try:
            with open(file_path, 'r', encoding=csv_format.detected_encoding, newline='') as f:
                reader = csv.reader(f)
                
                # Handle header
                if csv_format.has_header:
                    try:
                        header = next(reader)
                        
                        # Remove empty columns from header and track mapping
                        original_header = header[:]
                        non_empty_indices = [i for i, col in enumerate(header) if col.strip()]
                        header = [header[i] for i in non_empty_indices]
                        
                        if len(non_empty_indices) < len(original_header):
                            removed_count = len(original_header) - len(non_empty_indices)
                            validation_result.add_warning(f"Automatically removed {removed_count} empty columns during manual parsing")
                        
                        # Find email column index in cleaned header
                        email_column_name = csv_format.email_column
                        if email_column_name.startswith('\ufeff'):
                            email_column_name = email_column_name.replace('\ufeff', '')
                        
                        if email_column_name in header:
                            # Get the original index before column removal
                            cleaned_idx = header.index(email_column_name)
                            email_col_idx = non_empty_indices[cleaned_idx]
                        else:
                            validation_result.add_error(f"Email column '{email_column_name}' not found in header")
                            return [], validation_result
                    except StopIteration:
                        validation_result.add_error("CSV file is empty")
                        return [], validation_result
                else:
                    email_col_idx = 0  # Assume first column for headerless files
                
                # Read data rows
                for row_num, row in enumerate(reader, start=1):
                    try:
                        # Skip empty rows
                        if not any(cell.strip() for cell in row):
                            continue
                        
                        # Extract email content (handle inconsistent column counts gracefully)
                        if email_col_idx < len(row):
                            email_content = row[email_col_idx].strip()
                            if email_content:  # Only add non-empty emails
                                emails.append(email_content)
                        else:
                            validation_result.add_warning(f"Row {row_num}: Not enough columns, skipping")
                            
                    except Exception as e:
                        validation_result.add_warning(f"Row {row_num}: Error parsing row - {e}")
                        continue
                
                if not emails:
                    validation_result.add_error("No valid email content found in CSV file")
                else:
                    validation_result.add_warning(f"Used manual parsing due to inconsistent CSV format. Loaded {len(emails)} emails.")
                
                return emails, validation_result
                
        except Exception as e:
            validation_result.add_error(f"Error in manual CSV parsing: {e}")
            return [], validation_result