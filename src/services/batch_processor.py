"""
Batch processing service for handling multiple email classifications.

This module provides the BatchProcessor class for handling CSV file processing,
progress tracking, job status management, and result file generation.
"""

import os
import csv
import uuid
import time
import logging
from datetime import datetime, date
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
import pandas as pd

from src.models.data_models import BatchJob, ClassificationResult
from src.data_loader import ValidationResult
from src.flexible_csv_processor import FlexibleCSVProcessor


class BatchProcessor:
    """
    Handles batch processing of email classifications.
    
    Provides CSV file processing with progress tracking, job status management,
    and result file generation for large-scale email classification tasks.
    """
    
    def __init__(self, 
                 classification_service=None,
                 results_dir: str = "results",
                 max_batch_size: int = 1000):
        """
        Initialize the batch processor.
        
        Args:
            classification_service: ClassificationService for email classification
            results_dir: Directory to store batch processing results
            max_batch_size: Maximum number of emails per batch
        """
        self.classification_service = classification_service
        self.results_dir = results_dir
        self.max_batch_size = max_batch_size
        self.csv_processor = FlexibleCSVProcessor()
        self.logger = logging.getLogger(__name__)
        
        # Job tracking
        self.active_jobs: Dict[str, BatchJob] = {}
        
        # Daily statistics tracking
        self._daily_batch_stats = defaultdict(lambda: {
            'jobs_processed': 0,
            'emails_processed': 0,
            'total_processing_time': 0.0,
            'successful_jobs': 0,
            'failed_jobs': 0
        })
        
        # Ensure results directory exists
        os.makedirs(self.results_dir, exist_ok=True)
    
    def validate_csv_file(self, file_path: str) -> ValidationResult:
        """
        Validate CSV file format and structure before processing.
        
        Args:
            file_path: Path to the CSV file to validate
            
        Returns:
            ValidationResult with validation status and messages
        """
        try:
            # Use FlexibleCSVProcessor for validation
            validation_result = self.csv_processor.validate_csv_structure(file_path)
            
            # Add additional batch-specific validation
            if validation_result.is_valid:
                csv_format = self.csv_processor.detect_csv_format(file_path)
                
                # Check if format is suitable for batch processing
                if csv_format.format_type == "unknown":
                    validation_result.add_error("Cannot detect CSV format for batch processing")
                    # Add format suggestion
                    suggestion = self.csv_processor.suggest_format_fix(csv_format)
                    validation_result.add_error(f"Format suggestion: {suggestion}")
                elif csv_format.confidence_score < 0.5:
                    validation_result.add_warning(f"Low confidence in CSV format detection: {csv_format.confidence_score:.2f}")
                
                # Batch processing doesn't require labels, so accept both batch_only and training_data formats
                if csv_format.format_type in ["batch_only", "training_data"]:
                    validation_result.add_warning(f"Detected CSV format: {csv_format.format_type}")
            
            return validation_result
            
        except Exception as e:
            self.logger.error(f"Error validating CSV file {file_path}: {str(e)}")
            result = ValidationResult(is_valid=False, errors=[], warnings=[])
            result.add_error(f"Validation failed: {str(e)}")
            return result

    def get_csv_format_info(self, file_path: str) -> Dict:
        """
        Get information about CSV file format for debugging and user feedback.
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            Dictionary with format information
        """
        try:
            csv_format = self.csv_processor.detect_csv_format(file_path)
            return {
                'email_column': csv_format.email_column,
                'label_column': csv_format.label_column,
                'column_count': csv_format.column_count,
                'has_header': csv_format.has_header,
                'format_type': csv_format.format_type,
                'confidence_score': csv_format.confidence_score,
                'detected_encoding': csv_format.detected_encoding,
                'suggestion': self.csv_processor.suggest_format_fix(csv_format)
            }
        except Exception as e:
            self.logger.error(f"Error getting CSV format info for {file_path}: {str(e)}")
            return {
                'error': str(e),
                'suggestion': 'Please ensure the file is a valid CSV with email content'
            }

    def process_csv(self, file_path: str, job_id: Optional[str] = None) -> str:
        """
        Process emails from a CSV file with enhanced error handling and progress tracking.
        
        Args:
            file_path: Path to the CSV file containing emails
            job_id: Optional job ID, will generate one if not provided
            
        Returns:
            Job ID for tracking the batch processing
            
        Raises:
            ValueError: If file is invalid or exceeds batch size limit
            RuntimeError: If classification service is not ready
        """
        # Enhanced service initialization checks
        if self.classification_service is None:
            raise RuntimeError("Classification service not initialized. Please check application startup.")
        
        if not hasattr(self.classification_service, 'is_ready'):
            raise RuntimeError("Classification service does not support readiness checks. Service may be corrupted.")
        
        if not self.classification_service.is_ready():
            raise RuntimeError(
                "Classification service is not ready. Please ensure models are trained and preprocessing pipeline is fitted."
            )
        
        # Generate job ID if not provided
        if job_id is None:
            job_id = str(uuid.uuid4())
        
        # Initialize job tracking
        batch_job = None
        
        try:
            self.logger.info(f"Starting CSV processing for job {job_id}, file: {file_path}")
            
            # Enhanced file validation using FlexibleCSVProcessor
            csv_validation = self.validate_csv_file(file_path)
            if not csv_validation.is_valid:
                error_details = '; '.join(csv_validation.errors)
                self.logger.warning(f"CSV validation failed for job {job_id}: {error_details}")
                raise ValueError(f"Invalid CSV file format: {error_details}")
            
            # Log any warnings from CSV validation
            if csv_validation.warnings:
                for warning in csv_validation.warnings:
                    self.logger.info(f"CSV validation warning for job {job_id}: {warning}")
            
            # Load and validate emails from CSV with enhanced error handling using FlexibleCSVProcessor
            try:
                emails, validation_result = self.csv_processor.load_emails_from_csv(file_path)
            except Exception as load_error:
                self.logger.error(f"Failed to load emails from CSV: {str(load_error)}")
                # Try to provide format suggestion if loading fails
                try:
                    csv_format = self.csv_processor.detect_csv_format(file_path)
                    suggestion = self.csv_processor.suggest_format_fix(csv_format)
                    raise ValueError(f"Failed to read CSV file: {str(load_error)}. {suggestion}")
                except:
                    raise ValueError(f"Failed to read CSV file: {str(load_error)}. Please ensure the file is a valid CSV with proper encoding.")
            
            # Enhanced validation result checking
            if not validation_result.is_valid:
                error_details = '; '.join(validation_result.errors)
                self.logger.warning(f"Email validation failed for job {job_id}: {error_details}")
                raise ValueError(f"Invalid email data in CSV file: {error_details}")
            
            if not emails:
                # Try to provide helpful guidance based on detected format
                try:
                    csv_format = self.csv_processor.detect_csv_format(file_path)
                    suggestion = self.csv_processor.suggest_format_fix(csv_format)
                    raise ValueError(f"No valid emails found in CSV file. {suggestion}")
                except:
                    raise ValueError("No valid emails found in CSV file. Please ensure your file contains email data.")
            
            if len(emails) > self.max_batch_size:
                raise ValueError(
                    f"Batch size {len(emails)} exceeds maximum limit of {self.max_batch_size}. "
                    f"Please split your file into smaller batches."
                )
            
            # Create batch job with enhanced tracking
            batch_job = BatchJob(
                job_id=job_id,
                status="pending",
                total_emails=len(emails),
                processed_emails=0,
                created_at=datetime.now()
            )
            
            # Add error tracking fields
            batch_job.error_count = 0
            batch_job.errors = []
            
            self.active_jobs[job_id] = batch_job
            self.logger.info(f"Created batch job {job_id} with {len(emails)} emails")
            
            # Process emails with enhanced error recovery
            self._process_batch_sync_enhanced(job_id, emails)
            
            return job_id
            
        except Exception as e:
            self.logger.error(f"Error starting batch processing for job {job_id}: {str(e)}", exc_info=True)
            
            # Update job status if it was created
            if batch_job and job_id in self.active_jobs:
                self.active_jobs[job_id].status = "failed"
                self.active_jobs[job_id].completed_at = datetime.now()
                if hasattr(self.active_jobs[job_id], 'errors'):
                    self.active_jobs[job_id].errors.append(str(e))
            
            # Re-raise with appropriate error type
            if isinstance(e, (ValueError, RuntimeError)):
                raise
            else:
                raise RuntimeError(f"Unexpected error during batch processing initialization: {str(e)}")
    
    def _process_batch_sync(self, job_id: str, emails: List[str]) -> None:
        """
        Process batch of emails synchronously (legacy method).
        
        Args:
            job_id: Job identifier
            emails: List of email texts to process
        """
        # Delegate to enhanced method
        self._process_batch_sync_enhanced(job_id, emails)
    
    def _process_batch_sync_enhanced(self, job_id: str, emails: List[str]) -> None:
        """
        Process batch of emails with enhanced error handling and progress tracking.
        
        Args:
            job_id: Job identifier
            emails: List of email texts to process
        """
        batch_job = self.active_jobs[job_id]
        batch_job.status = "processing"
        
        results = []
        error_count = 0
        processing_errors = []
        
        try:
            self.logger.info(f"Starting processing of {len(emails)} emails for job {job_id}")
            
            for i, email in enumerate(emails):
                email_start_time = time.time()
                
                try:
                    # Validate individual email before processing
                    if not email or not email.strip():
                        raise ValueError("Empty email content")
                    
                    if len(email.strip()) < 10:
                        raise ValueError("Email content too short (minimum 10 characters)")
                    
                    # Classify the email with timeout protection
                    result = self.classification_service.classify_email(email)
                    
                    # Validate classification result
                    if not result:
                        raise ValueError("Classification service returned no result")
                    
                    processing_time = time.time() - email_start_time
                    
                    results.append({
                        'email_index': i + 1,
                        'email_text': email[:100] + "..." if len(email) > 100 else email,
                        'prediction': result.prediction,
                        'confidence': result.confidence,
                        'model_used': getattr(result, 'model_used', 'Unknown'),
                        'processing_time': getattr(result, 'processing_time', processing_time),
                        'timestamp': getattr(result, 'timestamp', datetime.now()).isoformat() if hasattr(getattr(result, 'timestamp', datetime.now()), 'isoformat') else str(getattr(result, 'timestamp', datetime.now())),
                        'status': 'success'
                    })
                    
                except Exception as e:
                    error_count += 1
                    error_msg = str(e)
                    processing_errors.append(f"Email {i+1}: {error_msg}")
                    
                    # Add error result with detailed information
                    results.append({
                        'email_index': i + 1,
                        'email_text': email[:100] + "..." if len(email) > 100 else email,
                        'prediction': 'Error',
                        'confidence': 0.0,
                        'model_used': 'None',
                        'processing_time': time.time() - email_start_time,
                        'timestamp': datetime.now().isoformat(),
                        'status': 'error',
                        'error_message': error_msg
                    })
                    
                    self.logger.warning(f"Error processing email {i+1} in job {job_id}: {error_msg}")
                    
                    # Stop processing if too many consecutive errors
                    if error_count > 10 and i < 50:  # More than 10 errors in first 50 emails
                        raise RuntimeError(f"Too many processing errors ({error_count}). Stopping batch processing.")
                
                # Update progress
                batch_job.processed_emails = i + 1
                
                # Update error tracking
                if hasattr(batch_job, 'error_count'):
                    batch_job.error_count = error_count
                if hasattr(batch_job, 'errors'):
                    batch_job.errors = processing_errors[-10:]  # Keep last 10 errors
                
                # Log progress at intervals
                if (i + 1) % 100 == 0 or (i + 1) == len(emails):
                    success_rate = ((i + 1 - error_count) / (i + 1)) * 100
                    self.logger.info(
                        f"Job {job_id}: Processed {i+1}/{len(emails)} emails "
                        f"(Success rate: {success_rate:.1f}%, Errors: {error_count})"
                    )
            
            # Save results to file with error handling
            try:
                result_file_path = self._save_results_enhanced(job_id, results)
            except Exception as save_error:
                self.logger.error(f"Failed to save results for job {job_id}: {str(save_error)}")
                raise RuntimeError(f"Processing completed but failed to save results: {str(save_error)}")
            
            # Update job status with comprehensive information
            batch_job.status = "completed"
            batch_job.completed_at = datetime.now()
            batch_job.result_file_path = result_file_path
            
            # Add summary information
            success_count = len(emails) - error_count
            success_rate = (success_count / len(emails)) * 100 if len(emails) > 0 else 0
            
            # Update daily statistics for successful job
            processing_time = (batch_job.completed_at - batch_job.created_at).total_seconds()
            self._update_daily_stats(len(emails), processing_time, success=True)
            
            self.logger.info(
                f"Batch job {job_id} completed. "
                f"Processed: {len(emails)}, Success: {success_count}, Errors: {error_count}, "
                f"Success rate: {success_rate:.1f}%. Results saved to {result_file_path}"
            )
            
            # Log warning if high error rate
            if error_count > 0 and success_rate < 90:
                self.logger.warning(
                    f"Job {job_id} completed with high error rate ({success_rate:.1f}% success). "
                    f"Please check the input data quality."
                )
            
        except Exception as e:
            batch_job.status = "failed"
            batch_job.completed_at = datetime.now()
            
            # Add error information to job
            if hasattr(batch_job, 'errors'):
                batch_job.errors.append(f"Processing failed: {str(e)}")
            
            # Update daily statistics for failed job
            processing_time = (batch_job.completed_at - batch_job.created_at).total_seconds()
            emails_processed = getattr(batch_job, 'processed_emails', 0)
            self._update_daily_stats(emails_processed, processing_time, success=False)
            
            self.logger.error(f"Batch job {job_id} failed after processing {batch_job.processed_emails}/{len(emails)} emails: {str(e)}", exc_info=True)
            
            # Try to save partial results if any were processed
            if results:
                try:
                    partial_file_path = self._save_results_enhanced(job_id, results, partial=True)
                    batch_job.result_file_path = partial_file_path
                    self.logger.info(f"Saved partial results for failed job {job_id} to {partial_file_path}")
                except Exception as partial_save_error:
                    self.logger.error(f"Failed to save partial results for job {job_id}: {str(partial_save_error)}")
            
            raise
    
    def _save_results(self, job_id: str, results: List[Dict]) -> str:
        """
        Save batch processing results to CSV file (legacy method).
        
        Args:
            job_id: Job identifier
            results: List of classification results
            
        Returns:
            Path to the saved results file
        """
        return self._save_results_enhanced(job_id, results)
    
    def _save_results_enhanced(self, job_id: str, results: List[Dict], partial: bool = False) -> str:
        """
        Save batch processing results to CSV file with enhanced error handling.
        
        Args:
            job_id: Job identifier
            results: List of classification results
            partial: Whether this is a partial result from a failed job
            
        Returns:
            Path to the saved results file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = "partial_results" if partial else "batch_results"
        filename = f"{prefix}_{job_id}_{timestamp}.csv"
        file_path = os.path.join(self.results_dir, filename)
        
        try:
            # Ensure results directory exists
            os.makedirs(self.results_dir, exist_ok=True)
            
            # Define standard fieldnames for consistency
            standard_fieldnames = [
                'email_index', 'email_text', 'prediction', 'confidence', 
                'model_used', 'processing_time', 'timestamp', 'status'
            ]
            
            # Add error_message field if any results have errors
            if any(result.get('status') == 'error' for result in results):
                standard_fieldnames.append('error_message')
            
            with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=standard_fieldnames, extrasaction='ignore')
                writer.writeheader()
                
                if results:
                    # Ensure all results have required fields
                    for result in results:
                        # Fill missing fields with defaults
                        for field in standard_fieldnames:
                            if field not in result:
                                if field == 'status':
                                    result[field] = 'success' if result.get('prediction', '') != 'Error' else 'error'
                                elif field == 'error_message':
                                    result[field] = result.get('error_message', '')
                                else:
                                    result[field] = ''
                    
                    writer.writerows(results)
                else:
                    # Write empty row for empty results
                    empty_row = {field: '' for field in standard_fieldnames}
                    empty_row['email_index'] = 'No data'
                    empty_row['email_text'] = 'No emails were processed'
                    writer.writerow(empty_row)
            
            # Verify file was created and has content
            if not os.path.exists(file_path):
                raise RuntimeError(f"Results file was not created: {file_path}")
            
            file_size = os.path.getsize(file_path)
            if file_size == 0:
                raise RuntimeError(f"Results file is empty: {file_path}")
            
            self.logger.info(f"Results saved to {file_path} (Size: {file_size} bytes, Partial: {partial})")
            return file_path
            
        except Exception as e:
            self.logger.error(f"Error saving results for job {job_id}: {str(e)}", exc_info=True)
            
            # Try to clean up partial file if it exists
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except Exception as cleanup_error:
                    self.logger.error(f"Failed to clean up partial results file {file_path}: {str(cleanup_error)}")
            
            raise RuntimeError(f"Failed to save batch processing results: {str(e)}")
    
    def get_processing_status(self, job_id: str) -> Optional[BatchJob]:
        """
        Get the processing status of a batch job.
        
        Args:
            job_id: Job identifier
            
        Returns:
            BatchJob instance with current status or None if job not found
        """
        return self.active_jobs.get(job_id)
    
    def get_progress_percentage(self, job_id: str) -> float:
        """
        Get the progress percentage of a batch job.
        
        Args:
            job_id: Job identifier
            
        Returns:
            Progress percentage (0.0 to 100.0) or 0.0 if job not found
        """
        job = self.active_jobs.get(job_id)
        if job and job.total_emails > 0:
            return (job.processed_emails / job.total_emails) * 100.0
        return 0.0
    
    def download_results(self, job_id: str) -> Optional[str]:
        """
        Get the file path for downloading batch processing results with enhanced validation.
        
        Args:
            job_id: Job identifier
            
        Returns:
            File path to results or None if not available
        """
        job = self.active_jobs.get(job_id)
        
        if not job:
            self.logger.warning(f"Job not found for download: {job_id}")
            return None
        
        # Allow downloads for both completed and failed jobs (failed jobs may have partial results)
        if job.status not in ["completed", "failed"]:
            self.logger.warning(f"Job {job_id} is not ready for download. Status: {job.status}")
            return None
        
        if not job.result_file_path:
            self.logger.warning(f"No result file path for job {job_id}")
            return None
        
        # Validate file existence and accessibility
        if not os.path.exists(job.result_file_path):
            self.logger.error(f"Results file not found for job {job_id}: {job.result_file_path}")
            # Try to clean up the job since the file is missing
            self._cleanup_job_with_missing_file(job_id)
            return None
        
        # Validate file is readable and not empty
        try:
            file_size = os.path.getsize(job.result_file_path)
            if file_size == 0:
                self.logger.warning(f"Results file is empty for job {job_id}: {job.result_file_path}")
                return None
            
            # Test file readability
            with open(job.result_file_path, 'r', encoding='utf-8') as test_file:
                test_file.read(100)  # Try to read first 100 characters
            
            self.logger.info(f"Results file validated for job {job_id}: {job.result_file_path} ({file_size} bytes)")
            return job.result_file_path
            
        except Exception as e:
            self.logger.error(f"Cannot validate results file for job {job_id}: {str(e)}")
            return None
    
    def list_active_jobs(self) -> List[BatchJob]:
        """
        Get a list of all active batch jobs.
        
        Returns:
            List of BatchJob instances
        """
        return list(self.active_jobs.values())
    
    def cleanup_completed_jobs(self, max_age_hours: int = 24) -> int:
        """
        Clean up completed jobs older than specified age with enhanced error handling.
        
        Args:
            max_age_hours: Maximum age in hours for keeping completed jobs
            
        Returns:
            Number of jobs cleaned up
        """
        current_time = datetime.now()
        jobs_to_remove = []
        files_removed = 0
        
        for job_id, job in self.active_jobs.items():
            if job.status in ["completed", "failed"] and job.completed_at:
                age_hours = (current_time - job.completed_at).total_seconds() / 3600
                if age_hours > max_age_hours:
                    jobs_to_remove.append(job_id)
                    
                    # Also remove result file if it exists
                    if job.result_file_path:
                        if os.path.exists(job.result_file_path):
                            try:
                                os.remove(job.result_file_path)
                                files_removed += 1
                                self.logger.info(f"Removed result file: {job.result_file_path}")
                            except Exception as e:
                                self.logger.error(f"Error removing result file {job.result_file_path}: {str(e)}")
                        else:
                            self.logger.debug(f"Result file already missing for job {job_id}: {job.result_file_path}")
        
        # Remove jobs from tracking
        for job_id in jobs_to_remove:
            del self.active_jobs[job_id]
        
        if jobs_to_remove:
            self.logger.info(f"Cleaned up {len(jobs_to_remove)} old batch jobs and {files_removed} result files")
        
        return len(jobs_to_remove)
    
    def cleanup_failed_jobs(self, max_age_hours: int = 1) -> int:
        """
        Clean up failed jobs older than specified age (shorter retention for failed jobs).
        
        Args:
            max_age_hours: Maximum age in hours for keeping failed jobs (default 1 hour)
            
        Returns:
            Number of failed jobs cleaned up
        """
        current_time = datetime.now()
        failed_jobs_to_remove = []
        
        for job_id, job in self.active_jobs.items():
            if job.status == "failed":
                # Use completed_at if available, otherwise use created_at
                reference_time = job.completed_at if job.completed_at else job.created_at
                age_hours = (current_time - reference_time).total_seconds() / 3600
                
                if age_hours > max_age_hours:
                    failed_jobs_to_remove.append(job_id)
                    
                    # Remove any partial result files
                    if job.result_file_path and os.path.exists(job.result_file_path):
                        try:
                            os.remove(job.result_file_path)
                            self.logger.info(f"Removed failed job result file: {job.result_file_path}")
                        except Exception as e:
                            self.logger.error(f"Error removing failed job result file {job.result_file_path}: {str(e)}")
        
        # Remove failed jobs from tracking
        for job_id in failed_jobs_to_remove:
            del self.active_jobs[job_id]
        
        if failed_jobs_to_remove:
            self.logger.info(f"Cleaned up {len(failed_jobs_to_remove)} failed batch jobs")
        
        return len(failed_jobs_to_remove)
    
    def _cleanup_job_with_missing_file(self, job_id: str) -> None:
        """
        Clean up a job whose result file is missing.
        
        Args:
            job_id: Job identifier
        """
        job = self.active_jobs.get(job_id)
        if job:
            self.logger.warning(f"Cleaning up job {job_id} due to missing result file")
            # Don't remove the job immediately, just mark the file path as None
            job.result_file_path = None
            
            # If the job is old enough, remove it completely
            if job.completed_at:
                age_hours = (datetime.now() - job.completed_at).total_seconds() / 3600
                if age_hours > 1:  # Remove jobs with missing files after 1 hour
                    del self.active_jobs[job_id]
                    self.logger.info(f"Removed job {job_id} with missing result file")
    
    def cleanup_orphaned_files(self) -> int:
        """
        Clean up result files that don't have corresponding active jobs.
        
        Returns:
            Number of orphaned files cleaned up
        """
        if not os.path.exists(self.results_dir):
            return 0
        
        # Get all result files in the directory
        result_files = []
        for filename in os.listdir(self.results_dir):
            if filename.startswith(('batch_results_', 'partial_results_')) and filename.endswith('.csv'):
                result_files.append(os.path.join(self.results_dir, filename))
        
        # Get all active job result file paths
        active_result_paths = set()
        for job in self.active_jobs.values():
            if job.result_file_path:
                active_result_paths.add(os.path.abspath(job.result_file_path))
        
        # Find orphaned files
        orphaned_files = []
        for file_path in result_files:
            abs_file_path = os.path.abspath(file_path)
            if abs_file_path not in active_result_paths:
                # Check file age - only remove files older than 24 hours
                try:
                    file_age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(file_path))
                    if file_age.total_seconds() > 24 * 3600:  # 24 hours
                        orphaned_files.append(file_path)
                except Exception as e:
                    self.logger.error(f"Error checking file age for {file_path}: {str(e)}")
        
        # Remove orphaned files
        removed_count = 0
        for file_path in orphaned_files:
            try:
                os.remove(file_path)
                removed_count += 1
                self.logger.info(f"Removed orphaned result file: {file_path}")
            except Exception as e:
                self.logger.error(f"Error removing orphaned file {file_path}: {str(e)}")
        
        if removed_count > 0:
            self.logger.info(f"Cleaned up {removed_count} orphaned result files")
        
        return removed_count
    
    def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a batch processing job.
        
        Args:
            job_id: Job identifier
            
        Returns:
            True if job was cancelled, False if job not found or already completed
        """
        job = self.active_jobs.get(job_id)
        if job and job.status in ["pending", "processing"]:
            job.status = "cancelled"
            job.completed_at = datetime.now()
            self.logger.info(f"Cancelled batch job {job_id}")
            return True
        
        return False
    
    def process_email_list(self, emails: List[str], job_id: Optional[str] = None) -> List[ClassificationResult]:
        """
        Process a list of emails directly (for API usage).
        
        Args:
            emails: List of email texts to process
            job_id: Optional job ID for tracking
            
        Returns:
            List of ClassificationResult objects
            
        Raises:
            ValueError: If email list is invalid
            RuntimeError: If classification service is not ready
        """
        if self.classification_service is None:
            raise RuntimeError("Classification service not initialized")
        
        if not self.classification_service.is_ready():
            raise RuntimeError("Classification service is not ready. Please train models first.")
        
        if not emails:
            raise ValueError("Email list cannot be empty")
        
        if len(emails) > self.max_batch_size:
            raise ValueError(f"Batch size {len(emails)} exceeds maximum limit of {self.max_batch_size}")
        
        # Generate job ID if not provided
        if job_id is None:
            job_id = str(uuid.uuid4())
        
        # Validate emails
        batch_validation, invalid_indices = self.data_loader.validator.validate_batch_emails(emails)
        if not batch_validation.is_valid:
            raise ValueError(f"Invalid email batch: {'; '.join(batch_validation.errors)}")
        
        # Process emails using classification service
        results = self.classification_service.classify_batch(emails)
        
        self.logger.info(f"Processed {len(emails)} emails in batch {job_id}")
        
        return results
    
    def get_job_summary(self, job_id: str) -> Optional[Dict]:
        """
        Get a summary of batch job results.
        
        Args:
            job_id: Job identifier
            
        Returns:
            Dictionary with job summary or None if job not found
        """
        job = self.active_jobs.get(job_id)
        if not job:
            return None
        
        summary = {
            'job_id': job.job_id,
            'status': job.status,
            'total_emails': job.total_emails,
            'processed_emails': job.processed_emails,
            'progress_percentage': self.get_progress_percentage(job_id),
            'created_at': job.created_at.isoformat(),
            'completed_at': job.completed_at.isoformat() if job.completed_at else None,
            'result_file_available': job.result_file_path is not None and os.path.exists(job.result_file_path) if job.result_file_path else False
        }
        
        # Add processing time if completed
        if job.completed_at and job.created_at:
            processing_time = (job.completed_at - job.created_at).total_seconds()
            summary['processing_time_seconds'] = processing_time
        
        return summary
    
    def _update_daily_stats(self, emails_processed: int, processing_time: float, success: bool) -> None:
        """
        Update daily batch processing statistics.
        
        Args:
            emails_processed: Number of emails processed in this job
            processing_time: Total processing time in seconds
            success: Whether the job completed successfully
        """
        try:
            today = date.today().isoformat()
            stats = self._daily_batch_stats[today]
            
            stats['jobs_processed'] += 1
            stats['emails_processed'] += emails_processed
            stats['total_processing_time'] += processing_time
            
            if success:
                stats['successful_jobs'] += 1
            else:
                stats['failed_jobs'] += 1
            
            self.logger.debug(f"Updated daily batch stats: {stats}")
            
        except Exception as e:
            self.logger.warning(f"Error updating daily batch stats: {str(e)}")
    
    def get_daily_stats(self) -> Dict[str, any]:
        """
        Get daily batch processing statistics.
        
        Returns:
            Dictionary containing daily batch processing statistics
        """
        try:
            today = date.today().isoformat()
            today_stats = self._daily_batch_stats[today]
            
            # Calculate averages
            avg_emails_per_job = 0.0
            avg_processing_time = 0.0
            success_rate = 0.0
            
            if today_stats['jobs_processed'] > 0:
                avg_emails_per_job = today_stats['emails_processed'] / today_stats['jobs_processed']
                avg_processing_time = today_stats['total_processing_time'] / today_stats['jobs_processed']
                success_rate = (today_stats['successful_jobs'] / today_stats['jobs_processed']) * 100
            
            stats = {
                'processed_today': today_stats['emails_processed'],
                'jobs_today': today_stats['jobs_processed'],
                'successful_jobs': today_stats['successful_jobs'],
                'failed_jobs': today_stats['failed_jobs'],
                'success_rate': success_rate,
                'avg_emails_per_job': avg_emails_per_job,
                'avg_processing_time': avg_processing_time,
                'total_processing_time': today_stats['total_processing_time'],
                'last_updated': datetime.now().isoformat()
            }
            
            self.logger.debug(f"Retrieved daily batch stats: {stats}")
            return stats
            
        except Exception as e:
            self.logger.error(f"Error getting daily batch stats: {str(e)}")
            return {
                'processed_today': 0,
                'jobs_today': 0,
                'successful_jobs': 0,
                'failed_jobs': 0,
                'success_rate': 0.0,
                'avg_emails_per_job': 0.0,
                'avg_processing_time': 0.0,
                'total_processing_time': 0.0,
                'last_updated': datetime.now().isoformat()
            }
    
    def reset_daily_batch_stats(self) -> None:
        """Reset daily batch statistics (typically called at midnight)."""
        try:
            today = date.today().isoformat()
            if today in self._daily_batch_stats:
                del self._daily_batch_stats[today]
            self.logger.info("Daily batch statistics reset")
        except Exception as e:
            self.logger.warning(f"Error resetting daily batch stats: {str(e)}")
    
    def get_batch_stats_summary(self) -> Dict[str, any]:
        """
        Get a comprehensive summary of batch processing statistics.
        
        Returns:
            Dictionary containing comprehensive batch processing statistics
        """
        try:
            daily_stats = self.get_daily_stats()
            active_jobs_count = len(self.active_jobs)
            
            # Count jobs by status
            status_counts = {'processing': 0, 'completed': 0, 'failed': 0, 'pending': 0}
            for job in self.active_jobs.values():
                status = job.status
                if status in status_counts:
                    status_counts[status] += 1
                else:
                    status_counts['pending'] += 1
            
            summary = {
                'service_ready': self.classification_service is not None and 
                               hasattr(self.classification_service, 'is_ready') and 
                               self.classification_service.is_ready(),
                'daily_stats': daily_stats,
                'active_jobs_count': active_jobs_count,
                'jobs_by_status': status_counts,
                'max_batch_size': self.max_batch_size,
                'results_directory': self.results_dir,
                'last_updated': datetime.now().isoformat()
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error getting batch stats summary: {str(e)}")
            return {
                'service_ready': False,
                'daily_stats': self.get_daily_stats(),
                'active_jobs_count': 0,
                'jobs_by_status': {'processing': 0, 'completed': 0, 'failed': 0, 'pending': 0},
                'max_batch_size': self.max_batch_size,
                'results_directory': self.results_dir,
                'last_updated': datetime.now().isoformat()
            }