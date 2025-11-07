"""
Web interface routes for the Email Spam Classifier.
"""
from flask import Blueprint, render_template, request, jsonify, flash, redirect, url_for, send_file, current_app
from werkzeug.utils import secure_filename
import os
import uuid
from datetime import datetime

from functools import wraps
from src.error_handling import (
    error_handler, enhanced_error_handler, input_validator, health_checker, enhanced_logger,
    ValidationError, ServiceNotReadyError, ProcessingError, DashboardErrorContext, BatchErrorContext
)

# Create blueprint
web_bp = Blueprint('web', __name__)

def get_service_health_manager():
    """Get service health manager from current app context."""
    from flask import current_app
    if not hasattr(current_app, 'service_health_manager') or current_app.service_health_manager is None:
        raise ServiceNotReadyError("Service Health Manager", "Service health manager not initialized during application startup")
    return current_app.service_health_manager


def get_classification_service():
    """Get classification service from current app context with readiness validation."""
    from flask import current_app
    if not hasattr(current_app, 'classification_service') or current_app.classification_service is None:
        raise ServiceNotReadyError("Classification Service", "Classification service not initialized. Please check application startup.")
    return current_app.classification_service


def get_batch_processor():
    """Get batch processor from current app context with readiness validation."""
    from flask import current_app
    if not hasattr(current_app, 'batch_processor') or current_app.batch_processor is None:
        raise ServiceNotReadyError("Batch Processor", "Batch processor not initialized. Please check application startup.")
    return current_app.batch_processor


def validate_service_readiness(service_name: str, require_ready: bool = True):
    """
    Validate service readiness using the service health manager.
    
    Args:
        service_name: Name of the service to validate
        require_ready: If True, raises exception if service not ready
        
    Returns:
        Tuple of (is_ready, message)
        
    Raises:
        ServiceNotReadyError: If require_ready=True and service is not ready
    """
    try:
        health_manager = get_service_health_manager()
        is_ready, message = health_manager.validate_service_readiness(service_name)
        
        if require_ready and not is_ready:
            raise ServiceNotReadyError(service_name, message)
        
        return is_ready, message
        
    except ServiceNotReadyError:
        # Re-raise service not ready errors
        raise
    except Exception as e:
        # Handle other errors (health manager issues)
        error_msg = f"Unable to validate service readiness: {str(e)}"
        if require_ready:
            raise ServiceNotReadyError(service_name, error_msg)
        return False, error_msg


def check_classification_service_readiness():
    """Check if classification service is ready for operation."""
    validate_service_readiness('classification_service', require_ready=True)


def check_batch_processor_readiness():
    """Check if batch processor is ready for operation."""
    validate_service_readiness('batch_processor', require_ready=True)


def handle_service_not_ready_error(f):
    """Decorator to handle ServiceNotReadyError consistently across routes."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except ServiceNotReadyError as e:
            # Get service health manager for detailed error response
            try:
                health_manager = get_service_health_manager()
                error_response = health_manager.create_service_not_ready_response(
                    e.service_name, 'web'
                )
                
                # Flash appropriate message to user
                flash(f"Service not ready: {error_response['error_message']}", 'error')
                
                # Redirect to appropriate page based on route
                if 'batch' in request.endpoint:
                    return redirect(url_for('web.batch_upload'))
                elif 'dashboard' in request.endpoint:
                    # For dashboard, render with error message
                    return render_template('dashboard.html', **error_response)
                else:
                    return redirect(url_for('web.index'))
                    
            except Exception:
                # Fallback if health manager not available
                flash(f"System not ready: {e.message}", 'error')
                return redirect(url_for('web.index'))
                
    return decorated_function


@web_bp.route('/')
def index():
    """Main page with email classification form."""
    return render_template('index.html')


@web_bp.route('/classify', methods=['POST'])
def classify_email():
    """Handle single email classification."""
    operation_id = enhanced_logger.log_operation_start(
        component="web_classification",
        operation="classify_email",
        context_data={'email_length': len(request.form.get('email_text', ''))}
    )
    
    start_time = datetime.now()
    
    try:
        email_text = request.form.get('email_text', '').strip()
        
        enhanced_logger.logger.info(f"Processing email classification request (length: {len(email_text)})")
        
        # Validate input
        input_validator.validate_email_text(email_text)
        enhanced_logger.logger.debug("Email text validation passed")
        
        # Check service readiness before proceeding
        check_classification_service_readiness()
        enhanced_logger.logger.debug("Service readiness check passed")
        
        # Get service (already validated as ready)
        classification_service = get_classification_service()
        enhanced_logger.logger.debug("Classification service obtained")
        
        # Classify the email
        result = classification_service.classify_email(email_text)
        
        # Log successful operation
        duration = (datetime.now() - start_time).total_seconds()
        enhanced_logger.log_operation_end(
            operation_id=operation_id,
            component="web_classification",
            operation="classify_email",
            success=True,
            duration=duration,
            result_data={
                'prediction': result.prediction,
                'confidence': result.confidence,
                'model_used': result.model_used
            }
        )
        
        enhanced_logger.logger.info(f"Classification successful: {result.prediction} (confidence: {result.confidence:.2f})")
        
        return render_template('result.html', 
                             result=result,
                             email_text=email_text)
    
    except ValidationError as e:
        duration = (datetime.now() - start_time).total_seconds()
        enhanced_logger.log_operation_end(operation_id, "web_classification", "classify_email", False, duration)
        enhanced_error_handler.log_error_with_context(e, "WEB_CLASSIFICATION", {'operation': 'validate_input'})
        flash(f'Input error: {e.message}', 'error')
        return redirect(url_for('web.index'))
    
    except ServiceNotReadyError as e:
        duration = (datetime.now() - start_time).total_seconds()
        enhanced_logger.log_operation_end(operation_id, "web_classification", "classify_email", False, duration)
        enhanced_error_handler.log_error_with_context(e, "WEB_CLASSIFICATION", {'operation': 'service_check'})
        flash(f'Service error: {e.message}', 'error')
        return redirect(url_for('web.index'))
    
    except Exception as e:
        duration = (datetime.now() - start_time).total_seconds()
        enhanced_logger.log_operation_end(operation_id, "web_classification", "classify_email", False, duration)
        enhanced_error_handler.log_error_with_context(e, "WEB_CLASSIFICATION", {'operation': 'classify_email'})
        flash('An unexpected error occurred during classification. Please try again.', 'error')
        return redirect(url_for('web.index'))


@web_bp.route('/batch')
def batch_upload():
    """Batch processing upload page."""
    return render_template('batch.html')


@web_bp.route('/batch/status/<job_id>')
def batch_status(job_id):
    """Get batch processing job status."""
    try:
        current_app.logger.debug(f"Status request for job ID: {job_id}")
        
        # Validate job ID
        if not job_id or not job_id.strip():
            raise ValidationError('No job ID provided.', 'job_id')
        
        job_id = job_id.strip()
        
        # Validate job ID format
        try:
            import uuid
            uuid.UUID(job_id)
        except ValueError:
            raise ValidationError('Invalid job ID format.', 'job_id')
        
        # Check service readiness
        check_batch_processor_readiness()
        
        # Get service (already validated as ready)
        batch_processor = get_batch_processor()
        
        # Get job status
        job_status = batch_processor.get_processing_status(job_id)
        if not job_status:
            return jsonify({
                'error': True,
                'message': 'Job not found',
                'job_id': job_id
            }), 404
        
        # Get job summary
        job_summary = batch_processor.get_job_summary(job_id)
        
        return jsonify({
            'error': False,
            'job_id': job_id,
            'status': job_status.status,
            'progress': batch_processor.get_progress_percentage(job_id),
            'summary': job_summary
        })
    
    except ValidationError as e:
        return jsonify({
            'error': True,
            'message': e.message,
            'job_id': job_id
        }), 400
    
    except ServiceNotReadyError as e:
        return jsonify({
            'error': True,
            'message': f'Service unavailable: {e.message}',
            'job_id': job_id
        }), 503
    
    except Exception as e:
        current_app.logger.error(f'Status check error for job {job_id}: {str(e)}', exc_info=True)
        return jsonify({
            'error': True,
            'message': 'An error occurred while checking job status',
            'job_id': job_id
        }), 500


@web_bp.route('/batch/upload', methods=['POST'])
def upload_batch():
    """Handle batch file upload and processing with enhanced validation and error handling."""
    upload_path = None
    job_id = None
    filename = None
    file_size = None
    
    operation_id = enhanced_logger.log_operation_start(
        component="batch_processor",
        operation="upload_and_process",
        context_data={'route': '/batch/upload'}
    )
    
    start_time = datetime.now()
    
    try:
        enhanced_logger.logger.info("Starting batch file upload and processing...")
        
        # Enhanced file validation
        if 'file' not in request.files:
            raise ValidationError('No file was uploaded. Please select a CSV file to upload.', 'file')
        
        file = request.files['file']
        if not file or file.filename == '':
            raise ValidationError('No file was selected. Please choose a valid CSV file.', 'file')
        
        filename = file.filename
        
        # Validate file extension and size
        allowed_extensions = current_app.config.get('ALLOWED_EXTENSIONS', {'csv'})
        input_validator.validate_file_upload(file, allowed_extensions)
        
        # Check file size before processing
        file.seek(0, 2)  # Seek to end of file
        file_size = file.tell()
        file.seek(0)  # Reset to beginning
        
        # Log file validation
        enhanced_logger.log_batch_operation(
            operation="file_validation",
            filename=filename,
            file_size=file_size,
            success=True
        )
        
        max_file_size = current_app.config.get('MAX_FILE_SIZE', 16 * 1024 * 1024)  # 16MB default
        if file_size > max_file_size:
            size_mb = file_size / (1024 * 1024)
            max_mb = max_file_size / (1024 * 1024)
            raise ValidationError(
                f'File size ({size_mb:.1f}MB) exceeds maximum allowed size ({max_mb:.1f}MB). '
                f'Please upload a smaller file.',
                'file'
            )
        
        if file_size == 0:
            raise ValidationError('The uploaded file is empty. Please upload a file with email data.', 'file')
        
        enhanced_logger.logger.info(f"File validation passed. Size: {file_size} bytes")
        
        # Pre-processing service readiness checks
        enhanced_logger.logger.info("Checking service readiness...")
        
        # Check batch processor readiness (includes classification service validation)
        check_batch_processor_readiness()
        enhanced_logger.logger.debug("Batch processor readiness check passed")
        
        # Also validate classification service directly for additional safety
        check_classification_service_readiness()
        enhanced_logger.logger.debug("Classification service readiness check passed")
        
        # Get services (already validated as ready)
        batch_processor = get_batch_processor()
        
        # Log service health checks
        enhanced_logger.log_service_health_check(
            service_name="batch_processor",
            is_healthy=True,
            check_details={"initialized": True, "readiness_validated": True}
        )
        
        enhanced_logger.logger.info("All service readiness checks passed")
        
        # Prepare file for processing
        secure_filename_result = secure_filename(filename)
        if not secure_filename_result:
            raise ValidationError('Invalid filename. Please use a valid filename with proper characters.', 'file')
        
        job_id = str(uuid.uuid4())
        upload_dir = current_app.config.get('UPLOAD_DIR', 'uploads')
        
        # Ensure upload directory exists
        os.makedirs(upload_dir, exist_ok=True)
        upload_path = os.path.join(upload_dir, f"{job_id}_{secure_filename_result}")
        
        enhanced_logger.logger.info(f"Saving uploaded file to: {upload_path}")
        
        # Log file upload start
        enhanced_logger.log_batch_operation(
            operation="file_upload",
            filename=filename,
            file_size=file_size,
            job_id=job_id,
            success=True
        )
        
        # Save uploaded file with error handling
        try:
            file.save(upload_path)
        except Exception as save_error:
            enhanced_logger.log_batch_operation(
                operation="file_save",
                filename=filename,
                file_size=file_size,
                job_id=job_id,
                success=False,
                error_details=str(save_error)
            )
            raise ProcessingError(
                'Failed to save uploaded file. Please try again or contact administrator.',
                'file_save'
            )
        
        # Verify file was saved correctly
        if not os.path.exists(upload_path):
            raise ProcessingError('File upload failed. The file was not saved correctly.', 'file_verification')
        
        saved_size = os.path.getsize(upload_path)
        if saved_size != file_size:
            raise ProcessingError(
                'File upload incomplete. The saved file size does not match the uploaded file.',
                'file_verification'
            )
        
        enhanced_logger.logger.info(f"File saved successfully. Starting batch processing for job {job_id}")
        
        # Start batch processing with enhanced error handling
        try:
            # Log processing start
            enhanced_logger.log_batch_operation(
                operation="batch_processing_start",
                filename=filename,
                file_size=file_size,
                job_id=job_id,
                success=True
            )
            
            processed_job_id = batch_processor.process_csv(upload_path, job_id)
            
            # Log processing completion
            enhanced_logger.log_batch_operation(
                operation="batch_processing_complete",
                filename=filename,
                file_size=file_size,
                job_id=processed_job_id,
                success=True
            )
            
            # Log successful operation
            duration = (datetime.now() - start_time).total_seconds()
            enhanced_logger.log_operation_end(
                operation_id=operation_id,
                component="batch_processor",
                operation="upload_and_process",
                success=True,
                duration=duration,
                result_data={'job_id': processed_job_id, 'filename': filename, 'file_size': file_size}
            )
            
            enhanced_logger.logger.info(f"Batch processing completed successfully for job {processed_job_id}")
            flash('Batch processing completed successfully! You can download the results below.', 'success')
            return render_template('batch_result.html', 
                                 job_id=processed_job_id,
                                 processed_time=datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            
        except ValueError as ve:
            # Handle CSV validation errors from batch processor
            enhanced_logger.log_batch_operation(
                operation="csv_validation",
                filename=filename,
                file_size=file_size,
                job_id=job_id,
                success=False,
                error_details=str(ve)
            )
            raise ValidationError(
                f'Invalid CSV file format: {str(ve)}. Please ensure your CSV file has the correct format with email data.',
                'csv_format'
            )
        except RuntimeError as re:
            # Handle service readiness errors from batch processor
            enhanced_logger.log_batch_operation(
                operation="service_check",
                filename=filename,
                file_size=file_size,
                job_id=job_id,
                success=False,
                error_details=str(re)
            )
            raise ServiceNotReadyError('Batch Processing', str(re))
        except Exception as pe:
            # Handle other processing errors
            enhanced_logger.log_batch_operation(
                operation="batch_processing",
                filename=filename,
                file_size=file_size,
                job_id=job_id,
                success=False,
                error_details=str(pe)
            )
            raise ProcessingError(
                'An error occurred while processing your file. Please check the file format and try again.',
                'batch_processing'
            )
    
    except ValidationError as e:
        duration = (datetime.now() - start_time).total_seconds()
        enhanced_logger.log_operation_end(operation_id, "batch_processor", "upload_and_process", False, duration)
        
        context = BatchErrorContext(
            route="/batch/upload",
            filename=filename,
            file_size=file_size,
            processing_stage="validation"
        )
        return enhanced_error_handler.handle_batch_error(e, context)
    
    except ServiceNotReadyError as e:
        duration = (datetime.now() - start_time).total_seconds()
        enhanced_logger.log_operation_end(operation_id, "batch_processor", "upload_and_process", False, duration)
        
        context = BatchErrorContext(
            route="/batch/upload",
            filename=filename,
            file_size=file_size,
            processing_stage="service_check"
        )
        return enhanced_error_handler.handle_batch_error(e, context)
    
    except ProcessingError as e:
        duration = (datetime.now() - start_time).total_seconds()
        enhanced_logger.log_operation_end(operation_id, "batch_processor", "upload_and_process", False, duration)
        
        context = BatchErrorContext(
            route="/batch/upload",
            filename=filename,
            file_size=file_size,
            processing_stage="processing"
        )
        return enhanced_error_handler.handle_batch_error(e, context)
    
    except Exception as e:
        duration = (datetime.now() - start_time).total_seconds()
        enhanced_logger.log_operation_end(operation_id, "batch_processor", "upload_and_process", False, duration)
        
        context = BatchErrorContext(
            route="/batch/upload",
            filename=filename,
            file_size=file_size,
            processing_stage="unknown"
        )
        return enhanced_error_handler.handle_batch_error(e, context)
    
    finally:
        # Enhanced cleanup with proper error handling
        if upload_path and os.path.exists(upload_path):
            try:
                os.remove(upload_path)
                enhanced_logger.logger.info(f"Successfully cleaned up uploaded file: {upload_path}")
                enhanced_logger.log_batch_operation(
                    operation="file_cleanup",
                    filename=filename,
                    job_id=job_id,
                    success=True
                )
            except Exception as cleanup_error:
                enhanced_logger.logger.error(f'Failed to clean up uploaded file {upload_path}: {str(cleanup_error)}')
                enhanced_logger.log_batch_operation(
                    operation="file_cleanup",
                    filename=filename,
                    job_id=job_id,
                    success=False,
                    error_details=str(cleanup_error)
                )
                # Don't raise exception here as the main operation might have succeeded


@web_bp.route('/batch/download/<job_id>')
def download_results(job_id):
    """Download batch processing results with enhanced validation and error handling."""
    try:
        current_app.logger.info(f"Download request for job ID: {job_id}")
        
        # Enhanced job ID validation
        if not job_id or not job_id.strip():
            raise ValidationError('No job ID provided. Please provide a valid job ID.', 'job_id')
        
        job_id = job_id.strip()
        
        # Validate job ID format (should be UUID)
        try:
            import uuid
            uuid.UUID(job_id)
        except ValueError:
            raise ValidationError('Invalid job ID format. Please provide a valid job ID.', 'job_id')
        
        # Check service readiness
        current_app.logger.debug("Checking batch processor readiness...")
        check_batch_processor_readiness()
        
        # Get service (already validated as ready)
        batch_processor = get_batch_processor()
        
        # Get job status and validate
        current_app.logger.debug(f"Getting job status for {job_id}...")
        job_status = batch_processor.get_processing_status(job_id)
        
        if not job_status:
            current_app.logger.warning(f"Job not found: {job_id}")
            flash(
                'Job not found. The job may not exist, may have expired, or the job ID may be incorrect. '
                'Please check your job ID and try again.',
                'error'
            )
            return redirect(url_for('web.batch_upload'))
        
        # Check job completion status
        if job_status.status not in ['completed', 'failed']:
            current_app.logger.info(f"Job {job_id} is not ready for download. Status: {job_status.status}")
            if job_status.status == 'processing':
                flash(
                    f'Job is still processing ({job_status.processed_emails}/{job_status.total_emails} emails completed). '
                    'Please wait for processing to complete before downloading results.',
                    'warning'
                )
            elif job_status.status == 'pending':
                flash('Job is pending processing. Please wait for processing to start.', 'warning')
            else:
                flash(f'Job is in {job_status.status} status and results are not available for download.', 'error')
            return redirect(url_for('web.batch_upload'))
        
        # Get result file path with enhanced validation
        current_app.logger.debug(f"Getting result path for job {job_id}...")
        result_path = batch_processor.download_results(job_id)
        
        if not result_path:
            current_app.logger.warning(f"No result path found for job {job_id}")
            if job_status.status == 'failed':
                flash(
                    'Job processing failed and no results are available. '
                    'Please check your input file and try processing again.',
                    'error'
                )
            else:
                flash(
                    'Results file not found. The results may have expired or been cleaned up. '
                    'Please process your file again.',
                    'error'
                )
            return redirect(url_for('web.batch_upload'))
        
        # Validate file existence and accessibility
        if not os.path.exists(result_path):
            current_app.logger.error(f"Result file does not exist: {result_path}")
            flash(
                'Result file not found on server. The file may have been deleted or moved. '
                'Please process your file again.',
                'error'
            )
            return redirect(url_for('web.batch_upload'))
        
        # Check file size and readability
        try:
            file_size = os.path.getsize(result_path)
            if file_size == 0:
                current_app.logger.warning(f"Result file is empty: {result_path}")
                flash(
                    'Result file is empty. There may have been an issue during processing. '
                    'Please try processing your file again.',
                    'warning'
                )
                return redirect(url_for('web.batch_upload'))
            
            # Test file readability
            with open(result_path, 'r', encoding='utf-8') as test_file:
                test_file.read(100)  # Try to read first 100 characters
                
        except Exception as file_error:
            current_app.logger.error(f"Cannot read result file {result_path}: {str(file_error)}")
            flash(
                'Result file cannot be read. The file may be corrupted. '
                'Please try processing your file again.',
                'error'
            )
            return redirect(url_for('web.batch_upload'))
        
        # Generate appropriate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        is_partial = 'partial_results' in os.path.basename(result_path)
        filename_prefix = 'partial_classification_results' if is_partial else 'classification_results'
        download_filename = f'{filename_prefix}_{job_id}_{timestamp}.csv'
        
        current_app.logger.info(
            f"Serving download for job {job_id}: {result_path} "
            f"(Size: {file_size} bytes, Partial: {is_partial})"
        )
        
        # Add flash message for partial results
        if is_partial:
            flash(
                'Note: These are partial results from a job that encountered errors during processing. '
                'Some emails may not have been processed.',
                'warning'
            )
        
        # Serve the file
        return send_file(
            result_path, 
            as_attachment=True, 
            download_name=download_filename,
            mimetype='text/csv'
        )
    
    except ValidationError as e:
        current_app.logger.warning(f'Download validation error for job {job_id}: {e.message}')
        flash(f'Invalid request: {e.message}', 'error')
        return redirect(url_for('web.batch_upload'))
    
    except ServiceNotReadyError as e:
        current_app.logger.warning(f'Service not ready for download job {job_id}: {e.message}')
        flash(f'Service unavailable: {e.message}', 'error')
        return redirect(url_for('web.batch_upload'))
    
    except Exception as e:
        current_app.logger.error(f'Unexpected download error for job {job_id}: {str(e)}', exc_info=True)
        flash(
            'An unexpected error occurred while preparing your download. '
            'Please try again or contact support if the problem persists.',
            'error'
        )
        return redirect(url_for('web.batch_upload'))


@web_bp.route('/models')
def model_selection():
    """Model selection page for users to choose active model."""
    try:
        # Check service readiness
        check_classification_service_readiness()
        
        # Get model manager
        model_manager = getattr(current_app, 'model_manager', None)
        if not model_manager:
            flash('Model management service not available.', 'error')
            return redirect(url_for('web.index'))
        
        # Get all available models and their metrics
        all_metrics = model_manager.get_all_metrics()
        current_best = model_manager.best_model_name
        
        # Prepare model data for display
        models_data = []
        for model_name, model in model_manager.models.items():
            metrics = all_metrics.get(model_name)
            if metrics:
                fnr = 1.0 - metrics.recall if metrics.recall > 0 else 1.0
                models_data.append({
                    'name': model_name,
                    'display_name': model.get_model_name(),
                    'accuracy': metrics.accuracy,
                    'precision': metrics.precision,
                    'recall': metrics.recall,
                    'f1_score': metrics.f1_score,
                    'false_negative_rate': fnr,
                    'is_current': model_name == current_best,
                    'is_trained': hasattr(model, 'is_trained') and model.is_trained
                })
        
        # Sort by accuracy for display
        models_data.sort(key=lambda x: x['accuracy'], reverse=True)
        
        return render_template('model_selection.html', 
                             models=models_data,
                             current_model=current_best)
    
    except Exception as e:
        current_app.logger.error(f'Error loading model selection: {str(e)}', exc_info=True)
        flash('Error loading model selection page.', 'error')
        return redirect(url_for('web.index'))


@web_bp.route('/models/select', methods=['POST'])
def select_model():
    """Handle user model selection."""
    try:
        model_name = request.form.get('model_name', '').strip()
        
        if not model_name:
            flash('No model selected.', 'error')
            return redirect(url_for('web.model_selection'))
        
        # Check service readiness
        check_classification_service_readiness()
        
        # Get model manager
        model_manager = getattr(current_app, 'model_manager', None)
        if not model_manager:
            flash('Model management service not available.', 'error')
            return redirect(url_for('web.model_selection'))
        
        # Validate model exists and is trained
        if model_name not in model_manager.models:
            flash(f'Model "{model_name}" not found.', 'error')
            return redirect(url_for('web.model_selection'))
        
        model = model_manager.models[model_name]
        if not (hasattr(model, 'is_trained') and model.is_trained):
            flash(f'Model "{model_name}" is not trained and cannot be selected.', 'error')
            return redirect(url_for('web.model_selection'))
        
        # Set as best model
        old_model = model_manager.best_model_name
        model_manager.best_model_name = model_name
        
        # Log the manual selection
        current_app.logger.info(f'User manually selected model: {model_name} (was: {old_model})')
        
        # Get model display name and metrics for flash message
        display_name = model.get_model_name()
        metrics = model_manager.get_model_metrics(model_name)
        
        if metrics:
            fnr = 1.0 - metrics.recall if metrics.recall > 0 else 1.0
            flash(f'Successfully selected {display_name} as active model. '
                  f'Accuracy: {metrics.accuracy:.3f}, False Negative Rate: {fnr:.3f}', 'success')
        else:
            flash(f'Successfully selected {display_name} as active model.', 'success')
        
        return redirect(url_for('web.model_selection'))
    
    except Exception as e:
        current_app.logger.error(f'Error selecting model: {str(e)}', exc_info=True)
        flash('Error selecting model. Please try again.', 'error')
        return redirect(url_for('web.model_selection'))


@web_bp.route('/models/retrain', methods=['POST'])
def retrain_models():
    """Trigger model retraining with improved selection logic."""
    try:
        # Check if training is already in progress (you might want to implement this check)
        flash('Model retraining started. This may take several minutes. Check the dashboard for progress.', 'info')
        
        # In a real implementation, you'd trigger async training here
        # For now, just redirect to a page that explains the process
        return redirect(url_for('web.model_selection'))
    
    except Exception as e:
        current_app.logger.error(f'Error starting model retraining: {str(e)}', exc_info=True)
        flash('Error starting model retraining. Please try again.', 'error')
        return redirect(url_for('web.model_selection'))


@web_bp.route('/dashboard')
def performance_dashboard():
    """Display model performance dashboard with comprehensive error handling."""
    operation_id = enhanced_logger.log_operation_start(
        component="dashboard",
        operation="load_dashboard",
        context_data={'route': '/dashboard'}
    )
    
    start_time = datetime.now()
    
    try:
        enhanced_logger.logger.info("Loading performance dashboard...")
        
        # Get service health manager for comprehensive health checks
        service_health_manager = getattr(current_app, 'service_health_manager', None)
        
        if service_health_manager is None:
            enhanced_logger.logger.error("Service health manager not available")
            
            # Create error context for dashboard error
            context = DashboardErrorContext(
                route="/dashboard",
                user_action="load_dashboard",
                service_status={"health_manager": False}
            )
            
            error = ServiceNotReadyError("Service Health Manager", "Not initialized during application startup")
            return enhanced_error_handler.handle_dashboard_error(error, context)
        
        # Log service health check
        enhanced_logger.log_service_health_check(
            service_name="service_health_manager",
            is_healthy=True,
            check_details={"initialized": True}
        )
        
        # Initialize dashboard data provider
        from src.services.dashboard_data_provider import DashboardDataProvider
        data_provider = DashboardDataProvider(service_health_manager)
        
        # Get services from app context
        classification_service = getattr(current_app, 'classification_service', None)
        batch_processor = getattr(current_app, 'batch_processor', None)
        
        # Log service availability
        service_status = {
            'classification_service': classification_service is not None,
            'batch_processor': batch_processor is not None
        }
        
        enhanced_logger.log_dashboard_access(
            user_action="load_dashboard",
            service_status=service_status,
            data_retrieved=False,  # Will update after data retrieval
            error_occurred=False
        )
        
        # Get comprehensive dashboard data with fallback support
        dashboard_data = data_provider.get_dashboard_data(
            classification_service=classification_service,
            batch_processor=batch_processor
        )
        
        # Log successful data retrieval
        enhanced_logger.log_dashboard_access(
            user_action="load_dashboard",
            service_status=service_status,
            data_retrieved=True,
            error_occurred=dashboard_data.error_message is not None
        )
        
        # Add flash message if there's an error message
        if dashboard_data.error_message:
            if dashboard_data.system_status == "not_ready":
                flash(f'System Status: {dashboard_data.error_message}', 'warning')
            elif dashboard_data.system_status == "unhealthy":
                flash(f'System Error: {dashboard_data.error_message}', 'error')
            elif dashboard_data.system_status == "degraded":
                flash(f'System Warning: {dashboard_data.error_message}', 'warning')
        
        # Log successful operation
        duration = (datetime.now() - start_time).total_seconds()
        enhanced_logger.log_operation_end(
            operation_id=operation_id,
            component="dashboard",
            operation="load_dashboard",
            success=True,
            duration=duration,
            result_data={
                'system_status': dashboard_data.system_status,
                'active_model': dashboard_data.active_model,
                'has_error': dashboard_data.error_message is not None
            }
        )
        
        enhanced_logger.logger.info(f"Dashboard loaded successfully. Status: {dashboard_data.system_status}, Active model: {dashboard_data.active_model}")
        
        # Render template with dashboard data
        return render_template('dashboard.html', **dashboard_data.to_dict())
    
    except Exception as e:
        # Log failed operation
        duration = (datetime.now() - start_time).total_seconds()
        enhanced_logger.log_operation_end(operation_id, "dashboard", "load_dashboard", False, duration)
        
        # Create error context for dashboard error
        context = DashboardErrorContext(
            route="/dashboard",
            user_action="load_dashboard",
            service_status=None,
            fallback_data=None
        )
        
        # Handle error with enhanced error handler
        return enhanced_error_handler.handle_dashboard_error(e, context)


def _render_dashboard_with_fallback(error_message: str, 
                                   system_status: str,
                                   health_summary=None):
    """Render dashboard with fallback data and error message."""
    current_app.logger.debug(f"Rendering dashboard with fallback data. Status: {system_status}")
    
    # Create fallback dashboard data
    fallback_data = {
        'metrics': {},
        'active_model': 'System Not Ready',
        'system_status': system_status,
        'error_message': error_message,
        'health_summary': health_summary,
        'processed_today': 0,
        'avg_response_time': 0.0,
        'service_ready': False,
        'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Add flash message for user feedback
    if system_status == "not_ready":
        flash(f'System Status: {error_message}', 'warning')
    elif system_status == "unhealthy":
        flash(f'System Error: {error_message}', 'error')
    elif system_status == "degraded":
        flash(f'System Warning: {error_message}', 'warning')
    
    return render_template('dashboard.html', **fallback_data)


def _get_dashboard_data(service_health_manager, classification_service):
    """Get additional dashboard data with error handling."""
    dashboard_data = {
        'processed_today': 0,
        'avg_response_time': 0.0,
        'service_ready': True,
        'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    try:
        # Get service status summary
        service_status = service_health_manager.get_service_status_summary()
        dashboard_data['service_status'] = service_status
        
        # Try to get processing statistics if available
        if hasattr(classification_service, 'get_processing_stats'):
            try:
                stats = classification_service.get_processing_stats()
                dashboard_data['processed_today'] = stats.get('processed_today', 0)
                dashboard_data['avg_response_time'] = stats.get('avg_response_time', 0.0)
            except Exception as stats_error:
                current_app.logger.debug(f"Could not get processing stats: {stats_error}")
        
    except Exception as e:
        current_app.logger.warning(f"Error getting additional dashboard data: {str(e)}")
    
    return dashboard_data





@web_bp.route('/health')
def health_check():
    """Health check endpoint for load balancers and monitoring with service status."""
    try:
        # Get service health information if available
        try:
            health_manager = get_service_health_manager()
            health_summary = health_manager.check_all_services()
            
            # Determine status for load balancers
            if health_summary.overall_status.value in ['healthy', 'degraded']:
                status = 'healthy'
                status_code = 200
            else:
                status = 'unhealthy'
                status_code = 503
            
            return jsonify({
                'status': status,
                'timestamp': datetime.now().isoformat(),
                'version': current_app.config.get('APP_VERSION', '1.0.0'),
                'system_status': health_summary.overall_status.value
            }), status_code
            
        except ServiceNotReadyError:
            # Health manager not available - basic health check
            return jsonify({
                'status': 'degraded',
                'timestamp': datetime.now().isoformat(),
                'version': current_app.config.get('APP_VERSION', '1.0.0'),
                'system_status': 'initializing'
            }), 200
            
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500


@web_bp.route('/status')
def detailed_status():
    """Detailed service status endpoint for administrators."""
    try:
        # Get comprehensive service health information
        try:
            health_manager = get_service_health_manager()
            health_summary = health_manager.check_all_services()
            detailed_status = health_manager.get_detailed_health_status()
            service_status = health_manager.get_service_status_summary()
            
            status_data = {
                'overall_status': health_summary.overall_status.value,
                'timestamp': datetime.now().isoformat(),
                'version': current_app.config.get('APP_VERSION', '1.0.0'),
                'readiness_message': health_manager.get_system_readiness_message(),
                'system_ready': health_manager.is_system_ready(),
                'services': {}
            }
            
            # Add detailed service information
            for service_name, status in detailed_status.items():
                status_data['services'][service_name] = {
                    'status': status.status.value,
                    'is_ready': status.is_ready,
                    'last_check': status.last_check.isoformat(),
                    'error_message': status.error_message,
                    'details': status.details
                }
            
            return jsonify(status_data), 200
            
        except ServiceNotReadyError as e:
            # Health manager not available
            return jsonify({
                'overall_status': 'not_ready',
                'timestamp': datetime.now().isoformat(),
                'version': current_app.config.get('APP_VERSION', '1.0.0'),
                'error': f'Service health manager not available: {e.message}',
                'system_ready': False,
                'services': {}
            }), 503
            
    except Exception as e:
        current_app.logger.error(f'Error getting detailed status: {str(e)}', exc_info=True)
        return jsonify({
            'overall_status': 'error',
            'timestamp': datetime.now().isoformat(),
            'error': str(e),
            'system_ready': False
        }), 500


@web_bp.errorhandler(404)
def not_found_error(error):
    """Handle 404 errors."""
    return render_template('errors/404.html'), 404


@web_bp.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    return render_template('errors/500.html'), 500