// Email Spam Classifier - Main JavaScript

document.addEventListener('DOMContentLoaded', function() {
    // Initialize all components
    initializeFormHandlers();
    initializeFileUpload();
    initializeProgressBars();
    initializeTooltips();
    initializeAnimations();
});

// Form Handlers
function initializeFormHandlers() {
    // Email classification form
    const classifyForm = document.getElementById('classifyForm');
    if (classifyForm) {
        classifyForm.addEventListener('submit', handleClassifySubmit);
    }

    // Batch upload form
    const batchForm = document.getElementById('batchForm');
    if (batchForm) {
        batchForm.addEventListener('submit', handleBatchSubmit);
    }

    // Clear form functionality
    const clearButtons = document.querySelectorAll('[onclick="clearForm()"]');
    clearButtons.forEach(button => {
        button.addEventListener('click', clearForm);
    });
}

// Handle email classification form submission
function handleClassifySubmit(event) {
    const form = event.target;
    const submitBtn = form.querySelector('button[type="submit"]');
    const spinner = form.querySelector('.spinner-border');
    const emailText = form.querySelector('#email_text').value.trim();

    // Validate input
    if (!emailText) {
        showAlert('Please enter email content to classify.', 'warning');
        event.preventDefault();
        return false;
    }

    // Show loading state
    setLoadingState(submitBtn, true, 'Processing...');
    
    // Add fade out animation to form
    form.classList.add('fade-out');
}

// Handle batch upload form submission
function handleBatchSubmit(event) {
    const form = event.target;
    const submitBtn = form.querySelector('button[type="submit"]');
    const fileInput = form.querySelector('#file');

    // Validate file selection
    if (!fileInput.files.length) {
        showAlert('Please select a CSV file to upload.', 'warning');
        event.preventDefault();
        return false;
    }

    // Validate file type
    const file = fileInput.files[0];
    if (!file.name.toLowerCase().endsWith('.csv')) {
        showAlert('Please select a valid CSV file.', 'danger');
        event.preventDefault();
        return false;
    }

    // Validate file size (16MB limit)
    if (file.size > 16 * 1024 * 1024) {
        showAlert('File size exceeds 16MB limit. Please choose a smaller file.', 'danger');
        event.preventDefault();
        return false;
    }

    // Show loading state
    setLoadingState(submitBtn, true, 'Uploading...');
    
    // Show progress indicator
    showUploadProgress();
}

// File Upload Enhancements
function initializeFileUpload() {
    const fileInput = document.getElementById('file');
    if (!fileInput) return;

    // File input change handler
    fileInput.addEventListener('change', function() {
        const file = this.files[0];
        if (file) {
            validateFile(file);
            updateFileInfo(file);
        }
    });

    // Drag and drop functionality
    const uploadArea = fileInput.closest('.card-body');
    if (uploadArea) {
        setupDragAndDrop(uploadArea, fileInput);
    }
}

// Setup drag and drop for file upload
function setupDragAndDrop(uploadArea, fileInput) {
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        uploadArea.addEventListener(eventName, preventDefaults, false);
    });

    ['dragenter', 'dragover'].forEach(eventName => {
        uploadArea.addEventListener(eventName, highlight, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        uploadArea.addEventListener(eventName, unhighlight, false);
    });

    uploadArea.addEventListener('drop', handleDrop, false);

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    function highlight() {
        uploadArea.classList.add('dragover');
    }

    function unhighlight() {
        uploadArea.classList.remove('dragover');
    }

    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        
        if (files.length > 0) {
            fileInput.files = files;
            validateFile(files[0]);
            updateFileInfo(files[0]);
        }
    }
}

// Validate uploaded file
function validateFile(file) {
    const errors = [];

    // Check file type
    if (!file.name.toLowerCase().endsWith('.csv')) {
        errors.push('File must be a CSV file');
    }

    // Check file size
    if (file.size > 16 * 1024 * 1024) {
        errors.push('File size must be less than 16MB');
    }

    // Check if file is empty
    if (file.size === 0) {
        errors.push('File cannot be empty');
    }

    if (errors.length > 0) {
        showAlert(errors.join('. '), 'danger');
        return false;
    }

    return true;
}

// Update file information display
function updateFileInfo(file) {
    const fileSize = formatFileSize(file.size);
    const fileName = file.name;
    
    // Create or update file info display
    let fileInfo = document.querySelector('.file-info');
    if (!fileInfo) {
        fileInfo = document.createElement('div');
        fileInfo.className = 'file-info mt-2 p-2 bg-light rounded';
        document.getElementById('file').parentNode.appendChild(fileInfo);
    }
    
    fileInfo.innerHTML = `
        <small class="text-muted">
            <i class="fas fa-file-csv text-success"></i>
            <strong>${fileName}</strong> (${fileSize})
        </small>
    `;
}

// Progress Bar Initialization
function initializeProgressBars() {
    // Animate progress bars on page load
    const progressBars = document.querySelectorAll('.progress-bar');
    progressBars.forEach(bar => {
        const width = bar.style.width;
        bar.style.width = '0%';
        setTimeout(() => {
            bar.style.transition = 'width 1s ease-in-out';
            bar.style.width = width;
        }, 500);
    });
}

// Show upload progress
function showUploadProgress() {
    const progressHtml = `
        <div class="upload-progress mt-3">
            <div class="d-flex justify-content-between align-items-center mb-2">
                <small class="text-muted">Uploading file...</small>
                <small class="text-muted">0%</small>
            </div>
            <div class="progress">
                <div class="progress-bar progress-bar-striped progress-bar-animated" 
                     role="progressbar" style="width: 0%"></div>
            </div>
        </div>
    `;
    
    const form = document.getElementById('batchForm');
    if (form && !form.querySelector('.upload-progress')) {
        form.insertAdjacentHTML('beforeend', progressHtml);
        
        // Simulate progress (since we can't track real upload progress easily)
        simulateProgress();
    }
}

// Simulate upload progress
function simulateProgress() {
    const progressBar = document.querySelector('.upload-progress .progress-bar');
    const progressText = document.querySelector('.upload-progress small:last-child');
    
    if (!progressBar) return;
    
    let progress = 0;
    const interval = setInterval(() => {
        progress += Math.random() * 15;
        if (progress > 90) progress = 90;
        
        progressBar.style.width = progress + '%';
        progressText.textContent = Math.round(progress) + '%';
        
        if (progress >= 90) {
            clearInterval(interval);
            progressText.textContent = 'Processing...';
        }
    }, 200);
}

// Initialize tooltips
function initializeTooltips() {
    // Initialize Bootstrap tooltips if available
    if (typeof bootstrap !== 'undefined' && bootstrap.Tooltip) {
        const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
        tooltipTriggerList.map(function (tooltipTriggerEl) {
            return new bootstrap.Tooltip(tooltipTriggerEl);
        });
    }
}

// Initialize animations
function initializeAnimations() {
    // Add fade-in animation to cards
    const cards = document.querySelectorAll('.card');
    cards.forEach((card, index) => {
        card.style.animationDelay = (index * 0.1) + 's';
        card.classList.add('fade-in');
    });

    // Add slide-up animation to alerts
    const alerts = document.querySelectorAll('.alert');
    alerts.forEach(alert => {
        alert.classList.add('slide-up');
    });
}

// Utility Functions
function setLoadingState(button, isLoading, text = 'Loading...') {
    if (isLoading) {
        button.disabled = true;
        button.innerHTML = `
            <span class="spinner-border spinner-border-sm me-2" role="status"></span>
            ${text}
        `;
    } else {
        button.disabled = false;
        // Restore original button content
        const originalText = button.getAttribute('data-original-text') || 'Submit';
        button.innerHTML = originalText;
    }
}

function showAlert(message, type = 'info') {
    // Create alert element
    const alertHtml = `
        <div class="alert alert-${type} alert-dismissible fade show slide-up" role="alert">
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        </div>
    `;
    
    // Insert at top of main content
    const main = document.querySelector('main.container');
    if (main) {
        main.insertAdjacentHTML('afterbegin', alertHtml);
        
        // Auto-dismiss after 5 seconds
        setTimeout(() => {
            const alert = main.querySelector('.alert');
            if (alert && alert.classList.contains('show')) {
                alert.classList.remove('show');
                setTimeout(() => alert.remove(), 150);
            }
        }, 5000);
    }
}

function clearForm() {
    const emailTextarea = document.getElementById('email_text');
    if (emailTextarea) {
        emailTextarea.value = '';
        emailTextarea.focus();
    }
    
    // Clear file input
    const fileInput = document.getElementById('file');
    if (fileInput) {
        fileInput.value = '';
        
        // Remove file info display
        const fileInfo = document.querySelector('.file-info');
        if (fileInfo) {
            fileInfo.remove();
        }
    }
    
    // Remove any progress indicators
    const uploadProgress = document.querySelector('.upload-progress');
    if (uploadProgress) {
        uploadProgress.remove();
    }
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

// Dashboard specific functions
function refreshDashboard() {
    // Refresh dashboard data
    if (window.location.pathname.includes('dashboard')) {
        location.reload();
    }
}

// Real-time updates for dashboard (if WebSocket is available)
function initializeRealTimeUpdates() {
    // This would connect to WebSocket for real-time updates
    // For now, we'll use periodic refresh
    if (window.location.pathname.includes('dashboard')) {
        setInterval(refreshDashboard, 30000); // Refresh every 30 seconds
    }
}

// Chart animations (if Chart.js is available)
function animateCharts() {
    if (typeof Chart !== 'undefined') {
        Chart.defaults.animation.duration = 1000;
        Chart.defaults.animation.easing = 'easeInOutQuart';
    }
}

// Error handling for AJAX requests
function handleAjaxError(xhr, status, error) {
    console.error('AJAX Error:', status, error);
    showAlert('An error occurred while processing your request. Please try again.', 'danger');
}

// Initialize real-time updates and chart animations
document.addEventListener('DOMContentLoaded', function() {
    initializeRealTimeUpdates();
    animateCharts();
});

// Export functions for global access
window.clearForm = clearForm;
window.showAlert = showAlert;
window.setLoadingState = setLoadingState;

// Enhanced User Experience Features

// Form validation enhancements
function initializeFormValidation() {
    const forms = document.querySelectorAll('form');
    forms.forEach(form => {
        form.addEventListener('submit', function(event) {
            if (!validateForm(this)) {
                event.preventDefault();
                event.stopPropagation();
            }
            this.classList.add('was-validated');
        });

        // Real-time validation
        const inputs = form.querySelectorAll('input, textarea, select');
        inputs.forEach(input => {
            input.addEventListener('blur', function() {
                validateField(this);
            });
            
            input.addEventListener('input', function() {
                if (this.classList.contains('is-invalid')) {
                    validateField(this);
                }
            });
        });
    });
}

// Validate individual form field
function validateField(field) {
    const value = field.value.trim();
    let isValid = true;
    let message = '';

    // Email content validation
    if (field.id === 'email_text') {
        if (!value) {
            isValid = false;
            message = 'Email content is required';
        } else if (value.length < 10) {
            isValid = false;
            message = 'Email content must be at least 10 characters';
        } else if (value.length > 10000) {
            isValid = false;
            message = 'Email content must be less than 10,000 characters';
        }
    }

    // File upload validation
    if (field.type === 'file') {
        if (field.files.length === 0) {
            isValid = false;
            message = 'Please select a file';
        } else {
            const file = field.files[0];
            if (!file.name.toLowerCase().endsWith('.csv')) {
                isValid = false;
                message = 'Please select a CSV file';
            } else if (file.size > 16 * 1024 * 1024) {
                isValid = false;
                message = 'File size must be less than 16MB';
            }
        }
    }

    // Update field state
    updateFieldValidation(field, isValid, message);
    return isValid;
}

// Update field validation state
function updateFieldValidation(field, isValid, message) {
    field.classList.remove('is-valid', 'is-invalid');
    
    // Remove existing feedback
    const existingFeedback = field.parentNode.querySelector('.invalid-feedback, .valid-feedback');
    if (existingFeedback) {
        existingFeedback.remove();
    }

    if (isValid) {
        field.classList.add('is-valid');
        if (field.value.trim()) {
            const feedback = document.createElement('div');
            feedback.className = 'valid-feedback';
            feedback.textContent = 'Looks good!';
            field.parentNode.appendChild(feedback);
        }
    } else {
        field.classList.add('is-invalid');
        const feedback = document.createElement('div');
        feedback.className = 'invalid-feedback';
        feedback.textContent = message;
        field.parentNode.appendChild(feedback);
    }
}

// Validate entire form
function validateForm(form) {
    const fields = form.querySelectorAll('input[required], textarea[required], select[required]');
    let isValid = true;

    fields.forEach(field => {
        if (!validateField(field)) {
            isValid = false;
        }
    });

    return isValid;
}

// Enhanced loading states
function showGlobalLoading(message = 'Loading...') {
    const overlay = document.createElement('div');
    overlay.className = 'loading-overlay';
    overlay.innerHTML = `
        <div class="loading-spinner">
            <div class="progress-circular"></div>
            <p class="mt-3 mb-0">${message}</p>
        </div>
    `;
    document.body.appendChild(overlay);
    document.body.style.overflow = 'hidden';
}

function hideGlobalLoading() {
    const overlay = document.querySelector('.loading-overlay');
    if (overlay) {
        overlay.remove();
        document.body.style.overflow = '';
    }
}

// Toast notifications
function showToast(message, type = 'info', duration = 5000) {
    const toastContainer = getOrCreateToastContainer();
    const toastId = 'toast-' + Date.now();
    
    const toastHtml = `
        <div class="toast" id="${toastId}" role="alert" aria-live="assertive" aria-atomic="true">
            <div class="toast-header">
                <i class="fas fa-${getToastIcon(type)} text-${type} me-2"></i>
                <strong class="me-auto">Notification</strong>
                <small class="text-muted">now</small>
                <button type="button" class="btn-close" data-bs-dismiss="toast" aria-label="Close"></button>
            </div>
            <div class="toast-body">
                ${message}
            </div>
        </div>
    `;
    
    toastContainer.insertAdjacentHTML('beforeend', toastHtml);
    
    const toastElement = document.getElementById(toastId);
    const toast = new bootstrap.Toast(toastElement, { delay: duration });
    toast.show();
    
    // Remove toast element after it's hidden
    toastElement.addEventListener('hidden.bs.toast', function() {
        this.remove();
    });
}

function getOrCreateToastContainer() {
    let container = document.querySelector('.toast-container');
    if (!container) {
        container = document.createElement('div');
        container.className = 'toast-container';
        document.body.appendChild(container);
    }
    return container;
}

function getToastIcon(type) {
    const icons = {
        success: 'check-circle',
        danger: 'exclamation-triangle',
        warning: 'exclamation-triangle',
        info: 'info-circle'
    };
    return icons[type] || 'info-circle';
}

// Keyboard navigation enhancements
function initializeKeyboardNavigation() {
    document.addEventListener('keydown', function(event) {
        // Escape key to close modals and overlays
        if (event.key === 'Escape') {
            const overlay = document.querySelector('.loading-overlay');
            if (overlay) {
                hideGlobalLoading();
            }
            
            const openToasts = document.querySelectorAll('.toast.show');
            openToasts.forEach(toast => {
                const bsToast = bootstrap.Toast.getInstance(toast);
                if (bsToast) {
                    bsToast.hide();
                }
            });
        }
        
        // Ctrl+Enter to submit forms
        if (event.ctrlKey && event.key === 'Enter') {
            const activeForm = document.activeElement.closest('form');
            if (activeForm) {
                const submitBtn = activeForm.querySelector('button[type="submit"]');
                if (submitBtn && !submitBtn.disabled) {
                    submitBtn.click();
                }
            }
        }
    });
}

// Accessibility enhancements
function initializeAccessibility() {
    // Skip link removed per user request
    
    // Add main content ID
    const mainContent = document.querySelector('main');
    if (mainContent) {
        mainContent.id = 'main-content';
    }
    
    // Enhance form labels
    const inputs = document.querySelectorAll('input, textarea, select');
    inputs.forEach(input => {
        if (!input.getAttribute('aria-label') && !input.getAttribute('aria-labelledby')) {
            const label = document.querySelector(`label[for="${input.id}"]`);
            if (label) {
                input.setAttribute('aria-labelledby', label.id || `label-${input.id}`);
                if (!label.id) {
                    label.id = `label-${input.id}`;
                }
            }
        }
    });
    
    // Add ARIA live regions for dynamic content
    const liveRegion = document.createElement('div');
    liveRegion.setAttribute('aria-live', 'polite');
    liveRegion.setAttribute('aria-atomic', 'true');
    liveRegion.className = 'sr-only';
    liveRegion.id = 'live-region';
    document.body.appendChild(liveRegion);
}

// Announce to screen readers
function announceToScreenReader(message) {
    const liveRegion = document.getElementById('live-region');
    if (liveRegion) {
        liveRegion.textContent = message;
        setTimeout(() => {
            liveRegion.textContent = '';
        }, 1000);
    }
}

// Performance monitoring
function initializePerformanceMonitoring() {
    // Monitor page load performance
    window.addEventListener('load', function() {
        if ('performance' in window) {
            const loadTime = performance.timing.loadEventEnd - performance.timing.navigationStart;
            console.log(`Page load time: ${loadTime}ms`);
            
            // Report slow page loads
            if (loadTime > 3000) {
                console.warn('Slow page load detected');
            }
        }
    });
    
    // Monitor form submission performance
    document.addEventListener('submit', function(event) {
        const form = event.target;
        const startTime = performance.now();
        
        form.addEventListener('load', function() {
            const endTime = performance.now();
            const submitTime = endTime - startTime;
            console.log(`Form submission time: ${submitTime}ms`);
        });
    });
}

// Error boundary for JavaScript errors
function initializeErrorHandling() {
    window.addEventListener('error', function(event) {
        console.error('JavaScript error:', event.error);
        
        // Show user-friendly error message
        showAlert('An unexpected error occurred. Please refresh the page and try again.', 'danger');
        
        // Log error details for debugging
        const errorDetails = {
            message: event.message,
            filename: event.filename,
            lineno: event.lineno,
            colno: event.colno,
            stack: event.error ? event.error.stack : null,
            timestamp: new Date().toISOString(),
            userAgent: navigator.userAgent,
            url: window.location.href
        };
        
        console.error('Error details:', errorDetails);
    });
    
    // Handle unhandled promise rejections
    window.addEventListener('unhandledrejection', function(event) {
        console.error('Unhandled promise rejection:', event.reason);
        showAlert('A network error occurred. Please check your connection and try again.', 'warning');
    });
}

// Initialize all enhanced features
document.addEventListener('DOMContentLoaded', function() {
    initializeFormValidation();
    initializeKeyboardNavigation();
    initializeAccessibility();
    initializePerformanceMonitoring();
    initializeErrorHandling();
});

// Enhanced utility functions
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

function throttle(func, limit) {
    let inThrottle;
    return function() {
        const args = arguments;
        const context = this;
        if (!inThrottle) {
            func.apply(context, args);
            inThrottle = true;
            setTimeout(() => inThrottle = false, limit);
        }
    };
}

// Enhanced file handling
function previewFile(file) {
    if (file.type === 'text/csv') {
        const reader = new FileReader();
        reader.onload = function(e) {
            const content = e.target.result;
            const lines = content.split('\n').slice(0, 5); // Show first 5 lines
            
            const preview = document.createElement('div');
            preview.className = 'file-preview mt-2 p-2 bg-light rounded';
            preview.innerHTML = `
                <h6>File Preview:</h6>
                <pre class="mb-0" style="font-size: 0.8rem; max-height: 150px; overflow-y: auto;">${lines.join('\n')}</pre>
                ${content.split('\n').length > 5 ? '<small class="text-muted">... and more</small>' : ''}
            `;
            
            // Remove existing preview
            const existingPreview = document.querySelector('.file-preview');
            if (existingPreview) {
                existingPreview.remove();
            }
            
            // Add new preview
            const fileInput = document.getElementById('file');
            if (fileInput) {
                fileInput.parentNode.appendChild(preview);
            }
        };
        reader.readAsText(file);
    }
}

// Export enhanced functions
window.showToast = showToast;
window.showGlobalLoading = showGlobalLoading;
window.hideGlobalLoading = hideGlobalLoading;
window.announceToScreenReader = announceToScreenReader;
window.previewFile = previewFile;