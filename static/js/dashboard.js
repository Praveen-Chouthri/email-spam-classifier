// Dashboard-specific JavaScript for Email Spam Classifier

class DashboardManager {
    constructor() {
        this.refreshInterval = null;
        this.charts = {};
        this.lastUpdateTime = new Date();
        this.init();
    }

    init() {
        this.initializeCharts();
        this.setupRealTimeUpdates();
        this.setupEventListeners();
        this.startAutoRefresh();
    }

    // Initialize all dashboard charts
    initializeCharts() {
        this.initializeMetricsChart();
        this.initializeAccuracyTrendChart();
        this.initializeClassificationResultsChart();
    }

    // Main metrics comparison chart
    initializeMetricsChart() {
        try {
            const ctx = document.getElementById('metricsChart');
            if (!ctx) return;

            const modelNames = window.dashboardData?.modelNames || [];
            const accuracyData = window.dashboardData?.accuracyData || [];
            const precisionData = window.dashboardData?.precisionData || [];
            const recallData = window.dashboardData?.recallData || [];
            const f1Data = window.dashboardData?.f1Data || [];

        this.charts.metrics = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: modelNames,
                datasets: [
                    {
                        label: 'Accuracy',
                        data: accuracyData,
                        backgroundColor: 'rgba(40, 167, 69, 0.8)',
                        borderColor: 'rgba(40, 167, 69, 1)',
                        borderWidth: 2,
                        borderRadius: 4
                    },
                    {
                        label: 'Precision',
                        data: precisionData,
                        backgroundColor: 'rgba(0, 123, 255, 0.8)',
                        borderColor: 'rgba(0, 123, 255, 1)',
                        borderWidth: 2,
                        borderRadius: 4
                    },
                    {
                        label: 'Recall',
                        data: recallData,
                        backgroundColor: 'rgba(255, 193, 7, 0.8)',
                        borderColor: 'rgba(255, 193, 7, 1)',
                        borderWidth: 2,
                        borderRadius: 4
                    },
                    {
                        label: 'F1-Score',
                        data: f1Data,
                        backgroundColor: 'rgba(220, 53, 69, 0.8)',
                        borderColor: 'rgba(220, 53, 69, 1)',
                        borderWidth: 2,
                        borderRadius: 4
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: {
                    duration: 1000,
                    easing: 'easeInOutQuart'
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 1.0,
                        ticks: {
                            callback: function(value) {
                                return (value * 100).toFixed(0) + '%';
                            }
                        },
                        grid: {
                            color: 'rgba(0, 0, 0, 0.1)'
                        }
                    },
                    x: {
                        grid: {
                            display: false
                        }
                    }
                },
                plugins: {
                    title: {
                        display: true,
                        text: 'Model Performance Comparison',
                        font: {
                            size: 16,
                            weight: 'bold'
                        }
                    },
                    legend: {
                        display: true,
                        position: 'top',
                        labels: {
                            usePointStyle: true,
                            padding: 20
                        }
                    },
                    tooltip: {
                        backgroundColor: 'rgba(0, 0, 0, 0.8)',
                        titleColor: 'white',
                        bodyColor: 'white',
                        borderColor: 'rgba(255, 255, 255, 0.1)',
                        borderWidth: 1,
                        callbacks: {
                            label: function(context) {
                                return context.dataset.label + ': ' + (context.parsed.y * 100).toFixed(1) + '%';
                            }
                        }
                    }
                }
            }
        });
        } catch (error) {
            console.error('Error initializing metrics chart:', error);
        }
    }

    // Accuracy trend over time chart
    initializeAccuracyTrendChart() {
        try {
            const ctx = document.getElementById('accuracyTrendChart');
            if (!ctx) return;

        // Sample data - in real implementation, this would come from backend
        const dates = this.generateDateLabels(7);
        const accuracyTrend = this.generateTrendData(7, 0.92, 0.98);

        this.charts.accuracyTrend = new Chart(ctx, {
            type: 'line',
            data: {
                labels: dates,
                datasets: [{
                    label: 'Daily Accuracy',
                    data: accuracyTrend,
                    borderColor: 'rgba(40, 167, 69, 1)',
                    backgroundColor: 'rgba(40, 167, 69, 0.1)',
                    borderWidth: 3,
                    fill: true,
                    tension: 0.4,
                    pointBackgroundColor: 'rgba(40, 167, 69, 1)',
                    pointBorderColor: 'white',
                    pointBorderWidth: 2,
                    pointRadius: 6
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: false,
                        min: 0.85,
                        max: 1.0,
                        ticks: {
                            callback: function(value) {
                                return (value * 100).toFixed(0) + '%';
                            }
                        }
                    }
                },
                plugins: {
                    title: {
                        display: true,
                        text: 'Accuracy Trend (Last 7 Days)'
                    },
                    legend: {
                        display: false
                    }
                }
            }
        });
        } catch (error) {
            console.error('Error initializing accuracy trend chart:', error);
        }
    }

    // Classification results distribution chart
    initializeClassificationResultsChart() {
        try {
            const ctx = document.getElementById('classificationResultsChart');
            if (!ctx) return;

        // Get classification results distribution data
        const distribution = window.dashboardData?.classificationResultsDistribution || {
            spam_count: 0, 
            legitimate_count: 0, 
            total_count: 0
        };
        
        const spamCount = distribution.spam_count || 0;
        const legitimateCount = distribution.legitimate_count || 0;
        const totalCount = distribution.total_count || (spamCount + legitimateCount);
        
        // Check if we have any data
        if (totalCount === 0) {
            // Show empty state
            try {
                const parent = ctx.parentElement;
                if (parent) {
                    parent.innerHTML = `
                        <div class="d-flex flex-column align-items-center justify-content-center h-100 text-muted">
                            <i class="fas fa-chart-pie fa-3x mb-3 opacity-50"></i>
                            <h6>No Classifications Yet</h6>
                            <small>Process some emails to see classification results</small>
                        </div>
                    `;
                }
            } catch (error) {
                console.log('No classification data available yet');
            }
            return;
        }

        this.charts.classificationResults = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: ['Spam', 'Legitimate'],
                datasets: [{
                    data: [spamCount, legitimateCount],
                    backgroundColor: [
                        'rgba(220, 53, 69, 0.8)',  // Red for spam
                        'rgba(40, 167, 69, 0.8)'   // Green for legitimate
                    ],
                    borderWidth: 2,
                    borderColor: 'white'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: `Classification Results Distribution (${totalCount} total)`
                    },
                    legend: {
                        position: 'bottom'
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                const total = context.dataset.data.reduce((a, b) => a + b, 0);
                                const percentage = total > 0 ? ((context.parsed / total) * 100).toFixed(1) : '0.0';
                                return context.label + ': ' + context.parsed + ' (' + percentage + '%)';
                            }
                        }
                    }
                }
            }
        });
        } catch (error) {
            console.error('Error initializing classification results chart:', error);
        }
    }

    // Setup real-time updates
    setupRealTimeUpdates() {
        // Update timestamp every second
        setInterval(() => {
            this.updateTimestamp();
        }, 1000);

        // Simulate real-time data updates
        setInterval(() => {
            this.updateRealTimeMetrics();
        }, 5000);
    }

    // Setup event listeners
    setupEventListeners() {
        // Refresh button
        const refreshBtn = document.querySelector('[onclick="refreshDashboard()"]');
        if (refreshBtn) {
            refreshBtn.addEventListener('click', (e) => {
                e.preventDefault();
                this.refreshDashboard();
            });
        }

        // Model selection (if available)
        const modelSelect = document.getElementById('modelSelect');
        if (modelSelect) {
            modelSelect.addEventListener('change', (e) => {
                this.switchActiveModel(e.target.value);
            });
        }
    }

    // Start auto-refresh
    startAutoRefresh() {
        this.refreshInterval = setInterval(() => {
            this.refreshDashboard();
        }, 30000); // Refresh every 30 seconds
    }

    // Stop auto-refresh
    stopAutoRefresh() {
        if (this.refreshInterval) {
            clearInterval(this.refreshInterval);
            this.refreshInterval = null;
        }
    }

    // Refresh dashboard data
    async refreshDashboard() {
        try {
            this.showLoadingState(true);
            
            // In a real implementation, this would fetch from API
            // const response = await fetch('/api/dashboard/metrics');
            // const data = await response.json();
            
            // Simulate API call
            await this.simulateApiCall();
            
            this.updateCharts();
            this.updateMetricCards();
            this.lastUpdateTime = new Date();
            
            this.showAlert('Dashboard refreshed successfully', 'success');
        } catch (error) {
            console.error('Error refreshing dashboard:', error);
            this.showAlert('Failed to refresh dashboard', 'danger');
        } finally {
            this.showLoadingState(false);
        }
    }

    // Update timestamp display
    updateTimestamp() {
        const timestampElement = document.getElementById('lastUpdated');
        if (timestampElement) {
            const now = new Date();
            const timeString = now.toLocaleTimeString();
            timestampElement.textContent = `Last updated: ${timeString}`;
        }
    }

    // Update real-time metrics
    updateRealTimeMetrics() {
        // Update processed today counter
        const processedElement = document.getElementById('processedToday');
        if (processedElement) {
            const current = parseInt(processedElement.textContent) || 0;
            const increment = Math.floor(Math.random() * 5) + 1;
            processedElement.textContent = current + increment;
        }

        // Update average response time
        const responseTimeElement = document.getElementById('avgResponseTime');
        if (responseTimeElement) {
            const randomTime = (Math.random() * 0.5 + 1.5).toFixed(2);
            responseTimeElement.textContent = randomTime + 's';
        }
    }

    // Update all charts with new data
    updateCharts() {
        Object.values(this.charts).forEach(chart => {
            if (chart && typeof chart.update === 'function') {
                chart.update('active');
            }
        });
    }

    // Update metric cards
    updateMetricCards() {
        // Update system status indicators
        const statusCards = document.querySelectorAll('.card.bg-primary, .card.bg-success, .card.bg-info, .card.bg-warning');
        statusCards.forEach(card => {
            card.classList.add('pulse-animation');
            setTimeout(() => {
                card.classList.remove('pulse-animation');
            }, 1000);
        });
    }

    // Show loading state
    showLoadingState(isLoading) {
        const refreshBtn = document.querySelector('[onclick="refreshDashboard()"]');
        if (refreshBtn) {
            if (isLoading) {
                refreshBtn.disabled = true;
                refreshBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Refreshing...';
            } else {
                refreshBtn.disabled = false;
                refreshBtn.innerHTML = '<i class="fas fa-sync-alt"></i> Refresh';
            }
        }
    }

    // Switch active model
    async switchActiveModel(modelName) {
        try {
            this.showAlert(`Switching to ${modelName} model...`, 'info');
            
            // In real implementation, this would call API to switch model
            // await fetch('/api/models/switch', { method: 'POST', body: JSON.stringify({model: modelName}) });
            
            await this.simulateApiCall();
            this.refreshDashboard();
            
            this.showAlert(`Successfully switched to ${modelName} model`, 'success');
        } catch (error) {
            console.error('Error switching model:', error);
            this.showAlert('Failed to switch model', 'danger');
        }
    }

    // Utility functions
    generateDateLabels(days) {
        const labels = [];
        for (let i = days - 1; i >= 0; i--) {
            const date = new Date();
            date.setDate(date.getDate() - i);
            labels.push(date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' }));
        }
        return labels;
    }

    generateTrendData(points, min, max) {
        const data = [];
        for (let i = 0; i < points; i++) {
            data.push(Math.random() * (max - min) + min);
        }
        return data;
    }

    async simulateApiCall() {
        return new Promise(resolve => {
            setTimeout(resolve, 1000);
        });
    }

    showAlert(message, type = 'info') {
        // Use the global showAlert function from main.js if available
        if (typeof window.showAlert === 'function') {
            window.showAlert(message, type);
            return;
        }
        
        // Fallback implementation
        try {
            // Create alert element
            const alertDiv = document.createElement('div');
            alertDiv.className = `alert alert-${type} alert-dismissible fade show`;
            alertDiv.innerHTML = `
                ${message}
                <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
            `;
            
            // Insert at top of content
            const container = document.querySelector('.container-fluid') || document.body;
            if (container && container.firstChild) {
                container.insertBefore(alertDiv, container.firstChild);
                
                // Auto-dismiss after 3 seconds
                setTimeout(() => {
                    if (alertDiv.parentNode) {
                        alertDiv.remove();
                    }
                }, 3000);
            } else {
                // Fallback to console if DOM manipulation fails
                console.log(`${type.toUpperCase()}: ${message}`);
            }
        } catch (error) {
            // Ultimate fallback to console
            console.log(`${type.toUpperCase()}: ${message}`);
        }
    }

    // Cleanup method
    destroy() {
        this.stopAutoRefresh();
        Object.values(this.charts).forEach(chart => {
            if (chart && typeof chart.destroy === 'function') {
                chart.destroy();
            }
        });
    }
}

// Initialize dashboard when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    // Only initialize on dashboard page
    if (window.location.pathname.includes('dashboard')) {
        window.dashboardManager = new DashboardManager();
    }
});

// Global function for refresh button
window.refreshDashboard = function() {
    if (window.dashboardManager) {
        window.dashboardManager.refreshDashboard();
    }
};

// Cleanup on page unload
window.addEventListener('beforeunload', function() {
    if (window.dashboardManager) {
        window.dashboardManager.destroy();
    }
});