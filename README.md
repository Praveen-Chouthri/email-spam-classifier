# Email Spam Classifier

A machine learning-powered email spam classification system built with Flask, scikit-learn, and modern web technologies. The system uses three different ML algorithms (Naive Bayes, Random Forest, Decision Tree) to classify emails as spam or legitimate with high accuracy.

## 🚀 Features

- **High Accuracy Classification**: 96.32% accuracy with the best-performing Naive Bayes model
- **Multiple ML Models**: Naive Bayes, Random Forest, and Decision Tree algorithms
- **Web Interface**: User-friendly web application for single email classification
- **Batch Processing**: Upload CSV files to classify multiple emails at once
- **REST API**: Complete API for integration with other applications
- **Performance Dashboard**: View model metrics and comparison
- **Production Ready**: Docker, Gunicorn, and Nginx configurations included
- **Health Monitoring**: Built-in health check endpoints
- **Comprehensive Logging**: Structured logging with rotation

## 📋 Requirements

- Python 3.8+
- Flask 2.3+
- scikit-learn 1.3+
- pandas 2.0+
- numpy 1.24+

## 🛠️ Installation

### Quick Start (Development)

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd email-spam-classifier
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set environment variables**
   ```bash
   # Windows PowerShell
   $env:SECRET_KEY="your-secret-key-here"
   $env:FLASK_ENV="development"
   
   # Linux/Mac
   export SECRET_KEY="your-secret-key-here"
   export FLASK_ENV="development"
   ```

4. **Train models (if not already trained)**
   ```bash
   python train_models.py
   ```

5. **Start the application**
   ```bash
   python app.py
   ```

6. **Open your browser**
   Navigate to `http://localhost:5000`

### Production Deployment

#### Option 1: Using Startup Scripts

**Windows:**
```bash
# Copy and configure environment
copy .env.production .env
# Edit .env with your production settings
start.bat
```

**Linux/Mac:**
```bash
# Copy and configure environment
cp .env.production .env
# Edit .env with your production settings
chmod +x start.sh
./start.sh
```

#### Option 2: Manual Production Setup

1. **Configure environment**
   ```bash
   cp .env.production .env
   # Edit .env file with your production settings
   ```

2. **Run deployment script**
   ```bash
   python deploy.py
   ```

3. **Start with Gunicorn**
   ```bash
   gunicorn --config gunicorn.conf.py wsgi:application
   ```

#### Option 3: Docker Deployment

1. **Build and run with Docker Compose**
   ```bash
   docker-compose up -d
   ```

2. **Access the application**
   - Web Interface: `http://localhost`
   - API: `http://localhost/api/v1`

## 🎯 Usage

### Web Interface

#### Single Email Classification
1. Navigate to the home page (`http://localhost:5000`)
2. Enter or paste email content in the text area
3. Click "Classify Email"
4. View the prediction result with confidence score

#### Batch Processing
1. Go to the Batch Processing page (`http://localhost:5000/batch`)
2. Upload a CSV file with email content
3. Download the results with classifications

**CSV Format for Batch Processing:**
```csv
email_text
"Subject: Meeting Tomorrow Hi John, reminder about our meeting..."
"URGENT! Click here to claim your FREE prize now!"
"Your order has been shipped and will arrive tomorrow."
```

#### Performance Dashboard
- Visit `/dashboard` to view model performance metrics
- Compare accuracy, precision, recall, and F1-scores
- See confusion matrices and training details

### REST API

#### Authentication
Some endpoints require authentication. Include the Bearer token in the Authorization header:
```bash
Authorization: Bearer demo-token
```

#### Classify Single Email
```bash
curl -X POST http://localhost:5000/api/v1/classify \
  -H "Content-Type: application/json" \
  -d '{
    "email_text": "URGENT! Click here to claim your FREE prize now!"
  }'
```

**Response:**
```json
{
  "error": false,
  "data": {
    "prediction": "Spam",
    "confidence": 0.95,
    "model_used": "Naive Bayes",
    "processing_time": 0.023,
    "timestamp": "2024-01-01T12:00:00Z"
  }
}
```

#### Batch Classification
```bash
curl -X POST http://localhost:5000/api/v1/classify/batch \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer demo-token" \
  -d '{
    "emails": [
      "Subject: Meeting Tomorrow Hi John, reminder...",
      "URGENT! Click now to claim your prize!"
    ]
  }'
```

#### Get Model Information
```bash
curl http://localhost:5000/api/v1/models
```

#### Health Check
```bash
curl http://localhost:5000/health
curl http://localhost:5000/api/v1/health
```

#### API Documentation
Visit `http://localhost:5000/api/v1/docs` for complete API documentation.

## 📊 Model Performance

The system includes three trained machine learning models:

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| **Naive Bayes** | **96.32%** | **96.43%** | **96.32%** | **96.08%** |
| Decision Tree | 95.87% | 95.78% | 95.87% | 95.69% |
| Random Forest | 90.40% | 91.36% | 90.40% | 87.97% |

*Naive Bayes is automatically selected as the best-performing model.*

## 🔧 Configuration

### Environment Variables

Key configuration options in `.env` file:

```bash
# Flask Configuration
FLASK_ENV=production
SECRET_KEY=your-secure-secret-key-here

# Application Settings
MAX_BATCH_SIZE=1000
CLASSIFICATION_TIMEOUT=3.0
MAX_FILE_SIZE=16777216

# Model Settings
DEFAULT_MODEL=naive_bayes
TRAINED_MODELS_DIR=models/trained

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/email_classifier.log

# Performance
WORKERS=4
THREADS=2
TIMEOUT=120
```

### Directory Structure

```
email-spam-classifier/
├── app.py                 # Main Flask application
├── wsgi.py               # WSGI entry point
├── config.py             # Configuration management
├── requirements.txt      # Python dependencies
├── train_models.py       # Model training script
├── deploy.py            # Deployment script
├── start.sh / start.bat # Startup scripts
├── src/                 # Source code
│   ├── models/          # ML model classes
│   ├── services/        # Business logic services
│   ├── preprocessing.py # Text preprocessing
│   └── error_handling.py # Error handling
├── routes/              # Flask routes
│   ├── web.py          # Web interface routes
│   └── api.py          # REST API routes
├── templates/           # HTML templates
├── static/             # CSS, JS, images
├── models/trained/     # Trained ML models
├── data/              # Training datasets
├── logs/              # Application logs
├── uploads/           # Uploaded files
└── results/           # Batch processing results
```

## 🐳 Docker Deployment

### Using Docker Compose (Recommended)

1. **Configure environment**
   ```bash
   cp .env.production .env
   # Edit .env with your settings
   ```

2. **Start services**
   ```bash
   docker-compose up -d
   ```

3. **View logs**
   ```bash
   docker-compose logs -f
   ```

4. **Stop services**
   ```bash
   docker-compose down
   ```

### Manual Docker Build

```bash
# Build image
docker build -t email-spam-classifier .

# Run container
docker run -d \
  -p 8000:8000 \
  -e SECRET_KEY="your-secret-key" \
  -e FLASK_ENV="production" \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/logs:/app/logs \
  email-spam-classifier
```

## 🔍 Monitoring and Health Checks

### Health Check Endpoints

- **Simple Health Check**: `GET /health`
- **Detailed Health Check**: `GET /api/v1/health`

### Logging

Logs are written to:
- Console output (development)
- `logs/email_classifier.log` (production)
- Rotating log files with configurable size limits

### Performance Monitoring

- Built-in performance dashboard at `/dashboard`
- Processing time tracking for each classification
- Model accuracy and metrics monitoring
- Request rate limiting and monitoring

## 🛡️ Security Features

- **CSRF Protection**: Enabled in production
- **Rate Limiting**: API endpoints have rate limits
- **Input Validation**: Comprehensive input sanitization
- **Security Headers**: X-Frame-Options, X-XSS-Protection, etc.
- **HTTPS Support**: SSL/TLS configuration ready
- **Authentication**: Bearer token authentication for sensitive endpoints

## 🧪 Testing

### Debug Script (Recommended First Step)
```bash
# Run the comprehensive debug script
python debug_app.py
```
This script will:
- Test all imports and dependencies
- Verify model files exist
- Test app creation and service initialization
- Test email classification functionality
- Provide detailed logging for troubleshooting

### Run Tests
```bash
# Install test dependencies
pip install pytest pytest-flask

# Run all tests
pytest

# Run with coverage
pytest --cov=src
```

### Manual Testing

1. **Test single classification**
   - Use the web interface at `/`
   - Try both spam and legitimate emails

2. **Test batch processing**
   - Upload a CSV file at `/batch`
   - Download and verify results

3. **Test API endpoints**
   - Use curl commands or Postman
   - Test authentication and rate limiting

4. **Test health checks**
   - Verify `/health` and `/api/v1/health` endpoints
   - Check response format and status codes

## 📝 API Rate Limits

| Endpoint | Rate Limit | Authentication |
|----------|------------|----------------|
| `/api/v1/classify` | 100 requests/hour | Not required |
| `/api/v1/classify/batch` | 10 requests/hour | Required |
| `/api/v1/models` | 50 requests/hour | Not required |
| `/api/v1/health` | No limit | Not required |

## 🚨 Troubleshooting

### Common Issues

1. **Models not found error**
   ```bash
   # Train models
   python train_models.py
   ```

2. **Permission denied on startup scripts**
   ```bash
   # Linux/Mac
   chmod +x start.sh
   
   # Windows: Run as Administrator
   ```

3. **Port already in use**
   ```bash
   # Change port in .env file
   PORT=8080
   ```

4. **Memory issues with large batches**
   ```bash
   # Reduce batch size in .env
   MAX_BATCH_SIZE=500
   ```

### Logs and Debugging

- Check application logs in `logs/email_classifier.log`
- Enable debug mode: `FLASK_DEBUG=True`
- Increase log level: `LOG_LEVEL=DEBUG`

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run the test suite
6. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For support and questions:
- Check the troubleshooting section above
- Review the API documentation at `/api/v1/docs`
- Check application logs for error details
- Ensure all environment variables are properly set

## 🔄 Updates and Maintenance

### Updating Models
```bash
# Retrain models with new data
python train_models.py

# Restart application to load new models
```

### Backup Important Data
- Trained models in `models/trained/`
- Configuration files (`.env`)
- Application logs in `logs/`
- Custom training data in `data/`

---

**Built with ❤️ using Flask, scikit-learn, and modern web technologies.**