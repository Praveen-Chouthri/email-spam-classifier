# Design Document

## Overview

This design implements advanced feature engineering for the Email Spam Classification System to extract spam-specific patterns and characteristics beyond basic text analysis. The solution adds sophisticated pattern detection for URLs, contact information, financial indicators, formatting anomalies, and linguistic patterns to create richer feature vectors that improve spam detection accuracy from the current 95.15% baseline.

The design integrates seamlessly with the existing preprocessing pipeline while maintaining backward compatibility and providing configurable feature extraction capabilities.

## Architecture

### High-Level Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Raw Email     │───▶│   Enhanced       │───▶│   Feature       │
│   Text Input    │    │   Preprocessing  │    │   Vector        │
│                 │    │   Pipeline       │    │   Output        │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              │
                              ▼
                       ┌──────────────────┐
                       │  Pattern         │
                       │  Extractors      │
                       │                  │
                       │ • URL Detector   │
                       │ • Contact Finder │
                       │ • Money Detector │
                       │ • Format Analyzer│
                       │ • Language Scorer│
                       └──────────────────┘
                              │
                              ▼
                       ┌──────────────────┐
                       │   Feature        │
                       │   Combiner       │
                       │                  │
                       │ TF-IDF + Custom  │
                       │ Features         │
                       └──────────────────┘
```

### Feature Extraction Pipeline

```
Email Text Input
       │
       ▼
┌─────────────────┐
│ Text Cleaning   │ ── Remove HTML, normalize whitespace
└─────────────────┘
       │
       ▼
┌─────────────────┐
│ Pattern         │ ── Extract URLs, phones, emails
│ Detection       │
└─────────────────┘
       │
       ▼
┌─────────────────┐
│ Linguistic      │ ── Analyze grammar, spelling, readability
│ Analysis        │
└─────────────────┘
       │
       ▼
┌─────────────────┐
│ Format          │ ── Measure caps, punctuation, structure
│ Analysis        │
└─────────────────┘
       │
       ▼
┌─────────────────┐
│ Feature         │ ── Combine all features into vector
│ Combination     │
└─────────────────┘
       │
       ▼
Enhanced Feature Vector
```

## Components and Interfaces

### 1. AdvancedFeatureExtractor Component

**Purpose**: Main orchestrator for all feature extraction processes

**Interface**:
```python
class AdvancedFeatureExtractor:
    def __init__(self, config: FeatureConfig)
    def extract_features(self, text: str) -> Dict[str, float]
    def fit_extractors(self, texts: List[str]) -> None
    def get_feature_names(self) -> List[str]
    def get_feature_importance(self) -> Dict[str, float]
```

**Key Methods**:
- `extract_features()`: Orchestrates all pattern extractors and returns combined feature vector
- `fit_extractors()`: Trains extractors that need fitting (like vocabulary-based features)
- `get_feature_names()`: Returns ordered list of all feature names for model training

### 2. URLPatternExtractor Component

**Purpose**: Detects and analyzes URL patterns in email content

**Interface**:
```python
class URLPatternExtractor:
    def __init__(self, suspicious_tlds: List[str], trusted_domains: List[str])
    def extract_url_features(self, text: str) -> Dict[str, float]
    def classify_url_suspicion(self, url: str) -> float
    def detect_shortened_urls(self, text: str) -> int
```

**Features Extracted**:
- Total URL count
- Suspicious URL ratio
- Shortened URL count (bit.ly, tinyurl, etc.)
- IP address URLs
- Suspicious TLD count (.tk, .ml, etc.)
- Domain reputation scores

### 3. ContactPatternExtractor Component

**Purpose**: Identifies contact information patterns (phones, emails, addresses)

**Interface**:
```python
class ContactPatternExtractor:
    def __init__(self, phone_patterns: Dict[str, str], suspicious_patterns: List[str])
    def extract_contact_features(self, text: str) -> Dict[str, float]
    def detect_phone_numbers(self, text: str) -> List[Dict[str, Any]]
    def analyze_email_addresses(self, text: str) -> Dict[str, float]
```

**Features Extracted**:
- Phone number count and types
- Premium rate number indicators
- International number patterns
- Email address count and domain analysis
- Contact information density

### 4. FinancialPatternExtractor Component

**Purpose**: Detects monetary amounts, currency symbols, and financial scam indicators

**Interface**:
```python
class FinancialPatternExtractor:
    def __init__(self, currency_symbols: List[str], scam_phrases: List[str])
    def extract_financial_features(self, text: str) -> Dict[str, float]
    def detect_monetary_amounts(self, text: str) -> List[float]
    def score_urgency_indicators(self, text: str) -> float
    def identify_scam_phrases(self, text: str) -> Dict[str, int]
```

**Features Extracted**:
- Currency symbol count
- Monetary amount presence and magnitude
- Urgency keyword score (URGENT, LIMITED TIME, ACT NOW)
- Financial scam phrase indicators
- Promotional language intensity

### 5. FormattingAnalyzer Component

**Purpose**: Analyzes text formatting and structural characteristics

**Interface**:
```python
class FormattingAnalyzer:
    def __init__(self, formatting_config: Dict[str, Any])
    def extract_formatting_features(self, text: str) -> Dict[str, float]
    def calculate_caps_ratio(self, text: str) -> float
    def analyze_punctuation_patterns(self, text: str) -> Dict[str, float]
    def measure_text_structure(self, text: str) -> Dict[str, float]
```

**Features Extracted**:
- Uppercase text ratio
- Excessive punctuation indicators (!!! ???)
- Special character density
- Sentence length variation
- Paragraph structure metrics
- Readability scores

### 6. LinguisticAnalyzer Component

**Purpose**: Analyzes language quality and linguistic patterns

**Interface**:
```python
class LinguisticAnalyzer:
    def __init__(self, language_models: Dict[str, Any])
    def extract_linguistic_features(self, text: str) -> Dict[str, float]
    def calculate_spelling_errors(self, text: str) -> float
    def analyze_grammar_quality(self, text: str) -> float
    def measure_language_complexity(self, text: str) -> Dict[str, float]
```

**Features Extracted**:
- Spelling error ratio
- Grammar quality score
- Vocabulary complexity
- Sentence structure quality
- Language detection confidence

## Data Models

### FeatureConfig

```python
@dataclass
class FeatureConfig:
    # URL Analysis
    enable_url_features: bool = True
    suspicious_tlds: List[str] = field(default_factory=lambda: ['.tk', '.ml', '.ga', '.cf'])
    trusted_domains: List[str] = field(default_factory=lambda: ['gmail.com', 'yahoo.com', 'outlook.com'])
    
    # Contact Analysis
    enable_contact_features: bool = True
    phone_patterns: Dict[str, str] = field(default_factory=dict)
    
    # Financial Analysis
    enable_financial_features: bool = True
    currency_symbols: List[str] = field(default_factory=lambda: ['$', '€', '£', '¥', '₹'])
    urgency_keywords: List[str] = field(default_factory=lambda: ['urgent', 'limited time', 'act now', 'expires'])
    
    # Formatting Analysis
    enable_formatting_features: bool = True
    caps_threshold: float = 0.3
    punctuation_threshold: float = 0.1
    
    # Linguistic Analysis
    enable_linguistic_features: bool = True
    language_model: str = 'en_core_web_sm'
    
    # Performance
    lightweight_mode: bool = False
    max_processing_time: float = 5.0
```

### ExtractedFeatures

```python
@dataclass
class ExtractedFeatures:
    # URL Features
    url_count: float = 0.0
    suspicious_url_ratio: float = 0.0
    shortened_url_count: float = 0.0
    ip_url_count: float = 0.0
    
    # Contact Features
    phone_count: float = 0.0
    premium_phone_indicators: float = 0.0
    email_count: float = 0.0
    contact_density: float = 0.0
    
    # Financial Features
    currency_count: float = 0.0
    monetary_amount_present: float = 0.0
    urgency_score: float = 0.0
    scam_phrase_count: float = 0.0
    
    # Formatting Features
    caps_ratio: float = 0.0
    punctuation_excess: float = 0.0
    special_char_density: float = 0.0
    structure_quality: float = 0.0
    
    # Linguistic Features
    spelling_error_ratio: float = 0.0
    grammar_quality: float = 0.0
    readability_score: float = 0.0
    language_confidence: float = 0.0
    
    def to_vector(self) -> np.ndarray:
        """Convert to numpy array for ML models."""
        return np.array([getattr(self, field.name) for field in fields(self)])
```

### EnhancedPreprocessingPipeline

```python
class EnhancedPreprocessingPipeline(PreprocessingPipeline):
    def __init__(self, 
                 feature_config: FeatureConfig,
                 traditional_params: Optional[Dict] = None):
        super().__init__(traditional_params)
        self.feature_extractor = AdvancedFeatureExtractor(feature_config)
        self.feature_scaler = StandardScaler()
        
    def fit(self, texts: List[str]) -> None:
        # Fit traditional TF-IDF pipeline
        super().fit(texts)
        
        # Fit advanced feature extractors
        self.feature_extractor.fit_extractors(texts)
        
        # Fit feature scaler
        advanced_features = [self.feature_extractor.extract_features(text) for text in texts]
        feature_matrix = np.array([list(features.values()) for features in advanced_features])
        self.feature_scaler.fit(feature_matrix)
    
    def transform(self, texts: List[str]) -> np.ndarray:
        # Get traditional TF-IDF features
        tfidf_features = super().transform(texts)
        
        # Extract advanced features
        advanced_features = []
        for text in texts:
            features = self.feature_extractor.extract_features(text)
            advanced_features.append(list(features.values()))
        
        # Scale advanced features
        advanced_matrix = self.feature_scaler.transform(np.array(advanced_features))
        
        # Combine TF-IDF and advanced features
        combined_features = np.hstack([tfidf_features.toarray(), advanced_matrix])
        
        return combined_features
```

## Error Handling

### Feature Extraction Errors

1. **Pattern Matching Failures**: Gracefully handle regex errors and malformed patterns
2. **Language Processing Errors**: Fall back to simpler analysis if NLP models fail
3. **Performance Timeouts**: Implement timeouts for expensive feature extraction
4. **Memory Constraints**: Use streaming processing for large email batches

### Graceful Degradation

- If advanced features fail, continue with traditional TF-IDF features
- Provide configurable fallback modes for different error scenarios
- Log feature extraction failures for debugging and monitoring
- Maintain model compatibility when features are unavailable

## Testing Strategy

### Unit Tests

1. **Pattern Extractor Tests**:
   - Test URL detection accuracy with various URL formats
   - Verify phone number pattern matching across international formats
   - Test financial pattern detection with different currencies and amounts

2. **Feature Quality Tests**:
   - Validate feature value ranges and distributions
   - Test feature extraction consistency and reproducibility
   - Verify feature scaling and normalization accuracy

3. **Performance Tests**:
   - Benchmark feature extraction speed on large datasets
   - Test memory usage with various email sizes and batch sizes
   - Validate timeout handling and graceful degradation

### Integration Tests

1. **Pipeline Integration**:
   - Test complete preprocessing pipeline with enhanced features
   - Verify model training compatibility with combined feature vectors
   - Test backward compatibility with existing models

2. **Model Performance Tests**:
   - Validate accuracy improvements with enhanced features
   - Test feature importance analysis and selection
   - Verify cross-validation performance with new features

### Validation Tests

1. **Feature Effectiveness**:
   - Measure individual feature contribution to spam detection
   - Test feature correlation and redundancy analysis
   - Validate feature interpretability and explainability

## Performance Considerations

### Processing Efficiency
- Implement caching for expensive pattern compilations
- Use vectorized operations for batch feature extraction
- Optimize regex patterns for common spam indicators

### Memory Management
- Stream processing for large email datasets
- Efficient feature vector storage and retrieval
- Garbage collection for temporary feature extraction objects

### Scalability
- Parallel processing for independent feature extractors
- Configurable feature extraction depth based on performance requirements
- Batch processing optimization for training and inference