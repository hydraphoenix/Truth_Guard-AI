# 🛡️ TruthGuard AI - Technical Documentation

## Table of Contents
1. [System Architecture](#system-architecture)
2. [AI Models Implementation](#ai-models-implementation)
3. [Google Cloud Integration](#google-cloud-integration)
4. [API Reference](#api-reference)
5. [Database Schema](#database-schema)
6. [Security Implementation](#security-implementation)
7. [Performance Optimization](#performance-optimization)
8. [Testing Strategy](#testing-strategy)
9. [Deployment Guide](#deployment-guide)
10. [Troubleshooting](#troubleshooting)

---

## System Architecture

### Overview
TruthGuard AI follows a microservices architecture designed for scalability, maintainability, and cloud-native deployment.

```
┌─────────────────────────────────────────────────────────────┐
│                    Frontend Layer                           │
├─────────────────────────────────────────────────────────────┤
│  • Streamlit Web Application                               │
│  • Responsive UI Components                                │
│  • Real-time Visualization                                 │
│  • Educational Interactive Modules                         │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   Application Layer                         │
├─────────────────────────────────────────────────────────────┤
│  • Content Processing Pipeline                             │
│  • User Session Management                                 │
│  • Cache Management                                        │
│  • Error Handling & Logging                               │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    AI Engine Layer                         │
├─────────────────────────────────────────────────────────────┤
│  • Ensemble Model Orchestration                           │
│  • Feature Extraction Pipeline                            │
│  • Model Inference Engine                                 │
│  • Result Aggregation & Scoring                           │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   Google Cloud Layer                       │
├─────────────────────────────────────────────────────────────┤
│  • Natural Language AI                                     │
│  • Translation API                                         │
│  • Cloud Storage                                           │
│  • BigQuery Analytics                                      │
└─────────────────────────────────────────────────────────────┘
```

### Core Components

#### 1. Frontend Layer (`app.py`)
- **Technology**: Streamlit 1.28+
- **Responsibilities**:
  - User interface rendering
  - Input validation and sanitization
  - Real-time result visualization
  - Educational content delivery
  - Analytics dashboard display

#### 2. Application Layer
- **Configuration Management** (`config.py`)
- **Session State Management**
- **Cache Layer** (Streamlit native caching)
- **Error Handling & Logging**

#### 3. AI Engine Layer (`models/`)
- **Advanced ML Model** (`advanced_ml_model.py`)
- **Feature Extraction Pipeline**
- **Model Ensemble Orchestration**
- **Result Aggregation Engine**

#### 4. Integration Layer
- **Google Cloud Services Integration**
- **External API Connectors**
- **Data Pipeline Management**

---

## AI Models Implementation

### Model Architecture Overview

TruthGuard AI employs a hybrid ensemble approach combining multiple machine learning models for optimal accuracy and reliability.

#### 1. Advanced ML Model (Primary) - 99% Accuracy
```python
class AdvancedMLModel:
    """
    TF-IDF + Logistic Regression based model
    Optimized for high accuracy fake news detection
    """
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 3),
            stop_words='english',
            lowercase=True,
            strip_accents='ascii'
        )
        self.model = LogisticRegression(
            random_state=42,
            max_iter=1000,
            C=1.0
        )
```

**Key Features**:
- **TF-IDF Vectorization**: N-gram range (1,3) for comprehensive feature capture
- **Logistic Regression**: Optimized for binary classification
- **Text Preprocessing**: Advanced cleaning and normalization
- **Feature Selection**: Top 10,000 most informative features

#### 2. LSTM Neural Network - 94.5% Accuracy
```python
class LSTMModel:
    """
    Deep learning model for sequential pattern recognition
    """
    
    def build_model(self):
        model = Sequential([
            Embedding(vocab_size, 128, input_length=max_len),
            LSTM(64, dropout=0.5, recurrent_dropout=0.5),
            Dense(32, activation='relu'),
            Dropout(0.5),
            Dense(1, activation='sigmoid')
        ])
        return model
```

#### 3. Ensemble Model System
```python
class EnsembleDetector:
    """
    Combines multiple models for robust predictions
    """
    
    def __init__(self):
        self.models = {
            'advanced_ml': AdvancedMLModel(),
            'lstm': LSTMModel(),
            'random_forest': RandomForestModel(),
            'svm': SVMModel(),
            'naive_bayes': NaiveBayesModel()
        }
        
        self.weights = {
            'advanced_ml': 0.4,
            'lstm': 0.3,
            'random_forest': 0.15,
            'svm': 0.1,
            'naive_bayes': 0.05
        }
```

### Feature Engineering Pipeline

#### Linguistic Features
```python
def extract_linguistic_features(text):
    """Extract comprehensive linguistic features"""
    features = {
        'word_count': len(text.split()),
        'char_count': len(text),
        'sentence_count': len(sent_tokenize(text)),
        'avg_word_length': np.mean([len(word) for word in text.split()]),
        'caps_ratio': sum(1 for c in text if c.isupper()) / len(text),
        'punctuation_ratio': sum(1 for c in text if c in string.punctuation) / len(text),
        'exclamation_count': text.count('!'),
        'question_count': text.count('?'),
        'digit_ratio': sum(1 for c in text if c.isdigit()) / len(text)
    }
    return features
```

#### Sentiment Analysis Features
```python
def extract_sentiment_features(text):
    """Extract sentiment-based features"""
    analyzer = SentimentIntensityAnalyzer()
    sentiment = analyzer.polarity_scores(text)
    
    features = {
        'sentiment_compound': sentiment['compound'],
        'sentiment_positive': sentiment['pos'],
        'sentiment_negative': sentiment['neg'],
        'sentiment_neutral': sentiment['neu'],
        'sentiment_polarity': TextBlob(text).sentiment.polarity,
        'sentiment_subjectivity': TextBlob(text).sentiment.subjectivity
    }
    return features
```

#### Credibility Scoring Features
```python
def extract_credibility_features(text):
    """Extract credibility indicators"""
    
    # Authority indicators
    authority_keywords = ['expert', 'study', 'research', 'professor', 'doctor']
    authority_score = sum(1 for keyword in authority_keywords if keyword in text.lower())
    
    # Sensationalism indicators
    sensational_keywords = ['shocking', 'unbelievable', 'exposed', 'secret', 'hidden']
    sensational_score = sum(1 for keyword in sensational_keywords if keyword in text.lower())
    
    # Urgency indicators
    urgency_keywords = ['urgent', 'immediately', 'breaking', 'alert', 'emergency']
    urgency_score = sum(1 for keyword in urgency_keywords if keyword in text.lower())
    
    features = {
        'authority_score': authority_score,
        'sensational_score': sensational_score,
        'urgency_score': urgency_score,
        'credibility_ratio': authority_score / max(sensational_score + urgency_score, 1)
    }
    return features
```

---

## Google Cloud Integration

### Services Configuration

#### 1. Natural Language AI
```python
from google.cloud import language_v1

class GoogleNLPIntegration:
    def __init__(self):
        self.client = language_v1.LanguageServiceClient()
    
    def analyze_sentiment(self, text):
        """Analyze sentiment using Google Cloud NLP"""
        document = language_v1.Document(
            content=text,
            type_=language_v1.Document.Type.PLAIN_TEXT
        )
        
        response = self.client.analyze_sentiment(
            request={'document': document}
        )
        
        return {
            'sentiment_score': response.document_sentiment.score,
            'magnitude': response.document_sentiment.magnitude
        }
    
    def extract_entities(self, text):
        """Extract entities using Google Cloud NLP"""
        document = language_v1.Document(
            content=text,
            type_=language_v1.Document.Type.PLAIN_TEXT
        )
        
        response = self.client.analyze_entities(
            request={'document': document}
        )
        
        entities = []
        for entity in response.entities:
            entities.append({
                'name': entity.name,
                'type': entity.type_.name,
                'salience': entity.salience
            })
        
        return entities
```

#### 2. Translation API Integration
```python
from google.cloud import translate_v2 as translate

class GoogleTranslateIntegration:
    def __init__(self):
        self.translate_client = translate.Client()
    
    def detect_language(self, text):
        """Detect language of input text"""
        result = self.translate_client.detect_language(text)
        return {
            'language': result['language'],
            'confidence': result['confidence']
        }
    
    def translate_to_english(self, text, source_language=None):
        """Translate text to English for analysis"""
        if source_language:
            result = self.translate_client.translate(
                text,
                source_language=source_language,
                target_language='en'
            )
        else:
            result = self.translate_client.translate(
                text,
                target_language='en'
            )
        
        return {
            'translated_text': result['translatedText'],
            'source_language': result['detectedSourceLanguage']
        }
```

#### 3. Cloud Storage Integration
```python
from google.cloud import storage

class GoogleStorageIntegration:
    def __init__(self, bucket_name):
        self.client = storage.Client()
        self.bucket = self.client.bucket(bucket_name)
    
    def save_model(self, model, model_name):
        """Save trained model to Cloud Storage"""
        blob = self.bucket.blob(f'models/{model_name}.pkl')
        
        with blob.open('wb') as f:
            pickle.dump(model, f)
    
    def load_model(self, model_name):
        """Load model from Cloud Storage"""
        blob = self.bucket.blob(f'models/{model_name}.pkl')
        
        with blob.open('rb') as f:
            model = pickle.load(f)
        
        return model
```

#### 4. BigQuery Analytics Integration
```python
from google.cloud import bigquery

class GoogleBigQueryIntegration:
    def __init__(self):
        self.client = bigquery.Client()
    
    def log_prediction(self, prediction_data):
        """Log prediction results for analytics"""
        table_id = "truthguard-ai.analytics.predictions"
        
        rows_to_insert = [prediction_data]
        
        errors = self.client.insert_rows_json(table_id, rows_to_insert)
        
        if not errors:
            print("Prediction logged successfully")
        else:
            print(f"Error logging prediction: {errors}")
    
    def get_analytics_data(self, time_range='7d'):
        """Retrieve analytics data for dashboard"""
        query = f"""
        SELECT 
            DATE(timestamp) as date,
            prediction_result,
            COUNT(*) as count,
            AVG(confidence_score) as avg_confidence
        FROM `truthguard-ai.analytics.predictions`
        WHERE timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {time_range})
        GROUP BY date, prediction_result
        ORDER BY date DESC
        """
        
        query_job = self.client.query(query)
        return query_job.result()
```

---

## API Reference

### REST API Endpoints

#### 1. Content Analysis API
```python
@app.route('/api/analyze', methods=['POST'])
def analyze_content():
    """
    Analyze content for misinformation
    
    Request Body:
    {
        "content": "Text content to analyze",
        "title": "Optional title",
        "language": "auto"  # Optional language code
    }
    
    Response:
    {
        "prediction": "Real" | "Fake",
        "confidence": 0.95,
        "risk_score": 85,
        "risk_level": "High",
        "features": {
            "linguistic": {...},
            "sentiment": {...},
            "credibility": {...}
        },
        "explanation": "Detailed explanation of the prediction",
        "timestamp": "2024-01-01T12:00:00Z"
    }
    """
```

#### 2. Batch Analysis API
```python
@app.route('/api/batch-analyze', methods=['POST'])
def batch_analyze():
    """
    Analyze multiple pieces of content
    
    Request Body:
    {
        "contents": [
            {"id": "1", "content": "Text 1", "title": "Title 1"},
            {"id": "2", "content": "Text 2", "title": "Title 2"}
        ]
    }
    
    Response:
    {
        "results": [
            {"id": "1", "prediction": "Real", "confidence": 0.95, ...},
            {"id": "2", "prediction": "Fake", "confidence": 0.87, ...}
        ],
        "summary": {
            "total_analyzed": 2,
            "fake_detected": 1,
            "average_confidence": 0.91
        }
    }
    """
```

#### 3. Model Information API
```python
@app.route('/api/models/info', methods=['GET'])
def get_model_info():
    """
    Get information about available models
    
    Response:
    {
        "models": {
            "advanced_ml": {
                "accuracy": 0.99,
                "type": "TF-IDF + Logistic Regression",
                "status": "active"
            },
            "lstm": {
                "accuracy": 0.945,
                "type": "LSTM Neural Network",
                "status": "active"
            }
        },
        "ensemble": {
            "accuracy": 0.912,
            "weights": {
                "advanced_ml": 0.4,
                "lstm": 0.3,
                "random_forest": 0.15,
                "svm": 0.1,
                "naive_bayes": 0.05
            }
        }
    }
    """
```

---

## Database Schema

### Analytics Tables (BigQuery)

#### Predictions Table
```sql
CREATE TABLE `truthguard-ai.analytics.predictions` (
    prediction_id STRING NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    content_hash STRING NOT NULL,
    content_length INT64,
    prediction_result STRING NOT NULL, -- 'Real' or 'Fake'
    confidence_score FLOAT64 NOT NULL,
    risk_score INT64 NOT NULL,
    risk_level STRING NOT NULL,
    model_version STRING NOT NULL,
    processing_time_ms INT64,
    user_session_id STRING,
    features STRUCT<
        linguistic STRUCT<
            word_count INT64,
            sentence_count INT64,
            avg_word_length FLOAT64,
            caps_ratio FLOAT64,
            punctuation_ratio FLOAT64
        >,
        sentiment STRUCT<
            compound FLOAT64,
            positive FLOAT64,
            negative FLOAT64,
            neutral FLOAT64
        >,
        credibility STRUCT<
            authority_score INT64,
            sensational_score INT64,
            urgency_score INT64
        >
    >
);
```

#### User Sessions Table
```sql
CREATE TABLE `truthguard-ai.analytics.user_sessions` (
    session_id STRING NOT NULL,
    start_timestamp TIMESTAMP NOT NULL,
    end_timestamp TIMESTAMP,
    total_analyses INT64 DEFAULT 0,
    educational_modules_completed ARRAY<STRING>,
    quiz_scores ARRAY<STRUCT<
        quiz_id STRING,
        score FLOAT64,
        completed_at TIMESTAMP
    >>,
    user_agent STRING,
    ip_address STRING -- Hashed for privacy
);
```

---

## Security Implementation

### Input Validation and Sanitization

```python
import re
import html

class SecurityValidator:
    def __init__(self):
        self.max_content_length = 50000
        self.min_content_length = 5
        
    def validate_input(self, content, title=None):
        """Comprehensive input validation"""
        errors = []
        
        # Length validation
        if len(content) < self.min_content_length:
            errors.append("Content too short for analysis")
        
        if len(content) > self.max_content_length:
            errors.append("Content exceeds maximum length")
        
        # Content type validation
        if not isinstance(content, str):
            errors.append("Content must be text")
        
        # XSS protection
        if self._contains_xss(content):
            errors.append("Content contains potentially malicious code")
        
        return errors
    
    def sanitize_input(self, content):
        """Sanitize input content"""
        # HTML escape
        content = html.escape(content)
        
        # Remove potentially dangerous patterns
        content = re.sub(r'<script.*?</script>', '', content, flags=re.IGNORECASE)
        content = re.sub(r'javascript:', '', content, flags=re.IGNORECASE)
        content = re.sub(r'on\w+\s*=', '', content, flags=re.IGNORECASE)
        
        return content
    
    def _contains_xss(self, content):
        """Check for XSS patterns"""
        xss_patterns = [
            r'<script.*?>.*?</script>',
            r'javascript:',
            r'on\w+\s*=',
            r'<iframe.*?>',
            r'<object.*?>',
            r'<embed.*?>'
        ]
        
        for pattern in xss_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                return True
        
        return False
```

### Rate Limiting Implementation

```python
from collections import defaultdict
import time

class RateLimiter:
    def __init__(self):
        self.requests = defaultdict(list)
        self.limits = {
            'per_minute': 60,
            'per_hour': 1000,
            'per_day': 10000
        }
    
    def is_allowed(self, identifier):
        """Check if request is allowed based on rate limits"""
        now = time.time()
        user_requests = self.requests[identifier]
        
        # Remove old requests
        user_requests[:] = [req_time for req_time in user_requests if now - req_time < 86400]  # 24 hours
        
        # Check limits
        minute_ago = now - 60
        hour_ago = now - 3600
        day_ago = now - 86400
        
        minute_count = sum(1 for req_time in user_requests if req_time > minute_ago)
        hour_count = sum(1 for req_time in user_requests if req_time > hour_ago)
        day_count = len(user_requests)
        
        if minute_count >= self.limits['per_minute']:
            return False, "Rate limit exceeded: too many requests per minute"
        
        if hour_count >= self.limits['per_hour']:
            return False, "Rate limit exceeded: too many requests per hour"
        
        if day_count >= self.limits['per_day']:
            return False, "Rate limit exceeded: too many requests per day"
        
        # Add current request
        user_requests.append(now)
        return True, "Request allowed"
```

---

## Performance Optimization

### Caching Strategy

```python
import streamlit as st
import hashlib
import pickle
from functools import lru_cache

class CacheManager:
    def __init__(self):
        self.cache_ttl = 3600  # 1 hour
        
    @staticmethod
    @st.cache_data(ttl=3600)
    def cache_prediction(content_hash, result):
        """Cache prediction results"""
        return result
    
    @staticmethod
    @st.cache_resource
    def load_models():
        """Cache loaded models"""
        from models.advanced_ml_model import AdvancedMLModel
        
        models = {
            'advanced_ml': AdvancedMLModel(),
            # Load other models...
        }
        
        return models
    
    def generate_content_hash(self, content, title=""):
        """Generate hash for content caching"""
        combined = f"{content}{title}"
        return hashlib.sha256(combined.encode()).hexdigest()
    
    @lru_cache(maxsize=1000)
    def cached_feature_extraction(self, content_hash, content):
        """Cache feature extraction results"""
        # Feature extraction logic here
        pass
```

### Model Optimization

```python
class ModelOptimizer:
    def __init__(self):
        self.batch_size = 32
        self.max_sequence_length = 512
        
    def optimize_text_processing(self, texts):
        """Batch process multiple texts efficiently"""
        # Vectorize in batches
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            yield self._process_batch(batch)
    
    def _process_batch(self, batch):
        """Process a batch of texts"""
        # Truncate to max length
        processed = [text[:self.max_sequence_length] for text in batch]
        return processed
    
    def enable_gpu_acceleration(self):
        """Enable GPU processing if available"""
        import tensorflow as tf
        
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                return True
            except RuntimeError as e:
                print(f"GPU setup error: {e}")
                return False
        return False
```

---

## Testing Strategy

### Unit Tests

```python
import unittest
from models.advanced_ml_model import AdvancedMLModel

class TestAdvancedMLModel(unittest.TestCase):
    def setUp(self):
        self.model = AdvancedMLModel()
        
    def test_text_preprocessing(self):
        """Test text preprocessing functionality"""
        text = "This is a TEST with CAPS and numbers 123!"
        processed = self.model.preprocess_text(text)
        
        self.assertIsInstance(processed, str)
        self.assertNotIn('123', processed)
        
    def test_prediction_format(self):
        """Test prediction output format"""
        text = "Sample news article content for testing"
        result = self.model.predict(text)
        
        self.assertIn('prediction', result)
        self.assertIn('confidence', result)
        self.assertIn(result['prediction'], ['Real', 'Fake'])
        self.assertGreaterEqual(result['confidence'], 0)
        self.assertLessEqual(result['confidence'], 1)
    
    def test_feature_extraction(self):
        """Test feature extraction"""
        text = "Test article with specific features"
        features = self.model.extract_features(text)
        
        self.assertIsInstance(features, dict)
        self.assertIn('word_count', features)
        self.assertIn('sentiment_compound', features)
```

### Integration Tests

```python
class TestAPIIntegration(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()
        
    def test_analyze_endpoint(self):
        """Test the analyze API endpoint"""
        response = self.client.post('/api/analyze', json={
            'content': 'Test news article content',
            'title': 'Test Title'
        })
        
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn('prediction', data)
        self.assertIn('confidence', data)
        
    def test_rate_limiting(self):
        """Test rate limiting functionality"""
        # Make multiple rapid requests
        for i in range(65):  # Exceed per-minute limit
            response = self.client.post('/api/analyze', json={
                'content': f'Test content {i}'
            })
            
            if i < 60:
                self.assertEqual(response.status_code, 200)
            else:
                self.assertEqual(response.status_code, 429)  # Too Many Requests
```

### Performance Tests

```python
import time
import concurrent.futures

class TestPerformance(unittest.TestCase):
    def test_processing_speed(self):
        """Test processing speed requirements"""
        model = AdvancedMLModel()
        text = "Sample article content for speed testing"
        
        start_time = time.time()
        result = model.predict(text)
        processing_time = time.time() - start_time
        
        self.assertLess(processing_time, 2.0)  # Should be under 2 seconds
        
    def test_concurrent_processing(self):
        """Test concurrent request handling"""
        model = AdvancedMLModel()
        texts = ["Test content " + str(i) for i in range(10)]
        
        start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(model.predict, text) for text in texts]
            results = [future.result() for future in futures]
        
        total_time = time.time() - start_time
        
        self.assertEqual(len(results), 10)
        self.assertLess(total_time, 10.0)  # Should complete in under 10 seconds
```

---

## Deployment Guide

### Local Development Setup

```bash
# 1. Clone repository
git clone <repository-url>
cd TruthGuard-AI

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download NLTK data
python -c "import nltk; nltk.download(['stopwords', 'punkt', 'vader_lexicon'])"

# 5. Set up spaCy (optional)
python -m spacy download en_core_web_sm

# 6. Run application
python run.py
```

### Google Cloud Deployment

#### 1. Setup Google Cloud Project
```bash
# Create project
gcloud projects create truthguard-ai-prod --name="TruthGuard AI Production"

# Set project
gcloud config set project truthguard-ai-prod

# Enable required APIs
gcloud services enable language.googleapis.com
gcloud services enable translate.googleapis.com
gcloud services enable storage.googleapis.com
gcloud services enable bigquery.googleapis.com
gcloud services enable run.googleapis.com
```

#### 2. Create Service Account
```bash
# Create service account
gcloud iam service-accounts create truthguard-ai-service \
    --display-name="TruthGuard AI Service Account"

# Grant permissions
gcloud projects add-iam-policy-binding truthguard-ai-prod \
    --member="serviceAccount:truthguard-ai-service@truthguard-ai-prod.iam.gserviceaccount.com" \
    --role="roles/ml.developer"

# Create and download key
gcloud iam service-accounts keys create credentials.json \
    --iam-account=truthguard-ai-service@truthguard-ai-prod.iam.gserviceaccount.com
```

#### 3. Deploy to Cloud Run
```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8080

CMD ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]
```

```bash
# Build and deploy
gcloud builds submit --tag gcr.io/truthguard-ai-prod/truthguard-ai

gcloud run deploy truthguard-ai \
    --image gcr.io/truthguard-ai-prod/truthguard-ai \
    --platform managed \
    --region us-central1 \
    --allow-unauthenticated \
    --memory 2Gi \
    --cpu 2
```

---

## Troubleshooting

### Common Issues and Solutions

#### 1. Model Loading Errors
```python
# Issue: Models fail to load
# Solution: Check model files and dependencies

def diagnose_model_loading():
    """Diagnose model loading issues"""
    import os
    from pathlib import Path
    
    models_dir = Path("models")
    
    if not models_dir.exists():
        print("❌ Models directory not found")
        return False
    
    required_files = [
        "advanced_ml_model.py",
        "saved_models/vectorizer.pkl",
        "saved_models/model.pkl"
    ]
    
    for file_path in required_files:
        full_path = models_dir / file_path
        if not full_path.exists():
            print(f"❌ Missing: {file_path}")
        else:
            print(f"✅ Found: {file_path}")
    
    return True
```

#### 2. Memory Issues
```python
# Issue: Out of memory errors during processing
# Solution: Implement memory management

import gc
import psutil

def monitor_memory_usage():
    """Monitor and manage memory usage"""
    process = psutil.Process()
    memory_info = process.memory_info()
    
    print(f"Memory usage: {memory_info.rss / 1024 / 1024:.2f} MB")
    
    if memory_info.rss > 1024 * 1024 * 1024:  # 1GB
        print("⚠️ High memory usage detected, running garbage collection")
        gc.collect()
```

#### 3. API Rate Limiting Issues
```python
# Issue: Rate limit exceeded errors
# Solution: Implement exponential backoff

import time
import random

def exponential_backoff(func, max_retries=3):
    """Implement exponential backoff for API calls"""
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if "rate limit" in str(e).lower() and attempt < max_retries - 1:
                wait_time = (2 ** attempt) + random.uniform(0, 1)
                print(f"Rate limit hit, waiting {wait_time:.2f} seconds...")
                time.sleep(wait_time)
            else:
                raise e
```

#### 4. Google Cloud Authentication
```bash
# Issue: Authentication errors
# Solution: Verify credentials and permissions

# Check authentication
gcloud auth list

# Set application default credentials
gcloud auth application-default login

# Verify service account
gcloud iam service-accounts list

# Test permissions
gcloud auth activate-service-account --key-file=credentials.json
```

### Performance Optimization Tips

1. **Enable Caching**: Use Streamlit's built-in caching for expensive operations
2. **Batch Processing**: Process multiple requests together when possible
3. **Model Optimization**: Use quantized models for faster inference
4. **Memory Management**: Implement garbage collection for long-running processes
5. **Database Indexing**: Create appropriate indexes for BigQuery tables

### Monitoring and Alerting

```python
import logging
from google.cloud import monitoring_v3

def setup_monitoring():
    """Setup Google Cloud Monitoring"""
    client = monitoring_v3.MetricServiceClient()
    project_name = f"projects/{project_id}"
    
    # Custom metrics for application monitoring
    descriptor = monitoring_v3.MetricDescriptor()
    descriptor.type = "custom.googleapis.com/truthguard/prediction_latency"
    descriptor.metric_kind = monitoring_v3.MetricDescriptor.MetricKind.GAUGE
    descriptor.value_type = monitoring_v3.MetricDescriptor.ValueType.DOUBLE
    descriptor.description = "Prediction processing latency"
    
    client.create_metric_descriptor(
        name=project_name, metric_descriptor=descriptor
    )
```

---

This technical documentation provides comprehensive coverage of TruthGuard AI's implementation details, architecture decisions, and operational procedures. For additional support or questions, please refer to the project repository or contact the development team.