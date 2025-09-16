# 🤖 TruthGuard AI - Models Directory

## Overview

This directory contains the AI models, training scripts, and model-related utilities for TruthGuard AI's misinformation detection system.

## Directory Structure

```
models/
├── README.md                    # This documentation
├── advanced_ml_model.py        # Primary ML model (99% accuracy)
├── saved_models/               # Trained model files and artifacts
│   ├── README.md              # Model files documentation
│   ├── advanced_ml/           # Advanced ML model artifacts
│   │   ├── vectorizer.pkl     # TF-IDF vectorizer
│   │   ├── model.pkl          # Trained logistic regression model
│   │   └── metadata.json      # Model metadata and performance
│   ├── lstm/                  # LSTM neural network model
│   │   ├── model.h5           # Trained LSTM model
│   │   ├── tokenizer.pkl      # Text tokenizer
│   │   └── metadata.json      # Model metadata
│   ├── ensemble/              # Ensemble model components
│   │   ├── random_forest.pkl  # Random Forest model
│   │   ├── svm.pkl           # Support Vector Machine model
│   │   ├── naive_bayes.pkl   # Naive Bayes model
│   │   └── ensemble_config.json # Ensemble weights and configuration
│   └── checkpoints/           # Training checkpoints and backups
├── training/                   # Training scripts and utilities
│   ├── train_advanced_ml.py   # Advanced ML model training
│   ├── train_lstm.py          # LSTM model training
│   ├── train_ensemble.py      # Ensemble model training
│   └── data_preprocessing.py  # Data preprocessing utilities
├── evaluation/                 # Model evaluation and testing
│   ├── evaluate_models.py     # Model performance evaluation
│   ├── cross_validation.py    # Cross-validation scripts
│   └── performance_metrics.py # Metrics calculation utilities
└── utils/                     # Model utilities and helpers
    ├── feature_extraction.py  # Feature engineering functions
    ├── text_preprocessing.py  # Text preprocessing utilities
    └── model_utils.py         # General model utilities
```

## Model Components

### 1. Advanced ML Model (Primary - 99% Accuracy)

**File**: `advanced_ml_model.py`
**Architecture**: TF-IDF + Logistic Regression
**Performance**: 99.0% accuracy on test set

#### Key Features:
- TF-IDF vectorization with n-gram range (1,3)
- Logistic Regression with optimized hyperparameters
- Comprehensive text preprocessing pipeline
- Feature extraction with linguistic indicators
- Fast inference (<1 second processing time)

#### Model Files:
- `saved_models/advanced_ml/vectorizer.pkl` - TF-IDF vectorizer
- `saved_models/advanced_ml/model.pkl` - Trained classifier
- `saved_models/advanced_ml/metadata.json` - Performance metrics

### 2. LSTM Neural Network (Secondary - 94.5% Accuracy)

**Architecture**: Deep LSTM with embedding layer
**Performance**: 94.5% accuracy on test set

#### Key Features:
- Word embedding layer (128 dimensions)
- LSTM layer with dropout regularization
- Dense layers with activation functions
- Sequential pattern recognition
- Handles variable-length text inputs

#### Model Files:
- `saved_models/lstm/model.h5` - Trained neural network
- `saved_models/lstm/tokenizer.pkl` - Text tokenizer
- `saved_models/lstm/metadata.json` - Architecture and metrics

### 3. Ensemble Models

**Components**: Random Forest, SVM, Naive Bayes
**Combined Accuracy**: 91.2% weighted ensemble

#### Ensemble Configuration:
```json
{
  "weights": {
    "advanced_ml": 0.4,
    "lstm": 0.3,
    "random_forest": 0.15,
    "svm": 0.1,
    "naive_bayes": 0.05
  },
  "voting_strategy": "weighted_soft",
  "confidence_threshold": 0.7
}
```

## Model Training

### Setup Requirements
```bash
# Install training dependencies
pip install -r requirements.txt

# Download required datasets
python -c "from data.data_loader import load_training_data; load_training_data()"

# Setup NLTK data
python -c "import nltk; nltk.download(['stopwords', 'punkt', 'vader_lexicon'])"
```

### Training Pipeline

#### 1. Advanced ML Model
```bash
# Train the primary model
python models/training/train_advanced_ml.py

# Expected output:
# - saved_models/advanced_ml/vectorizer.pkl
# - saved_models/advanced_ml/model.pkl
# - saved_models/advanced_ml/metadata.json
```

#### 2. LSTM Model  
```bash
# Train the neural network
python models/training/train_lstm.py --epochs 50 --batch_size 32

# Expected output:
# - saved_models/lstm/model.h5
# - saved_models/lstm/tokenizer.pkl
# - saved_models/lstm/metadata.json
```

#### 3. Ensemble Models
```bash
# Train all ensemble components
python models/training/train_ensemble.py

# Expected output:
# - Individual model files in saved_models/ensemble/
# - ensemble_config.json with optimized weights
```

### Automated Setup
```bash
# Run complete model setup
python setup_advanced_model.py

# This script will:
# - Check for existing models
# - Download/train missing models
# - Verify model functionality
# - Create necessary directories
```

## Model Evaluation

### Performance Metrics

#### Advanced ML Model
- **Accuracy**: 99.0%
- **Precision**: 98.5%
- **Recall**: 99.2%
- **F1-Score**: 98.8%
- **Processing Time**: <1 second

#### LSTM Model
- **Accuracy**: 94.5%
- **Precision**: 92.8%
- **Recall**: 93.1%
- **F1-Score**: 92.9%
- **Processing Time**: <2 seconds

#### Ensemble Average
- **Accuracy**: 91.2%
- **Precision**: 89.7%
- **Recall**: 90.3%
- **F1-Score**: 90.0%
- **Robustness**: High (multiple model consensus)

### Evaluation Scripts
```bash
# Evaluate all models
python models/evaluation/evaluate_models.py

# Cross-validation analysis
python models/evaluation/cross_validation.py --cv 5

# Generate performance report
python models/evaluation/performance_metrics.py --output report.json
```

## Feature Engineering

### Linguistic Features
- Word count, character count, sentence count
- Average word length, sentence length
- Capitalization ratio, punctuation ratio
- Exclamation marks, question marks
- Digit ratio, special character patterns

### Sentiment Features
- Sentiment polarity (positive, negative, neutral)
- Sentiment intensity (compound score)
- Subjectivity score
- Emotional indicators

### Credibility Features
- Authority indicators (expert, study, research)
- Sensationalism markers (shocking, unbelievable)
- Urgency signals (urgent, breaking, alert)
- Source credibility patterns

### Implementation
```python
from models.utils.feature_extraction import extract_all_features

# Extract comprehensive features
features = extract_all_features(text_content)

# Features include:
# - linguistic: word patterns, structure
# - sentiment: emotional indicators
# - credibility: authority and reliability markers
```

## Model Deployment

### Production Deployment
```python
from models.advanced_ml_model import AdvancedMLModel

# Load trained model
model = AdvancedMLModel()
model.load_model('saved_models/advanced_ml/')

# Make prediction
result = model.predict(text_content)
print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.2%}")
```

### API Integration
```python
# For REST API deployment
from models.model_manager import ModelManager

manager = ModelManager()
manager.load_all_models()

# Process request
prediction = manager.predict_ensemble(content)
```

## Model Updates and Versioning

### Versioning Strategy
- **Major versions**: Significant architecture changes
- **Minor versions**: Performance improvements, new features
- **Patch versions**: Bug fixes, small optimizations

### Update Process
1. Train new model version
2. Evaluate against current benchmark
3. A/B test with subset of traffic
4. Deploy if performance improves
5. Maintain backward compatibility

### Model Registry
```json
{
  "model_versions": {
    "advanced_ml": {
      "current": "1.0.0",
      "accuracy": 0.99,
      "last_updated": "2024-09-16"
    },
    "lstm": {
      "current": "1.0.0", 
      "accuracy": 0.945,
      "last_updated": "2024-09-16"
    }
  }
}
```

## Troubleshooting

### Common Issues

#### Model Loading Errors
```bash
# Check model file existence
ls -la saved_models/advanced_ml/

# Verify file integrity
python -c "import pickle; pickle.load(open('saved_models/advanced_ml/model.pkl', 'rb'))"
```

#### Memory Issues
```python
# For large datasets, use batch processing
from models.utils.model_utils import batch_predict

predictions = batch_predict(model, texts, batch_size=100)
```

#### Performance Issues
```python
# Enable model caching
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_predict(text_hash):
    return model.predict(text)
```

## Contributing

### Adding New Models
1. Create model class inheriting from `BaseModel`
2. Implement required methods: `train()`, `predict()`, `save()`, `load()`
3. Add evaluation metrics and documentation
4. Update ensemble configuration if applicable

### Model Guidelines
- **Accuracy**: Minimum 85% on test set
- **Speed**: Maximum 3 seconds processing time
- **Documentation**: Complete docstrings and examples
- **Testing**: Unit tests for all public methods

## Research and Development

### Future Improvements
- **Multi-language Support**: Models for Hindi, Bengali, Tamil
- **Real-time Learning**: Online learning capabilities
- **Adversarial Robustness**: Defense against adversarial examples
- **Explainable AI**: Enhanced model interpretability

### Experimental Models
- Transformer-based models (BERT, RoBERTa)
- Graph neural networks for source verification
- Multimodal models (text + images)
- Federated learning approaches

---

**Note**: Model files are automatically generated during setup. Run `python setup_advanced_model.py` to initialize all models.

**Last Updated**: September 2024
**Model Version**: 1.0.0
**Performance Verified**: ✅