# 💾 Saved Models Directory

## Overview

This directory contains trained model files, serialized artifacts, and model metadata for TruthGuard AI's misinformation detection system.

## Directory Structure

```
saved_models/
├── README.md                   # This documentation
├── advanced_ml/               # Advanced ML model artifacts (99% accuracy)
│   ├── vectorizer.pkl         # TF-IDF vectorizer (auto-generated)
│   ├── model.pkl             # Logistic regression model (auto-generated)
│   └── metadata.json         # Model performance metrics (auto-generated)
├── lstm/                     # LSTM neural network artifacts (94.5% accuracy)
│   ├── model.h5              # Keras/TensorFlow model (auto-generated)
│   ├── tokenizer.pkl         # Text tokenizer (auto-generated)
│   └── metadata.json         # Model architecture & metrics (auto-generated)
├── ensemble/                 # Ensemble model components (91.2% combined)
│   ├── random_forest.pkl     # Random Forest classifier (auto-generated)
│   ├── svm.pkl              # Support Vector Machine (auto-generated)
│   ├── naive_bayes.pkl      # Naive Bayes classifier (auto-generated)
│   └── ensemble_config.json  # Ensemble weights & configuration (auto-generated)
└── checkpoints/              # Training checkpoints and backups
    ├── advanced_ml_backup.pkl    # Backup of advanced ML model
    ├── lstm_checkpoint_epoch_50.h5 # LSTM training checkpoint
    └── training_history.json     # Training history and metrics
```

## Model Generation

### Automatic Model Creation

All model files in this directory are **automatically generated** when you run:

```bash
# Primary setup command
python setup_advanced_model.py

# Or through the main launcher
python run.py
```

### What Gets Created

#### Advanced ML Model Files
- **vectorizer.pkl**: TF-IDF vectorizer fitted on training data
- **model.pkl**: Trained logistic regression classifier
- **metadata.json**: Performance metrics and configuration

#### LSTM Model Files  
- **model.h5**: Complete Keras/TensorFlow neural network
- **tokenizer.pkl**: Text tokenizer with vocabulary mapping
- **metadata.json**: Architecture details and performance metrics

#### Ensemble Model Files
- **Individual classifiers**: RF, SVM, NB model files
- **ensemble_config.json**: Optimized weights for model combination

### File Sizes (Approximate)

| File | Typical Size | Description |
|------|-------------|-------------|
| `vectorizer.pkl` | 50-100 MB | TF-IDF vectorizer with vocabulary |
| `model.pkl` | 1-5 MB | Logistic regression weights |
| `model.h5` | 10-50 MB | LSTM neural network |
| `tokenizer.pkl` | 5-20 MB | Text tokenizer vocabulary |
| `random_forest.pkl` | 20-100 MB | Random Forest ensemble |
| `svm.pkl` | 10-50 MB | Support Vector Machine |
| `naive_bayes.pkl` | 1-10 MB | Naive Bayes classifier |

## Model Metadata Format

### Advanced ML Metadata Example
```json
{
  "model_type": "TF-IDF + Logistic Regression",
  "version": "1.0.0",
  "created_date": "2024-09-16",
  "training_data_size": 146372,
  "performance_metrics": {
    "accuracy": 0.990,
    "precision": 0.985,
    "recall": 0.992,
    "f1_score": 0.988,
    "auc_roc": 0.995
  },
  "hyperparameters": {
    "tfidf_max_features": 10000,
    "tfidf_ngram_range": [1, 3],
    "logistic_regression_C": 1.0,
    "random_state": 42
  },
  "feature_count": 10000,
  "processing_time_ms": 850,
  "model_size_mb": 75
}
```

### LSTM Metadata Example
```json
{
  "model_type": "LSTM Neural Network",
  "version": "1.0.0", 
  "created_date": "2024-09-16",
  "training_data_size": 146372,
  "performance_metrics": {
    "accuracy": 0.945,
    "precision": 0.928,
    "recall": 0.931,
    "f1_score": 0.929,
    "auc_roc": 0.978
  },
  "architecture": {
    "embedding_dim": 128,
    "lstm_units": 64,
    "dense_units": 32,
    "dropout_rate": 0.5,
    "max_sequence_length": 512
  },
  "training_config": {
    "epochs": 50,
    "batch_size": 32,
    "optimizer": "adam",
    "learning_rate": 0.001
  },
  "vocabulary_size": 20000,
  "processing_time_ms": 1200,
  "model_size_mb": 35
}
```

## Model Loading and Usage

### Loading Models in Code

#### Advanced ML Model
```python
import pickle
import json

# Load vectorizer
with open('saved_models/advanced_ml/vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Load model
with open('saved_models/advanced_ml/model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load metadata
with open('saved_models/advanced_ml/metadata.json', 'r') as f:
    metadata = json.load(f)

# Make prediction
text_vector = vectorizer.transform([text_content])
prediction = model.predict(text_vector)[0]
confidence = model.predict_proba(text_vector)[0].max()
```

#### LSTM Model
```python
import pickle
import json
from tensorflow.keras.models import load_model

# Load model
model = load_model('saved_models/lstm/model.h5')

# Load tokenizer
with open('saved_models/lstm/tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# Load metadata
with open('saved_models/lstm/metadata.json', 'r') as f:
    metadata = json.load(f)

# Make prediction
sequence = tokenizer.texts_to_sequences([text_content])
padded = pad_sequences(sequence, maxlen=metadata['architecture']['max_sequence_length'])
prediction = model.predict(padded)[0][0]
```

### Using the Model Manager
```python
from models.advanced_ml_model import AdvancedMLModel

# Simplified loading through model class
model = AdvancedMLModel()
model.load_model('saved_models/advanced_ml/')

# Direct prediction
result = model.predict(text_content)
print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.2%}")
```

## Backup and Recovery

### Automatic Backups
The setup process creates backups of critical models:

```bash
# Backup locations
saved_models/checkpoints/advanced_ml_backup.pkl
saved_models/checkpoints/lstm_checkpoint_epoch_50.h5
saved_models/checkpoints/training_history.json
```

### Manual Backup
```bash
# Create backup of all models
tar -czf models_backup_$(date +%Y%m%d).tar.gz saved_models/

# Restore from backup
tar -xzf models_backup_20240916.tar.gz
```

### Recovery Process
```bash
# If models are corrupted, regenerate
rm -rf saved_models/advanced_ml/
python setup_advanced_model.py

# Or restore from backup
cp saved_models/checkpoints/advanced_ml_backup.pkl saved_models/advanced_ml/model.pkl
```

## Performance Monitoring

### Model Performance Tracking
```json
{
  "performance_history": [
    {
      "date": "2024-09-16",
      "accuracy": 0.990,
      "processing_time_ms": 850,
      "memory_usage_mb": 2048
    }
  ],
  "alerts": {
    "accuracy_threshold": 0.85,
    "performance_threshold_ms": 3000,
    "memory_threshold_mb": 4096
  }
}
```

### Health Checks
```python
# Model health verification
def verify_model_health():
    """Verify all models are functional"""
    try:
        # Test advanced ML model
        from models.advanced_ml_model import AdvancedMLModel
        model = AdvancedMLModel()
        test_result = model.predict("This is a test article.")
        assert 'prediction' in test_result
        assert 'confidence' in test_result
        print("✅ Advanced ML model: Healthy")
        
        return True
    except Exception as e:
        print(f"❌ Model health check failed: {e}")
        return False
```

## Troubleshooting

### Common Issues

#### Missing Model Files
```bash
# Error: FileNotFoundError: saved_models/advanced_ml/model.pkl
# Solution: Run model setup
python setup_advanced_model.py
```

#### Corrupted Model Files
```bash
# Error: pickle.UnpicklingError or similar
# Solution: Regenerate models
rm -rf saved_models/advanced_ml/
python setup_advanced_model.py
```

#### Version Compatibility
```bash
# Error: sklearn/tensorflow version mismatch
# Solution: Update packages and regenerate
pip install --upgrade scikit-learn tensorflow
python setup_advanced_model.py
```

#### Memory Issues
```python
# For memory-constrained environments
import gc

# Load model on demand
def load_model_on_demand():
    model = load_model()
    # Use model
    result = model.predict(text)
    # Clean up
    del model
    gc.collect()
    return result
```

## Model Security

### File Integrity
```bash
# Generate checksums for model files
find saved_models -name "*.pkl" -o -name "*.h5" | xargs sha256sum > model_checksums.txt

# Verify integrity
sha256sum -c model_checksums.txt
```

### Access Control
- Model files should be read-only in production
- Restrict write access to training processes only
- Use proper file permissions (644 for files, 755 for directories)

## Contributing

### Adding New Model Formats
1. Create appropriate subdirectory in `saved_models/`
2. Implement loading/saving functions in model class
3. Add metadata.json with performance metrics
4. Update this documentation

### Model Validation
All saved models must include:
- ✅ Serialized model file
- ✅ Metadata with performance metrics
- ✅ Loading verification test
- ✅ Documentation update

---

**Important Notes**:
- All model files are automatically generated during setup
- Do not manually edit model files
- Run `python setup_advanced_model.py` if models are missing
- Check `test_setup.py` to verify model functionality

**Last Updated**: September 2024
**Auto-Generation**: ✅ Enabled
**Health Status**: To be verified after setup