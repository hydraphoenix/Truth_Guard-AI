# 🧪 TruthGuard AI - Testing Suite

## Overview

This directory contains comprehensive test suites for TruthGuard AI, including unit tests, integration tests, performance tests, and evaluation scripts for the Google Cloud AI Challenge submission.

## Test Structure

```
tests/
├── README.md                   # This documentation
├── __init__.py                # Test package initialization
├── conftest.py                # Pytest configuration and fixtures
├── unit/                      # Unit tests for individual components
│   ├── __init__.py           # Unit tests package
│   ├── test_advanced_ml_model.py  # Advanced ML model tests
│   ├── test_feature_extraction.py # Feature engineering tests
│   ├── test_text_preprocessing.py # Text preprocessing tests
│   └── test_config.py         # Configuration tests
├── integration/               # Integration tests for system components
│   ├── __init__.py           # Integration tests package
│   ├── test_app_integration.py    # Streamlit app integration
│   ├── test_model_pipeline.py     # End-to-end model pipeline
│   └── test_google_cloud.py       # Google Cloud services integration
├── performance/               # Performance and load testing
│   ├── __init__.py           # Performance tests package
│   ├── test_processing_speed.py   # Speed benchmarks
│   ├── test_memory_usage.py       # Memory consumption tests
│   └── test_load_testing.py       # Concurrent request testing
├── evaluation/                # Model evaluation and accuracy tests
│   ├── __init__.py           # Evaluation package
│   ├── test_model_accuracy.py     # Accuracy verification
│   ├── test_cross_validation.py   # Cross-validation tests
│   └── test_benchmark_datasets.py # Standard dataset evaluation
├── fixtures/                  # Test data and fixtures
│   ├── sample_real_news.txt   # Real news examples for testing
│   ├── sample_fake_news.txt   # Fake news examples for testing
│   └── test_config.json      # Test configuration
└── reports/                   # Test reports and coverage
    ├── coverage_report.html   # Coverage analysis (generated)
    ├── performance_report.json # Performance benchmarks (generated)
    └── accuracy_report.json   # Model accuracy results (generated)
```

## Running Tests

### Quick Test Suite (for Judges)
```bash
# Run essential functionality tests
python -m pytest tests/unit/test_advanced_ml_model.py -v

# Run integration tests
python -m pytest tests/integration/test_app_integration.py -v

# Run performance benchmarks
python -m pytest tests/performance/test_processing_speed.py -v
```

### Complete Test Suite
```bash
# Run all tests with coverage
python -m pytest tests/ --cov=. --cov-report=html

# Run specific test categories
python -m pytest tests/unit/ -v        # Unit tests only
python -m pytest tests/integration/ -v # Integration tests only
python -m pytest tests/performance/ -v # Performance tests only
```

### Test Configuration
```bash
# Install test dependencies
pip install pytest pytest-cov pytest-mock pytest-benchmark

# Run with specific markers
python -m pytest -m "fast" -v      # Quick tests only
python -m pytest -m "slow" -v      # Comprehensive tests
python -m pytest -m "gpu" -v       # GPU-dependent tests (if available)
```

## Test Categories

### Unit Tests
**Purpose**: Test individual functions and classes in isolation
**Speed**: Fast (< 1 second per test)
**Coverage**: Core functionality, edge cases, error handling

#### Key Test Files:
- `test_advanced_ml_model.py`: AI model functionality
- `test_feature_extraction.py`: Feature engineering pipeline
- `test_text_preprocessing.py`: Text cleaning and normalization
- `test_config.py`: Configuration management

### Integration Tests
**Purpose**: Test component interactions and system workflows
**Speed**: Medium (1-10 seconds per test)
**Coverage**: End-to-end functionality, API integration

#### Key Test Files:
- `test_app_integration.py`: Streamlit application testing
- `test_model_pipeline.py`: Complete prediction pipeline
- `test_google_cloud.py`: Cloud services integration (if configured)

### Performance Tests
**Purpose**: Verify speed, memory, and scalability requirements
**Speed**: Variable (depends on test complexity)
**Coverage**: Processing speed, memory usage, concurrent handling

#### Key Test Files:
- `test_processing_speed.py`: Sub-2-second processing requirement
- `test_memory_usage.py`: Memory consumption monitoring
- `test_load_testing.py`: Concurrent request handling

### Evaluation Tests
**Purpose**: Verify model accuracy and performance metrics
**Speed**: Slow (may take minutes for full evaluation)
**Coverage**: Model accuracy, cross-validation, benchmark comparison

#### Key Test Files:
- `test_model_accuracy.py`: Accuracy verification on test sets
- `test_cross_validation.py`: Robust evaluation strategies
- `test_benchmark_datasets.py`: Standard dataset comparisons

## Test Data and Fixtures

### Sample Content
**Real News Examples**: Verified authentic news articles
**Fake News Examples**: Confirmed misinformation samples
**Edge Cases**: Short text, long text, special characters

### Test Configuration
```json
{
  "performance_thresholds": {
    "max_processing_time_seconds": 2.0,
    "max_memory_usage_mb": 2048,
    "min_accuracy": 0.85
  },
  "test_datasets": {
    "small_sample": 100,
    "medium_sample": 1000,
    "large_sample": 10000
  },
  "google_cloud": {
    "enable_cloud_tests": false,
    "test_project_id": "test-project",
    "mock_services": true
  }
}
```

## Expected Test Results

### Performance Benchmarks
- **Processing Speed**: < 2 seconds per article
- **Memory Usage**: < 2GB RAM for normal operations
- **Accuracy**: > 90% on provided test sets
- **Startup Time**: < 30 seconds for application launch

### Coverage Targets
- **Unit Test Coverage**: > 85%
- **Integration Coverage**: > 70%
- **Critical Path Coverage**: 100%

## Continuous Integration

### GitHub Actions Workflow
```yaml
name: TruthGuard AI Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.9
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-cov
    - name: Run tests
      run: pytest tests/ --cov=. --cov-report=xml
    - name: Upload coverage
      uses: codecov/codecov-action@v1
```

### Pre-commit Hooks
```yaml
repos:
-   repo: local
    hooks:
    -   id: pytest-check
        name: pytest-check
        entry: pytest tests/unit/
        language: system
        pass_filenames: false
        always_run: true
```

## Testing Guidelines for Contributors

### Writing New Tests
1. **Follow naming convention**: `test_*.py` for files, `test_*` for functions
2. **Use descriptive names**: Clear test purpose and expected outcome
3. **Include docstrings**: Explain what the test verifies
4. **Test edge cases**: Not just happy path scenarios
5. **Use appropriate markers**: `@pytest.mark.slow` for lengthy tests

### Test Quality Standards
- **Isolated**: Tests should not depend on each other
- **Deterministic**: Same input should always produce same result
- **Fast**: Unit tests should complete in < 1 second
- **Clear**: Easy to understand test purpose and assertions
- **Maintainable**: Easy to update when code changes

### Mock and Fixtures
```python
# Example test with fixtures
@pytest.fixture
def sample_news_article():
    return {
        'title': 'Test News Title',
        'content': 'This is a sample news article for testing.',
        'expected_label': 'Real'
    }

def test_model_prediction(sample_news_article):
    model = AdvancedMLModel()
    result = model.predict(sample_news_article['content'])
    assert result['prediction'] in ['Real', 'Fake']
    assert 0 <= result['confidence'] <= 1
```

## Judge Evaluation Tests

### Quick Verification Suite
For Google Cloud AI Challenge judges, a streamlined test suite verifies core functionality:

```bash
# Essential functionality test (30 seconds)
python -m pytest tests/evaluation/test_judge_verification.py -v

# Expected output:
# ✅ Model loading and prediction
# ✅ Processing speed under 2 seconds
# ✅ Accuracy above 90% threshold
# ✅ Application startup and basic UI
# ✅ Error handling and edge cases
```

### Demonstration Tests
```bash
# Interactive demonstration test
python tests/evaluation/demo_test.py

# This will:
# - Load sample real and fake news
# - Run predictions and show results
# - Display performance metrics
# - Verify all major features work
```

## Troubleshooting Tests

### Common Test Issues

#### Import Errors
```bash
# Ensure PYTHONPATH includes project root
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Or run tests from project root
cd TruthGuard-AI
python -m pytest tests/
```

#### Missing Dependencies
```bash
# Install test dependencies
pip install pytest pytest-cov pytest-mock pytest-benchmark

# For GPU tests
pip install tensorflow-gpu  # if GPU available
```

#### Model Loading Errors
```bash
# Ensure models are generated first
python setup_advanced_model.py

# Then run tests
python -m pytest tests/unit/test_advanced_ml_model.py
```

### Performance Test Issues
```bash
# Skip slow tests during development
python -m pytest tests/ -m "not slow"

# Run performance tests with relaxed thresholds
python -m pytest tests/performance/ --benchmark-min-rounds=1
```

## Test Reports and Analytics

### Coverage Reports
```bash
# Generate HTML coverage report
python -m pytest tests/ --cov=. --cov-report=html

# View report
open htmlcov/index.html
```

### Performance Reports
```bash
# Generate performance benchmark report
python -m pytest tests/performance/ --benchmark-json=benchmark.json

# View performance trends
python scripts/analyze_benchmarks.py benchmark.json
```

### Accuracy Reports
```bash
# Generate model accuracy report
python tests/evaluation/generate_accuracy_report.py

# Output: tests/reports/accuracy_report.json
```

---

**Note**: Test files are provided as templates and examples. Full test implementation depends on final model architecture and specific requirements.

**Last Updated**: September 2024
**Test Framework**: pytest
**Coverage Tool**: pytest-cov
**Status**: Test structure ready, implementation pending