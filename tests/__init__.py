"""
TruthGuard AI - Test Suite

This package contains comprehensive tests for the TruthGuard AI misinformation detection system.
Includes unit tests, integration tests, performance tests, and evaluation scripts.

Test Categories:
- unit/: Individual component testing
- integration/: System interaction testing  
- performance/: Speed and memory benchmarks
- evaluation/: Model accuracy and validation

Usage:
    python -m pytest tests/                    # Run all tests
    python -m pytest tests/unit/               # Unit tests only
    python -m pytest tests/performance/        # Performance tests only
    python -m pytest tests/ -v                 # Verbose output
    python -m pytest tests/ --cov=.            # With coverage

For judges: 
    python -m pytest tests/evaluation/test_judge_verification.py -v
"""

__version__ = "1.0.0"
__author__ = "TruthGuard AI Team"

# Test configuration
TEST_CONFIG = {
    "performance_thresholds": {
        "max_processing_time_seconds": 2.0,
        "max_memory_usage_mb": 2048,
        "min_accuracy": 0.90
    },
    "test_data_paths": {
        "sample_real_news": "tests/fixtures/sample_real_news.txt",
        "sample_fake_news": "tests/fixtures/sample_fake_news.txt",
        "test_config": "tests/fixtures/test_config.json"
    }
}

# Common test utilities
def get_test_config():
    """Get test configuration dictionary"""
    return TEST_CONFIG

def get_sample_content():
    """Get sample content for testing"""
    return {
        "real_news": "Scientists at Stanford University have developed a new treatment for COVID-19 that shows promising results in early clinical trials. The research team published their findings in a peer-reviewed journal after extensive testing.",
        "fake_news": "SHOCKING! Government secretly putting mind control chips in vaccines! Scientists are TERRIFIED to speak out! Share this before it gets DELETED by Big Tech!"
    }