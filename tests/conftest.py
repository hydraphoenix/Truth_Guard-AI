"""
TruthGuard AI - Pytest Configuration and Fixtures

This file contains pytest configuration, fixtures, and shared test utilities
for the TruthGuard AI test suite.
"""

import pytest
import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Test configuration
@pytest.fixture(scope="session")
def test_config():
    """Provide test configuration for all tests"""
    return {
        "performance_thresholds": {
            "max_processing_time_seconds": 2.0,
            "max_memory_usage_mb": 2048,
            "min_accuracy": 0.90
        },
        "sample_sizes": {
            "small": 10,
            "medium": 100,
            "large": 1000
        },
        "timeouts": {
            "unit_test": 5,
            "integration_test": 30,
            "performance_test": 60
        }
    }

@pytest.fixture(scope="session")
def sample_real_news():
    """Provide sample real news content for testing"""
    return [
        {
            "title": "Stanford Researchers Develop New COVID-19 Treatment",
            "content": "Scientists at Stanford University have developed a new treatment for COVID-19 that shows promising results in early clinical trials. The treatment, based on monoclonal antibodies, has shown a 67% reduction in hospitalization rates among high-risk patients. The research team, led by Dr. Sarah Johnson, published their findings in the New England Journal of Medicine after a 6-month study involving 2,000 participants across 15 medical centers.",
            "expected_label": "Real",
            "confidence_threshold": 0.8
        },
        {
            "title": "Tech Company Announces New AI Initiative",
            "content": "A major technology company announced today that it will be investing $2 billion in artificial intelligence research over the next five years. The initiative will focus on developing AI solutions for healthcare, education, and environmental sustainability. The company's CEO stated that this investment represents their commitment to using technology for social good.",
            "expected_label": "Real",
            "confidence_threshold": 0.7
        },
        {
            "title": "Local Election Results Certified",
            "content": "The state election board has officially certified the results of last week's municipal elections. Voter turnout was reported at 68%, the highest in the city's recent history. Election officials praised the smooth operation of polling stations and the accuracy of electronic voting systems.",
            "expected_label": "Real",
            "confidence_threshold": 0.8
        }
    ]

@pytest.fixture(scope="session")
def sample_fake_news():
    """Provide sample fake news content for testing"""
    return [
        {
            "title": "SHOCKING: 5G Towers Control Your Mind!",
            "content": "URGENT ALERT!!! Secret government documents EXPOSED revealing that 5G towers are actually mind control devices designed to control the population! Scientists are TERRIFIED to speak out but one brave researcher has leaked the TRUTH! The radiation from these towers can alter your DNA and make you obey government commands! Share this before it gets DELETED! Big Tech doesn't want you to know this SHOCKING secret!",
            "expected_label": "Fake",
            "confidence_threshold": 0.9
        },
        {
            "title": "Doctors Don't Want You to Know This Simple Trick",
            "content": "Medical professionals HATE this one weird trick that cures EVERYTHING! A local mom discovered this ancient secret that Big Pharma has been hiding for decades. Just drink this special water mixture and you'll never need medicine again! Doctors are losing their minds over this discovery! Click here to learn the secret they don't want you to know!",
            "expected_label": "Fake",
            "confidence_threshold": 0.95
        },
        {
            "title": "BREAKING: Aliens Have Landed in Nevada",
            "content": "EXCLUSIVE FOOTAGE shows alien spacecraft landing near Area 51! Government sources confirm that extraterrestrial contact has been made. The aliens have been negotiating with world leaders for months. This is being covered up by mainstream media! Only independent journalists are reporting the TRUTH! Share this before the government shuts down the internet!",
            "expected_label": "Fake",
            "confidence_threshold": 0.9
        }
    ]

@pytest.fixture(scope="session")
def edge_cases():
    """Provide edge case content for testing"""
    return {
        "very_short": "Breaking news!",
        "very_long": "Lorem ipsum dolor sit amet, consectetur adipiscing elit. " * 1000,
        "special_characters": "Test with émojis 🚨, sp3c!@l ch@rs, and FORMATTING issues...",
        "empty_string": "",
        "whitespace_only": "   \n\t   ",
        "numbers_only": "12345 67890 2024 99.9%",
        "mixed_languages": "This is English mixed with español and français",
        "html_content": "<script>alert('test')</script>This is a test article with HTML.",
        "urls_and_emails": "Visit https://example.com or email test@example.com for more info."
    }

@pytest.fixture
def mock_model():
    """Provide a mock model for testing without loading actual models"""
    class MockModel:
        def __init__(self):
            self.loaded = True
            
        def predict(self, text):
            # Simple mock prediction based on keywords
            fake_indicators = ['shocking', 'urgent', 'secret', 'exposed', 'deleted']
            is_fake = any(indicator.lower() in text.lower() for indicator in fake_indicators)
            
            return {
                'prediction': 'Fake' if is_fake else 'Real',
                'confidence': 0.95 if is_fake else 0.85,
                'risk_score': 85 if is_fake else 15,
                'risk_level': 'High' if is_fake else 'Low',
                'processing_time': 0.5,
                'features': {
                    'word_count': len(text.split()),
                    'sentiment_compound': -0.5 if is_fake else 0.3
                }
            }
            
        def predict_batch(self, texts):
            return [self.predict(text) for text in texts]
    
    return MockModel()

@pytest.fixture
def performance_timer():
    """Provide a timer for performance testing"""
    import time
    
    class Timer:
        def __init__(self):
            self.start_time = None
            self.end_time = None
            
        def start(self):
            self.start_time = time.time()
            
        def stop(self):
            self.end_time = time.time()
            
        def elapsed(self):
            if self.start_time and self.end_time:
                return self.end_time - self.start_time
            return None
            
        def __enter__(self):
            self.start()
            return self
            
        def __exit__(self, *args):
            self.stop()
    
    return Timer()

@pytest.fixture
def memory_monitor():
    """Provide memory usage monitoring for tests"""
    import psutil
    import os
    
    class MemoryMonitor:
        def __init__(self):
            self.process = psutil.Process(os.getpid())
            self.initial_memory = None
            
        def start(self):
            self.initial_memory = self.process.memory_info().rss
            
        def current_usage_mb(self):
            return self.process.memory_info().rss / 1024 / 1024
            
        def memory_increase_mb(self):
            if self.initial_memory:
                current = self.process.memory_info().rss
                return (current - self.initial_memory) / 1024 / 1024
            return 0
    
    return MemoryMonitor()

# Pytest markers
def pytest_configure(config):
    """Configure pytest markers"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (may take several seconds)"
    )
    config.addinivalue_line(
        "markers", "fast: marks tests as fast (should complete in <1 second)"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", "performance: marks tests as performance benchmarks"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests that require GPU (skipped if not available)"
    )
    config.addinivalue_line(
        "markers", "cloud: marks tests that require Google Cloud setup"
    )

# Test environment setup
@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Setup test environment before running tests"""
    # Ensure we're in the correct directory
    os.chdir(project_root)
    
    # Set environment variables for testing
    os.environ['TRUTHGUARD_DEBUG'] = 'true'
    os.environ['TRUTHGUARD_TEST_MODE'] = 'true'
    
    # Suppress warnings for cleaner test output
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    
    yield
    
    # Cleanup after tests
    # (Any cleanup code would go here)

# Skip conditions
def pytest_runtest_setup(item):
    """Setup conditions for skipping tests"""
    # Skip GPU tests if no GPU available
    if "gpu" in item.keywords:
        try:
            import tensorflow as tf
            if not tf.config.list_physical_devices('GPU'):
                pytest.skip("GPU not available")
        except ImportError:
            pytest.skip("TensorFlow not available")
    
    # Skip cloud tests if not configured
    if "cloud" in item.keywords:
        if not os.getenv('GOOGLE_APPLICATION_CREDENTIALS'):
            pytest.skip("Google Cloud credentials not configured")

# Custom assertions
def assert_prediction_format(result):
    """Assert that prediction result has correct format"""
    required_keys = ['prediction', 'confidence', 'risk_score', 'risk_level']
    for key in required_keys:
        assert key in result, f"Missing required key: {key}"
    
    assert result['prediction'] in ['Real', 'Fake'], f"Invalid prediction: {result['prediction']}"
    assert 0 <= result['confidence'] <= 1, f"Invalid confidence: {result['confidence']}"
    assert 0 <= result['risk_score'] <= 100, f"Invalid risk score: {result['risk_score']}"
    assert result['risk_level'] in ['Low', 'Medium', 'High'], f"Invalid risk level: {result['risk_level']}"

# Add custom assertion to pytest
pytest.assert_prediction_format = assert_prediction_format