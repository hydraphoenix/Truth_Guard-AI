"""
TruthGuard AI - Configuration Settings

This module contains all configuration settings for the TruthGuard AI application.
"""

import os
from pathlib import Path

# Application Information
APP_NAME = "TruthGuard AI"
APP_VERSION = "1.0.0"
APP_DESCRIPTION = "Advanced AI-Powered Misinformation Detection System"
APP_AUTHOR = "TruthGuard AI Team"

# Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
ASSETS_DIR = BASE_DIR / "assets"
LOGS_DIR = BASE_DIR / "logs"

# Ensure directories exist
for directory in [DATA_DIR, MODELS_DIR, ASSETS_DIR, LOGS_DIR]:
    directory.mkdir(exist_ok=True)

# Streamlit Configuration
STREAMLIT_CONFIG = {
    'server.port': int(os.getenv('TRUTHGUARD_PORT', 8501)),
    'server.headless': True,
    'server.runOnSave': True,
    'browser.gatherUsageStats': False,
    'client.showErrorDetails': False
}

# Model Configuration
MODEL_CONFIG = {
    'ensemble_weights': {
        'lstm': 0.4,
        'random_forest': 0.2,
        'svm': 0.15,
        'naive_bayes': 0.1,
        'heuristic': 0.15
    },
    'confidence_threshold': 0.7,
    'risk_thresholds': {
        'low': (0, 25),
        'low_medium': (25, 45),
        'medium': (45, 70),
        'high': (70, 100)
    }
}

# Detection Parameters
DETECTION_CONFIG = {
    'max_text_length': 10000,
    'min_text_length': 5,
    'default_language': 'en',
    'supported_languages': ['en', 'hi', 'es', 'fr'],
    'timeout_seconds': 30
}

# Feature Extraction Settings
FEATURE_CONFIG = {
    'linguistic_features': [
        'word_count', 'char_count', 'sentence_count',
        'avg_word_length', 'avg_sentence_length',
        'caps_ratio', 'punctuation_ratio'
    ],
    'sentiment_features': [
        'sentiment_compound', 'sentiment_positive',
        'sentiment_negative', 'sentiment_neutral'
    ],
    'credibility_features': [
        'credibility_score', 'authority_score',
        'urgency_score', 'sensational_score', 'fear_score'
    ]
}

# Educational Content
EDUCATION_CONFIG = {
    'enable_quizzes': True,
    'track_progress': True,
    'show_explanations': True,
    'case_studies_enabled': True,
    'interactive_learning': True
}

# Analytics Settings
ANALYTICS_CONFIG = {
    'enable_tracking': False,  # Set to True to enable usage analytics
    'retention_days': 30,
    'aggregate_only': True,
    'privacy_mode': True
}

# Google Cloud Settings (Optional)
GOOGLE_CLOUD_CONFIG = {
    'project_id': os.getenv('GOOGLE_CLOUD_PROJECT', ''),
    'credentials_path': os.getenv('GOOGLE_APPLICATION_CREDENTIALS', ''),
    'services': {
        'natural_language': True,
        'translation': True,
        'automl': False,
        'bigquery': False,
        'pubsub': False
    },
    'regions': {
        'default': 'us-central1',
        'backup': 'us-east1'
    }
}

# API Configuration
API_CONFIG = {
    'enable_api': False,  # Set to True to enable REST API
    'api_key_required': False,
    'rate_limit': {
        'requests_per_minute': 60,
        'requests_per_hour': 1000
    },
    'cors_origins': ['http://localhost:3000', 'http://localhost:8000']
}

# Security Settings
SECURITY_CONFIG = {
    'input_sanitization': True,
    'xss_protection': True,
    'csrf_protection': True,
    'rate_limiting': True,
    'ip_whitelist': [],  # Empty list means no IP restrictions
    'max_content_length': 50000  # Maximum characters in input
}

# Logging Configuration
LOGGING_CONFIG = {
    'level': os.getenv('TRUTHGUARD_LOG_LEVEL', 'INFO'),
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file_logging': True,
    'console_logging': True,
    'log_rotation': True,
    'max_file_size': '10MB',
    'backup_count': 5
}

# UI Theme and Styling
UI_CONFIG = {
    'theme': 'light',  # 'light' or 'dark'
    'primary_color': '#1f77b4',
    'background_color': '#ffffff',
    'text_color': '#000000',
    'sidebar_color': '#f0f2f6',
    'font_family': 'Arial, sans-serif',
    'show_loading_spinner': True,
    'animation_enabled': True
}

# Cache Settings
CACHE_CONFIG = {
    'enable_caching': True,
    'cache_ttl': 3600,  # 1 hour in seconds
    'max_cache_size': 1000,  # Maximum number of cached results
    'clear_cache_on_startup': False
}

# Development Settings
DEV_CONFIG = {
    'debug_mode': os.getenv('TRUTHGUARD_DEBUG', 'false').lower() == 'true',
    'show_debug_info': False,
    'enable_profiling': False,
    'mock_external_apis': False,
    'test_mode': False
}

# Feature Flags
FEATURE_FLAGS = {
    'content_analysis': True,
    'educational_hub': True,
    'analytics_dashboard': True,
    'realtime_monitor': False,  # Coming soon
    'api_endpoints': False,  # Coming soon
    'mobile_support': True,
    'multi_language': False,  # Coming soon
    'google_cloud_integration': False,  # Optional
    'user_authentication': False,  # Future feature
    'collaboration_tools': False  # Future feature
}

# Error Messages
ERROR_MESSAGES = {
    'invalid_input': "Please provide valid text content for analysis.",
    'content_too_short': "Content must be at least 5 words long for accurate analysis.",
    'content_too_long': "Content exceeds maximum length limit. Please shorten your input.",
    'processing_error': "An error occurred while processing your request. Please try again.",
    'model_unavailable': "Analysis model is temporarily unavailable. Please try again later.",
    'rate_limit_exceeded': "Too many requests. Please wait before submitting another analysis.",
    'service_unavailable': "Service is temporarily unavailable. Please try again in a few minutes."
}

# Success Messages
SUCCESS_MESSAGES = {
    'analysis_complete': "Analysis completed successfully!",
    'model_loaded': "AI models loaded successfully.",
    'cache_cleared': "Cache cleared successfully.",
    'settings_saved': "Settings saved successfully.",
    'feedback_submitted': "Thank you for your feedback!"
}

# External URLs
EXTERNAL_URLS = {
    'documentation': 'https://truthguard-ai.org/docs',
    'support': 'https://truthguard-ai.org/support',
    'github': 'https://github.com/truthguard-ai/truthguard',
    'privacy_policy': 'https://truthguard-ai.org/privacy',
    'terms_of_service': 'https://truthguard-ai.org/terms'
}

def get_config_summary():
    """Return a summary of current configuration"""
    return {
        'app_name': APP_NAME,
        'version': APP_VERSION,
        'debug_mode': DEV_CONFIG['debug_mode'],
        'features_enabled': sum(FEATURE_FLAGS.values()),
        'total_features': len(FEATURE_FLAGS),
        'google_cloud_enabled': bool(GOOGLE_CLOUD_CONFIG['project_id']),
        'api_enabled': API_CONFIG['enable_api'],
        'caching_enabled': CACHE_CONFIG['enable_caching']
    }

def validate_config():
    """Validate configuration settings"""
    errors = []
    warnings = []
    
    # Check required directories
    for dir_path in [DATA_DIR, MODELS_DIR, ASSETS_DIR, LOGS_DIR]:
        if not dir_path.exists():
            warnings.append(f"Directory does not exist: {dir_path}")
    
    # Validate port number
    port = STREAMLIT_CONFIG['server.port']
    if not (1000 <= port <= 65535):
        errors.append(f"Invalid port number: {port}")
    
    # Check Google Cloud configuration
    if GOOGLE_CLOUD_CONFIG['project_id'] and not GOOGLE_CLOUD_CONFIG['credentials_path']:
        warnings.append("Google Cloud project ID set but credentials path is missing")
    
    return {
        'valid': len(errors) == 0,
        'errors': errors,
        'warnings': warnings
    }

if __name__ == "__main__":
    # Configuration validation and summary
    print(f"{APP_NAME} v{APP_VERSION} - Configuration Summary")
    print("=" * 50)
    
    config_summary = get_config_summary()
    for key, value in config_summary.items():
        print(f"{key.replace('_', ' ').title()}: {value}")
    
    print("\n" + "=" * 50)
    print("Configuration Validation:")
    
    validation = validate_config()
    if validation['valid']:
        print("✅ Configuration is valid")
    else:
        print("❌ Configuration has errors:")
        for error in validation['errors']:
            print(f"   - {error}")
    
    if validation['warnings']:
        print("⚠️  Warnings:")
        for warning in validation['warnings']:
            print(f"   - {warning}")