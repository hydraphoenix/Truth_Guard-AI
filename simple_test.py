#!/usr/bin/env python3
"""
Simple test script for TruthGuard AI
Tests basic functionality without Unicode characters for Windows compatibility
"""

import sys
import os
from pathlib import Path

def test_basic_setup():
    """Test basic setup"""
    print("TruthGuard AI - Basic Test")
    print("=" * 40)
    
    # Test Python version
    print("Testing Python version...")
    if sys.version_info >= (3, 8):
        print(f"OK: Python {sys.version.split()[0]} is compatible")
        python_ok = True
    else:
        print(f"ERROR: Python {sys.version.split()[0]} requires 3.8+")
        python_ok = False
    
    # Test core imports
    print("\nTesting core imports...")
    core_modules = ['os', 'sys', 'pathlib', 're', 'string']
    core_ok = True
    
    for module in core_modules:
        try:
            __import__(module)
            print(f"OK: {module}")
        except ImportError as e:
            print(f"ERROR: {module} - {e}")
            core_ok = False
    
    # Test optional imports
    print("\nTesting optional imports...")
    optional_modules = {
        'streamlit': 'Web framework',
        'pandas': 'Data processing', 
        'numpy': 'Numerical computing',
        'matplotlib': 'Plotting',
        'plotly': 'Interactive plots',
        'textblob': 'NLP processing',
        'sklearn': 'Machine learning'
    }
    
    available_optional = []
    missing_optional = []
    
    for module, description in optional_modules.items():
        try:
            __import__(module)
            print(f"OK: {module} ({description})")
            available_optional.append(module)
        except ImportError:
            print(f"MISSING: {module} ({description})")
            missing_optional.append(module)
    
    # Test file structure
    print("\nTesting file structure...")
    required_files = ['app.py', 'run.py', 'requirements.txt', 'README.md', 'config.py']
    files_ok = True
    
    for file in required_files:
        if Path(file).exists():
            print(f"OK: {file}")
        else:
            print(f"MISSING: {file}")
            files_ok = False
    
    # Test basic functionality
    print("\nTesting basic functionality...")
    try:
        # Test text processing
        test_text = "  HELLO WORLD TEST!  "
        cleaned = test_text.lower().strip()
        assert cleaned == "hello world test!", f"Text processing failed: {cleaned}"
        
        # Test basic feature extraction
        words = cleaned.split()
        features = {
            'word_count': len(words),
            'char_count': len(cleaned)
        }
        assert features['word_count'] == 3, f"Word count failed: {features}"
        
        print("OK: Basic text processing works")
        functionality_ok = True
        
    except Exception as e:
        print(f"ERROR: Basic functionality failed - {e}")
        functionality_ok = False
    
    # Summary
    print("\n" + "=" * 40)
    print("TEST SUMMARY")
    print("=" * 40)
    
    if python_ok:
        print("Python Version: COMPATIBLE")
    else:
        print("Python Version: INCOMPATIBLE")
    
    if core_ok:
        print("Core Modules: ALL AVAILABLE")
    else:
        print("Core Modules: SOME MISSING")
    
    print(f"Optional Modules: {len(available_optional)}/{len(optional_modules)} available")
    
    if files_ok:
        print("Application Files: COMPLETE")
    else:
        print("Application Files: SOME MISSING")
    
    if functionality_ok:
        print("Basic Functionality: WORKING")
    else:
        print("Basic Functionality: FAILED")
    
    # Overall status
    ready = python_ok and core_ok and files_ok and functionality_ok
    
    print("\n" + "=" * 40)
    if ready:
        if len(available_optional) >= 4:  # Need at least core modules
            print("STATUS: READY TO RUN!")
            print("Launch with: python run.py")
        else:
            print("STATUS: NEEDS PACKAGES")
            print("Install packages with: pip install -r requirements.txt")
            print("Then launch with: python run.py")
    else:
        print("STATUS: SETUP REQUIRED")
        if missing_optional:
            print("Missing packages:", ', '.join(missing_optional))
            print("Run: pip install -r requirements.txt")
    
    print("=" * 40)
    
    return ready

if __name__ == "__main__":
    success = test_basic_setup()
    print(f"\nTest completed. Ready: {success}")
    input("Press Enter to exit...")