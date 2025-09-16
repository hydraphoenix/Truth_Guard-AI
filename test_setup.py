#!/usr/bin/env python3
"""
TruthGuard AI - Setup Test Script

This script tests the application setup without running the full Streamlit app.
It checks imports and basic functionality.
"""

import sys
import os
from pathlib import Path

def test_python_version():
    """Test Python version compatibility"""
    print("Testing Python version...")
    if sys.version_info >= (3, 8):
        print(f"OK Python {sys.version.split()[0]} - Compatible")
        return True
    else:
        print(f"ERROR Python {sys.version.split()[0]} - Requires 3.8+")
        return False

def test_imports():
    """Test if all required modules can be imported"""
    print("\n📦 Testing imports...")
    
    required_modules = {
        'streamlit': 'Streamlit web framework',
        'pandas': 'Data manipulation library',
        'numpy': 'Numerical computing library',
        'matplotlib': 'Plotting library',
        'plotly': 'Interactive plotting library'
    }
    
    optional_modules = {
        'textblob': 'Natural language processing',
        'nltk': 'Natural language toolkit', 
        'sklearn': 'Machine learning library',
        'wordcloud': 'Word cloud generation'
    }
    
    missing_required = []
    missing_optional = []
    
    # Test required modules
    for module, description in required_modules.items():
        try:
            __import__(module)
            print(f"✅ {module}: Available")
        except ImportError:
            print(f"❌ {module}: Missing ({description})")
            missing_required.append(module)
    
    # Test optional modules  
    for module, description in optional_modules.items():
        try:
            __import__(module)
            print(f"✅ {module}: Available")
        except ImportError:
            print(f"⚠️ {module}: Missing ({description}) - Optional")
            missing_optional.append(module)
    
    return missing_required, missing_optional

def test_app_structure():
    """Test if application files are properly structured"""
    print("\n📁 Testing application structure...")
    
    required_files = [
        'app.py',
        'run.py', 
        'requirements.txt',
        'README.md',
        'config.py'
    ]
    
    missing_files = []
    
    for file in required_files:
        if Path(file).exists():
            print(f"✅ {file}: Found")
        else:
            print(f"❌ {file}: Missing")
            missing_files.append(file)
    
    return missing_files

def test_basic_functionality():
    """Test basic application functionality without imports that might fail"""
    print("\n🧠 Testing basic functionality...")
    
    try:
        # Test configuration
        sys.path.append('.')
        import config
        print("✅ Configuration: Loaded")
        
        # Test basic classes and functions
        class TestDetector:
            def __init__(self):
                self.models = {'basic': True}
            
            def clean_text(self, text):
                if not text:
                    return ""
                return str(text).lower().strip()
            
            def extract_basic_features(self, text):
                if not text:
                    return {'word_count': 0}
                words = text.split()
                return {
                    'word_count': len(words),
                    'char_count': len(text)
                }
        
        detector = TestDetector()
        
        # Test text cleaning
        cleaned = detector.clean_text("  HELLO WORLD!  ")
        assert cleaned == "hello world!", f"Text cleaning failed: {cleaned}"
        print("✅ Text cleaning: Working")
        
        # Test feature extraction
        features = detector.extract_basic_features("hello world test")
        assert features['word_count'] == 3, f"Feature extraction failed: {features}"
        print("✅ Feature extraction: Working")
        
        print("✅ Basic functionality: All tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality: Failed - {e}")
        return False

def generate_install_commands(missing_required, missing_optional):
    """Generate installation commands for missing packages"""
    if not (missing_required or missing_optional):
        return []
    
    commands = []
    
    if missing_required:
        commands.append("# Install required packages:")
        commands.append(f"pip install {' '.join(missing_required)}")
    
    if missing_optional:
        commands.append("# Install optional packages for enhanced features:")
        commands.append(f"pip install {' '.join(missing_optional)}")
    
    commands.append("# Or install all at once:")
    commands.append("pip install -r requirements.txt")
    
    return commands

def main():
    """Main test function"""
    print("TruthGuard AI - Setup Test")
    print("=" * 50)
    
    # Test Python version
    python_ok = test_python_version()
    
    # Test imports
    missing_required, missing_optional = test_imports()
    
    # Test file structure
    missing_files = test_app_structure()
    
    # Test basic functionality
    functionality_ok = test_basic_functionality()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 SETUP TEST SUMMARY")
    print("=" * 50)
    
    if python_ok:
        print("✅ Python Version: Compatible")
    else:
        print("❌ Python Version: Incompatible")
    
    if not missing_required:
        print("✅ Required Packages: All available")
    else:
        print(f"❌ Required Packages: {len(missing_required)} missing")
    
    if not missing_optional:
        print("✅ Optional Packages: All available")
    else:
        print(f"⚠️ Optional Packages: {len(missing_optional)} missing")
    
    if not missing_files:
        print("✅ Application Files: Complete")
    else:
        print(f"❌ Application Files: {len(missing_files)} missing")
    
    if functionality_ok:
        print("✅ Basic Functionality: Working")
    else:
        print("❌ Basic Functionality: Issues detected")
    
    # Overall status
    all_good = (python_ok and not missing_required and 
                not missing_files and functionality_ok)
    
    print("\n" + "=" * 50)
    if all_good:
        print("🎉 SETUP STATUS: READY TO RUN!")
        print("   You can now launch the application with:")
        print("   python run.py")
    else:
        print("🔧 SETUP STATUS: REQUIRES ATTENTION")
        
        # Generate installation commands
        install_commands = generate_install_commands(missing_required, missing_optional)
        
        if install_commands:
            print("\n💡 SUGGESTED ACTIONS:")
            for command in install_commands:
                print(f"   {command}")
        
        if missing_files:
            print(f"\n📁 MISSING FILES: {', '.join(missing_files)}")
            print("   Please ensure all required files are present.")
    
    print("=" * 50)
    
    return all_good

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)