#!/usr/bin/env python3
"""
Setup script for TruthGuard AI Advanced ML Model
Downloads required spaCy models and verifies installation
"""

import subprocess
import sys
import importlib.util

def check_package(package_name):
    """Check if a package is installed"""
    spec = importlib.util.find_spec(package_name)
    return spec is not None

def install_package(package_name):
    """Install a package using pip"""
    print(f"Installing {package_name}...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])

def download_spacy_model(model_name="en_core_web_sm"):
    """Download spaCy language model"""
    print(f"Downloading spaCy model: {model_name}")
    subprocess.check_call([sys.executable, "-m", "spacy", "download", model_name])

def main():
    print("🤖 Setting up TruthGuard AI Advanced ML Model...")
    print("=" * 50)
    
    # Required packages
    required_packages = {
        'spacy': 'spacy>=3.6.0',
        'swifter': 'swifter>=1.3.0',
        'sklearn': 'scikit-learn>=1.3.0',
        'pandas': 'pandas>=1.5.0',
        'numpy': 'numpy>=1.24.0'
    }
    
    # Check and install packages
    for package, pip_name in required_packages.items():
        if not check_package(package):
            print(f"❌ {package} not found. Installing...")
            try:
                install_package(pip_name)
                print(f"✅ {package} installed successfully!")
            except subprocess.CalledProcessError as e:
                print(f"❌ Failed to install {package}: {e}")
                return False
        else:
            print(f"✅ {package} already installed")
    
    # Download spaCy model
    print("\n📚 Setting up spaCy language model...")
    try:
        download_spacy_model()
        print("✅ spaCy model downloaded successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to download spaCy model: {e}")
        print("You can try manually: python -m spacy download en_core_web_sm")
        return False
    
    # Verify installation
    print("\n🔍 Verifying installation...")
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        print("✅ spaCy model verification successful!")
    except Exception as e:
        print(f"❌ spaCy model verification failed: {e}")
        return False
    
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        print("✅ scikit-learn verification successful!")
    except Exception as e:
        print(f"❌ scikit-learn verification failed: {e}")
        return False
    
    print("\n🎉 Advanced ML Model setup completed successfully!")
    print("You can now use the '🤖 Advanced ML Model' tab in TruthGuard AI.")
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        print("\n❌ Setup failed. Please check the error messages above.")
        sys.exit(1)
    else:
        print("\n✅ Setup completed. Run 'streamlit run app.py' to start TruthGuard AI.")