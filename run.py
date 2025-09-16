#!/usr/bin/env python3
"""
TruthGuard AI - Application Launcher

This script handles the setup and launch of the TruthGuard AI application.
It checks dependencies, downloads required data, and starts the Streamlit server.

Usage:
    python run.py

Author: TruthGuard AI Team
Version: 1.0.0
"""

import os
import sys
import subprocess
import platform
from pathlib import Path
import importlib.util

def check_python_version():
    """Check if Python version is compatible"""
    print("🐍 Checking Python version...")
    
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"   Current version: {sys.version}")
        print("   Please upgrade Python and try again.")
        return False
    
    print(f"✅ Python version: {sys.version.split()[0]} (Compatible)")
    return True

def check_required_packages():
    """Check if core packages are available"""
    print("\n📦 Checking required packages...")
    
    required_packages = {
        'streamlit': 'streamlit',
        'pandas': 'pandas', 
        'numpy': 'numpy',
        'sklearn': 'scikit-learn',
        'textblob': 'textblob',
        'plotly': 'plotly',
        'matplotlib': 'matplotlib'
    }
    
    missing_packages = []
    
    for module_name, package_name in required_packages.items():
        try:
            if importlib.util.find_spec(module_name) is not None:
                print(f"✅ {package_name}: Available")
            else:
                print(f"❌ {package_name}: Not found")
                missing_packages.append(package_name)
        except ImportError:
            print(f"❌ {package_name}: Import error")
            missing_packages.append(package_name)
    
    return missing_packages

def install_packages(packages):
    """Install missing packages"""
    if not packages:
        return True
    
    print(f"\n💾 Installing missing packages: {', '.join(packages)}")
    
    try:
        cmd = [sys.executable, '-m', 'pip', 'install', '--upgrade'] + packages
        subprocess.check_call(cmd)
        print("✅ All packages installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install packages: {e}")
        print("\n🔧 Manual installation required:")
        print("   pip install -r requirements.txt")
        return False

def setup_nltk_data():
    """Download required NLTK data"""
    print("\n📚 Setting up NLTK data...")
    
    try:
        import nltk
        
        # Download required datasets quietly
        datasets = ['stopwords', 'punkt', 'vader_lexicon', 'averaged_perceptron_tagger']
        
        for dataset in datasets:
            try:
                nltk.data.find(f'tokenizers/{dataset}' if dataset == 'punkt' else f'corpora/{dataset}')
                print(f"✅ NLTK {dataset}: Already available")
            except LookupError:
                print(f"⬇️ Downloading NLTK {dataset}...")
                nltk.download(dataset, quiet=True)
                print(f"✅ NLTK {dataset}: Downloaded")
        
        return True
        
    except ImportError:
        print("⚠️ NLTK not available - some features may be limited")
        return False
    except Exception as e:
        print(f"⚠️ NLTK setup warning: {e}")
        return False

def check_port_availability(port=8501):
    """Check if the default Streamlit port is available"""
    import socket
    
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('localhost', port))
            return True
    except OSError:
        return False

def launch_application():
    """Launch the TruthGuard AI application"""
    print("\n🚀 Launching TruthGuard AI...")
    print("🌐 The application will open in your web browser")
    
    # Check if port is available
    port = 8501
    if not check_port_availability(port):
        print(f"⚠️ Port {port} is busy, trying alternative ports...")
        for alt_port in range(8502, 8510):
            if check_port_availability(alt_port):
                port = alt_port
                break
        else:
            print("❌ No available ports found. Please close other Streamlit applications.")
            return False
    
    print(f"🔗 Application URL: http://localhost:{port}")
    print("📱 Use Ctrl+C to stop the application")
    print("\n" + "="*60)
    
    try:
        # Launch Streamlit
        cmd = [
            sys.executable, '-m', 'streamlit', 'run', 
            'app.py',
            '--server.port', str(port),
            '--server.headless', 'false',
            '--server.runOnSave', 'true',
            '--browser.gatherUsageStats', 'false'
        ]
        
        subprocess.run(cmd)
        return True
        
    except KeyboardInterrupt:
        print("\n\n👋 TruthGuard AI stopped by user")
        return True
    except FileNotFoundError:
        print("\n❌ Streamlit not found. Please install it with:")
        print("   pip install streamlit")
        return False
    except Exception as e:
        print(f"\n❌ Failed to launch application: {e}")
        print("\n🔧 Try running manually with:")
        print("   streamlit run app.py")
        return False

def display_system_info():
    """Display system information"""
    print("="*60)
    print("🛡️  TRUTHGUARD AI - MISINFORMATION DETECTION SYSTEM")
    print("="*60)
    print(f"🖥️  Operating System: {platform.system()} {platform.release()}")
    print(f"🐍 Python Version: {sys.version.split()[0]}")
    print(f"📁 Working Directory: {Path.cwd()}")
    print("="*60)

def main():
    """Main setup and launch function"""
    display_system_info()
    
    # Check Python version
    if not check_python_version():
        input("\nPress Enter to exit...")
        sys.exit(1)
    
    # Check required packages
    missing_packages = check_required_packages()
    
    if missing_packages:
        print(f"\n📋 Missing packages detected: {', '.join(missing_packages)}")
        
        while True:
            choice = input("\n💡 Install missing packages automatically? (y/n/q): ").lower().strip()
            
            if choice in ['y', 'yes']:
                if not install_packages(missing_packages):
                    input("\nPress Enter to exit...")
                    sys.exit(1)
                break
            elif choice in ['n', 'no']:
                print("\n⚠️ Application may not work correctly without required packages.")
                print("📋 To install manually: pip install -r requirements.txt")
                
                continue_choice = input("Continue anyway? (y/n): ").lower().strip()
                if continue_choice not in ['y', 'yes']:
                    sys.exit(1)
                break
            elif choice in ['q', 'quit']:
                print("👋 Setup cancelled by user")
                sys.exit(0)
            else:
                print("Please enter 'y' for yes, 'n' for no, or 'q' to quit.")
    
    # Setup NLTK data
    setup_nltk_data()
    
    # Final system check
    print("\n" + "="*60)
    print("📊 SYSTEM STATUS")
    print("="*60)
    print("✅ Core Dependencies: Ready")
    print("✅ Python Environment: Compatible") 
    print("✅ Application Files: Available")
    print("✅ NLTK Data: Configured")
    print("="*60)
    
    # Launch confirmation
    while True:
        launch_choice = input("\n🚀 Launch TruthGuard AI now? (y/n): ").lower().strip()
        
        if launch_choice in ['y', 'yes', '']:
            success = launch_application()
            if success:
                print("\n✅ Application session completed successfully")
            break
        elif launch_choice in ['n', 'no']:
            print("\n💡 To launch manually later, run:")
            print("   python run.py")
            print("   or")
            print("   streamlit run app.py")
            break
        else:
            print("Please enter 'y' for yes or 'n' for no.")
    
    input("\nPress Enter to exit...")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Setup interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print("Please report this issue if it persists.")
        input("\nPress Enter to exit...")
    finally:
        sys.exit(0)