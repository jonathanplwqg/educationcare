#!/usr/bin/env python3
"""
Quick Start Script for EducationCare Streamlit App
Usage: python quick_start.py
"""

import subprocess
import sys
from pathlib import Path

def main():
    print("🎓 EducationCare - Quick Start")
    print("=" * 30)
    print()
    
    # Check Python version
    version = sys.version.split()[0]
    print(f"✅ Python found: {version}")
    print()
    
    # Install requirements
    print("📦 Installing dependencies...")
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
            check=True
        )
        print("✅ Dependencies installed successfully")
    except subprocess.CalledProcessError:
        print("❌ Failed to install dependencies")
        return 1
    
    print()
    print("=" * 30)
    print("🎉 Setup complete!")
    print("=" * 30)
    print()
    
    # Check for model files
    if Path("model.pkl").exists():
        print("✅ Model files found - Full mode available")
    else:
        print("⚠️  Model files not found - Will run in demo mode")
        print()
        print("To enable full predictions:")
        print("1. Open Final.ipynb")
        print("2. Run the cell from save_model_cell.py")
        print("3. Restart the app")
    
    print()
    print("🚀 Starting Streamlit app...")
    print()
    print("The app will open in your browser at http://localhost:8501")
    print("Press Ctrl+C to stop the app")
    print()
    
    # Start Streamlit
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])
    except KeyboardInterrupt:
        print("\n\n👋 App stopped")
        return 0

if __name__ == "__main__":
    sys.exit(main())
