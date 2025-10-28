#!/bin/bash

# Quick Start Script for EducationCare Streamlit App
# Usage: bash quick_start.sh

echo "🎓 EducationCare - Quick Start"
echo "=============================="
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

echo "✅ Python found: $(python3 --version)"
echo ""

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip is not installed. Please install pip."
    exit 1
fi

echo "✅ pip found"
echo ""

# Install requirements
echo "📦 Installing dependencies..."
pip3 install -r requirements.txt

if [ $? -eq 0 ]; then
    echo "✅ Dependencies installed successfully"
else
    echo "❌ Failed to install dependencies"
    exit 1
fi

echo ""
echo "=============================="
echo "🎉 Setup complete!"
echo "=============================="
echo ""

# Check for model files
if [ -f "model.pkl" ]; then
    echo "✅ Model files found - Full mode available"
else
    echo "⚠️  Model files not found - Will run in demo mode"
    echo ""
    echo "To enable full predictions:"
    echo "1. Open Final.ipynb"
    echo "2. Run the cell from save_model_cell.py"
    echo "3. Restart the app"
fi

echo ""
echo "🚀 Starting Streamlit app..."
echo ""
echo "The app will open in your browser at http://localhost:8501"
echo "Press Ctrl+C to stop the app"
echo ""

# Start Streamlit
streamlit run app.py
