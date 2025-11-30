#!/bin/bash

echo "=========================================="
echo "Breast Guideline RAG - Setup Script"
echo "=========================================="

# Create virtual environment
echo ""
echo "1. Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo ""
echo "2. Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo ""
echo "3. Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo ""
echo "4. Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Edit .streamlit/secrets.toml and add your OpenAI API key"
echo "2. Run: python build_vectorstore.py"
echo "3. Run: streamlit run Chat_UI.py"
echo ""
echo "=========================================="
