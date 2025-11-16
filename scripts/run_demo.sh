#!/bin/bash
# run_demo.sh - Easy launcher for system demonstration

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║     Multimodal Authentication System - Demo Launcher          ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python 3 is not installed"
    echo "Please install Python 3.8 or higher"
    exit 1
fi

echo "✓ Python found: $(python3 --version)"
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
    echo "✓ Virtual environment created"
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📥 Installing dependencies..."
pip install -q -r requirements_demo.txt

echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                     Select Demo Mode                          ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "1. Simple Demo (Recommended - No models required)"
echo "2. Full System Demo (Requires trained models)"
echo "3. Exit"
echo ""

read -p "Enter your choice (1-3): " choice

case $choice in
    1)
        echo ""
        echo "🚀 Launching Simple Demo..."
        python3 simple_demo.py
        ;;
    2)
        echo ""
        echo "🚀 Launching Full System Demo..."
        python3 system_demo.py
        ;;
    3)
        echo "👋 Goodbye!"
        exit 0
        ;;
    *)
        echo "❌ Invalid choice"
        exit 1
        ;;
esac

# Deactivate virtual environment
deactivate

echo ""
echo "✅ Demo completed"
