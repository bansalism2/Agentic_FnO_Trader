#!/bin/bash

# AlgoTrade Agent Dashboard UI Launcher
# This script starts the web-based dashboard UI

echo "🤖 AlgoTrade Agent Dashboard UI Launcher"
echo "========================================"

# Check if conda environment exists
if ! conda env list | grep -q "kite_auto"; then
    echo "❌ Error: Conda environment 'kite_auto' not found!"
    echo "Please create the environment first:"
    echo "conda create -n kite_auto python=3.12"
    echo "conda activate kite_auto"
    echo "pip install -r requirements_ui.txt"
    exit 1
fi

# Activate conda environment
echo "🔧 Activating conda environment 'kite_auto'..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate kite_auto

# Check if Flask is installed
if ! python -c "import flask" 2>/dev/null; then
    echo "📦 Installing UI dependencies..."
    pip install -r requirements_ui.txt
fi

# Check if logs directory exists
if [ ! -d "logs" ]; then
    echo "📁 Creating logs directory..."
    mkdir -p logs
fi

echo "🚀 Starting AlgoTrade Agent Dashboard UI..."
echo "📊 Dashboard will be available at: http://localhost:8080"
echo "🔄 Auto-refreshing every 2 seconds"
echo "Press Ctrl+C to stop"
echo ""

# Start the dashboard UI
python dashboard_ui.py 