#!/bin/bash

# AlgoTrade Web Dashboard Launcher
# This script launches the web-based dashboard

echo "🚀 Launching AlgoTrade Web Dashboard..."
echo "======================================"

# Change to the dashboard directory
cd "$(dirname "$0")"

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo "❌ Python is not installed or not in PATH"
    exit 1
fi

# Check if required files exist
if [ ! -f "web_dashboard.py" ]; then
    echo "❌ web_dashboard.py not found"
    exit 1
fi

# Check if Flask is installed
if ! python -c "import flask" &> /dev/null; then
    echo "⚠️  Flask not found. Installing requirements..."
    pip install -r requirements_web.txt
fi

# Launch the web dashboard
echo "✅ Starting web dashboard..."
echo "📊 Open your browser and go to: http://localhost:5001"
echo "⏰ Dashboard will auto-refresh every 30 seconds"
echo "🛑 Press Ctrl+C to stop"
echo ""

python web_dashboard.py 