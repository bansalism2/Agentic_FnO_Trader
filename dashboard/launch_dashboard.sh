#!/bin/bash

# AlgoTrade Enhanced Dashboard Launcher
# This script launches the enhanced dashboard with auto-refresh

echo "🚀 Launching AlgoTrade Enhanced Dashboard..."
echo "=========================================="

# Change to the dashboard directory
cd "$(dirname "$0")"

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo "❌ Python is not installed or not in PATH"
    exit 1
fi

# Check if required files exist
if [ ! -f "enhanced_dashboard.py" ]; then
    echo "❌ enhanced_dashboard.py not found"
    exit 1
fi

# Launch the dashboard
echo "✅ Starting dashboard with auto-refresh..."
echo "Press Ctrl+C to stop"
echo ""

python enhanced_dashboard.py 