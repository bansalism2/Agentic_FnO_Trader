#!/usr/bin/env python3
"""
AlgoTrade Agent Dashboard Web UI

Flask web application to provide a real-time dashboard interface for monitoring
agent activities, decisions, and trading actions.

@author: AlgoTrade Team
"""

from flask import Flask, render_template, jsonify, request
import json
import os
from pathlib import Path
from datetime import datetime
import pytz
import threading
import time

app = Flask(__name__)

# Global variables
DASHBOARD_DATA = {}
DASHBOARD_FILE = None
LAST_UPDATE = None

def find_latest_dashboard():
    """Find the latest dashboard file"""
    log_dir = Path("logs")
    if not log_dir.exists():
        return None
    
    dashboard_files = list(log_dir.glob("agent_dashboard_*.json"))
    if not dashboard_files:
        return None
    
    # Sort by modification time (newest first)
    dashboard_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return dashboard_files[0]

def load_dashboard_data():
    """Load dashboard data from file"""
    global DASHBOARD_DATA, DASHBOARD_FILE, LAST_UPDATE
    
    dashboard_file = find_latest_dashboard()
    if not dashboard_file:
        return
    
    try:
        # Check if file has been modified
        file_mtime = dashboard_file.stat().st_mtime
        if DASHBOARD_FILE == str(dashboard_file) and LAST_UPDATE == file_mtime:
            return  # No changes
        
        with open(dashboard_file, 'r') as f:
            DASHBOARD_DATA = json.load(f)
            DASHBOARD_FILE = str(dashboard_file)
            LAST_UPDATE = file_mtime
            
    except Exception as e:
        print(f"Error loading dashboard: {e}")

def background_updater():
    """Background thread to update dashboard data"""
    while True:
        load_dashboard_data()
        time.sleep(2)  # Update every 2 seconds

@app.route('/')
def index():
    """Main dashboard page"""
    load_dashboard_data()
    return render_template('dashboard.html')

@app.route('/api/dashboard')
def api_dashboard():
    """API endpoint to get dashboard data"""
    load_dashboard_data()
    return jsonify(DASHBOARD_DATA)

@app.route('/api/current_execution')
def api_current_execution():
    """API endpoint to get current execution data"""
    load_dashboard_data()
    return jsonify(DASHBOARD_DATA.get('current_execution', {}))

@app.route('/api/summary')
def api_summary():
    """API endpoint to get summary data"""
    load_dashboard_data()
    return jsonify(DASHBOARD_DATA.get('summary', {}))

@app.route('/api/executions')
def api_executions():
    """API endpoint to get recent executions"""
    load_dashboard_data()
    return jsonify(DASHBOARD_DATA.get('executions', []))

if __name__ == '__main__':
    # Start background updater thread
    updater_thread = threading.Thread(target=background_updater, daemon=True)
    updater_thread.start()
    
    print("🤖 AlgoTrade Agent Dashboard UI Starting...")
    print("📊 Dashboard will be available at: http://localhost:8080")
    print("🔄 Auto-refreshing every 2 seconds")
    print("Press Ctrl+C to stop")
    
    app.run(debug=True, host='0.0.0.0', port=8080) 