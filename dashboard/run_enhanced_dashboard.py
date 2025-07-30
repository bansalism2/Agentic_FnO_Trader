#!/usr/bin/env python3
"""
Run Enhanced Dashboard with Auto-Refresh

Simple script to run the enhanced dashboard with automatic refresh every 30 seconds.
"""

import time
import os
import sys
from pathlib import Path

# Add the dashboard directory to the path
sys.path.insert(0, os.path.dirname(__file__))

from enhanced_dashboard import EnhancedDashboard

def clear_screen():
    """Clear the terminal screen"""
    os.system('cls' if os.name == 'nt' else 'clear')

def main():
    """
    Main function to run the enhanced dashboard with auto-refresh
    """
    print("🚀 Starting AlgoTrade Enhanced Dashboard...")
    print("Press Ctrl+C to stop")
    
    dashboard = EnhancedDashboard()
    
    try:
        while True:
            clear_screen()
            dashboard.update_dashboard()
            dashboard.print_dashboard()
            
            print(f"\n⏰ Next refresh in 30 seconds... (Press Ctrl+C to stop)")
            time.sleep(30)
            
    except KeyboardInterrupt:
        print("\n\n🛑 Dashboard stopped by user")
        print("📊 Final dashboard saved...")
        dashboard.save_dashboard("final_dashboard.json")
        print("✅ Goodbye!")

if __name__ == "__main__":
    main() 