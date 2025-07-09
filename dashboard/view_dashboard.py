#!/usr/bin/env python3
"""
Dashboard Viewer for AlgoTrade Agent Activities

Simple script to view the latest dashboard data and monitor agent activities.

@author: AlgoTrade Team
"""

import json
import os
from pathlib import Path
from datetime import datetime
import pytz

def find_latest_dashboard():
    """
    Find the latest dashboard file
    """
    log_dir = Path("logs")
    if not log_dir.exists():
        print("No logs directory found")
        return None
    
    dashboard_files = list(log_dir.glob("agent_dashboard_*.json"))
    if not dashboard_files:
        print("No dashboard files found")
        return None
    
    # Sort by modification time (newest first)
    dashboard_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return dashboard_files[0]

def load_dashboard(file_path):
    """
    Load dashboard data from file
    """
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading dashboard: {e}")
        return None

def print_detailed_dashboard(data):
    """
    Print detailed dashboard information
    """
    print("\n" + "="*100)
    print("🤖 ALGOTRADE AGENT DASHBOARD - DETAILED VIEW")
    print("="*100)
    
    # Session info
    print(f"Session ID: {data['session_id']}")
    print(f"Start Time: {data['start_time']}")
    print(f"Total Executions: {data['summary']['total_executions']}")
    print(f"Successful: {data['summary']['successful_executions']}")
    print(f"Failed: {data['summary']['failed_executions']}")
    print(f"Trades Executed: {data['summary']['trades_executed']}")
    print(f"Positions Managed: {data['summary']['positions_managed']}")
    print(f"Global Market Checks: {data['summary']['global_market_checks']}")
    
    # Current execution
    if data['current_execution']:
        current = data['current_execution']
        print(f"\n🔄 CURRENT EXECUTION:")
        print(f"  ID: {current['execution_id']}")
        print(f"  Type: {current['execution_type']}")
        print(f"  Status: {current['status']}")
        print(f"  Start Time: {current['start_time']}")
        if current['end_time']:
            print(f"  End Time: {current['end_time']}")
            print(f"  Duration: {current['duration_seconds']}s")
        
        # Agent activities
        if current['agents_activity']:
            print(f"\n📝 AGENT ACTIVITIES:")
            for activity in current['agents_activity'][-10:]:  # Show last 10
                print(f"  [{activity['timestamp']}] {activity['agent']}: {activity['activity']}")
        
        # Decisions
        if current['decisions_made']:
            print(f"\n🎯 DECISIONS MADE:")
            for decision in current['decisions_made'][-5:]:  # Show last 5
                confidence_str = f" (Confidence: {decision['confidence']:.1%})" if decision['confidence'] else ""
                print(f"  [{decision['timestamp']}] {decision['agent']}: {decision['decision']}{confidence_str}")
                if decision['reasoning']:
                    print(f"    Reasoning: {decision['reasoning']}")
        
        # Actions
        if current['actions_taken']:
            print(f"\n⚡ ACTIONS TAKEN:")
            for action in current['actions_taken'][-5:]:  # Show last 5
                print(f"  [{action['timestamp']}] {action['action_type']}: {action['status']}")
                if action['action_details']:
                    for key, value in action['action_details'].items():
                        print(f"    {key}: {value}")
        
        # Errors
        if current['errors']:
            print(f"\n❌ ERRORS:")
            for error in current['errors'][-3:]:  # Show last 3
                print(f"  [{error['timestamp']}] {error['error_type']}: {error['error_message']}")
        
        # Warnings
        if current['warnings']:
            print(f"\n⚠️ WARNINGS:")
            for warning in current['warnings'][-3:]:  # Show last 3
                print(f"  [{warning['timestamp']}] {warning['warning_message']}")
    
    # Recent executions
    if data['executions']:
        print(f"\n📋 RECENT EXECUTIONS:")
        for exec_data in data['executions'][-3:]:  # Show last 3
            print(f"  {exec_data['execution_id']}: {exec_data['status']} ({exec_data['duration_seconds']}s)")
            if exec_data['summary']:
                print(f"    Summary: {exec_data['summary']}")
    
    print("="*100)

def print_simple_summary(data):
    """
    Print a simple summary of the dashboard
    """
    print("\n" + "="*60)
    print("🤖 ALGOTRADE AGENT DASHBOARD - SUMMARY")
    print("="*60)
    print(f"Session: {data['session_id']}")
    print(f"Start: {data['start_time']}")
    print(f"Executions: {data['summary']['total_executions']} (✓{data['summary']['successful_executions']} ✗{data['summary']['failed_executions']})")
    print(f"Trades: {data['summary']['trades_executed']} | Positions: {data['summary']['positions_managed']} | Global Checks: {data['summary']['global_market_checks']}")
    
    if data['current_execution']:
        current = data['current_execution']
        print(f"Current: {current['status']} | Activities: {len(current['agents_activity'])} | Decisions: {len(current['decisions_made'])} | Actions: {len(current['actions_taken'])}")
    
    print("="*60)

def main():
    """
    Main function to view dashboard
    """
    import argparse
    parser = argparse.ArgumentParser(description='View AlgoTrade Agent Dashboard')
    parser.add_argument('--detailed', action='store_true', help='Show detailed dashboard view')
    parser.add_argument('--file', type=str, help='Specific dashboard file to view')
    args = parser.parse_args()
    
    # Find dashboard file
    if args.file:
        dashboard_file = Path(args.file)
        if not dashboard_file.exists():
            print(f"Dashboard file not found: {args.file}")
            return
    else:
        dashboard_file = find_latest_dashboard()
        if not dashboard_file:
            return
    
    print(f"Loading dashboard: {dashboard_file}")
    
    # Load and display dashboard
    data = load_dashboard(dashboard_file)
    if not data:
        return
    
    if args.detailed:
        print_detailed_dashboard(data)
    else:
        print_simple_summary(data)

if __name__ == "__main__":
    main() 