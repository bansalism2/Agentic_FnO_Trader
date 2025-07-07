#!/usr/bin/env python3
"""
Agent Dashboard for AlgoTrade Crew Execution

Real-time dashboard to monitor agent activities, decisions, and trading actions
during crew execution.

@author: AlgoTrade Team
"""

import json
import logging
from datetime import datetime
import pytz
from typing import Dict, List, Optional
from pathlib import Path
import os

# Configure logging
logger = logging.getLogger(__name__)

class AgentDashboard:
    def __init__(self, log_dir: str = "logs"):
        """
        Initialize the agent dashboard
        
        Args:
            log_dir: Directory to store dashboard logs
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Dashboard data structure
        self.session_data = {
            'session_id': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'start_time': datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%Y-%m-%d %H:%M:%S %Z'),
            'executions': [],
            'current_execution': None,
            'summary': {
                'total_executions': 0,
                'successful_executions': 0,
                'failed_executions': 0,
                'trades_executed': 0,
                'positions_managed': 0,
                'global_market_checks': 0,
                'last_execution_time': None
            }
        }
        
        # Create dashboard file
        self.dashboard_file = self.log_dir / f"agent_dashboard_{self.session_data['session_id']}.json"
        self.save_dashboard()
        
        logger.info(f"Agent Dashboard initialized: {self.dashboard_file}")
    
    def start_execution(self, execution_type: str = "crew_agent"):
        """
        Start a new execution session
        """
        execution_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        self.session_data['current_execution'] = {
            'execution_id': execution_id,
            'execution_type': execution_type,
            'start_time': datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%Y-%m-%d %H:%M:%S %Z'),
            'status': 'RUNNING',
            'agents_activity': [],
            'decisions_made': [],
            'actions_taken': [],
            'errors': [],
            'warnings': [],
            'end_time': None,
            'duration_seconds': None
        }
        
        self.session_data['summary']['total_executions'] += 1
        self.session_data['summary']['last_execution_time'] = datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%Y-%m-%d %H:%M:%S %Z')
        
        self.save_dashboard()
        logger.info(f"Started execution: {execution_id}")
    
    def log_agent_activity(self, agent_name: str, activity: str, details: Dict = None):
        """
        Log agent activity during execution
        """
        if not self.session_data['current_execution']:
            logger.warning("No active execution to log activity")
            return
        
        activity_entry = {
            'timestamp': datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%Y-%m-%d %H:%M:%S %Z'),
            'agent': agent_name,
            'activity': activity,
            'details': details or {}
        }
        
        self.session_data['current_execution']['agents_activity'].append(activity_entry)
        self.save_dashboard()
        
        logger.info(f"Agent Activity - {agent_name}: {activity}")
    
    def log_decision(self, agent_name: str, decision: str, reasoning: str = None, confidence: float = None):
        """
        Log a decision made by an agent
        """
        if not self.session_data['current_execution']:
            logger.warning("No active execution to log decision")
            return
        
        decision_entry = {
            'timestamp': datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%Y-%m-%d %H:%M:%S %Z'),
            'agent': agent_name,
            'decision': decision,
            'reasoning': reasoning,
            'confidence': confidence
        }
        
        self.session_data['current_execution']['decisions_made'].append(decision_entry)
        self.save_dashboard()
        
        logger.info(f"Decision - {agent_name}: {decision}")
    
    def log_action(self, action_type: str, action_details: Dict, status: str = "PENDING"):
        """
        Log an action taken by the system
        """
        if not self.session_data['current_execution']:
            logger.warning("No active execution to log action")
            return
        
        action_entry = {
            'timestamp': datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%Y-%m-%d %H:%M:%S %Z'),
            'action_type': action_type,
            'action_details': action_details,
            'status': status
        }
        
        self.session_data['current_execution']['actions_taken'].append(action_entry)
        
        # Update summary based on action type
        if action_type == "TRADE_EXECUTION":
            self.session_data['summary']['trades_executed'] += 1
        elif action_type == "POSITION_MANAGEMENT":
            self.session_data['summary']['positions_managed'] += 1
        elif action_type == "GLOBAL_MARKET_CHECK":
            self.session_data['summary']['global_market_checks'] += 1
        
        self.save_dashboard()
        logger.info(f"Action - {action_type}: {status}")
    
    def log_error(self, error_message: str, error_type: str = "ERROR", details: Dict = None):
        """
        Log an error during execution
        """
        if not self.session_data['current_execution']:
            logger.warning("No active execution to log error")
            return
        
        error_entry = {
            'timestamp': datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%Y-%m-%d %H:%M:%S %Z'),
            'error_type': error_type,
            'error_message': error_message,
            'details': details or {}
        }
        
        self.session_data['current_execution']['errors'].append(error_entry)
        self.save_dashboard()
        
        logger.error(f"Dashboard Error - {error_type}: {error_message}")
    
    def log_warning(self, warning_message: str, details: Dict = None):
        """
        Log a warning during execution
        """
        if not self.session_data['current_execution']:
            logger.warning("No active execution to log warning")
            return
        
        warning_entry = {
            'timestamp': datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%Y-%m-%d %H:%M:%S %Z'),
            'warning_message': warning_message,
            'details': details or {}
        }
        
        self.session_data['current_execution']['warnings'].append(warning_entry)
        self.save_dashboard()
        
        logger.warning(f"Dashboard Warning: {warning_message}")
    
    def end_execution(self, status: str = "COMPLETED", summary: str = None):
        """
        End the current execution session
        """
        if not self.session_data['current_execution']:
            logger.warning("No active execution to end")
            return
        
        end_time = datetime.now(pytz.timezone('Asia/Kolkata'))
        start_time = datetime.strptime(
            self.session_data['current_execution']['start_time'], 
            '%Y-%m-%d %H:%M:%S %Z'
        ).replace(tzinfo=pytz.timezone('Asia/Kolkata'))
        
        duration = (end_time - start_time).total_seconds()
        
        self.session_data['current_execution'].update({
            'end_time': end_time.strftime('%Y-%m-%d %H:%M:%S %Z'),
            'status': status,
            'duration_seconds': round(duration, 2),
            'summary': summary
        })
        
        # Move current execution to executions list
        self.session_data['executions'].append(self.session_data['current_execution'])
        self.session_data['current_execution'] = None
        
        # Update summary
        if status == "COMPLETED":
            self.session_data['summary']['successful_executions'] += 1
        else:
            self.session_data['summary']['failed_executions'] += 1
        
        self.save_dashboard()
        logger.info(f"Ended execution with status: {status} (Duration: {duration:.2f}s)")
    
    def save_dashboard(self):
        """
        Save dashboard data to JSON file
        """
        try:
            with open(self.dashboard_file, 'w') as f:
                json.dump(self.session_data, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save dashboard: {e}")
    
    def get_dashboard_summary(self) -> Dict:
        """
        Get a summary of the dashboard data
        """
        return {
            'session_id': self.session_data['session_id'],
            'start_time': self.session_data['start_time'],
            'summary': self.session_data['summary'],
            'current_execution': self.session_data['current_execution'],
            'recent_executions': self.session_data['executions'][-5:] if self.session_data['executions'] else []
        }
    
    def print_dashboard_summary(self):
        """
        Print a formatted summary of the dashboard
        """
        summary = self.get_dashboard_summary()
        
        print("\n" + "="*80)
        print("🤖 ALGOTRADE AGENT DASHBOARD")
        print("="*80)
        print(f"Session ID: {summary['session_id']}")
        print(f"Start Time: {summary['start_time']}")
        print(f"Dashboard File: {self.dashboard_file}")
        
        print(f"\n📊 EXECUTION SUMMARY:")
        print(f"  Total Executions: {summary['summary']['total_executions']}")
        print(f"  Successful: {summary['summary']['successful_executions']}")
        print(f"  Failed: {summary['summary']['failed_executions']}")
        print(f"  Trades Executed: {summary['summary']['trades_executed']}")
        print(f"  Positions Managed: {summary['summary']['positions_managed']}")
        print(f"  Global Market Checks: {summary['summary']['global_market_checks']}")
        print(f"  Last Execution: {summary['summary']['last_execution_time']}")
        
        if summary['current_execution']:
            current = summary['current_execution']
            print(f"\n🔄 CURRENT EXECUTION:")
            print(f"  ID: {current['execution_id']}")
            print(f"  Type: {current['execution_type']}")
            print(f"  Status: {current['status']}")
            print(f"  Start Time: {current['start_time']}")
            print(f"  Agent Activities: {len(current['agents_activity'])}")
            print(f"  Decisions Made: {len(current['decisions_made'])}")
            print(f"  Actions Taken: {len(current['actions_taken'])}")
            print(f"  Errors: {len(current['errors'])}")
            print(f"  Warnings: {len(current['warnings'])}")
        
        if summary['recent_executions']:
            print(f"\n📋 RECENT EXECUTIONS:")
            for exec_data in summary['recent_executions'][-3:]:
                print(f"  {exec_data['execution_id']}: {exec_data['status']} ({exec_data['duration_seconds']}s)")
        
        print("="*80)

# Global dashboard instance
dashboard = None

def get_dashboard() -> AgentDashboard:
    """
    Get the global dashboard instance
    """
    global dashboard
    if dashboard is None:
        dashboard = AgentDashboard()
    return dashboard

def log_agent_activity(agent_name: str, activity: str, details: Dict = None):
    """
    Convenience function to log agent activity
    """
    dash = get_dashboard()
    dash.log_agent_activity(agent_name, activity, details)

def log_decision(agent_name: str, decision: str, reasoning: str = None, confidence: float = None):
    """
    Convenience function to log agent decision
    """
    dash = get_dashboard()
    dash.log_decision(agent_name, decision, reasoning, confidence)

def log_action(action_type: str, action_details: Dict, status: str = "PENDING"):
    """
    Convenience function to log action
    """
    dash = get_dashboard()
    dash.log_action(action_type, action_details, status)

def log_error(error_message: str, error_type: str = "ERROR", details: Dict = None):
    """
    Convenience function to log error
    """
    dash = get_dashboard()
    dash.log_error(error_message, error_type, details)

def log_warning(warning_message: str, details: Dict = None):
    """
    Convenience function to log warning
    """
    dash = get_dashboard()
    dash.log_warning(warning_message, details)

if __name__ == "__main__":
    # Test the dashboard
    dashboard = AgentDashboard()
    dashboard.start_execution("test")
    
    dashboard.log_agent_activity("Market Analyst", "Analyzing market conditions")
    dashboard.log_decision("Market Analyst", "BUY", "Strong bullish signals", 0.85)
    dashboard.log_action("TRADE_EXECUTION", {"strategy": "long_call", "strike": 19000})
    
    dashboard.end_execution("COMPLETED", "Test execution completed successfully")
    dashboard.print_dashboard_summary() 