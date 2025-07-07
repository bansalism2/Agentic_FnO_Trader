#!/usr/bin/env python3
"""
AlgoTrade Crew Driver - Automated trading crew execution

This script runs the AlgoTrade crew agent every 15 minutes during market hours
(9:30 AM to 3:30 PM IST) for automated F&O trading analysis and execution.

@author: AlgoTrade Team
"""

import time
import schedule
import logging
from datetime import datetime, time as dt_time
import pytz
import sys
import os
import traceback
import subprocess
from pathlib import Path

# Import dashboard
try:
    from agent_dashboard import get_dashboard, log_agent_activity, log_decision, log_action, log_error, log_warning
except ImportError as e:
    print(f"Warning: Could not import agent_dashboard: {e}")
    # Create dummy functions if dashboard is not available
    def get_dashboard():
        return None
    def log_agent_activity(*args, **kwargs):
        pass
    def log_decision(*args, **kwargs):
        pass
    def log_action(*args, **kwargs):
        pass
    def log_error(*args, **kwargs):
        pass
    def log_warning(*args, **kwargs):
        pass

# Add current directory to path for imports
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

# Configure logging
def setup_logging():
    """Setup logging configuration"""
    log_dir = current_dir / "logs"
    log_dir.mkdir(exist_ok=True)
    
    log_file = log_dir / f"crew_driver_{datetime.now().strftime('%Y%m%d')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

# Market hours configuration
MARKET_START_TIME = dt_time(9, 15)  # 9:15 AM - Pre-opening session starts
MARKET_END_TIME = dt_time(15, 30)   # 3:30 PM
PRE_MARKET_TIME = dt_time(9, 0)     # 9:00 AM - Pre-market data fetch
IST_TIMEZONE = pytz.timezone('Asia/Kolkata')

def refresh_access_token():
    """
    Refresh the Kite Connect access token by running get_access_token.py
    """
    try:
        logger.info("Refreshing Kite Connect access token...")
        
        # Run get_access_token.py as a subprocess
        result = subprocess.run(
            [sys.executable, 'get_access_token.py'],
            capture_output=True,
            text=True,
            cwd=current_dir,
            timeout=120  # 2 minute timeout
        )
        
        if result.returncode == 0:
            logger.info("Access token refreshed successfully")
            logger.info(f"Output: {result.stdout}")
            return True
        else:
            logger.error(f"Failed to refresh access token. Return code: {result.returncode}")
            logger.error(f"Error output: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error("Access token refresh timed out after 2 minutes")
        return False
    except Exception as e:
        logger.error(f"Error refreshing access token: {e}")
        return False

def fetch_pre_market_data():
    """
    Fetch pre-market global market data before market opens
    """
    try:
        logger.info("Fetching pre-market global market data...")
        
        # Log to dashboard
        log_agent_activity("Crew Driver", "Fetching pre-market global market data")
        log_action("GLOBAL_MARKET_CHECK", {"type": "pre_market_data"}, "STARTED")
        
        # Import pre-market data module
        from pre_market_data import fetch_pre_market_data as fetch_data, get_json_output
        
        # Fetch the data
        data = fetch_data()
        
        if data and data['metadata']['successful_fetches'] > 0:
            # Convert to JSON and log summary
            json_output = get_json_output(data)
            
            # Save JSON report to file
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            json_filename = f"pre_market_data_{timestamp}.json"
            json_path = current_dir / json_filename
            
            with open(json_path, 'w') as f:
                f.write(json_output)
            
            # Log key information
            summary = data['summary']
            logger.info("Pre-market analysis summary:")
            logger.info(f"Expected NIFTY move: {summary['expected_nifty_move']}")
            logger.info(f"Gap type: {summary['gap_type']}")
            logger.info(f"Confidence: {summary['confidence']}%")
            logger.info(f"Recommendation: {summary['recommendation']}")
            logger.info(f"Market sentiment: {summary['overall_sentiment']}")
            
            # Log to dashboard
            log_decision("Pre-Market Analysis", 
                        f"Expected {summary['expected_nifty_move']} gap", 
                        f"Gap type: {summary['gap_type']}, Confidence: {summary['confidence']}%",
                        summary['confidence'] / 100.0)
            
            log_action("GLOBAL_MARKET_CHECK", {
                "expected_move": summary['expected_nifty_move'],
                "gap_type": summary['gap_type'],
                "confidence": summary['confidence'],
                "recommendation": summary['recommendation'],
                "sentiment": summary['overall_sentiment'],
                "file_saved": str(json_path)
            }, "COMPLETED")
            
            # Log alerts if any
            if data['key_alerts']:
                logger.info("Key alerts:")
                for alert in data['key_alerts'][:3]:  # Show top 3 alerts
                    logger.info(f"  {alert}")
                    log_warning(f"Global market alert: {alert}")
            
            logger.info(f"Pre-market data saved to: {json_path}")
            return True
        else:
            logger.warning("No pre-market data fetched successfully")
            log_warning("No pre-market data fetched successfully")
            log_action("GLOBAL_MARKET_CHECK", {"status": "no_data"}, "FAILED")
            return False
            
    except ImportError as e:
        error_msg = f"Failed to import pre_market_data module: {e}"
        logger.error(error_msg)
        logger.error("Make sure pre_market_data.py is in the same directory")
        log_error(error_msg, "IMPORT_ERROR")
        log_action("GLOBAL_MARKET_CHECK", {"error": error_msg}, "FAILED")
        return False
    except Exception as e:
        error_msg = f"Error fetching pre-market data: {e}"
        logger.error(error_msg)
        logger.error(f"Traceback: {traceback.format_exc()}")
        log_error(error_msg, "EXECUTION_ERROR", {"traceback": traceback.format_exc()})
        log_action("GLOBAL_MARKET_CHECK", {"error": error_msg}, "FAILED")
        return False

def is_market_open(force_run=False):
    """
    Check if the market is currently open (9:30 AM to 3:30 PM IST)
    If force_run is True, always return True (for testing)
    """
    if force_run:
        logger.info("[TEST MODE] Forcing market open for testing.")
        return True
    try:
        # Get current time in IST
        ist_now = datetime.now(IST_TIMEZONE)
        current_time = ist_now.time()
        # Check if it's a weekday (Monday = 0, Sunday = 6)
        is_weekday = ist_now.weekday() < 5
        # Check if current time is within market hours
        is_market_hours = MARKET_START_TIME <= current_time <= MARKET_END_TIME
        market_open = is_weekday and is_market_hours
        logger.info(f"Current IST time: {ist_now.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        logger.info(f"Weekday: {is_weekday}, Market hours: {is_market_hours}, Market open: {market_open}")
        return market_open
    except Exception as e:
        logger.error(f"Error checking market status: {e}")
        return False

def run_crew_agent(force_run=False):
    """
    Run the AlgoTrade crew agent for analysis and trading decisions
    If force_run is True, skip market open check (for testing)
    """
    # Initialize dashboard
    dashboard = get_dashboard()
    if dashboard:
        dashboard.start_execution("crew_agent")
        log_agent_activity("Crew Driver", "Starting crew agent execution")
    
    try:
        logger.info("=" * 80)
        logger.info("STARTING ALGOTRADE CREW AGENT EXECUTION")
        logger.info("=" * 80)
        
        # Check if market is open
        if not is_market_open(force_run=force_run):
            logger.info("Market is closed. Skipping crew agent execution.")
            if dashboard:
                log_warning("Market is closed, skipping execution")
                dashboard.end_execution("SKIPPED", "Market closed")
            return
        
        # Refresh access token before running crew agent
        logger.info("Step 1: Refreshing access token...")
        if dashboard:
            log_agent_activity("Crew Driver", "Refreshing access token")
        
        token_refreshed = refresh_access_token()
        if not token_refreshed:
            logger.warning("Failed to refresh access token. Proceeding with existing token...")
            if dashboard:
                log_warning("Failed to refresh access token")
        
        # Import and run the crew agent
        from crew_agent import trading_crew
        logger.info("Step 2: Executing trading crew...")
        if dashboard:
            log_agent_activity("Crew Driver", "Executing trading crew")
        
        result = trading_crew.kickoff()
        
        logger.info("=" * 80)
        logger.info("CREW AGENT EXECUTION COMPLETED")
        logger.info("=" * 80)
        logger.info(f"Result: {result}")
        logger.info("=" * 80)
        
        # Log completion to dashboard
        if dashboard:
            log_agent_activity("Crew Driver", "Crew agent execution completed")
            dashboard.end_execution("COMPLETED", f"Crew execution completed with result: {result}")
            dashboard.print_dashboard_summary()
        
    except ImportError as e:
        error_msg = f"Failed to import crew_agent: {e}"
        logger.error(error_msg)
        logger.error("Make sure crew_agent.py is in the same directory")
        if dashboard:
            log_error(error_msg, "IMPORT_ERROR")
            dashboard.end_execution("FAILED", error_msg)
    except Exception as e:
        error_msg = f"Error running crew agent: {e}"
        logger.error(error_msg)
        logger.error(f"Traceback: {traceback.format_exc()}")
        if dashboard:
            log_error(error_msg, "EXECUTION_ERROR", {"traceback": traceback.format_exc()})
            dashboard.end_execution("FAILED", error_msg)

def schedule_crew_execution(force_run=False):
    """
    Schedule crew agent execution with dynamic frequency based on market time
    """
    logger.info("Setting up crew agent schedule...")
    
    # Schedule pre-market data fetch at 9:00 AM
    schedule.every().day.at("09:00").do(fetch_pre_market_data)
    
    # Schedule every 15 minutes for normal trading hours (9:30 AM - 2:30 PM)
    schedule.every(15).minutes.do(run_crew_agent, force_run=force_run)
    
    # Schedule every 5 minutes for position management hours (2:30 PM - 3:30 PM)
    # This will be handled dynamically in the main loop
    
    logger.info("Crew agent scheduled with dynamic frequency:")
    logger.info("- 9:00 AM: Pre-market data fetch")
    logger.info("- 9:15 AM - 2:30 PM: Every 15 minutes (trading and analysis)")
    logger.info("- 2:30 PM - 3:30 PM: Every 5 minutes (position management)")
    logger.info(f"Market hours: {MARKET_START_TIME.strftime('%H:%M')} - {MARKET_END_TIME.strftime('%H:%M')} IST")
    logger.info("Press Ctrl+C to stop the driver")

def main(force_run=False):
    """
    Main function to run the crew driver
    """
    try:
        logger.info("=" * 80)
        logger.info("ALGOTRADE CREW DRIVER STARTING")
        logger.info("=" * 80)
        logger.info(f"Start time: {datetime.now(IST_TIMEZONE).strftime('%Y-%m-%d %H:%M:%S %Z')}")
        
        # Setup schedule
        schedule_crew_execution(force_run=force_run)
        
        # Run initial execution if market is open
        if is_market_open(force_run=force_run):
            logger.info("Market is open. Running initial crew execution...")
            run_crew_agent(force_run=force_run)
        else:
            logger.info("Market is closed. Waiting for market hours...")
        
        # Variables for dynamic scheduling
        last_execution_time = None
        position_management_mode = False
        
        # Main loop
        while True:
            try:
                # Check current time for dynamic scheduling
                ist_now = datetime.now(IST_TIMEZONE)
                current_time = ist_now.time()
                
                # Define position management time (2:30 PM - 3:30 PM)
                position_management_start = dt_time(14, 30)  # 2:30 PM
                
                # Check if we're in position management mode
                in_position_management = position_management_start <= current_time <= MARKET_END_TIME
                
                # Log mode change
                if in_position_management and not position_management_mode:
                    logger.info("=" * 50)
                    logger.info("SWITCHING TO POSITION MANAGEMENT MODE (2:30 PM - 3:30 PM)")
                    logger.info("Increased frequency: Every 5 minutes")
                    logger.info("Focus: Position management and closures only")
                    logger.info("=" * 50)
                    position_management_mode = True
                elif not in_position_management and position_management_mode:
                    logger.info("=" * 50)
                    logger.info("SWITCHING TO NORMAL TRADING MODE (9:15 AM - 2:30 PM)")
                    logger.info("Frequency: Every 15 minutes")
                    logger.info("Focus: Trading and analysis")
                    logger.info("=" * 50)
                    position_management_mode = False
                
                # Run scheduled tasks (15-minute intervals)
                schedule.run_pending()
                
                # Dynamic execution for position management mode
                if in_position_management and is_market_open(force_run=force_run):
                    current_minute = ist_now.minute
                    
                    # Execute every 5 minutes (at minutes 0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55)
                    if current_minute % 5 == 0:
                        # Check if we haven't executed in this minute yet
                        if last_execution_time is None or last_execution_time.minute != current_minute:
                            logger.info(f"Position management execution at {ist_now.strftime('%H:%M:%S')}")
                            run_crew_agent(force_run=force_run)
                            last_execution_time = ist_now
                
                time.sleep(60)  # Check every minute
                
            except KeyboardInterrupt:
                logger.info("Received interrupt signal. Shutting down...")
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                time.sleep(60)  # Wait before retrying
        
        logger.info("Crew driver stopped successfully")
        
    except Exception as e:
        logger.error(f"Fatal error in crew driver: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)

def run_single_execution(force_run=False):
    """
    Run a single execution of the crew agent (for testing)
    """
    logger.info("Running single crew agent execution...")
    run_crew_agent(force_run=force_run)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='AlgoTrade Crew Driver')
    parser.add_argument('--single', action='store_true', 
                       help='Run single execution instead of continuous scheduling')
    parser.add_argument('--test', action='store_true',
                       help='Test mode - run regardless of market hours')
    args = parser.parse_args()
    force_run = args.test
    if args.single:
        run_single_execution(force_run=force_run)
    else:
        main(force_run=force_run) 