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
MARKET_START_TIME = dt_time(9, 30)  # 9:30 AM
MARKET_END_TIME = dt_time(15, 30)   # 3:30 PM
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
    try:
        logger.info("=" * 80)
        logger.info("STARTING ALGOTRADE CREW AGENT EXECUTION")
        logger.info("=" * 80)
        # Check if market is open
        if not is_market_open(force_run=force_run):
            logger.info("Market is closed. Skipping crew agent execution.")
            return
        # Refresh access token before running crew agent
        logger.info("Step 1: Refreshing access token...")
        token_refreshed = refresh_access_token()
        if not token_refreshed:
            logger.warning("Failed to refresh access token. Proceeding with existing token...")
        # Import and run the crew agent
        from crew_agent import trading_crew
        logger.info("Step 2: Executing trading crew...")
        result = trading_crew.kickoff()
        logger.info("=" * 80)
        logger.info("CREW AGENT EXECUTION COMPLETED")
        logger.info("=" * 80)
        logger.info(f"Result: {result}")
        logger.info("=" * 80)
    except ImportError as e:
        logger.error(f"Failed to import crew_agent: {e}")
        logger.error("Make sure crew_agent.py is in the same directory")
    except Exception as e:
        logger.error(f"Error running crew agent: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")

def schedule_crew_execution(force_run=False):
    """
    Schedule crew agent execution with dynamic frequency based on market time
    """
    logger.info("Setting up crew agent schedule...")
    
    # Schedule every 15 minutes for normal trading hours (9:30 AM - 2:30 PM)
    schedule.every(15).minutes.do(run_crew_agent, force_run=force_run)
    
    # Schedule every 5 minutes for position management hours (2:30 PM - 3:30 PM)
    # This will be handled dynamically in the main loop
    
    logger.info("Crew agent scheduled with dynamic frequency:")
    logger.info("- 9:30 AM - 2:30 PM: Every 15 minutes (trading and analysis)")
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
                    logger.info("SWITCHING TO NORMAL TRADING MODE (9:30 AM - 2:30 PM)")
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