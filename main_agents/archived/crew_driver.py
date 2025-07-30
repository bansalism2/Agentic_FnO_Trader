#!/usr/bin/env python3
"""
AlgoTrade Crew Driver - Automated trading crew execution (WITH CRITICAL SAFETY FIXES)

This script runs the AlgoTrade agents with split functionality:
- Opportunity Hunter Agent: Every 15 minutes (finds new trading opportunities with three-stage analysis + CRITICAL SAFETY FIXES)
- Position Manager Agent: Every 15 minutes (manages existing positions)

CRITICAL SAFETY FIXES IMPLEMENTED:
- Multi-factor emergency execution safety checks
- Market regime-aware strategy selection
- Contextual PCR analysis
- Technical integration with options strategies
- Volatility term structure analysis

Agents run sequentially to avoid conflicts during market hours (9:15 AM to 3:30 PM IST).
Three-stage analysis enables optimal token efficiency and speed for all market conditions.

@author: AlgoTrade Team
"""

import time
import schedule
import logging
import json
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
MARKET_START_TIME = dt_time(9, 15)  # 9:15 AM - Pre-opening session starts
MARKET_END_TIME = dt_time(15, 30)   # 3:30 PM
PRE_MARKET_TIME = dt_time(9, 0)     # 9:00 AM - Pre-market data fetch
IST_TIMEZONE = pytz.timezone('Asia/Kolkata')

def cleanup_profit_target_marker():
    """
    Clean up the profit target marker file at the start of each trading day
    """
    try:
        import os
        profit_target_file = "/tmp/algotrade_no_more_trades_today"
        if os.path.exists(profit_target_file):
            os.remove(profit_target_file)
            logger.info("✅ Profit target marker file cleaned up for new trading day")
        else:
            logger.info("ℹ️  No profit target marker file found - continuing normally")
    except Exception as e:
        logger.error(f"Error cleaning up profit target marker: {e}")

def refresh_access_token():
    """
    Refresh the Kite Connect access token by running get_access_token.py
    """
    try:
        logger.info("Refreshing Kite Connect access token...")
        
        # Run get_access_token.py as a subprocess
        result = subprocess.run(
            [sys.executable, '../utilities/get_access_token.py'],
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

def check_pre_market_data_exists():
    """
    Check if pre-market data was already fetched today
    """
    try:
        today = datetime.now().strftime('%Y%m%d')
        # Look for any pre-market data file from today
        for file in current_dir.glob(f"pre_market_data_{today}_*.json"):
            return True
        return False
    except Exception:
        return False

def fetch_pre_market_data():
    """
    Fetch pre-market global market data before market opens
    """
    try:
        # Check if pre-market data was already fetched today
        if check_pre_market_data_exists():
            logger.info("Pre-market data already fetched today - skipping duplicate fetch")
            return True
            
        logger.info("Fetching pre-market global market data...")
        
        # Import pre-market data module
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
        from data.pre_market_data import fetch_pre_market_data as fetch_data, get_json_output
        
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
            
            # Log alerts if any
            if data['key_alerts']:
                logger.info("Key alerts:")
                for alert in data['key_alerts'][:3]:  # Show top 3 alerts
                    logger.info(f"  {alert}")
            
            logger.info(f"Pre-market data saved to: {json_path}")
            return True
        else:
            logger.warning("No pre-market data fetched successfully")
            return False
            
    except ImportError as e:
        error_msg = f"Failed to import pre_market_data module: {e}"
        logger.error(error_msg)
        logger.error("Make sure data/pre_market_data.py exists in the agent_tools directory")
        return False
    except Exception as e:
        error_msg = f"Error fetching pre-market data: {e}"
        logger.error(error_msg)
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

def refresh_iv_cache():
    """
    Refresh IV cache data for enhanced analysis
    """
    try:
        logger.info("Refreshing IV cache data...")
        
        # Import IV data manager
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
        from iv_data_manager import IVDataManager
        
        # Initialize IV manager and refresh data
        iv_manager = IVDataManager()
        refresh_success = iv_manager.refresh_if_needed(force_refresh=True)
        
        if refresh_success:
            logger.info("IV cache refreshed successfully")
            
            # Get comprehensive analysis for logging
            analysis = iv_manager.get_comprehensive_analysis()
            if analysis.get('status') == 'SUCCESS':
                summary = analysis.get('summary', {})
                logger.info("IV Analysis Summary:")
                logger.info(f"  Current IV: {summary.get('current_iv', 'N/A')}")
                logger.info(f"  IV Percentile: {summary.get('iv_percentile', 'N/A')}")
                logger.info(f"  Market Condition: {summary.get('market_condition', 'N/A')}")
                logger.info(f"  Volatility Regime: {summary.get('volatility_regime', 'N/A')}")
                logger.info(f"  Recommendation: {summary.get('recommendation', 'N/A')}")
            
            return True
        else:
            logger.warning("IV cache refresh failed")
            return False
            
    except ImportError as e:
        logger.error(f"Failed to import IV data manager: {e}")
        return False
    except Exception as e:
        logger.error(f"Error refreshing IV cache: {e}")
        return False

def cleanup_active_trades():
    """
    Clean up active_trades.json and portfolio_history.json at market close (3:30 PM)
    """
    try:
        logger.info("=" * 80)
        logger.info("MARKET CLOSE CLEANUP - CLEANING ACTIVE TRADES AND PORTFOLIO HISTORY")
        logger.info("=" * 80)
        
        # Path to active_trades.json (relative to main_agents directory)
        active_trades_path = current_dir / ".." / "trade_storage" / "active_trades.json"
        if active_trades_path.exists():
            # Read current active trades
            with open(active_trades_path, 'r') as f:
                active_trades = json.load(f)
            logger.info(f"Found {len(active_trades)} active trades before cleanup")
            # Clear active trades (all positions should be closed by broker at 3:20 PM)
            with open(active_trades_path, 'w') as f:
                json.dump({}, f, indent=2)
            logger.info("Active trades cleared successfully")
            logger.info("All positions should have been auto-closed by broker at 3:20 PM")
        else:
            logger.info("No active_trades.json found - nothing to clean")

        # Path to portfolio_history.json (relative to core_tools directory)
        portfolio_history_path = current_dir / ".." / "core_tools" / "portfolio_history.json"
        if portfolio_history_path.exists():
            try:
                with open(portfolio_history_path, 'w') as f:
                    json.dump({"snapshots": []}, f, indent=2)
                logger.info("Portfolio history cleared successfully")
            except Exception as e:
                logger.error(f"Error clearing portfolio_history.json: {e}")
        else:
            logger.info("No portfolio_history.json found - nothing to clean")

        logger.info("=" * 80)
        logger.info("MARKET CLOSE CLEANUP COMPLETED")
        logger.info("=" * 80)
        
    except Exception as e:
        error_msg = f"Error during market close cleanup: {e}"
        logger.error(error_msg)
        logger.error(f"Traceback: {traceback.format_exc()}")

def is_market_open(force_run=False):
    """
    Check if the market is currently open (9:15 AM to 3:30 PM IST)
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

def run_opportunity_hunter(force_run=False):
    """
    Run the Opportunity Hunter Agent for finding new trading opportunities
    If force_run is True, skip market open check (for testing)
    """
    try:
        logger.info("=" * 80)
        logger.info("STARTING OPPORTUNITY HUNTER AGENT EXECUTION")
        logger.info("=" * 80)
        
        # Check if market is open
        if not is_market_open(force_run=force_run):
            logger.info("Market is closed. Skipping opportunity hunter execution.")
            return
        
        # Check if after 2:30 PM IST (TEMPORARILY DISABLED FOR TESTING)
        ist_now = datetime.now(IST_TIMEZONE)
        if ist_now.time() >= dt_time(14, 30):
            logger.info("After 2:30 PM IST. No new trades allowed. Skipping opportunity hunter execution.")
            # logger.info("[TESTING MODE] 2:30 PM check temporarily disabled - continuing execution")
            return  # RE-ENABLED: No new trades after 2:30 PM
        
        # Import and run the ENHANCED THREE-STAGE OPPORTUNITY HUNTER WITH CRITICAL SAFETY FIXES
        from opportunity_hunter_agent_enhanced_three_stage_fixed import run_enhanced_three_stage_opportunity_hunter
        logger.info("Executing enhanced three-stage opportunity hunter with CRITICAL SAFETY FIXES...")
        
        # Run the enhanced three-stage opportunity hunter with full AI agent logic
        result = run_enhanced_three_stage_opportunity_hunter()
        
        logger.info("=" * 80)
        logger.info("THREE-STAGE OPPORTUNITY HUNTER EXECUTION COMPLETED (WITH CRITICAL SAFETY FIXES)")
        logger.info("=" * 80)
        logger.info(f"Result: {result}")
        logger.info("=" * 80)
        
    except ImportError as e:
        error_msg = f"Failed to import opportunity_hunter_agent_enhanced_three_stage_fixed: {e}"
        logger.error(error_msg)
        logger.error("Make sure opportunity_hunter_agent_enhanced_three_stage_fixed.py is in the same directory")
        logger.info("Skipping this run and waiting for next scheduled execution...")
        return False
    except Exception as e:
        error_msg = f"Error running opportunity hunter agent: {e}"
        logger.error(error_msg)
        logger.error(f"Traceback: {traceback.format_exc()}")
        logger.info("Skipping this run and waiting for next scheduled execution...")
        return False
    
    return True

def run_position_manager(force_run=False):
    """
    Run the Position Manager Agent for managing existing positions
    If force_run is True, skip market open check (for testing)
    """
    try:
        logger.info("=" * 80)
        logger.info("STARTING POSITION MANAGER AGENT EXECUTION")
        logger.info("=" * 80)
        
        # Check if market is open
        if not is_market_open(force_run=force_run):
            logger.info("Market is closed. Skipping position manager execution.")
            return
        
        # Import and run the hybrid position manager agent (history version)
        from position_manager_agent_hybrid_history import run_ultra_conservative_position_manager_hybrid
        logger.info("Executing position manager (hybrid history)...")
        
        result = run_ultra_conservative_position_manager_hybrid()
        
        logger.info("=" * 80)
        logger.info("POSITION MANAGER EXECUTION COMPLETED")
        logger.info("=" * 80)
        logger.info(f"Result: {result}")
        logger.info("=" * 80)
        
    except ImportError as e:
        error_msg = f"Failed to import position_manager_agent_hybrid_history: {e}"
        logger.error(error_msg)
        logger.error("Make sure position_manager_agent_hybrid_history.py is in the same directory")
        logger.info("Skipping this run and waiting for next scheduled execution...")
        return False
    except Exception as e:
        error_msg = f"Error running position manager agent: {e}"
        logger.error(error_msg)
        logger.error(f"Traceback: {traceback.format_exc()}")
        logger.info("Skipping this run and waiting for next scheduled execution...")
        return False
    
    return True

def force_square_off_all_positions():
    """
    Force square-off of all open NIFTY positions at 3:15 PM, regardless of other processes.
    """
    try:
        logger.info("=" * 80)
        logger.info("FORCED SQUARE-OFF TRIGGERED AT 3:15 PM - CLOSING ALL POSITIONS")
        logger.info("=" * 80)
        from position_manager_agent_hybrid_history import get_portfolio_positions, square_off_positions
        
        positions_result = get_portfolio_positions()
        if positions_result.get('status') == 'SUCCESS':
            positions = positions_result.get('positions', [])
            nifty_positions = [p for p in positions if 'NIFTY' in p.get('tradingsymbol', '') or 'NIFTY' in p.get('symbol', '')]
            if nifty_positions:
                logger.info(f"Found {len(nifty_positions)} NIFTY positions to force close.")
                square_off_positions(nifty_positions)
                logger.info("All NIFTY positions forcibly squared off at 3:15 PM.")
            else:
                logger.info("No NIFTY positions to force close at 3:15 PM.")
        else:
            logger.error(f"Failed to fetch positions for forced square-off: {positions_result.get('message', 'Unknown error')}")
        logger.info("=" * 80)
        logger.info("FORCED SQUARE-OFF COMPLETED")
        logger.info("=" * 80)
    except Exception as e:
        logger.error(f"Error during forced square-off: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")

def schedule_agent_execution(force_run=False):
    """
    Schedule agent execution with specified frequencies
    - Pre-market data: At 9:00 AM or immediately if started late
    - Opportunity Hunter: Every 15 minutes (fast-track enabled)
    - Position Manager: Every 15 minutes
    - Market Close Cleanup: At 3:30 PM
    """
    logger.info("Setting up agent schedule...")
    
    # Check if we need to run pre-market data fetch immediately
    ist_now = datetime.now(IST_TIMEZONE)
    current_time = ist_now.time()
    pre_market_time = dt_time(9, 0)  # 9:00 AM
    
    # If it's after 9:00 AM but before 9:30 AM, run pre-market data immediately
    if pre_market_time <= current_time < dt_time(9, 30):
        logger.info("Started after 9:00 AM - running pre-market data fetch immediately...")
        fetch_pre_market_data()
        logger.info("Pre-market data fetch completed")
    else:
        # Schedule pre-market data fetch at 9:00 AM for next day
        schedule.every().day.at("09:00").do(fetch_pre_market_data)
        logger.info("Pre-market data fetch scheduled for 9:00 AM")
    
    # Schedule opportunity hunter every 15 minutes (fast-track optimized)
    schedule.every(15).minutes.do(run_opportunity_hunter, force_run=force_run)
    
    # Schedule position manager every 15 minutes
    schedule.every(15).minutes.do(run_position_manager, force_run=force_run)
    
    # Schedule market close cleanup at 3:35 PM (after 3:15 PM square-off)
    schedule.every().day.at("15:35").do(cleanup_active_trades)
    
    # Schedule forced square-off at 3:15 PM
    schedule.every().day.at("15:15").do(force_square_off_all_positions)
    
    logger.info("Agent schedule configured:")
    if pre_market_time <= current_time < dt_time(9, 30):
        logger.info("- Pre-market data: Already fetched (late start)")
    else:
        logger.info("- 9:00 AM: Pre-market data fetch")
    logger.info("- Every 15 minutes: Three-Stage Opportunity Hunter Agent (new opportunities)")
    logger.info("- Every 15 minutes: Position Manager Agent (existing positions)")
    logger.info("- 3:35 PM: Market close cleanup (clear active_trades.json)")
    logger.info(f"Market hours: {MARKET_START_TIME.strftime('%H:%M')} - {MARKET_END_TIME.strftime('%H:%M')} IST")
    logger.info("Agents run sequentially to avoid conflicts")
    logger.info("Three-stage analysis enables optimal token efficiency and speed")
    logger.info("Failed runs are skipped (no retries) - wait for next scheduled execution")
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
        
        # Refresh access token ONCE at the start
        logger.info("Step 1: Refreshing access token...")
        token_refreshed = refresh_access_token()
        if not token_refreshed:
            logger.warning("Failed to refresh access token. Proceeding with existing token...")
        
        # Refresh IV cache for enhanced analysis
        logger.info("Step 2: Refreshing IV cache...")
        iv_refreshed = refresh_iv_cache()
        if not iv_refreshed:
            logger.warning("Failed to refresh IV cache. Proceeding with basic analysis...")
        
        # Clean up profit target marker at the start of each trading day
        cleanup_profit_target_marker()
        
        # Setup schedule
        schedule_agent_execution(force_run=force_run)
        
        # Run initial execution if market is open
        if is_market_open(force_run=force_run):
            logger.info("Market is open. Running initial agent executions...")
            
            # Run both agents initially (no token refresh here)
            logger.info("Step 2: Running initial position manager...")
            pm_success = run_position_manager(force_run=force_run)
            if not pm_success:
                logger.warning("Initial position manager run failed - will retry at next scheduled time")
            
            logger.info("Step 3: Running initial opportunity hunter...")
            oh_success = run_opportunity_hunter(force_run=force_run)
            if not oh_success:
                logger.warning("Initial opportunity hunter run failed - will retry at next scheduled time")
        else:
            logger.info("Market is closed. Waiting for market hours...")
        
        # Main loop
        while True:
            try:
                # Run scheduled tasks
                schedule.run_pending()
                
                time.sleep(60)  # Check every minute
                
            except KeyboardInterrupt:
                logger.info("Received interrupt signal. Shutting down...")
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                logger.info("Waiting 2 minutes before continuing...")
                time.sleep(120)  # Wait 2 minutes before retrying
        
        logger.info("Crew driver stopped successfully")
        
    except Exception as e:
        logger.error(f"Fatal error in crew driver: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)

def run_single_execution(force_run=False):
    """
    Run a single execution of both agents (for testing)
    """
    logger.info("Running single agent executions...")
    
    # Refresh access token ONCE at the start (not before each agent)
    logger.info("Refreshing access token...")
    refresh_access_token()
    
    # Refresh IV cache for enhanced analysis
    logger.info("Refreshing IV cache...")
    refresh_iv_cache()
    
    # Run position manager first
    logger.info("Running position manager...")
    pm_success = run_position_manager(force_run=force_run)
    if not pm_success:
        logger.warning("Position manager run failed")
    
    # Run opportunity hunter second
    # logger.info("Running opportunity hunter...")
    # oh_success = run_opportunity_hunter(force_run=force_run)
    # if not oh_success:
    #     logger.warning("Opportunity hunter run failed")
    
    logger.info("Single execution completed")

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