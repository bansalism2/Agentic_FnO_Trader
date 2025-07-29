#!/usr/bin/env python3
"""
AlgoTrade Hybrid Advanced Crew Driver - Best of Both Worlds

This script combines the advanced features of the pure scalping driver with hybrid agent capabilities:
- Advanced pre-market data integration
- IV cache management and token refresh
- Forced square-off and profit target logic
- Hybrid opportunity hunter and position manager agents
- Clear mode separation between scalping and premium selling

ADVANCED FEATURES RETAINED:
- Pre-market data fetch at 9:00 AM
- IV cache refresh for enhanced analysis
- Access token management
- Forced square-off at 3:15 PM
- Profit target logic and cleanup
- High-frequency execution scheduling

HYBRID CAPABILITIES:
- Scalping Mode: Quick directional trades with tight stops
- Premium Selling Mode: Time-decay strategies with IV-based entries
- Clear mode separation prevents contradictions
- Adaptive strategy selection based on market conditions

Execution Schedule:
- Pre-market data: 9:00 AM daily
- Hybrid Opportunity Hunter: Every 5 minutes
- Hybrid Position Manager: Every 3 minutes
- Forced square-off: 3:15 PM daily
- Cleanup: 3:35 PM daily

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

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the new, corrected opportunity hunter
from opportunity_hunter_hybrid_scalping_v2 import run_hybrid_scalping_opportunity_hunter

# Import other necessary functions
from core_tools.connect_data_tools import initialize_connection
from core_tools.trade_storage import get_active_trades, move_trade_to_history, update_trade_status
from core_tools.execution_portfolio_tools import get_portfolio_positions

# --- Position Manager Import ---
# from main_agents.position_manager_scalping import ScalpingPositionManager  # Original manager (commented out for testing)
# NOTE: Replaced original position manager from 'position_manager_scalping.py' with SimplePositionManager from 'position_manager_simple.py' for testing
from main_agents.position_manager_simple import SimplePositionManager

# --- Forced Square-Off Logic ---
def force_square_off_all_positions():
    print("\n" + "="*80)
    print("⏰ FORCED SQUARE-OFF: Closing all open NIFTY positions at 3:20 PM (crew driver safety net)")
    from core_tools.execution_portfolio_tools import get_portfolio_positions, execute_options_strategy
    positions_result = get_portfolio_positions()
    if positions_result.get('status') != 'SUCCESS':
        print("❌ Could not get portfolio positions for forced square-off.")
        return
    open_positions = [
        p for p in positions_result.get('positions', [])
        if 'NIFTY' in p.get('symbol', '')
    ]
    if not open_positions:
        print("✅ No open NIFTY positions to square off.")
        return
    closing_legs = []
    for pos in open_positions:
        symbol = pos.get('symbol')
        quantity = pos.get('quantity')
        action = 'SELL' if quantity > 0 else 'BUY'
        closing_legs.append({
            'symbol': symbol,
            'action': action,
            'quantity': abs(quantity)
        })
        print(f"  🔄 FORCED EXIT: {symbol} | Qty: {quantity} | Action: {action} | Reason: Crew Driver Forced Square-Off")
    if closing_legs:
        result = execute_options_strategy(closing_legs, order_type='Closing')
        print(f"  📝 Forced square-off result: {json.dumps(result, indent=2)}")
    print("="*80)

# --- Schedule forced square-off as a final safety net (even if position manager also handles it) ---
schedule.every().day.at("15:20").do(force_square_off_all_positions)

# Configure logging
def setup_logging():
    """Setup logging configuration"""
    log_dir = current_dir / "logs"
    log_dir.mkdir(exist_ok=True)
    
    log_file = log_dir / f"crew_driver_hybrid_advanced_{datetime.now().strftime('%Y%m%d')}.log"
    
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
        
        # CORRECTED: Call the original, robust get_access_token.py script
        script_path = os.path.join(current_dir, "..", "utilities", "get_access_token.py")
        
        # Check if the script exists and is executable
        if not os.access(script_path, os.X_OK):
            os.chmod(script_path, 0o755) # Make it executable
            
        result = subprocess.run([sys.executable, script_path], capture_output=True, text=True)
        
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
        
        # First ensure connection is established
        from core_tools.connect_data_tools import initialize_connection
        from dotenv import load_dotenv
        import os
        
        # Load credentials
        load_dotenv(dotenv_path='../.env')
        api_key = os.getenv("kite_api_key")
        api_secret = os.getenv("kite_api_secret")
        access_token = None
        try:
            with open("../data/access_token.txt", "r") as f:
                access_token = f.read().strip()
        except Exception as e:
            logger.error(f"Could not read access token: {e}")
            return False
        
        # Initialize connection first
        init_result = initialize_connection(api_key, api_secret, access_token)
        if init_result.get('status') != 'SUCCESS':
            logger.error(f"Failed to initialize connection for IV refresh: {init_result.get('message', 'Unknown error')}")
            return False
        
        # Set up import paths before importing IV data manager
        import sys
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.join(current_dir, '..')
        core_tools_dir = os.path.join(parent_dir, 'core_tools')
        
        # Add paths to sys.path
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
        if core_tools_dir not in sys.path:
            sys.path.insert(0, core_tools_dir)
        
        # Now import IV data manager
        from iv_data_manager import IVDataManager
        
        # Initialize and refresh IV data
        iv_manager = IVDataManager()
        refresh_success = iv_manager.refresh_if_needed(force_refresh=True)
        
        if refresh_success:
            logger.info("✅ IV cache data refreshed successfully")
            return True
        else:
            logger.warning("IV cache refresh failed")
            return False
            
    except ImportError as e:
        logger.warning(f"IV data manager not available: {e}")
        logger.info("Proceeding with basic IV analysis...")
        return False
    except Exception as e:
        logger.error(f"Error refreshing IV cache: {e}")
        return False

def cleanup_active_trades():
    """
    Clean up active_trades.json at market close
    """
    try:
        logger.info("Cleaning up active_trades.json at market close...")
        
        # Import trade storage module
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
        from core_tools.trade_storage import clear_active_trades
        
        clear_active_trades()
        logger.info("✅ Active trades cleaned up successfully")
        
    except ImportError as e:
        logger.warning(f"Trade storage module not available: {e}")
    except Exception as e:
        logger.error(f"Error cleaning up active trades: {e}")

def is_market_open(force_run=False):
    """
    Check if market is currently open
    If force_run is True, always return True (for testing)
    """
    if force_run:
        return True
    
    ist_now = datetime.now(IST_TIMEZONE)
    current_time = ist_now.time()
    
    return MARKET_START_TIME <= current_time <= MARKET_END_TIME

def run_opportunity_hunter():
    """Wrapper function to run the hybrid opportunity hunter agent."""
    logger.info("=" * 80)
    logger.info("STARTING HYBRID OPPORTUNITY HUNTER AGENT EXECUTION")
    logger.info("=" * 80)
    
    # Initialize connection for this run
    try:
        # CORRECTED: Read the access token directly from the file
        access_token = None
        try:
            with open(os.path.join(current_dir, "..", "data", "access_token.txt"), "r") as f:
                access_token = f.read().strip()
        except FileNotFoundError:
            logger.error("access_token.txt not found. Please run the authentication script.")
            return

        init_result = initialize_connection(os.getenv("kite_api_key"), os.getenv("kite_api_secret"), access_token)
        if init_result.get('status') != 'SUCCESS':
            logger.error(f"Kite connection failed for opportunity hunter: {init_result.get('message')}")
            return
    except Exception as e:
        logger.error(f"Kite connection failed for opportunity hunter: {e}")
        return

    logger.info("Executing hybrid opportunity hunter for adaptive trading...")
    result = run_hybrid_scalping_opportunity_hunter() # Using the new v2 hunter
    
    logger.info("=" * 80)
    logger.info("HYBRID OPPORTUNITY HUNTER EXECUTION COMPLETED")
    logger.info("=" * 80)
    # Pretty-print the full result
    logger.info("Result:\n" + json.dumps(result, indent=2))
    logger.info("=" * 80)
    
    return True

def run_position_manager():
    """
    Run the Hybrid Position Manager Agent for managing existing positions
    If force_run is True, skip market open check (for testing)
    """
    try:
        logger.info("=" * 80)
        logger.info("STARTING HYBRID POSITION MANAGER AGENT EXECUTION")
        logger.info("=" * 80)
        
        # Check if market is open
        if not is_market_open(force_run=False): # Changed force_run to False as it's not a force run
            logger.info("Market is closed. Skipping hybrid position manager execution.")
            return
        
        # Import and run the hybrid position manager agent
        # from position_manager_scalping import run_intraday_scalping_position_manager
        # logger.info("Executing hybrid position manager for adaptive monitoring...")
        
        # result = run_intraday_scalping_position_manager()
        manager = SimplePositionManager()  # Using simple manager for testing
        result = manager.manage_positions()
        
        logger.info("=" * 80)
        logger.info("HYBRID POSITION MANAGER EXECUTION COMPLETED")
        logger.info("=" * 80)
        # Pretty-print the full result
        logger.info("Result:\n" + json.dumps(result, indent=2))
        logger.info("=" * 80)
        
    except ImportError as e:
        error_msg = f"Failed to import position_manager_hybrid_scalping: {e}"
        logger.error(error_msg)
        logger.error("Make sure position_manager_hybrid_scalping.py is in the same directory")
        logger.info("Skipping this run and waiting for next scheduled execution...")
        return False
    except Exception as e:
        error_msg = f"Error running hybrid position manager agent: {e}"
        logger.error(error_msg)
        logger.error(f"Traceback: {traceback.format_exc()}")
        logger.info("Skipping this run and waiting for next scheduled execution...")
        return False
    
    return True

def force_square_off_all_positions():
    """
    Force square-off of all open positions.
    This is now handled by the position manager's internal logic.
    """
    logger.info("=" * 80)
    logger.info("FORCED SQUARE-OFF TRIGGERED (DEPRECATED - HANDLED BY MANAGER)")
    logger.info("=" * 80)
    # The new position manager handles this automatically at 3:15 PM.
    # We can call it one last time to be sure.
    run_position_manager()

def cleanup_market_close():
    """
    Clean up active_trades.json at market close
    """
    try:
        logger.info("🧹 Cleaning up active_trades.json at market close...")
        
        # Import trade storage module
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
        from core_tools.trade_storage import clear_active_trades, get_active_trades
        
        # Check if there are any active trades before clearing
        active_trades = get_active_trades()
        if active_trades:
            logger.warning(f"⚠️ Found {len(active_trades)} active trades at market close - clearing them")
            for trade_id in active_trades.keys():
                logger.warning(f"  - Clearing trade: {trade_id}")
        else:
            logger.info("✅ No active trades found - clean state")
        
        # Clear active trades
        result = clear_active_trades()
        if result.get('status') == 'SUCCESS':
            logger.info("✅ Active trades cleaned up successfully")
        else:
            logger.error(f"❌ Failed to clear active trades: {result.get('message')}")
        
    except ImportError as e:
        logger.warning(f"Trade storage module not available: {e}")
    except Exception as e:
        logger.error(f"Error cleaning up active trades: {e}")

def schedule_agent_execution(force_run=False):
    """
    Schedule agent execution with enhanced frequencies for hybrid trading
    - Pre-market data: At 9:00 AM or immediately if started late
    - Hybrid Opportunity Hunter: Every 5 minutes (adaptive opportunity detection)
    - Hybrid Position Manager: Every 3 minutes (adaptive position monitoring)
    - Market Close Cleanup: At 3:35 PM
    """
    logger.info("Setting up hybrid advanced agent schedule...")
    
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
    
    # Schedule hybrid opportunity hunter every 2 minutes (was 5)
    schedule.every(2).minutes.do(run_opportunity_hunter)
    
    # Schedule position manager - More frequent around market close
    schedule.every(3).minutes.do(run_position_manager)
    
    # Additional position manager runs around market close for safety
    schedule.every().day.at("15:18").do(run_position_manager)  # 2 minutes before forced square-off
    schedule.every().day.at("15:19").do(run_position_manager)  # 1 minute before forced square-off
    schedule.every().day.at("15:20").do(run_position_manager)  # At forced square-off time
    schedule.every().day.at("15:21").do(run_position_manager)  # 1 minute after forced square-off
    
    # Schedule forced square-off - Backup safety net at 3:20 PM
    schedule.every().day.at("15:20").do(force_square_off_all_positions)
    
    # Schedule market close cleanup
    schedule.every().day.at("15:35").do(cleanup_market_close)
    
    logger.info("Hybrid advanced agent schedule configured:")
    if pre_market_time <= current_time < dt_time(9, 30):
        logger.info("- Pre-market data: Already fetched (late start)")
    else:
        logger.info("- 9:00 AM: Pre-market data fetch")
    logger.info("- Every 5 minutes: Hybrid Opportunity Hunter Agent (adaptive mode detection)")
    logger.info("- Every 3 minutes: Hybrid Position Manager Agent (adaptive monitoring)")
    logger.info("- 3:18-3:21 PM: Enhanced position monitoring (every minute)")
    logger.info("- 3:20 PM: Forced square-off (backup safety net)")
    logger.info("- 3:35 PM: Market close cleanup")
    logger.info(f"Market hours: {MARKET_START_TIME.strftime('%H:%M')} - {MARKET_END_TIME.strftime('%H:%M')} IST")
    logger.info("Agents run sequentially to avoid conflicts")
    logger.info("Hybrid system with clear mode separation")
    logger.info("Advanced features: pre-market data, IV cache, token management")
    logger.info("Failed runs are skipped (no retries) - wait for next scheduled execution")
    logger.info("Press Ctrl+C to stop the driver")

def main(force_run=False):
    """
    Main function to run the hybrid advanced crew driver
    """
    try:
        logger.info("=" * 80)
        logger.info("ALGOTRADE HYBRID ADVANCED CREW DRIVER STARTING")
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
        
        # Clear any leftover profit target markers from previous day
        cleanup_profit_target_marker()

        # Setup schedule
        schedule_agent_execution(force_run=force_run)
        
        # Run initial execution if market is open
        if is_market_open(force_run=force_run):
            logger.info("Market is open. Running initial hybrid agent executions...")
            
            # Run both agents initially (no token refresh here)
            logger.info("Step 3: Running initial hybrid position manager...")
            pm_success = run_position_manager()
            if not pm_success:
                logger.warning("Initial hybrid position manager run failed - will retry at next scheduled time")
            
            logger.info("Step 4: Running initial hybrid opportunity hunter...")
            oh_success = run_opportunity_hunter()
            if not oh_success:
                logger.warning("Initial hybrid opportunity hunter run failed - will retry at next scheduled time")
        else:
            logger.info("Market is closed. Waiting for market hours...")
        
        # Main loop
        while True:
            try:
                # Check if market has closed (after 3:30 PM)
                current_time = datetime.now(IST_TIMEZONE).time()
                if current_time >= dt_time(15, 30):  # 3:30 PM or later
                    logger.info("Market has closed (3:30 PM). Stopping crew driver...")
                    logger.info("All positions should have been squared off by 3:20 PM")
                    
                    # Final cleanup of active trades before shutdown
                    logger.info("🧹 Final cleanup of active_trades.json before shutdown...")
                    try:
                        from core_tools.trade_storage import clear_active_trades, get_active_trades
                        active_trades = get_active_trades()
                        if active_trades:
                            logger.warning(f"⚠️ Final cleanup: Found {len(active_trades)} active trades - clearing them")
                            clear_active_trades()
                            logger.info("✅ Final cleanup completed")
                        else:
                            logger.info("✅ Final cleanup: No active trades found")
                    except Exception as e:
                        logger.error(f"❌ Final cleanup failed: {e}")
                    
                    break
                
                # Run scheduled tasks
                schedule.run_pending()
                
                time.sleep(30)  # Check every 30 seconds for high-frequency execution
                
            except KeyboardInterrupt:
                logger.info("Received interrupt signal. Shutting down...")
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                logger.info("Waiting 1 minute before continuing...")
                time.sleep(60)  # Wait 1 minute before retrying
        
        logger.info("Hybrid advanced crew driver stopped successfully")
        
    except Exception as e:
        logger.error(f"Fatal error in hybrid advanced crew driver: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)

def run_single_execution(force_run=False):
    """
    Run a single execution of both hybrid agents (for testing)
    """
    logger.info("Running single hybrid agent executions...")
    
    # Refresh access token ONCE at the start (not before each agent)
    logger.info("Refreshing access token...")
    refresh_access_token()
    
    # Refresh IV cache for enhanced analysis
    logger.info("Refreshing IV cache...")
    refresh_iv_cache()
    
    # Run hybrid position manager first
    logger.info("Running hybrid position manager...")
    pm_success = run_position_manager()
    if not pm_success:
        logger.warning("Hybrid position manager run failed")
    
    # Run hybrid opportunity hunter second
    logger.info("Running hybrid opportunity hunter...")
    oh_success = run_opportunity_hunter()
    if not oh_success:
        logger.warning("Hybrid opportunity hunter run failed")
    
    logger.info("Single hybrid execution completed")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='AlgoTrade Hybrid Advanced Crew Driver')
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