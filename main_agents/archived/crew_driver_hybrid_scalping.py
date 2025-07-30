#!/usr/bin/env python3
"""
AlgoTrade Hybrid Scalping Crew Driver - Clear Mode Separation

This script runs the hybrid scalping agents with clear separation between:
- SCALPING MODE: Quick directional trades with tight stops
- PREMIUM SELLING MODE: Time-decay strategies with IV-based entries

NO MORE CONTRADICTIONS:
- Each mode has appropriate execution frequency
- Mode-specific risk management
- Clear separation of strategies
- LLM validation for each mode

Execution Schedule:
- Hybrid Opportunity Hunter: Every 5 minutes
- Hybrid Position Manager: Every 3 minutes
- Main loop: Every 30 seconds

Agents run sequentially to avoid conflicts during market hours (9:15 AM to 3:30 PM IST).
"""

import sys
import os
import time
import json
import logging
import argparse
from datetime import datetime, time as dt_time
from typing import Dict, Any

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from opportunity_hunter_hybrid_scalping import run_hybrid_scalping_opportunity_hunter
from position_manager_hybrid_scalping import run_hybrid_scalping_position_manager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'crew_driver_hybrid_scalping_{datetime.now().strftime("%Y%m%d")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants
MARKET_START_TIME = dt_time(9, 15)
MARKET_END_TIME = dt_time(15, 30)
IST_TIMEZONE = "Asia/Kolkata"

# Execution intervals (in seconds)
OPPORTUNITY_HUNTER_INTERVAL = 300  # 5 minutes
POSITION_MANAGER_INTERVAL = 180    # 3 minutes
MAIN_LOOP_INTERVAL = 30            # 30 seconds

def is_market_hours() -> bool:
    """Check if current time is within market hours"""
    current_time = datetime.now().time()
    return MARKET_START_TIME <= current_time <= MARKET_END_TIME

def get_intraday_trading_mode(current_time: dt_time) -> str:
    """
    Different modes for different times of day
    """
    if dt_time(9, 15) <= current_time <= dt_time(11, 30):
        return 'BOTH'  # Opening volatility - both modes possible
    elif dt_time(11, 30) <= current_time <= dt_time(14, 0):
        return 'PREMIUM_SELLING_ONLY'  # Mid-day calm - premium selling
    elif dt_time(14, 0) <= current_time <= dt_time(15, 20):
        return 'SCALPING_ONLY'  # Closing volatility - scalping only
    else:
        return 'NONE'  # No trading

def run_single_execution() -> Dict[str, Any]:
    """
    Run a single execution cycle of both agents
    """
    results = {}
    
    try:
        # Run hybrid opportunity hunter
        logger.info("=" * 80)
        logger.info("STARTING HYBRID OPPORTUNITY HUNTER AGENT EXECUTION")
        logger.info("=" * 80)
        
        opportunity_result = run_hybrid_scalping_opportunity_hunter()
        results['opportunity_hunter'] = opportunity_result
        
        logger.info("=" * 80)
        logger.info("HYBRID OPPORTUNITY HUNTER EXECUTION COMPLETED")
        logger.info("=" * 80)
        logger.info(f"Result: {opportunity_result}")
        logger.info("=" * 80)
        
        # Run hybrid position manager
        logger.info("=" * 80)
        logger.info("STARTING HYBRID POSITION MANAGER AGENT EXECUTION")
        logger.info("=" * 80)
        
        position_result = run_hybrid_scalping_position_manager()
        results['position_manager'] = position_result
        
        logger.info("=" * 80)
        logger.info("HYBRID POSITION MANAGER EXECUTION COMPLETED")
        logger.info("=" * 80)
        logger.info(f"Result: {position_result}")
        logger.info("=" * 80)
        
        return results
        
    except Exception as e:
        logger.error(f"Error in single execution: {str(e)}")
        return {'error': str(e)}

def run_continuous_execution() -> None:
    """
    Run continuous execution with proper intervals
    """
    logger.info("🚀 Starting HYBRID SCALPING CREW DRIVER...")
    logger.info("📋 Mode: Clear separation between scalping and premium selling")
    logger.info("📋 Opportunity Hunter: Every 5 minutes")
    logger.info("📋 Position Manager: Every 3 minutes")
    logger.info("⚡ NO MORE CONTRADICTIONS - Each mode optimized for its conditions")
    logger.info("⏱️  Token Efficiency: Optimal for all scenarios")
    
    last_opportunity_hunter = 0
    last_position_manager = 0
    
    while True:
        try:
            current_time = time.time()
            
            # Check if market is open
            if not is_market_hours():
                logger.info("Market closed. Waiting for next market session...")
                time.sleep(60)  # Sleep for 1 minute
                continue
            
            # Get current trading mode
            current_trading_mode = get_intraday_trading_mode(datetime.now().time())
            logger.info(f"Current trading mode: {current_trading_mode}")
            
            # Run opportunity hunter (every 5 minutes)
            if current_time - last_opportunity_hunter >= OPPORTUNITY_HUNTER_INTERVAL:
                logger.info("🔄 Executing hybrid opportunity hunter...")
                opportunity_result = run_hybrid_scalping_opportunity_hunter()
                last_opportunity_hunter = current_time
                
                logger.info(f"Opportunity Hunter Result: {opportunity_result.get('decision', 'UNKNOWN')}")
                logger.info(f"Mode: {opportunity_result.get('mode', 'UNKNOWN')}")
                logger.info(f"Strategy: {opportunity_result.get('strategy', 'NONE')}")
            
            # Run position manager (every 3 minutes)
            if current_time - last_position_manager >= POSITION_MANAGER_INTERVAL:
                logger.info("🔄 Executing hybrid position manager...")
                position_result = run_hybrid_scalping_position_manager()
                last_position_manager = current_time
                
                logger.info(f"Position Manager Result: {position_result.get('decision', 'UNKNOWN')}")
                logger.info(f"Positions: {position_result.get('nifty_positions_count', 0)}")
                logger.info(f"Exits: {position_result.get('exits_executed', 0)}")
            
            # Sleep for main loop interval
            time.sleep(MAIN_LOOP_INTERVAL)
            
        except KeyboardInterrupt:
            logger.info("🛑 Hybrid crew driver stopped by user")
            break
        except Exception as e:
            logger.error(f"Error in continuous execution: {str(e)}")
            time.sleep(MAIN_LOOP_INTERVAL)

def main():
    """
    Main function with command line argument parsing
    """
    parser = argparse.ArgumentParser(description='Hybrid Scalping Crew Driver')
    parser.add_argument('--single', action='store_true', help='Run single execution cycle')
    parser.add_argument('--test', action='store_true', help='Run in test mode')
    parser.add_argument('--continuous', action='store_true', help='Run continuous execution')
    
    args = parser.parse_args()
    
    if args.single:
        logger.info("Running single execution cycle...")
        result = run_single_execution()
        print(json.dumps(result, indent=2))
    elif args.test:
        logger.info("Running test execution...")
        result = run_single_execution()
        print(json.dumps(result, indent=2))
    elif args.continuous:
        logger.info("Running continuous execution...")
        run_continuous_execution()
    else:
        # Default to single execution
        logger.info("Running single execution cycle (default)...")
        result = run_single_execution()
        print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main() 