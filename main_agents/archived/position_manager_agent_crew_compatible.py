#!/usr/bin/env python3
"""
AlgoTrade - Ultra-Conservative Position Manager Agent (CREW-COMPATIBLE)
======================================================================

Crew-compatible version that returns analysis results for crew agents to process.
This version does NOT execute crew tasks directly - it provides analysis for the crew.

Author: AlgoTrade Team
Version: 2.0 (Ultra-Conservative + Crew-Compatible)
"""

# ============================================================================
# CREDENTIALS AND CONFIG
# ============================================================================

from dotenv import load_dotenv
import os
from datetime import datetime, timedelta, time as dt_time
import json
from pathlib import Path

# Load .env using relative path (consistent with opportunity hunter)
load_dotenv(dotenv_path='../.env')
print(f"[DEBUG] OPENAI_API_KEY loaded from env: {os.environ.get('OPENAI_API_KEY')}")
api_key = os.getenv("kite_api_key")
api_secret = os.getenv("kite_api_secret")
access_token = None
try:
    with open("../data/access_token.txt", "r") as f:
        access_token = f.read().strip()
        print(f"✅ Successfully loaded access token: {access_token[:10]}...")
except Exception as e:
    print(f"❌ Could not read ../data/access_token.txt: {e}")

# === LLM Model Selection ===
llm_model = "gemini/gemini-2.5-pro"  # Using Gemini for position management
anthropic_key = os.getenv("ANTHROPIC_API_KEY")
openai_key = os.getenv("OPENAI_API_KEY")
gemini_key = os.getenv("GEMINI_API_KEY")

# Set API key for provider
if llm_model.startswith("claude") or llm_model.startswith("anthropic"):
    if anthropic_key:
        os.environ["ANTHROPIC_API_KEY"] = anthropic_key
        print(f"Using Anthropic model: {llm_model}")
    else:
        print("Warning: ANTHROPIC_KEY not found in environment variables")
elif llm_model.startswith("gemini"):
    if gemini_key:
        os.environ["GOOGLE_API_KEY"] = gemini_key
        print(f"Using Google Gemini model: {llm_model}")
    else:
        print("Warning: GEMINI_API_KEY not found in environment variables")
else:
    if openai_key:
        os.environ["OPENAI_API_KEY"] = openai_key
        print(f"Using OpenAI model: {llm_model}")
    else:
        print("Warning: OPENAI_API_KEY not found in environment variables")

# Get current date and time
current_date = datetime.now().strftime('%Y-%m-%d')
current_time = datetime.now().strftime('%H:%M:%S')
current_datetime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

print(f"Starting Ultra-Conservative Position Manager (Crew-Compatible) at {current_datetime}...")

# ============================================================================
# TOOL IMPORTS
# ============================================================================

# Add parent directory to Python path for imports
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import all required tools from their appropriate modules
try:
    from core_tools.connect_data_tools import (
        get_nifty_spot_price_safe, debug_kite_instruments, get_nifty_instruments,
        fetch_historical_data, initialize_connection, get_options_chain_safe,
        get_nifty_expiry_dates, get_historical_volatility
    )
    print("✅ Successfully imported connect_data_tools")
except ImportError as e:
    print(f"❌ Warning: Could not import connect_data_tools: {e}")

try:
    from core_tools.master_indicators import (
        get_nifty_technical_analysis_tool, get_nifty_daily_technical_analysis_wrapper,
        calculate_pcr_technical_analysis_wrapper, analyze_pcr_extremes_wrapper
    )
    print("✅ Successfully imported master_indicators")
except ImportError as e:
    print(f"❌ Warning: Could not import master_indicators: {e}")

try:
    from core_tools.execution_portfolio_tools import (
        get_portfolio_positions, get_account_margins, get_orders_history,
        get_daily_trading_summary, get_risk_metrics, execute_options_strategy,
        validate_trading_capital, calculate_realistic_pricing, analyze_position_conflicts,
        validate_general_capital
    )
    print("✅ Successfully imported execution_portfolio_tools")
except ImportError as e:
    print(f"❌ Warning: Could not import execution_portfolio_tools: {e}")

try:
    from core_tools.calculate_analysis_tools import (
        calculate_option_greeks, calculate_implied_volatility, calculate_strategy_pnl, 
        calculate_portfolio_greeks, calculate_volatility_surface, 
        calculate_probability_of_profit, analyze_vix_integration_wrapper,
        calculate_iv_rank_analysis_wrapper, detect_market_regime_wrapper,
        calculate_pnl_percentage
    )
    print("✅ Successfully imported calculate_analysis_tools")
except ImportError as e:
    print(f"❌ Warning: Could not import calculate_analysis_tools: {e}")

try:
    from core_tools.trade_storage import (
        get_active_trades, update_trade_status, get_trade_history,
        write_successful_trade, get_trade_summary
    )
    print("✅ Successfully imported trade_storage")
except ImportError as e:
    print(f"❌ Warning: Could not import trade_storage: {e}")

# Initialize Kite Connect session
try:
    init_result = initialize_connection(api_key, api_secret, access_token)
    print("Kite Connect initialization result:", init_result)
except Exception as e:
    print(f"Warning: Kite Connect initialization failed: {e}")
    init_result = {'status': 'ERROR', 'message': str(e)}

# ============================================================================
# CREW-COMPATIBLE POSITION MANAGEMENT FUNCTIONS
# ============================================================================

def get_position_analysis():
    """
    Get comprehensive position analysis for crew processing
    """
    try:
        print("\n" + "="*80)
        print("🔍 POSITION ANALYSIS FOR CREW PROCESSING")
        print(f"📅 Current Time: {current_datetime}")
        print("="*80)
        
        # Get current positions
        positions_result = get_portfolio_positions()
        
        # DEBUG: Print raw positions result
        print(f"🔍 DEBUG: Raw positions result: {positions_result}")
        
        if positions_result.get('status') == 'SUCCESS':
            positions = positions_result.get('positions', [])
            print(f"🔍 DEBUG: Raw positions list: {positions}")
            nifty_positions = [p for p in positions if 'NIFTY' in p.get('tradingsymbol', '') or 'NIFTY' in p.get('symbol', '')]
            
            print(f"📊 Total Positions: {len(positions)}")
            print(f"📈 NIFTY Positions: {len(nifty_positions)}")
            
            if len(nifty_positions) == 0:
                print("✅ NO POSITIONS TO MANAGE - TASK COMPLETE")
                return {
                    'status': 'SUCCESS',
                    'decision': 'NO_POSITIONS',
                    'positions_count': 0,
                    'nifty_positions_count': 0,
                    'recommendation': 'No positions to manage - system idle',
                    'timestamp': datetime.now().isoformat()
                }
            
            # Get current spot price
            spot_result = get_nifty_spot_price_safe()
            current_spot = spot_result.get('spot_price', 0) if spot_result else 0
            
            # Analyze each position
            position_analysis = []
            total_pnl = 0
            total_exposure = 0
            
            for position in nifty_positions:
                symbol = position.get('tradingsymbol', '')
                quantity = position.get('quantity', 0)
                average_price = position.get('average_price', 0)
                last_price = position.get('last_price', 0)
                
                # Calculate P&L
                if quantity > 0:  # Long position
                    pnl = (last_price - average_price) * quantity
                else:  # Short position
                    pnl = (average_price - last_price) * abs(quantity)
                
                pnl_percentage = (pnl / (average_price * abs(quantity))) * 100 if average_price > 0 else 0
                
                # Calculate time metrics (simplified)
                entry_time = position.get('entry_time', current_datetime)
                minutes_since_entry = 0  # Simplified for crew compatibility
                minutes_to_close = 0  # Simplified for crew compatibility
                
                position_info = {
                    'symbol': symbol,
                    'quantity': quantity,
                    'average_price': average_price,
                    'last_price': last_price,
                    'pnl': pnl,
                    'pnl_percentage': pnl_percentage,
                    'entry_time': entry_time,
                    'minutes_since_entry': minutes_since_entry,
                    'minutes_to_close': minutes_to_close,
                    'exposure': abs(quantity * last_price)
                }
                
                position_analysis.append(position_info)
                total_pnl += pnl
                total_exposure += position_info['exposure']
                
                print(f"   📊 {symbol}: Qty={quantity}, P&L=₹{pnl:.2f} ({pnl_percentage:.2f}%)")
            
            # Determine management recommendation
            recommendation = "HOLD - Ultra-conservative management"
            decision = "HOLD_POSITIONS"
            
            # Check for emergency conditions
            for pos in position_analysis:
                if pos['pnl_percentage'] < -30:  # Catastrophic loss
                    recommendation = f"EMERGENCY EXIT - Catastrophic loss in {pos['symbol']}"
                    decision = "EMERGENCY_EXIT"
                    break
                elif pos['pnl_percentage'] < -20:  # Significant loss
                    recommendation = f"MONITOR CLOSELY - Significant loss in {pos['symbol']}"
                    decision = "MONITOR"
                    break
            
            # Check time-based conditions
            current_time_obj = datetime.now().time()
            close_time = dt_time(15, 20)  # 3:20 PM
            
            if current_time_obj >= close_time:
                recommendation = "MARKET CLOSE - Positions will auto square off"
                decision = "AUTO_SQUARE_OFF"
            
            result = {
                'status': 'SUCCESS',
                'decision': decision,
                'positions_count': len(positions),
                'nifty_positions_count': len(nifty_positions),
                'total_pnl': total_pnl,
                'total_exposure': total_exposure,
                'current_spot_price': current_spot,
                'position_analysis': position_analysis,
                'recommendation': recommendation,
                'management_philosophy': 'Ultra-conservative: Hold for time decay, exit only for emergencies',
                'timestamp': datetime.now().isoformat()
            }
            
            print(f"\n📋 POSITION MANAGEMENT SUMMARY:")
            print(f"   Decision: {decision}")
            print(f"   Total P&L: ₹{total_pnl:.2f}")
            print(f"   Total Exposure: ₹{total_exposure:.2f}")
            print(f"   Recommendation: {recommendation}")
            print("="*80)
            
            return result
            
        else:
            print(f"❌ Failed to get positions: {positions_result.get('message', 'Unknown error')}")
            return {
                'status': 'ERROR',
                'decision': 'ERROR',
                'error': positions_result.get('message', 'Failed to get positions'),
                'recommendation': 'Error in position analysis',
                'timestamp': datetime.now().isoformat()
            }
            
    except Exception as e:
        print(f"❌ Position analysis failed: {str(e)}")
        return {
            'status': 'ERROR',
            'decision': 'ERROR',
            'error': str(e),
            'recommendation': 'Error in position analysis',
            'timestamp': datetime.now().isoformat()
        }

def run_ultra_conservative_position_manager_crew_compatible():
    """
    Crew-compatible version of ultra-conservative position manager
    Returns analysis results instead of running crew tasks
    """
    try:
        print("\n" + "="*80)
        print("🎯 ULTRA-CONSERVATIVE POSITION MANAGER (CREW-COMPATIBLE)")
        print(f"📅 Current Time: {current_datetime}")
        print("🔄 Mission: Analyze positions for crew processing")
        print("="*80)
        
        # Get position analysis
        analysis_result = get_position_analysis()
        
        # Create summary for crew
        if analysis_result.get('status') == 'SUCCESS':
            if analysis_result.get('decision') == 'NO_POSITIONS':
                summary = f"""
# POSITION MANAGER ANALYSIS - {current_datetime}

## DECISION: NO POSITIONS TO MANAGE

## ANALYSIS SUMMARY
- **Positions Found**: 0
- **NIFTY Positions**: 0
- **Status**: System idle - no positions to manage
- **Recommendation**: No action required

## ULTRA-CONSERVATIVE PHILOSOPHY
- Default Action: HOLD positions for time decay
- Emergency Exits: Only when absolutely necessary
- Cost-Aware: Every exit economically justified
- Time Decay: Primary source of F&O profits
- Capital Protection: Through patience, not premature action

## CONCLUSION
No positions to manage. System is idle and ready for new opportunities.
"""
            else:
                positions_count = analysis_result.get('nifty_positions_count', 0)
                total_pnl = analysis_result.get('total_pnl', 0)
                recommendation = analysis_result.get('recommendation', 'Unknown')
                
                summary = f"""
# POSITION MANAGER ANALYSIS - {current_datetime}

## DECISION: {analysis_result.get('decision', 'UNKNOWN')}

## ANALYSIS SUMMARY
- **Total Positions**: {analysis_result.get('positions_count', 0)}
- **NIFTY Positions**: {positions_count}
- **Total P&L**: ₹{total_pnl:.2f}
- **Total Exposure**: ₹{analysis_result.get('total_exposure', 0):.2f}
- **Current Spot**: ₹{analysis_result.get('current_spot_price', 0):.2f}
- **Recommendation**: {recommendation}

## POSITION DETAILS
"""
                
                for pos in analysis_result.get('position_analysis', []):
                    summary += f"""
### {pos['symbol']}
- **Quantity**: {pos['quantity']}
- **Average Price**: ₹{pos['average_price']:.2f}
- **Last Price**: ₹{pos['last_price']:.2f}
- **P&L**: ₹{pos['pnl']:.2f} ({pos['pnl_percentage']:.2f}%)
- **Exposure**: ₹{pos['exposure']:.2f}
"""
                
                summary += f"""
## ULTRA-CONSERVATIVE PHILOSOPHY
- Default Action: HOLD positions for time decay
- Emergency Exits: Only when absolutely necessary
- Cost-Aware: Every exit economically justified
- Time Decay: Primary source of F&O profits
- Capital Protection: Through patience, not premature action

## CONCLUSION
{recommendation}
"""
            
            print(summary)
            
            return {
                'status': 'SUCCESS',
                'analysis': analysis_result,
                'summary': summary,
                'final_recommendation': analysis_result.get('recommendation', 'Unknown'),
                'timestamp': datetime.now().isoformat()
            }
            
        else:
            error_summary = f"""
# POSITION MANAGER ANALYSIS - {current_datetime}

## DECISION: ERROR

## ERROR DETAILS
- **Error**: {analysis_result.get('error', 'Unknown error')}
- **Status**: {analysis_result.get('status', 'ERROR')}
- **Recommendation**: {analysis_result.get('recommendation', 'Error in analysis')}

## CONCLUSION
Position analysis failed. Manual intervention may be required.
"""
            
            print(error_summary)
            
            return {
                'status': 'ERROR',
                'analysis': analysis_result,
                'summary': error_summary,
                'final_recommendation': 'Error in position analysis',
                'timestamp': datetime.now().isoformat()
            }
            
    except Exception as e:
        print(f"❌ Ultra-conservative position manager failed: {str(e)}")
        return {
            'status': 'ERROR',
            'error': str(e),
            'final_recommendation': 'Error in position manager',
            'timestamp': datetime.now().isoformat()
        }

# ============================================================================
# MAIN EXECUTION - CREW-COMPATIBLE
# ============================================================================

if __name__ == "__main__":
    print("🚀 Starting Ultra-Conservative Position Manager (Crew-Compatible)...")
    print("📋 Mission: Provide position analysis for crew processing")
    print("📋 Note: This version does NOT execute crew tasks directly")
    print("💰 Philosophy: Ultra-conservative position management")
    print("⚡ Features: Position analysis + P&L tracking")
    print("⏱️  Token Efficiency: Optimal for crew processing")
    print("-" * 50)

    # Run position manager analysis
    result = run_ultra_conservative_position_manager_crew_compatible()
    
    print("\n" + "="*80)
    print(f"🏁 ULTRA-CONSERVATIVE POSITION MANAGER (CREW-COMPATIBLE) COMPLETED - {current_datetime}")
    print("="*80)
    print(f"📊 Final Status: {result.get('status', 'UNKNOWN')}")
    print(f"💡 Recommendation: {result.get('final_recommendation', 'No recommendation')}")
    print("💰 Token Efficiency: Optimal for crew processing")
    print("="*80) 