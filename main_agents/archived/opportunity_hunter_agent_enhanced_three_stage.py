#!/usr/bin/env python3
"""
AlgoTrade - Opportunity Hunter Agent (SIMPLIFIED)
=================================================

Specialized agent for finding and executing NEW NIFTY F&O trading opportunities ONLY.
This agent assumes Position Manager has already cleaned the portfolio.

Author: AlgoTrade Team
"""

# ============================================================================
# CREDENTIALS AND CONFIG (Same as before)
# ============================================================================

from dotenv import load_dotenv
import os
from datetime import datetime

# Load credentials
load_dotenv(dotenv_path='../.env')
api_key = os.getenv("kite_api_key")
api_secret = os.getenv("kite_api_secret")
access_token = None
try:
    with open("../data/access_token.txt", "r") as f:
        access_token = f.read().strip()
except Exception as e:
    print("Could not read ../data/access_token.txt:", e)

# === LLM Model Selection ===
# Add these to your .env:
# LLM_MODEL= (e.g. gpt-4o, gpt-4, claude-3-sonnet-20240229, gemini/gemini-2.5-pro, etc)
# ANTHROPIC_API_KEY= (your anthropic API key, if using Anthropic)
# GEMINI_API_KEY= (your google gemini API key, if using Gemini)
llm_model = os.getenv("LLM_MODEL", "gpt-4o")
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

print(f"Starting Opportunity Hunter at {current_datetime}...")

# ============================================================================
# TOOL IMPORTS (Reduced set)
# ============================================================================

from crewai import Agent, Task, Crew
from crewai.tools import tool
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# === LLM Provider Selection for CrewAI ===
# CrewAI uses LLM class for model configuration
llm_kwargs = {}
try:
    from crewai import LLM
    if llm_model.startswith("claude") or llm_model.startswith("anthropic"):
        # For Anthropic models, use the format: anthropic/claude-3-sonnet-20240229
        if not llm_model.startswith("anthropic/"):
            llm_model = f"anthropic/{llm_model}"
        llm_kwargs = {"llm": LLM(model=llm_model, temperature=0.7)}
        print(f"✅ Configured Anthropic LLM: {llm_model}")
    elif llm_model.startswith("gemini"):
        # For Google Gemini models, use the format: gemini/gemini-2.5-pro
        llm_kwargs = {"llm": LLM(model=llm_model, temperature=0.7)}
        print(f"✅ Configured Google Gemini LLM: {llm_model}")
    else:
        # For OpenAI models
        llm_kwargs = {"llm": LLM(model=llm_model, temperature=0.7)}
        print(f"✅ Configured OpenAI LLM: {llm_model}")
except ImportError:
    print("crewAI.LLM not available, using default LLM configuration.")
    llm_kwargs = {}

# Import only tools needed for NEW opportunities
try:
    from core_tools.connect_data_tools import (
        get_nifty_spot_price_safe, get_nifty_expiry_dates, get_options_chain_safe, 
        get_historical_volatility, analyze_options_flow_safe, initialize_connection, 
        get_nifty_instruments, get_global_market_conditions, debug_kite_instruments,
        fetch_historical_data, get_available_expiry_dates_with_analysis
    )
    from core_tools.calculate_analysis_tools import (
        calculate_option_greeks, calculate_implied_volatility, calculate_strategy_pnl, 
        find_arbitrage_opportunities, calculate_probability_of_profit,
        calculate_portfolio_greeks, calculate_volatility_surface,
        # NEW ADVANCED ANALYSIS TOOLS
        analyze_vix_integration_wrapper, calculate_iv_rank_analysis_wrapper, detect_market_regime_wrapper,
        # ENHANCED IV ANALYSIS WITH REALIZED VOLATILITY AND LIQUIDITY
        calculate_realized_volatility, get_realized_volatility_from_kite, analyze_options_liquidity
    )
    from core_tools.execution_portfolio_tools import (
        get_portfolio_positions, get_account_margins, execute_options_strategy, 
        calculate_strategy_margins, analyze_position_conflicts, analyze_position_conflicts_wrapper, validate_trading_capital,
        get_risk_metrics, get_orders_history, get_daily_trading_summary,
        validate_general_capital, execute_and_store_strategy
    )
    from core_tools.strategy_creation_tools import (
        create_long_straddle_strategy, create_short_strangle_strategy, 
        create_iron_condor_strategy, create_butterfly_spread_strategy, 
        # NEW PREMIUM SELLING STRATEGIES
        create_bull_put_spread_strategy, create_bear_call_spread_strategy, create_calendar_spread_strategy,
        recommend_options_strategy, analyze_strategy_greeks,
        # NEW COMPREHENSIVE ANALYSIS TOOL
        comprehensive_advanced_analysis_wrapper,
        # CREWAI-COMPATIBLE WRAPPER FUNCTIONS
        create_long_straddle_wrapper, create_short_strangle_wrapper, 
        create_iron_condor_wrapper, create_butterfly_spread_wrapper,
        create_bull_put_spread_wrapper, create_bear_call_spread_wrapper, create_calendar_spread_wrapper
    )
    from core_tools.master_indicators import (
        get_nifty_technical_analysis_tool, get_nifty_daily_technical_analysis_wrapper,
        # NEW PCR ANALYSIS TOOLS
        calculate_pcr_technical_analysis_wrapper, analyze_pcr_extremes_wrapper
    )
    from core_tools.trade_storage import write_successful_trade
    print("✅ Successfully imported all tools")
except ImportError as e:
    print(f"❌ CRITICAL ERROR: Could not import required tools: {e}")
    print("This usually means:")
    print("1. Missing dependencies (kiteconnect, pandas, numpy, etc.)")
    print("2. Incorrect file paths")
    print("3. Missing core_tools modules")
    print("\nPlease ensure all dependencies are installed:")
    print("pip install kiteconnect pandas numpy scipy yfinance ta-lib")
    print("\nAnd verify the core_tools directory structure is correct.")
    raise ImportError(f"Cannot proceed without required tools: {e}")

# Initialize connection
try:
    init_result = initialize_connection(api_key, api_secret, access_token)
    print("Kite Connect initialization result:", init_result)
except Exception as e:
    print(f"Warning: Kite Connect initialization failed: {e}")

# ============================================================================
# ENHANCED IV ANALYSIS INTEGRATION
# ============================================================================

# Initialize IV Data Manager for enhanced analysis
try:
    from iv_data_manager import IVDataManager
    iv_manager = IVDataManager()
    print("✅ IV Data Manager initialized for enhanced analysis")
except Exception as e:
    print(f"⚠️  IV Data Manager initialization failed: {e}")
    iv_manager = None

# Add enhanced IV analysis tools to imports
try:
    from calculate_analysis_tools import (
        calculate_comprehensive_volatility_surface, analyze_volatility_regime
    )
    print("✅ Enhanced IV analysis tools imported")
except Exception as e:
    print(f"⚠️  Enhanced IV analysis tools import failed: {e}")

# === FAST-TRACK ANALYSIS & STRATEGY SELECTION IMPROVEMENTS ===

def refined_emergency_strategy_selection(emergency_signal_type):
    """
    Refined emergency strategy selection - clean and practical approach
    """
    try:
        print(f"🎯 REFINED STRATEGY SELECTION:")
        print(f"   Emergency Signal: {emergency_signal_type}")
        
        # COMMENTED OUT: Low IV emergency logic removed - Low IV is common and not truly "emergency"
        # if 'LOW_IV' in emergency_signal_type:
        #     # LOW IV = Cheap options = Take advantage with long strategies
        #     technical = get_nifty_technical_analysis_tool()
        #     signal = technical.get('signal', 'NEUTRAL')
        #     rsi = technical.get('rsi', 50)
        #     
        #     print(f"   Technical Signal: {signal} (RSI: {rsi:.1f})")
        #     
        #     # Strong directional signals with confirmation
        #     if signal == 'STRONG_BUY' and rsi < 70:  # Not overbought
        #         strategy = 'Long Call'
        #         reasoning = "Low IV + Strong bullish signal = Long Call"
        #         
        #     elif signal == 'STRONG_SELL' and rsi > 30:  # Not oversold
        #         strategy = 'Long Put'  
        #         reasoning = "Low IV + Strong bearish signal = Long Put"
        #         
        #     else:
        #         # No clear direction or conflicting signals
        #         strategy = 'Long Straddle'
        #         reasoning = "Low IV + Neutral/weak signal = Long Straddle"
        #         
        #     print(f"   ✅ LOW IV Strategy: {strategy}")
        #     print(f"   💡 Reasoning: {reasoning}")
            
        if 'HIGH_IV' in emergency_signal_type:
            # HIGH IV = Expensive options = Prefer safe hedged selling
            strategy = 'Iron Condor'
            reasoning = "High IV + Emergency = Conservative hedged selling (Iron Condor)"
            
            print(f"   ✅ HIGH IV Strategy: {strategy}")
            print(f"   💡 Reasoning: {reasoning}")
            
        else:
            # Unknown signal or mixed conditions = Conservative default
            strategy = 'Iron Condor'
            reasoning = "Unknown/mixed conditions = Conservative default (Iron Condor)"
            
            print(f"   ✅ DEFAULT Strategy: {strategy}")
            print(f"   💡 Reasoning: {reasoning}")
        
        return {
            'strategy': strategy,
            'reasoning': reasoning,
            'signal_type': emergency_signal_type
        }
        
    except Exception as e:
        print(f"❌ Error in strategy selection: {e}")
        # Ultra-safe fallback
        return {
            'strategy': 'Iron Condor',
            'reasoning': 'Error occurred - using ultra-safe fallback',
            'signal_type': 'ERROR'
        }


# COMMENTED OUT: Low IV emergency functions removed - Low IV is common and not truly "emergency"
# def create_long_call_emergency(expiry_date, spot_price, quantity=75):
#     """
#     Create Long Call for emergency execution in low IV environment
#     """
#     try:
#         options_chain = get_options_chain_safe(expiry_date=expiry_date)
#         if not options_chain or options_chain.get('status') != 'SUCCESS':
#             return {'status': 'FAILED', 'reason': 'Could not get options chain'}
#         chain_data = options_chain.get('options_chain', [])
#         target_strike = None
#         for option in chain_data:
#             strike = option.get('strike', 0)
#             if strike > spot_price + 50 and strike <= spot_price + 100:
#                 if option.get('CE_ltp', 0) > 0:
#                     target_strike = strike
#                     break
#         if not target_strike:
#             return {'status': 'FAILED', 'reason': 'Could not find suitable call option'}
#         call_option = next((opt for opt in chain_data if opt['strike'] == target_strike), None)
#         if not call_option:
#             return {'status': 'FAILED', 'reason': 'Target strike not found'}
#         legs = [{
#             'symbol': call_option.get('CE_symbol', ''),
#             'action': 'BUY',
#             'quantity': quantity,
#             'strike': target_strike,
#             'option_type': 'CE',
#             'price': call_option.get('CE_ltp', 0)
#         }]
#         return {
#             'status': 'SUCCESS',
#             'strategy_name': 'Long Call',
#             'legs': legs,
#             'expiry_date': expiry_date,
#             'target_strike': target_strike
#         }
#     except Exception as e:
#         return {'status': 'FAILED', 'reason': f'Long Call creation failed: {str(e)}'}


# def create_long_put_emergency(expiry_date, spot_price, quantity=75):
#     """
#     Create Long Put for emergency execution in low IV environment
#     """
#     try:
#         options_chain = get_options_chain_safe(expiry_date=expiry_date)
#         if not options_chain or options_chain.get('status') != 'SUCCESS':
#             return {'status': 'FAILED', 'reason': 'Could not get options chain'}
#         chain_data = options_chain.get('options_chain', [])
#         target_strike = None
#         for option in reversed(chain_data):
#             strike = option.get('strike', 0)
#             if strike < spot_price - 50 and strike >= spot_price - 100:
#                 if option.get('PE_ltp', 0) > 0:
#                     target_strike = strike
#                     break
#         if not target_strike:
#             return {'status': 'FAILED', 'reason': 'Could not find suitable put option'}
#         put_option = next((opt for opt in chain_data if opt['strike'] == target_strike), None)
#         if not put_option:
#             return {'status': 'FAILED', 'reason': 'Target strike not found'}
#         legs = [{
#             'symbol': put_option.get('PE_symbol', ''),
#             'action': 'BUY',
#             'quantity': quantity,
#             'strike': target_strike,
#             'option_type': 'PE',
#             'price': put_option.get('PE_ltp', 0)
#         }]
#         return {
#             'status': 'SUCCESS',
#             'strategy_name': 'Long Put',
#             'legs': legs,
#             'expiry_date': expiry_date,
#             'target_strike': target_strike
#         }
#     except Exception as e:
#         return {'status': 'FAILED', 'reason': f'Long Put creation failed: {str(e)}'}

def emergency_fast_track():
    """
    Enhanced emergency fast-track with comprehensive IV analysis including realized volatility and liquidity
    """
    try:
        # Import required modules for timezone handling
        from datetime import datetime
        import pytz
        IST_TIMEZONE = pytz.timezone('Asia/Kolkata')
        
        print("🚨 ENHANCED EMERGENCY FAST-TRACK WITH REALIZED VOLATILITY & LIQUIDITY:")
        
        # Get basic market data
        spot_price = get_nifty_spot_price_safe()
        if not spot_price:
            return {'status': 'ERROR', 'message': 'Cannot get spot price'}
        
        options_chain = get_options_chain_safe()
        if not options_chain:
            return {'status': 'ERROR', 'message': 'Cannot get options chain'}
        
        # Get realized volatility
        realized_vol = get_realized_volatility_from_kite()
        
        # Get PCR analysis for market sentiment validation
        pcr_analysis = calculate_pcr_technical_analysis_wrapper()
        pcr_extremes = analyze_pcr_extremes_wrapper()
        print(f"📊 PCR Technical: {pcr_analysis.get('entry_signal', 'N/A')} (PCR: {pcr_analysis.get('put_call_ratio', 'N/A')})")
        print(f"📊 PCR Extremes: {pcr_extremes.get('extreme_signal', 'N/A')}")
        
        # Enhanced IV analysis with realized volatility and liquidity
        iv_analysis = calculate_iv_rank_analysis_wrapper()
        
        if iv_analysis.get('status') == 'SUCCESS':
            current_iv = iv_analysis.get('current_iv', 0)
            iv_percentile = iv_analysis.get('iv_percentile', 0)
            iv_validation = iv_analysis.get('iv_validation', {})
            liquidity_analysis = iv_analysis.get('liquidity_analysis', {})
            
            print(f"📊 ENHANCED IV ANALYSIS:")
            print(f"   Current IV: {current_iv:.4f}")
            print(f"   IV Percentile: {iv_percentile:.1%}")
            
            if realized_vol:
                print(f"   Realized Vol: {realized_vol:.4f}")
                iv_ratio = current_iv / realized_vol if realized_vol > 0 else 0
                print(f"   IV/Realized Ratio: {iv_ratio:.2f}")
            
            # Check liquidity
            if liquidity_analysis.get('status') == 'SUCCESS':
                liquidity_score = liquidity_analysis.get('liquidity_metrics', {}).get('overall_liquidity_score', 0)
                liquidity_status = liquidity_analysis.get('liquidity_metrics', {}).get('liquidity_status', 'UNKNOWN')
                print(f"   Liquidity Score: {liquidity_score}/100 ({liquidity_status})")
            
            # Enhanced decision logic
            emergency_signal = 'NO_EMERGENCY'
            
            # Check for high IV emergency (more conservative thresholds)
            if current_iv > 0.30 and iv_percentile > 0.8:  # More conservative
                if realized_vol and current_iv / realized_vol > 1.5:  # More stringent overpriced check
                    emergency_signal = 'HIGH_IV_OVERPRICED'
                    print(f"🚨 HIGH IV EMERGENCY: IV={current_iv:.4f}, Overpriced vs Realized")
                else:
                    emergency_signal = 'HIGH_IV_FAIR'
                    print(f"🚨 HIGH IV EMERGENCY: IV={current_iv:.4f}, Fairly Priced")
            
            # Check for low IV opportunity (more conservative thresholds)
            # COMMENTED OUT: Low IV is common and not truly "emergency"
            # elif current_iv < 0.08 and iv_percentile < 0.15:  # Much more conservative
            #     if realized_vol and current_iv / realized_vol < 0.6:  # More stringent underpriced check
            #         emergency_signal = 'LOW_IV_UNDERPRICED'
            #         print(f"🚨 LOW IV OPPORTUNITY: IV={current_iv:.4f}, Underpriced vs Realized")
            #     else:
            #         emergency_signal = 'LOW_IV_FAIR'
            #         print(f"🚨 LOW IV OPPORTUNITY: IV={current_iv:.4f}, Fairly Priced")
            
            # Check liquidity constraints
            if liquidity_analysis.get('status') == 'SUCCESS':
                liquidity_score = liquidity_analysis.get('liquidity_metrics', {}).get('overall_liquidity_score', 0)
                if liquidity_score < 40:  # Poor liquidity
                    if emergency_signal != 'NO_EMERGENCY':
                        emergency_signal = f"{emergency_signal}_POOR_LIQUIDITY"
                        print(f"⚠️  POOR LIQUIDITY: Score={liquidity_score}/100")
                    else:
                        emergency_signal = 'POOR_LIQUIDITY'
                        print(f"⚠️  POOR LIQUIDITY ONLY: Score={liquidity_score}/100")
            
            # PCR confirmation logic
            pcr_signal = pcr_analysis.get('entry_signal', '').upper()
            pcr_extreme = (pcr_extremes.get('extreme_signal', '') or '').upper()
            if emergency_signal.startswith('HIGH_IV'):
                if pcr_signal == 'STRONG_BUY' or pcr_extreme == 'EXTREME_BULL':
                    print(f"❌ PCR BLOCK: HIGH_IV emergency blocked due to bullish PCR ({pcr_signal}, {pcr_extreme})")
                    emergency_signal = 'NO_EMERGENCY_PCR_BLOCKED'
            # COMMENTED OUT: Low IV emergency logic removed
            # elif emergency_signal.startswith('LOW_IV'):
            #     if pcr_signal == 'STRONG_SELL' or pcr_extreme == 'EXTREME_BEAR':
            #         print(f"❌ PCR BLOCK: LOW_IV emergency blocked due to bearish PCR ({pcr_signal}, {pcr_extreme})")
            #         emergency_signal = 'NO_EMERGENCY_PCR_BLOCKED'
            
            return {
                'status': 'SUCCESS',
                'emergency_signal': emergency_signal,
                'current_iv': current_iv,
                'iv_percentile': iv_percentile,
                'realized_volatility': realized_vol,
                'liquidity_score': liquidity_analysis.get('liquidity_metrics', {}).get('overall_liquidity_score', 0) if liquidity_analysis.get('status') == 'SUCCESS' else None,
                'spot_price': spot_price,
                'pcr_signal': pcr_analysis.get('entry_signal', None),
                'pcr_value': pcr_analysis.get('put_call_ratio', None),
                'pcr_extreme_signal': pcr_extremes.get('extreme_signal', None),
                'timestamp': datetime.now(IST_TIMEZONE).isoformat()
            }
        else:
            print(f"❌ IV Analysis failed: {iv_analysis.get('message', 'Unknown error')}")
            return {'status': 'ERROR', 'message': 'IV analysis failed'}
            
    except Exception as e:
        print(f"❌ Emergency fast-track failed: {str(e)}")
        return {'status': 'ERROR', 'message': f'Emergency fast-track failed: {str(e)}'}

def run_emergency_execution(emergency_signal):
    """
    Stage 2: Emergency execution for high IV/time-sensitive opportunities
    
    Args:
        emergency_signal (dict): Emergency signal from fast-track
        
    Returns:
        dict: Execution result
    """
    print("\n" + "="*80)
    print("🚨 EMERGENCY EXECUTION MODE")
    print(f"📅 Current Time: {current_datetime}")
    print(f"🎯 Emergency Signal: {emergency_signal.get('emergency_signal', 'Unknown')}")
    print(f"📊 Current IV: {emergency_signal.get('current_iv', 'Unknown')}")
    print(f"📈 IV Percentile: {emergency_signal.get('iv_percentile', 'Unknown')}")
    print(f"💧 Liquidity Score: {emergency_signal.get('liquidity_score', 'Unknown')}")
    print("="*80)
    
    # Check market hours before execution
    try:
        from datetime import datetime, time as dt_time
        import pytz
        
        IST_TIMEZONE = pytz.timezone('Asia/Kolkata')
        MARKET_START_TIME = dt_time(9, 15)  # 9:15 AM
        MARKET_END_TIME = dt_time(15, 30)   # 3:30 PM
        
        ist_now = datetime.now(IST_TIMEZONE)
        current_time = ist_now.time()
        is_weekday = ist_now.weekday() < 5
        is_market_hours = MARKET_START_TIME <= current_time <= MARKET_END_TIME
        market_open = is_weekday and is_market_hours
        
        if not market_open:
            print(f"❌ Market is closed. Current time: {ist_now.strftime('%Y-%m-%d %H:%M:%S %Z')}")
            print(f"❌ Market hours: {MARKET_START_TIME.strftime('%H:%M')} - {MARKET_END_TIME.strftime('%H:%M')} IST")
            return {
                'decision': 'EMERGENCY_EXECUTION_FAILED',
                'reason': 'Market is closed - cannot place orders',
                'fallback': 'WAIT - Market closed'
            }
        else:
            print(f"✅ Market is open. Current time: {ist_now.strftime('%Y-%m-%d %H:%M:%S %Z')}")
            
    except Exception as e:
        print(f"⚠️  Could not check market hours: {e}. Proceeding with caution.")
    
    try:
        # Get essential data for emergency execution
        spot_price = get_nifty_spot_price_safe()
        # Force-fix: ensure spot_price is a float
        if isinstance(spot_price, dict):
            spot_price = spot_price.get('spot_price', 0)
        expiry_analysis = get_available_expiry_dates_with_analysis()
        
        # Debug expiry analysis
        print(f"📅 Expiry Analysis: {expiry_analysis}")
        if expiry_analysis and 'available_expiries' in expiry_analysis:
            print(f"📅 Available Expiries: {len(expiry_analysis['available_expiries'])}")
            for i, expiry in enumerate(expiry_analysis['available_expiries'][:3]):  # Show first 3
                print(f"   {i+1}. {expiry.get('date', 'N/A')} - {expiry.get('category', 'N/A')}")
        else:
            print("❌ No expiry analysis data available")
        
        chosen_expiry = select_optimal_expiry(expiry_analysis)
        print(f"🎯 Chosen Expiry: {chosen_expiry}")
        
        if not chosen_expiry:
            return {
                'decision': 'EMERGENCY_EXECUTION_FAILED',
                'reason': 'No suitable expiry found',
                'fallback': 'WAIT'
            }
        
        # Get options chain
        options_chain = get_options_chain_safe(expiry_date=chosen_expiry)
        
        emergency_signal_type = emergency_signal.get('emergency_signal', 'UNKNOWN')
        print(f"🔍 DEBUG: Emergency signal type = {emergency_signal_type}")
        print(f"🔍 DEBUG: Full emergency signal = {emergency_signal}")
        
        # Conservative parameters for emergency execution
        if emergency_signal.get('emergency_level', 'MEDIUM') == 'HIGH':
            risk_per_trade = 0.5  # 0.5% of capital
            max_width = 200  # Narrow spreads
        else:
            risk_per_trade = 1.0  # 1% of capital
            max_width = 300  # Standard spreads
        
        # Execute strategy with refined logic
        execution_result = execute_emergency_strategy(
            strategy=None,  # Not used anymore
            spot_price=spot_price,
            expiry_date=chosen_expiry,
            options_chain=options_chain,
            risk_per_trade=risk_per_trade,
            max_width=max_width,
            emergency_level=emergency_signal.get('emergency_level', 'MEDIUM'),
            emergency_signal_type=emergency_signal_type
        )
        
        # Check if execution was successful (handle multiple success statuses)
        execution_status = execution_result.get('status', 'UNKNOWN')
        success_statuses = ['SUCCESS', 'BASKET_SUCCESS', 'PARTIAL_SUCCESS', 'EXECUTED']
        if execution_status in success_statuses:
            decision = 'EMERGENCY_EXECUTION_COMPLETE'
        else:
            decision = 'EMERGENCY_EXECUTION_FAILED'
        
        return {
            'decision': decision,
            'strategy': execution_result.get('strategy_name', 'Unknown'),
            'emergency_level': emergency_signal.get('emergency_level', 'MEDIUM'),
            'execution_result': execution_result,
            'reason': emergency_signal.get('reason', 'Emergency conditions detected'),
            'token_efficiency': 'HIGH - Emergency execution used ~300 tokens'
        }
        
    except Exception as e:
        print(f"❌ Emergency execution failed: {e}")
        return {
            'decision': 'EMERGENCY_EXECUTION_FAILED',
            'error': str(e),
            'fallback': 'WAIT - Emergency execution failed'
        }

def execute_emergency_strategy(strategy, spot_price, expiry_date, options_chain, risk_per_trade, max_width, emergency_level, emergency_signal_type):
    """
    Execute emergency strategy with refined selection logic
    """
    try:
        # === STRICT TIME CHECK: NO TRADES AFTER 2:30 PM IST ===
        from datetime import datetime, time as dt_time
        import pytz
        IST_TIMEZONE = pytz.timezone('Asia/Kolkata')
        now = datetime.now(IST_TIMEZONE)
        if now.time() >= dt_time(14, 30):
            print("\n⏰ Trade execution blocked: No new trades allowed after 2:30 PM IST.")
            return {'status': 'FAILED', 'reason': 'Trade execution blocked: No new trades allowed after 2:30 PM IST.'}
        
        # === PROFIT TARGET CHECK: BLOCK NEW TRADES IF TARGET REACHED ===
        import os
        profit_target_file = "/tmp/algotrade_no_more_trades_today"
        if os.path.exists(profit_target_file):
            print("\n🚫 PROFIT TARGET BLOCK: Day's profit target (₹5,000) already reached. No new trades allowed.")
            return {'status': 'FAILED', 'reason': 'Day profit target (₹5,000) reached - no new trades allowed'}
        
        # Use refined strategy selection
        strategy_info = refined_emergency_strategy_selection(emergency_signal_type)
        selected_strategy = strategy_info['strategy']
        reasoning = strategy_info['reasoning']
        print(f"🚨 Executing refined emergency strategy: {selected_strategy}")
        print(f"💡 Selection reasoning: {reasoning}")
        if isinstance(spot_price, dict):
            spot_price = spot_price.get('spot_price', 0)
        margins = get_account_margins()
        if margins.get('status') != 'SUCCESS':
            return {'status': 'FAILED', 'reason': 'Could not fetch account margins'}
        capital = margins['equity'].get('live_balance', 0)
        if capital < 50000:
            return {'status': 'FAILED', 'reason': 'Insufficient capital'}
        # COMMENTED OUT: Low IV emergency strategies removed
        # if selected_strategy == 'Long Call':
        #     strategy_result = create_long_call_emergency(
        #         expiry_date=expiry_date,
        #         spot_price=spot_price,
        #         quantity=75
        #     )
        # elif selected_strategy == 'Long Put':
        #     strategy_result = create_long_put_emergency(
        #         expiry_date=expiry_date,
        #         spot_price=spot_price,
        #         quantity=75
        #     )
        # elif selected_strategy == 'Long Straddle':
        if selected_strategy == 'Long Straddle':
            strategy_result = create_long_straddle_wrapper(
                expiry_date=expiry_date,
                expiry_type='weekly',
                quantity=75
            )
        elif selected_strategy == 'Iron Condor':
            strategy_result = create_iron_condor_wrapper(
                expiry_date=expiry_date,
                expiry_type='weekly',
                wing_width=100,
                quantity=75
            )
        else:
            return {'status': 'FAILED', 'reason': f'Unknown strategy: {selected_strategy}'}
        if strategy_result.get('status') == 'SUCCESS':
            from core_tools.execution_portfolio_tools import execute_and_store_strategy
            strategy_result.update({
                'emergency_level': emergency_level,
                'emergency_signal': emergency_signal_type,
                'execution_time': datetime.now(IST_TIMEZONE).isoformat(),
                'strategy_reasoning': reasoning,
                'selection_logic': 'refined_emergency'
            })
            result = execute_and_store_strategy(
                strategy_legs=strategy_result.get('legs', []),
                trade_metadata=strategy_result
            )
        else:
            result = strategy_result
        return result
    except Exception as e:
        print(f"❌ Emergency strategy execution failed: {e}")
        return {'status': 'FAILED', 'reason': f'Execution error: {str(e)}'}

def quick_opportunity_check():
    """
    Stage 1: Quick opportunity check with realized volatility and liquidity analysis
    """
    try:
        print("\n" + "="*80)
        print("🔍 QUICK OPPORTUNITY CHECK WITH REALIZED VOLATILITY & LIQUIDITY (STAGE 1)")
        print(f"📅 Current Time: {current_datetime}")
        print("="*80)
        
        # Get basic market data
        spot_price = get_nifty_spot_price_safe()
        if not spot_price:
            return {'status': 'ERROR', 'message': 'Cannot get spot price'}
        
        options_chain = get_options_chain_safe()
        if not options_chain:
            return {'status': 'ERROR', 'message': 'Cannot get options chain'}
        
        # Get realized volatility
        realized_vol = get_realized_volatility_from_kite()
        
        # Enhanced IV analysis with realized volatility and liquidity
        iv_analysis = calculate_iv_rank_analysis_wrapper()
        
        if iv_analysis.get('status') == 'SUCCESS':
            current_iv = iv_analysis.get('current_iv', 0)
            iv_percentile = iv_analysis.get('iv_percentile', 0)
            iv_validation = iv_analysis.get('iv_validation', {})
            liquidity_analysis = iv_analysis.get('liquidity_analysis', {})
            
            print(f"📊 ENHANCED QUICK CHECK:")
            print(f"   Current IV: {current_iv:.4f}")
            print(f"   IV Percentile: {iv_percentile:.1%}")
            
            if realized_vol:
                print(f"   Realized Vol: {realized_vol:.4f}")
                iv_ratio = current_iv / realized_vol if realized_vol > 0 else 0
                print(f"   IV/Realized Ratio: {iv_ratio:.2f}")
            
            # Check liquidity
            liquidity_score = 0
            if liquidity_analysis.get('status') == 'SUCCESS':
                liquidity_score = liquidity_analysis.get('liquidity_metrics', {}).get('overall_liquidity_score', 0)
                liquidity_status = liquidity_analysis.get('liquidity_metrics', {}).get('liquidity_status', 'UNKNOWN')
                print(f"   Liquidity Score: {liquidity_score}/100 ({liquidity_status})")
            
            # Enhanced decision logic with realized volatility and liquidity
            opportunity_detected = False
            reason = "No opportunity detected"
            
            # Check IV conditions (more conservative thresholds)
            if current_iv > 0.20 or iv_percentile > 0.6:  # More conservative
                opportunity_detected = True
                reason = "IV conditions met"
                
                # Check realized volatility validation
                if realized_vol:
                    if current_iv / realized_vol > 1.4:  # IV overpriced (more stringent)
                        reason = "IV overpriced vs realized volatility - premium selling opportunity"
                    elif current_iv / realized_vol < 0.7:  # IV underpriced (more stringent)
                        reason = "IV underpriced vs realized volatility - long opportunity"
                    else:
                        reason = "IV fairly priced vs realized volatility"
                
                # Check liquidity constraints
                if liquidity_score < 30:  # Very poor liquidity
                    opportunity_detected = False
                    reason = "Poor liquidity prevents trading"
                elif liquidity_score < 50:  # Poor liquidity
                    reason += " - Limited by poor liquidity"
            
            if opportunity_detected:
                print(f"✅ OPPORTUNITY DETECTED: {reason} = ✅ YES (PROCEED TO STAGE 2)")
                return {
                    'status': 'SUCCESS',
                    'opportunity_detected': True,
                    'current_iv': current_iv,
                    'iv_percentile': iv_percentile,
                    'realized_volatility': realized_vol,
                    'liquidity_score': liquidity_score,
                    'reason': reason,
                    'next_stage': 'STAGE_2_EMERGENCY_CHECK'
                }
            else:
                print(f"❌ NO OPPORTUNITY: {reason} = ❌ NO (SKIP ANALYSIS)")
                return {
                    'status': 'SUCCESS',
                    'opportunity_detected': False,
                    'current_iv': current_iv,
                    'iv_percentile': iv_percentile,
                    'realized_volatility': realized_vol,
                    'liquidity_score': liquidity_score,
                    'reason': reason,
                    'token_efficiency': 'HIGH - Enhanced quick check used ~200 tokens'
                }
        else:
            print(f"❌ IV Analysis failed: {iv_analysis.get('message', 'Unknown error')}")
            return {'status': 'ERROR', 'message': 'IV analysis failed'}
            
    except Exception as e:
        print(f"❌ Quick opportunity check failed: {str(e)}")
        return {'status': 'ERROR', 'message': f'Quick opportunity check failed: {str(e)}'}

def run_detailed_analysis():
    """
    Execute the full detailed analysis pipeline when fast-track indicates potential
    """
    try:
        print("🔍 Running detailed analysis pipeline...")
        print("=" * 80)
        print("📊 DETAILED ANALYSIS HIERARCHY EXECUTION")
        print("=" * 80)
        
        # Core data gathering
        print("📈 Step 1: Core Data Gathering...")
        global_conditions = get_global_market_conditions()
        print(f"   ✅ Global Market Conditions: {global_conditions.get('market_sentiment', 'N/A')}")
        
        instruments = get_nifty_instruments()
        print(f"   ✅ NIFTY Instruments: {len(instruments) if instruments else 0} instruments")
        
        spot_price = get_nifty_spot_price_safe()
        print(f"   ✅ Spot Price: {spot_price.get('spot_price', 'N/A') if spot_price else 'N/A'}")
        
        expiry_analysis = get_available_expiry_dates_with_analysis()
        print(f"   ✅ Expiry Analysis: {len(expiry_analysis.get('available_expiries', [])) if expiry_analysis else 0} expiries")
        
        # Select appropriate expiry
        print("📅 Step 2: Expiry Selection...")
        chosen_expiry = select_optimal_expiry(expiry_analysis)
        print(f"   ✅ Chosen Expiry: {chosen_expiry}")
        
        # Get options chain for chosen expiry
        print("📊 Step 3: Options Chain Analysis...")
        options_chain = get_options_chain_safe(expiry_date=chosen_expiry)
        print(f"   ✅ Options Chain: {len(options_chain.get('options_chain', [])) if options_chain else 0} options")
        
        # Technical analysis
        print("📈 Step 4: Technical Analysis...")
        intraday_tech = get_nifty_technical_analysis_tool()
        print(f"   ✅ Intraday Technical: {intraday_tech.get('signal', 'N/A')} (RSI: {intraday_tech.get('rsi', 'N/A')})")
        
        daily_tech = get_nifty_daily_technical_analysis_wrapper()
        print(f"   ✅ Daily Technical: {daily_tech.get('signal', 'N/A')} (RSI: {daily_tech.get('rsi', 'N/A')})")
        
        # Advanced analysis
        print("🔬 Step 5: Advanced Analysis...")
        vix_analysis = analyze_vix_integration_wrapper()
        print(f"   ✅ VIX Analysis: {vix_analysis.get('volatility_regime', 'N/A')}")
        
        iv_rank = calculate_iv_rank_analysis_wrapper()
        print(f"   ✅ IV Rank: {iv_rank.get('current_iv', 'N/A')} (Percentile: {iv_rank.get('iv_percentile', 'N/A')}%)")
        
        pcr_tech = calculate_pcr_technical_analysis_wrapper()
        print(f"   ✅ PCR Technical: {pcr_tech.get('signal', 'N/A')} (PCR: {pcr_tech.get('pcr_value', 'N/A')})")
        
        pcr_extremes = analyze_pcr_extremes_wrapper()
        print(f"   ✅ PCR Extremes: {pcr_extremes.get('extreme_signal', 'N/A')}")
        
        regime = detect_market_regime_wrapper()
        print(f"   ✅ Market Regime: {regime.get('classification', 'N/A')}")
        
        comprehensive = comprehensive_advanced_analysis_wrapper()
        print(f"   ✅ Comprehensive Analysis: {comprehensive.get('overall_signal', 'N/A')}")
        
        # Strategy determination
        print("🎯 Step 6: Strategy Determination...")
        strategy_recommendation = determine_strategy_from_analysis(
            iv_rank, regime, intraday_tech, daily_tech, comprehensive
        )
        print(f"   ✅ Strategy Recommendation: {strategy_recommendation}")
        
        # Compile results
        detailed_result = {
            'decision': 'DETAILED_ANALYSIS_COMPLETE',
            'spot_price': spot_price,
            'chosen_expiry': chosen_expiry,
            'iv_rank': iv_rank,
            'regime': regime,
            'technical': {
                'intraday': intraday_tech,
                'daily': daily_tech
            },
            'advanced_analysis': {
                'vix': vix_analysis,
                'pcr_tech': pcr_tech,
                'pcr_extremes': pcr_extremes,
                'comprehensive': comprehensive
            },
            'strategy_recommendation': strategy_recommendation
        }
        
        # Print detailed analysis summary
        print("\n" + "=" * 80)
        print("📋 DETAILED ANALYSIS SUMMARY")
        print("=" * 80)
        print(f"📊 Market Regime: {regime.get('classification', 'N/A')}")
        print(f"📈 IV Status: {iv_rank.get('current_iv', 'N/A')} (Percentile: {iv_rank.get('iv_percentile', 'N/A')}%)")
        print(f"📉 Technical Signals: Intraday={intraday_tech.get('signal', 'N/A')}, Daily={daily_tech.get('signal', 'N/A')}")
        print(f"📊 PCR Status: {pcr_tech.get('signal', 'N/A')} (Value: {pcr_tech.get('pcr_value', 'N/A')})")
        print(f"🎯 Strategy Recommendation: {strategy_recommendation}")
        print(f"📅 Expiry Selected: {chosen_expiry}")
        print(f"💰 Spot Price: {spot_price.get('spot_price', 'N/A') if spot_price else 'N/A'}")
        print("=" * 80)
        
        return detailed_result
        
    except Exception as e:
        print(f"❌ Detailed analysis failed: {e}")
        return {
            'decision': 'DETAILED_ANALYSIS_FAILED',
            'error': str(e),
            'fallback': 'WAIT - Analysis incomplete'
        }

def select_optimal_expiry(expiry_analysis):
    """Helper function to select best expiry based on analysis"""
    try:
        if not expiry_analysis:
            return None
        
        # Handle different data structures
        expiry_list = None
        if 'available_expiries' in expiry_analysis:
            expiry_list = expiry_analysis['available_expiries']
        elif 'recommended_expiries' in expiry_analysis:
            expiry_list = expiry_analysis['recommended_expiries']
        elif 'expiry_dates' in expiry_analysis:
            expiry_list = expiry_analysis['expiry_dates']
        
        if not expiry_list:
            return None
        
        # Determine field names based on data structure
        date_field = 'date' if 'date' in expiry_list[0] else 'expiry_date'
        
        # Prefer MEDIUM_TERM expiries (8-14 days) for most strategies
        for expiry_info in expiry_list:
            if expiry_info.get('category') == 'MEDIUM_TERM':
                return expiry_info.get(date_field)
        
        # Fallback to LONG_TERM if no MEDIUM_TERM available
        for expiry_info in expiry_list:
            if expiry_info.get('category') == 'LONG_TERM':
                return expiry_info.get(date_field)
        
        # Last resort: first available expiry that's not TOO_CLOSE
        for expiry_info in expiry_list:
            if expiry_info.get('category') != 'TOO_CLOSE':
                return expiry_info.get(date_field)
        
        return None
    except Exception as e:
        print(f"Error selecting expiry: {e}")
        return None

def determine_strategy_from_analysis(iv_rank, regime, intraday_tech, daily_tech, comprehensive):
    """Helper function to determine strategy from all analysis data"""
    try:
        iv_percentile = iv_rank.get('iv_percentile', 0)
        regime_class = regime.get('classification', 'UNKNOWN')
        intraday_signal = intraday_tech.get('signal', 'NEUTRAL')
        daily_signal = daily_tech.get('signal', 'NEUTRAL')
        
        # Premium selling strategies (preferred) - further relaxed for current market
        if iv_percentile > 30:  # Further relaxed from 45% - more opportunities for premium selling
            if regime_class in ['RANGING', 'COMPRESSED']:
                return 'Short Strangle'
            elif regime_class == 'NEUTRAL':
                return 'Iron Condor'
            else:
                return 'Bull Put Spread'  # Conservative premium selling
        
        # Consider long strategies when IV is low and strong directional signal (further relaxed)
        if iv_percentile < 25:  # Further relaxed from 35% - allow more long strategy opportunities
            if intraday_signal in ['STRONG_BUY', 'STRONG_SELL'] and daily_signal in ['STRONG_BUY', 'STRONG_SELL']:
                if intraday_signal == 'STRONG_BUY' and daily_signal == 'STRONG_BUY':
                    return 'Long Call'
                elif intraday_signal == 'STRONG_SELL' and daily_signal == 'STRONG_SELL':
                    return 'Long Put'
        
        return 'WAIT - No clear opportunity'
        
    except Exception as e:
        print(f"Error determining strategy: {e}")
        return 'WAIT - Analysis error'

def optimized_market_analysis():
    """
    Optimized market analysis with fast-track logic to reduce analysis paralysis
    
    Returns:
        dict: Analysis result with decision and strategy recommendation
    """
    # Step 1: Emergency Fast-Track Check
    emergency_signal = emergency_fast_track()
    if emergency_signal.get('status') == 'SUCCESS':
        emergency_type = emergency_signal.get('emergency_signal', 'NO_EMERGENCY')
        if emergency_type != 'NO_EMERGENCY':
            # Return emergency signal for immediate execution
            return {
                'decision': 'EMERGENCY_EXECUTION',
                'emergency_signal': emergency_signal,
                'reason': f'Emergency detected: {emergency_type}'
            }
    
    # Step 2: Fast-Track Analysis (30 seconds)
    fast_result = quick_opportunity_check()
    if fast_result.get('opportunity_detected') == False:
        return {"decision": "NO_OPPORTUNITY", "reason": fast_result.get('reason', 'No opportunity detected')}
    
    # Step 3: Detailed Analysis (only if promising)
    if fast_result.get('opportunity_detected') == True:
        return {"decision": "PROCEED_TO_DETAILED", "reason": "Opportunity detected, proceeding to detailed analysis"}
    
    # Default case
    return {"decision": "NO_OPPORTUNITY", "reason": "No clear opportunity detected"}

# === STRATEGY SUITABILITY CHECK FUNCTIONS ===

def iron_condor_suitability_check():
    """
    Enhanced Iron Condor validation with trend filters
    
    Returns:
        dict: Approval status with confidence level and reasoning
    """
    try:
        regime = detect_market_regime_wrapper()
        technical = get_nifty_technical_analysis_tool()
        iv_rank = calculate_iv_rank_analysis_wrapper()
        
        regime_class = regime.get('classification', 'UNKNOWN')
        tech_signal = technical.get('signal', 'NEUTRAL')
        rsi = technical.get('rsi', 50)
        iv_percentile = iv_rank.get('iv_percentile', 0)
        
        # Disqualification Rules
        if regime_class in ['TRENDING_BULL', 'TRENDING_BEAR']:
            return {'approved': False, 'reason': f'Strong trend detected: {regime_class}'}
        
        if rsi < 30 or rsi > 70:
            return {'approved': False, 'reason': f'Momentum too strong (RSI: {rsi:.1f})'}
        
        if tech_signal in ['STRONG_BUY', 'STRONG_SELL']:
            return {'approved': False, 'reason': f'Breakout pattern: {tech_signal}'}
        
        if iv_percentile < 45:  # Relaxed from 60% - allow more iron condor opportunities
            return {'approved': False, 'reason': f'IV too low: {iv_percentile:.1f}%'}
        
        return {
            'approved': True,
            'confidence': 'HIGH' if iv_percentile > 70 else 'MEDIUM',  # Relaxed from 80%
            'reason': f'Suitable for Iron Condor (IV: {iv_percentile:.1f}%)'
        }
        
    except Exception as e:
        return {'approved': False, 'reason': f'Check failed: {e}'}

def short_strangle_enhanced_logic():
    """
    Enhanced Short Strangle validation with multiple conditions
    
    Returns:
        dict: Approval status with confidence level and conditions met
    """
    try:
        regime = detect_market_regime_wrapper()
        technical = get_nifty_technical_analysis_tool()
        pcr = calculate_pcr_technical_analysis_wrapper()
        iv_rank = calculate_iv_rank_analysis_wrapper()
        
        conditions = {
            'regime_suitable': regime.get('classification') in ['RANGING', 'COMPRESSED'],
            'technical_neutral': technical.get('signal') in ['NEUTRAL', 'WEAK_BUY', 'WEAK_SELL'],
            'no_extreme_sentiment': pcr.get('signal') not in ['STRONG_BUY', 'STRONG_SELL'],
            'volatility_elevated': iv_rank.get('iv_percentile', 0) > 45  # Relaxed from 60%
        }
        
        passed_conditions = sum(conditions.values())
        
        if passed_conditions >= 3:
            return {
                'approved': True,
                'confidence': 'HIGH' if passed_conditions == 4 else 'MEDIUM',
                'conditions_met': f"{passed_conditions}/4"
            }
        
        # High IV override (relaxed from 85%)
        if iv_rank.get('iv_percentile', 0) > 70:
            return {
                'approved': True,
                'confidence': 'MEDIUM',
                'reason': f"IV override: {iv_rank.get('iv_percentile'):.1f}%"
            }
        
        return {
            'approved': False,
            'conditions_met': f"{passed_conditions}/4",
            'reason': 'Insufficient conditions met'
        }
        
    except Exception as e:
        return {'approved': False, 'reason': f'Logic failed: {e}'}

# ============================================================================
# SIMPLIFIED AGENTS - FOCUS ON NEW OPPORTUNITIES ONLY
# ============================================================================

# Market Opportunity Analyst (with Intraday-Only Constraints)
market_analyst = Agent(
    role="Market Opportunity Analyst",
    goal="Identify high-probability trading opportunities in NIFTY F&O markets while preserving capital through patience and discipline.",
    backstory=f"""You are a market opportunity specialist focused ONLY on finding profitable trading setups.
    You assume the portfolio is already clean and managed by another system.
    
    CURRENT DATE AND TIME: {current_datetime}
    
    **Your Core Philosophy**: Capital preservation is paramount. You believe that patience and discipline 
    are more valuable than aggressive trading. You wait for clear, high-probability opportunities rather 
    than forcing trades in uncertain conditions. Your success comes from being selective and only taking 
    calculated risks when the odds are clearly in your favor. You understand that missing a trade is 
    better than losing money on a poor setup.
    
    **PREMIUM SELLING FIRST APPROACH**: You prioritize selling options over buying options because:
    - Selling options earns money from time decay (theta)
    - Buying options loses money from time decay (theta)
    - Premium selling has higher probability of profit in most market conditions
    - Only buy options when IV is low AND strong directional signal exists
    
    Your job is simple:
    1. Analyze current market conditions thoroughly
    2. Identify ONLY profitable, high-probability trading opportunities  
    3. **ALWAYS prefer premium selling strategies** (Short Strangle, Iron Condor, Credit spreads)
    4. Recommend long strategies ONLY when IV is low (<35% percentile) AND strong directional signal
    
    You do NOT manage existing positions - that's handled elsewhere.
    You focus purely on: "Is there a profitable premium selling opportunity right now?"
    
    **CRITICAL: MULTI-TIMEFRAME ANALYSIS**
    - **Intraday Analysis**: Use 15min/5min intervals for short-term momentum and entry timing
    - **Daily Analysis**: Use 60+ days daily data for medium-term trend direction and support/resistance
    - **Trend Alignment**: Prefer setups where intraday and daily trends align for higher probability
    - **Divergence Detection**: Be cautious when short-term and long-term trends diverge
    - During market hours: Focus on intraday momentum within daily trend context
    - Pre-market: Use daily data for gap analysis and trend continuation
    
    **PATIENCE DISCIPLINE**: When in doubt, WAIT. There will always be another opportunity.
    
    ---
    **INTRADAY-ONLY CONSTRAINTS:**
    - All trades are MIS (intraday only) and closed by 3:20 PM (broker auto square-off).
    - No overnight or multi-day trades are allowed.
    - Be extra cautious as the day progresses; avoid new trades if there is not enough time for the strategy to play out before 3:20 PM.
    - Only recommend trades with a high probability of being profitable within the same day.
    - If there is any doubt about intraday profitability, WAIT.
    ---
    """,
    tools=[
        # CRITICAL DATA TOOLS
        tool("Get correct instrument symbols for NIFTY trading")(get_nifty_instruments),
        tool("Debug Kite Connect instruments and symbols")(debug_kite_instruments),
        tool("Get NIFTY spot price")(get_nifty_spot_price_safe),
        tool("Get options chain data")(get_options_chain_safe),
        tool("Get NIFTY technical analysis")(get_nifty_technical_analysis_tool),
        tool("Get NIFTY daily technical analysis")(get_nifty_daily_technical_analysis_wrapper),
        tool("Get NIFTY expiry dates")(get_nifty_expiry_dates),
        tool("Get available expiry dates with analysis for decision making")(get_available_expiry_dates_with_analysis),
        tool("Get historical volatility")(get_historical_volatility),
        tool("Fetch historical OHLCV data for a symbol and date range")(fetch_historical_data),
        tool("Analyze options flow")(analyze_options_flow_safe),
        tool("Get global market conditions (pre-market only)")(get_global_market_conditions),
        # NEW ADVANCED ANALYSIS TOOLS
        tool("VIX Integration & Volatility Regime Analysis")(analyze_vix_integration_wrapper),
        tool("IV Rank Analysis for Premium Decisions")(calculate_iv_rank_analysis_wrapper),
        tool("PCR + Technical Analysis for Entry Timing")(calculate_pcr_technical_analysis_wrapper),
        tool("PCR Extremes Analysis for Contrarian Opportunities")(analyze_pcr_extremes_wrapper),
        tool("Market Regime Detection for Strategy Selection")(detect_market_regime_wrapper),
        tool("Comprehensive Advanced Analysis (All-in-One)")(comprehensive_advanced_analysis_wrapper),
        # === FAST-TRACK ANALYSIS TOOLS ===
        tool("Emergency Fast-Track Analysis for Obvious Opportunities")(emergency_fast_track),
        tool("Quick Opportunity Check to Skip Detailed Analysis")(quick_opportunity_check),
        tool("Optimized Market Analysis with Fast-Track Logic")(optimized_market_analysis),
        # NEW PREMIUM SELLING STRATEGY TOOLS
        tool("Create a Bull Put Spread (Credit Spread) strategy")(create_bull_put_spread_wrapper),
        tool("Create a Bear Call Spread (Credit Spread) strategy")(create_bear_call_spread_wrapper),
        tool("Create a Calendar Spread (Time Spread) strategy")(create_calendar_spread_wrapper),
    ],
    verbose=True,
    max_iter=3,
    **llm_kwargs
)

# Simple Capital Checker (Simplified)
capital_checker = Agent(
    role="Capital Availability Checker", 
    goal="Verify if sufficient capital exists for new trades and check basic portfolio constraints.",
    backstory=f"""You are a simple capital checker. Your job is straightforward:
    
    1. Check available capital
    2. Count existing positions  
    3. Give go/no-go for new trades
    
    SIMPLE RULES:
    - 0-5 positions: OK to trade (if capital available)
    - 5+ positions: NO new trades
    - After 14:30: NO new trades
    - Insufficient capital: NO trades
    
    You don't manage existing positions - just check if new ones are allowed.
    """,
    tools=[
        tool("Get portfolio positions count")(get_portfolio_positions),
        tool("Get account margins and cash")(get_account_margins),
        tool("Validate general capital availability")(validate_general_capital),
        tool("Calculate strategy margins")(calculate_strategy_margins),
        tool("Check position conflicts with new strategy")(analyze_position_conflicts_wrapper),
        tool("Validate if sufficient capital is available for trading strategy")(validate_trading_capital),
        tool("Get risk metrics for the account")(get_risk_metrics),
        tool("Get order history for the account")(get_orders_history),
        tool("Get daily trading summary for the account")(get_daily_trading_summary),
    ],
    verbose=True,
    max_iter=3,
    **llm_kwargs
)

# Trade Executor (with Intraday-Only Constraints)
trade_executor = Agent(
    role="New Trade Executor",
    goal="Execute NEW trades only when all conditions are perfect. Default is to WAIT and preserve capital.",
    backstory=f"""You are a disciplined trade executor focused ONLY on profitable new opportunities.
    
    CURRENT DATE AND TIME: {current_datetime}
    
    **Your Core Philosophy**: Capital preservation is paramount. You believe that patience and discipline 
    are more valuable than aggressive trading. You wait for clear, high-probability opportunities rather 
    than forcing trades in uncertain conditions. Your success comes from being selective and only taking 
    calculated risks when the odds are clearly in your favor. You understand that missing a trade is 
    better than losing money on a poor setup.
    
    **PREMIUM SELLING FIRST APPROACH**: You prioritize selling options over buying options because:
    - Selling options earns money from time decay (theta)
    - Buying options loses money from time decay (theta)
    - Premium selling has higher probability of profit in most market conditions
    - Only buy options when IV is low AND strong directional signal exists
    
    Your philosophy: "When in doubt, don't trade. Preserve capital for better opportunities."
    
    **MANDATORY EXECUTION PROCESS:**
    You ALWAYS follow a two-step process:
    1. **CREATE**: Use strategy creation tools to build strategy structure (no execution)
    2. **EXECUTE**: Use execute_and_store_strategy() to execute and store the trade
    
    **STRATEGY PRIORITY ORDER:**
    1. **Short Strangle** (preferred premium selling strategy)
    2. **Iron Condor** (range-bound premium selling)
    3. **Bull Put Spread** (bullish premium selling with defined risk)
    4. **Bear Call Spread** (bearish premium selling with defined risk)
    5. **Calendar Spread** (time decay premium selling)
    6. **Long Straddle/Strangle** (only if IV <35% AND strong directional signal)
    
    You execute trades ONLY when:
    1. Market analysis shows PROFITABLE setup (not just good, but profitable)
    2. Capital checker confirms availability  
    3. Time is before 14:30
    4. Risk-reward is compelling (>2:1)
    5. All volatility and movement conditions are perfectly aligned
    6. **Premium selling opportunity exists** (IV rank >45% OR low volatility regime)
    7. No doubts or uncertainties exist
    
    You do NOT:
    - Manage existing positions
    - Close conflicting positions  
    - Do complex portfolio analysis
    - Use strategy creation tools for execution (they only create structures)
    - Force trades in uncertain conditions
    
    You simply ask: "Is this a profitable new trade opportunity with clear high probability?"
    If yes: Create strategy → Execute and store. If no: Wait and preserve capital.
    
    **PATIENCE DISCIPLINE**: Better to miss 10 good trades than lose money on 1 bad trade.
    
    ---
    **INTRADAY-ONLY CONSTRAINTS:**
    - All trades are MIS (intraday only) and closed by 3:20 PM (broker auto square-off).
    - No overnight or multi-day trades are allowed.
    - Be extra cautious as the day progresses; avoid new trades if there is not enough time for the strategy to play out before 3:20 PM.
    - Only execute trades with a high probability of being profitable within the same day.
    - If there is any doubt about intraday profitability, WAIT.
    ---
    """,
    tools=[
        # Data fetching tools (needed for strategy creation)
        tool("Get NIFTY instruments")(get_nifty_instruments),
        tool("Get NIFTY spot price")(get_nifty_spot_price_safe),
        tool("Get NIFTY expiry dates")(get_nifty_expiry_dates),
        tool("Get NIFTY options chain")(get_options_chain_safe),
        
        # Strategy creation tools (using wrapper functions for CrewAI compatibility)
        tool("Create long straddle")(create_long_straddle_wrapper),
        tool("Create short strangle")(create_short_strangle_wrapper),
        tool("Create iron condor")(create_iron_condor_wrapper),
        tool("Create butterfly spread")(create_butterfly_spread_wrapper),
        tool("Create a Bull Put Spread (Credit Spread) strategy")(create_bull_put_spread_wrapper),
        tool("Create a Bear Call Spread (Credit Spread) strategy")(create_bear_call_spread_wrapper),
        tool("Create a Calendar Spread (Time Spread) strategy")(create_calendar_spread_wrapper),
        tool("Recommend strategy")(recommend_options_strategy),
        
        # Execution tools
        tool("Execute options strategy")(execute_options_strategy),
        tool("Execute strategy and store if successful")(execute_and_store_strategy),
        
        # Calculation tools
        tool("Calculate option Greeks")(calculate_option_greeks),
        tool("Calculate strategy P&L")(calculate_strategy_pnl),
        tool("Calculate probability of profit")(calculate_probability_of_profit),
        tool("Calculate implied volatility for an option")(calculate_implied_volatility),
        tool("Find arbitrage opportunities in the options chain")(find_arbitrage_opportunities),
        tool("Calculate portfolio Greeks for open positions")(calculate_portfolio_greeks),
        tool("Calculate the volatility surface for options data")(calculate_volatility_surface),
        tool("Analyze Greeks for a given options strategy")(analyze_strategy_greeks),
        
        # === STRATEGY SUITABILITY CHECK TOOLS ===
        tool("Iron Condor Suitability Check with Trend Filters")(iron_condor_suitability_check),
        tool("Short Strangle Enhanced Logic with Multi-Condition Check")(short_strangle_enhanced_logic),
    ],
    verbose=True,
    max_iter=3,
    **llm_kwargs
)

# ============================================================================
# SIMPLIFIED TASKS - NO POSITION MANAGEMENT
# ============================================================================

# Market Opportunity Analysis (with Intraday-Only Constraints)
opportunity_analysis_task = Task(
    description=f"""
    Find profitable NIFTY F&O trading opportunities for NEW positions while preserving capital:
    
    CURRENT DATE AND TIME: {current_datetime}
    
    **CRITICAL: Your output determines if subsequent tasks run.**
    - If you find NO_OPPORTUNITY, capital checker and trade executor will be SKIPPED
    - This saves significant token usage and execution time
    - Only proceed if you find PROFITABLE opportunities
    
    **PREMIUM SELLING FIRST MANDATE**: Your primary goal is to find premium selling opportunities that benefit from time decay. Only recommend long strategies when IV is extremely low (<30% percentile) AND strong directional signal exists.
    
    **CAPITAL PRESERVATION MANDATE**: Your secondary goal is to preserve capital. Only recommend trades 
    when there is a CLEAR, HIGH-PROBABILITY opportunity. When in doubt, WAIT. Better to miss a trade 
    than lose money on a poor setup.
    
    ---
    **INTRADAY-ONLY CONSTRAINTS:**
    - All trades are MIS (intraday only) and closed by 3:20 PM (broker auto square-off).
    - No overnight or multi-day trades are allowed.
    - Be extra cautious as the day progresses; avoid new trades if there is not enough time for the strategy to play out before 3:20 PM.
    - Only recommend trades with a high probability of being profitable within the same day.
    - If there is any doubt about intraday profitability, WAIT.
    ---
    
    1. **FAST-TRACK ANALYSIS APPROACH - HONOR HIERARCHICAL DECISIONS:**
       
       **STEP 1: Run optimized_market_analysis() tool ONLY**
       - This tool automatically runs emergency_fast_track() and quick_opportunity_check()
       - **CRITICAL**: Do NOT override the tool's decision - honor it completely
       
       **STEP 2: Process the result based on tool output:**
       
       **A) If result contains "NO_OPPORTUNITY" or "SKIP_DETAILED_ANALYSIS":**
          - Output: "NO CLEAR OPPORTUNITY - WAIT TO PRESERVE CAPITAL"
          - **STOP HERE** - Do NOT run any additional analysis
          - **REASON**: Fast-track determined no opportunity exists
       
       **B) If result contains "IMMEDIATE_PREMIUM_SELLING" or emergency signal:**
          - Proceed directly to strategy recommendation using the tool's output
          - **STOP HERE** - Do NOT run detailed analysis
          - **REASON**: Emergency fast-track identified obvious opportunity
       
       **C) If result contains "PROCEED_TO_DETAILED" or "DETAILED_ANALYSIS_COMPLETE":**
          - Use the detailed analysis already provided by the tool
          - **STOP HERE** - Do NOT run additional analysis
          - **REASON**: Tool already completed detailed analysis
       
       **CRITICAL RULES:**
       - **NEVER override fast-track decisions**
       - **NEVER run detailed analysis if tool says "SKIP"**
       - **NEVER run additional tools if tool provides complete analysis**
       - **ALWAYS honor the tool's hierarchical decision logic**
       
       **TOOL OUTPUT INTERPRETATION:**
       
       The optimized_market_analysis() tool provides comprehensive analysis including:
       
       a) **Emergency Fast-Track Results**: VIX spikes, extreme IV, obvious opportunities
       b) **Quick Check Results**: IV too low, conflicting signals, no clear edge
       c) **Detailed Analysis Results**: Complete market analysis when opportunity exists
       
       **Your job**: Interpret the tool's output and provide clear decision
    
    2. **Intelligent Expiry Selection Strategy:**
       - **CRITICAL**: Use get_available_expiry_dates_with_analysis() to get comprehensive expiry analysis
       - **AGENT DECISION**: You must choose the most appropriate expiry based on your strategy and market analysis
       - **SAFETY RULE**: NEVER use expiries with ≤3 days to expiry (categorized as TOO_CLOSE)
       
       **EXPIRY CATEGORIES AND YOUR DECISION LOGIC:**
       
       a) **SHORT_TERM (4-7 days)**: High theta decay, high gamma risk
          * **Choose for**: High-conviction directional trades, momentum plays, event-driven trades
          * **Avoid for**: Low conviction, high volatility uncertainty, complex strategies
          * **Your decision**: Only use if you have very high conviction and clear directional bias
       
       b) **MEDIUM_TERM (8-14 days)**: Balanced theta/gamma, preferred for most strategies
          * **Choose for**: Most strategies including straddles, strangles, iron condors
          * **Optimal range**: 10-14 days for balanced risk/reward
          * **Your decision**: Default choice for most strategies unless specific conditions require different timing
       
       c) **LONG_TERM (15+ days)**: Lower theta decay, use for longer-term positions
          * **Choose for**: Longer-term directional views, low volatility strategies, theta decay strategies
          * **Risk**: Lower gamma, less responsive to short-term moves
          * **Your decision**: Use when you expect gradual moves or want to reduce time decay impact
       
       **YOUR EXPIRY SELECTION PROCESS:**
       1. Get available expiries with analysis using get_available_expiry_dates_with_analysis()
       2. Review the categorization (SHORT_TERM, MEDIUM_TERM, LONG_TERM, TOO_CLOSE)
       3. Consider your strategy requirements and market conditions
       4. Choose the expiry that best matches your strategy and risk tolerance
       5. Pass your chosen expiry date to subsequent analysis functions
       
       **STRATEGY-EXPIRY MATCHING:**
       - **Directional strategies**: Choose based on expected move timing (SHORT_TERM for quick moves, LONG_TERM for gradual moves)
       - **Volatility strategies**: MEDIUM_TERM usually optimal for balanced theta/gamma
       - **Premium selling**: LONG_TERM for theta decay, SHORT_TERM for gamma risk management
       - **Event-driven**: SHORT_TERM for immediate events, MEDIUM_TERM for extended events
    
    3. **Multi-Timeframe Strategy Recommendations:**
       - **Daily Trend Analysis**: Use daily data to determine overall market direction and key levels
       - **Intraday Momentum**: Use intraday data for precise entry timing and momentum confirmation
       - **Trend Alignment Check**: Ensure intraday momentum aligns with daily trend direction
       - **Volatility Regime + IV Rank**: Determine premium buying vs selling approach
       - **PCR + Technical**: Determine entry timing and directional bias
       - **Market Regime**: Select appropriate strategy types
       - **Comprehensive Analysis**: Final strategy recommendation with risk management
       
       **STRATEGY SELECTION LOGIC (PREMIUM SELLING PRIORITY):**
       - **PREMIUM SELLING FIRST APPROACH**: Always prefer selling options to benefit from time decay (theta)
         * **Primary Preference**: Short Strangle, Iron Condor, Calendar spreads, Credit spreads
         * **Secondary Preference**: Long strategies only when exceptional directional opportunity exists
         * **Theta Advantage**: Selling options earns money from time decay, buying options loses money from time decay
       
       - **Volatility-Based Strategy Selection**:
         * **High Volatility (>2% daily range)**: Short Strangle, Iron Condor, Credit spreads (prefer selling high IV)
         * **Low Volatility (<1% daily range)**: Short Strangle, Iron Condor, Calendar spreads (ideal for selling)
         * **Medium Volatility (1-2% daily range)**: Short Strangle, Iron Condor, Credit spreads (balanced selling)
         * **Long Strategies**: Only consider Long Straddle/Strangle if IV is extremely low (<20% percentile) AND strong directional signal
       
       - **Movement-Based Strategy Selection**:
         * **Strong Trend + High Volatility**: Credit spreads in trend direction (sell OTM options)
         * **Sideways + Low Volatility**: Short Strangle, Iron Condor (perfect for premium selling)
         * **Mixed Signals**: Conservative premium selling strategies or WAIT
       
       - **IV Rank Strategy Selection**:
         * **High IV Rank (>80%)**: Strongly prefer selling options (expensive premiums)
         * **Medium IV Rank (40-80%)**: Balanced approach, still prefer selling
         * **Low IV Rank (<40%)**: Consider buying only if exceptional directional setup
       
       - **Final Decision Logic**:
         * **DEFAULT**: Always look for premium selling opportunities first
         * **Premium Selling Criteria**: IV rank >60% OR low volatility regime OR sideways market
         * **Long Strategy Criteria**: IV rank <30% AND strong directional signal AND high conviction
         * If Comprehensive Analysis shows NEUTRAL: Use Short Strangle or Iron Condor
         * If Comprehensive Analysis shows WAIT: No trade (preserve capital)
         * **PATIENCE RULE**: If ANY doubt exists about the setup: WAIT and preserve capital
         * **PROFITABLE OPPORTUNITY RULE**: Only execute if the opportunity is truly profitable, not just good
       
       - Specify exact strikes, expiry, and quantity (minimum 75 lots)
       - Include risk management guidelines from the analysis
       - Expected profit potential and risk-reward ratio
    
    **CRITICAL CONSTRAINTS:**
    - **CAPITAL PRESERVATION FIRST**: Preserve capital is the primary goal - only take calculated risks
    - **PATIENCE DISCIPLINE**: Wait for CLEAR, HIGH-PROBABILITY opportunities - better to miss a trade than lose money
    - **CONSERVATIVE APPROACH**: When in doubt, WAIT - there will always be another opportunity
    - Focus ONLY on NEW opportunities
    - **FAST-TRACK HONORING**: ALWAYS honor the optimized_market_analysis() tool's decisions
    - **NO OVERRIDE**: Never override fast-track decisions with additional analysis
    - **EFFICIENCY FOCUS**: Use tool output efficiently - don't duplicate analysis
    - Ignore existing portfolio - assume it's clean
    - **SAFETY RULE**: NEVER use expiries with ≤3 days to expiry (categorized as TOO_CLOSE)
    - **DEFAULT BEHAVIOR**: WAIT unless profitable opportunity exists - patience is profitable
    
    Output: Fast-track decision based on optimized_market_analysis() tool output:
    - If tool says "NO_OPPORTUNITY": Output "NO CLEAR OPPORTUNITY - WAIT TO PRESERVE CAPITAL"
    - If tool says "IMMEDIATE_PREMIUM_SELLING": Output strategy recommendation with reasoning
    - If tool provides detailed analysis: Output strategy recommendation with analysis summary
    - Always include the tool's reasoning in your output
    """,
    agent=market_analyst,
    expected_output="Market opportunity analysis with specific new trade recommendations or clear statement that no opportunities exist."
)

# Simple Capital Check (with Intraday-Only Constraints)
capital_check_task = Task(
    description=f"""
    Simple capital and position count check for new trades with capital preservation focus:
    
    CURRENT DATE AND TIME: {current_datetime}
    
    **CRITICAL: This task only runs if Market Analyst found an opportunity.**
    - If Market Analyst output contains "NO_OPPORTUNITY", this task will be SKIPPED
    - This saves significant token usage and execution time
    - Only validate capital when there's a real opportunity to validate
    
    **CAPITAL PRESERVATION MANDATE**: When in doubt about capital availability, say NO to preserve capital.
    
    ---
    **INTRADAY-ONLY CONSTRAINTS:**
    - All trades are MIS (intraday only) and closed by 3:20 PM (broker auto square-off).
    - No overnight or multi-day trades are allowed.
    - Be extra cautious as the day progresses; avoid new trades if there is not enough time for the strategy to play out before 3:20 PM.
    - Only recommend trades with a high probability of being profitable within the same day.
    - If there is any doubt about intraday profitability, WAIT.
    ---
    
    **Simple Checks Only:**
    
    1. **Position Count Check:**
       - Count current open positions using get_portfolio_positions()
       - If 3+ positions: STOP - no new trades allowed
       - If 0-2 positions: Continue to capital check
    
    2. **Time Check:**
       - If after 14:30: STOP - no new trades (liquidity concerns)
       - If before 14:30: Continue
    
    3. **General Capital Check:**
       - Use validate_general_capital() to check overall capital availability
       - Get account margins using get_account_margins() for detailed cash info
       - Get risk metrics using get_risk_metrics() for additional validation
       - Check if we have sufficient capital for typical NIFTY options trades (75+ lots)
       - Simple yes/no answer based on general capital availability
    
    4. **Basic Position Conflict Check:**
       - Use analyze_position_conflicts() to automatically fetch and check for basic conflicts
       - Focus on major conflicts only (directional, expiry concentration)
       - Don't do detailed analysis - just flag major issues
    
    **NO COMPLEX ANALYSIS:**
    - Don't analyze existing positions in detail
    - Don't do detailed conflict analysis
    - Don't manage portfolio risk
    - Just answer: "Can we take a new trade? Yes/No and why"
    
    **CAPITAL VALIDATION LOGIC:**
    - Use validate_general_capital() for overall capital assessment
    - Check if available cash can support typical NIFTY options trades
    - Consider intraday_payin amounts in total available capital
    - Ensure emergency buffer remains after potential trades
    
    Output: Simple GO/NO-GO decision with reason
    """,
    agent=capital_checker,
    expected_output="Simple GO/NO-GO decision for new trades with clear reasoning (position count, time, or capital constraints).",
    context=[opportunity_analysis_task]
)

# Simple Trade Execution (with Intraday-Only Constraints)
trade_execution_task = Task(
    description=f"""
    Execute new trade ONLY if profitable opportunity exists with capital preservation focus:
    
    CURRENT DATE AND TIME: {current_datetime}
    
    **CRITICAL: This task only runs if both Market Analyst found opportunity AND Capital Checker approved.**
    - If Market Analyst output contains "NO_OPPORTUNITY", this task will be SKIPPED
    - If Capital Checker output contains "CAPITAL_REJECTED", this task will be SKIPPED
    - This saves significant token usage and execution time
    - Only execute when there's both opportunity AND capital approval
    
    **CAPITAL PRESERVATION MANDATE**: When in doubt, WAIT. Better to miss a trade than lose money.
    
    ---
    **INTRADAY-ONLY CONSTRAINTS:**
    - All trades are MIS (intraday only) and closed by 3:20 PM (broker auto square-off).
    - No overnight or multi-day trades are allowed.
    - Be extra cautious as the day progresses; avoid new trades if there is not enough time for the strategy to play out before 3:20 PM.
    - Only execute trades with a high probability of being profitable within the same day.
    - If there is any doubt about intraday profitability, WAIT.
    ---
    
    **Simple Decision Tree:**
    
    1. **Check Prerequisites:**
       - Did capital checker say GO? If NO: output "WAIT - [reason from capital checker]"
       - Is there a clear opportunity from market analysis? If NO: output "WAIT - No clear setup"
       - Any position conflicts identified? If YES: output "WAIT - Position conflicts detected"
    
    2. **Validate Strategy Details:**
       - Ensure minimum 75 lot quantity for NIFTY options
       - **EXPIRY VALIDATION**: Confirm expiry selection logic and validate chosen expiry:
         * Verify expiry date is available using get_nifty_expiry_dates()
         * **CRITICAL SAFETY CHECK**: Ensure expiry has >3 days to expiry (NEVER use ≤3 days)
         * Prefer 15-25 days to expiry for most strategies
         * For weekly expiries: Ensure high conviction, short-term outlook, and minimum 4 days to expiry
         * For monthly expiries: Verify adequate time for strategy to work
         * Check liquidity and open interest for chosen expiry
       - Verify strategy alignment with advanced analysis results:
         * Volatility regime recommendations
         * IV rank analysis decisions
         * PCR + technical analysis signals
         * Market regime strategy preferences
         * **Volatility & Movement Compatibility**: Ensure strategy matches current market conditions
       - Check bid-ask spreads are reasonable (<8% of option price)
       - **EXPIRY-SPECIFIC CHECKS**:
         * Days to expiry vs strategy type compatibility
         * Time decay impact on strategy profitability
         * Liquidity across different expiries for the same strategy
    
    3. **Quick Opportunity Assessment:**
       - Is the setup truly profitable? (High probability, good risk-reward >2:1)
       - Does the strategy align with comprehensive advanced analysis results?
       - Does the strategy match the volatility regime and IV rank recommendations?
       - Does the entry timing align with PCR + technical analysis signals?
       - **Does current NIFTY volatility support the strategy requirements?**
       - **Does NIFTY movement pattern match strategy expectations?**
       - Can we execute with good liquidity (OI > 500, volume > 50)?
       - **EXPIRY ASSESSMENT**:
         * **SAFETY FIRST**: Does expiry have >3 days to expiry? (REJECT if ≤3 days)
         * Does timeframe match expiry selection (quick trades = weekly, strategic = monthly)?
         * Is the chosen expiry optimal for the strategy type?
         * Are there better expiries available with better liquidity or pricing?
         * Does the expiry provide adequate time for the strategy to work?
         * Is the time decay profile suitable for the expected holding period?
    
    3. **Execution Decision:**
       - If ALL criteria met: Create and EXECUTE the strategy
       - If ANY doubt exists: output "WAIT - [specific reason]"
       - Default is always WAIT unless compelling opportunity
    
    4. **MANDATORY TWO-STEP EXECUTION PROCESS:**
       **STEP 1: Create Strategy Structure**
       - Use strategy creation tools (Create long straddle, Create short strangle, etc.) to build strategy structure
       - These tools ONLY create the strategy - they do NOT execute orders
       - Get strategy legs, pricing, and metrics from the creation tool
       
       **STEP 2: Execute and Store Strategy**
       - Use execute_and_store_strategy() with the strategy legs from Step 1
       - This function handles BOTH execution AND storage in one call
       - Only stores the trade if execution is successful
       - Include trade metadata: strategy name, analysis data, risk management, expiry, spot price
       - Report the combined execution and storage result
    
    **EXECUTION FLOW:**
    1. Validate all prerequisites (capital, time, opportunity)
    2. **STEP 1**: Create strategy using creation tools (get strategy legs)
    3. **STEP 2**: Execute and store using execute_and_store_strategy() with the strategy legs
    4. Check the combined result status
    5. Report success or failure based on the result
    
    **CRITICAL**: Never use strategy creation tools for execution - they only create strategy structures!
    
    **WHAT YOU DON'T DO:**
    - Position conflict analysis
    - Complex portfolio management
    - Closing existing positions
    - Managing multiple scenarios
    
    **WHAT YOU DO:**
    - Simple yes/no on execution
    - If yes: Use execute_and_store_strategy() for combined execution and storage
    - If no: Clear reason why waiting
    
    Remember: Missing a trade is better than taking a bad trade.
    """,
    agent=trade_executor,
    expected_output="Clear EXECUTE (with actual trade execution) or WAIT decision with specific reasoning. Default should be WAIT unless profitable opportunity.",
    context=[opportunity_analysis_task, capital_check_task]
)

# ============================================================================
# SIMPLIFIED CREW
# ============================================================================

opportunity_hunter_crew = Crew(
    agents=[market_analyst, capital_checker, trade_executor],
    tasks=[opportunity_analysis_task, capital_check_task, trade_execution_task],
    verbose=True,
    process="sequential",
    max_rpm=15,  # Reduced API calls
    planning=True,  # Enable better coordination between agents
    **llm_kwargs
)

# ============================================================================
# FINAL REASONING TASK
# ============================================================================

# Add a final reasoning task that consolidates all decisions
final_reasoning_task = Task(
    description=f"""
    Provide a comprehensive final reasoning summary for the Opportunity Hunter decision:
    
    CURRENT DATE AND TIME: {current_datetime}
    
    **CRITICAL: HONOR FAST-TRACK DECISIONS**
    - If Market Analyst output contains "NO_OPPORTUNITY" or "SKIP_DETAILED_ANALYSIS": 
      * Provide summary based ONLY on fast-track results
      * Do NOT run additional analysis tools
      * Focus on why fast-track decided to skip detailed analysis
    - If Market Analyst found opportunity: Provide comprehensive analysis as below
    
    **YOUR JOB**: Analyze the results from all previous agents and provide a clear, detailed explanation of why the final decision was made, with emphasis on premium selling opportunities.
    
    **FAST-TRACK SCENARIO (NO_OPPORTUNITY)**:
    - **DO NOT RUN ADDITIONAL ANALYSIS TOOLS**
    - Summarize the fast-track decision and reasoning
    - Explain why detailed analysis was skipped
    - Focus on capital preservation benefits of waiting
    
    **OPPORTUNITY SCENARIO (PROCEED_TO_DETAILED)**:
    - Provide comprehensive analysis as detailed below
    
    **REQUIRED ANALYSIS (ONLY IF OPPORTUNITY EXISTS)**:
    
    1. **Market Analysis Summary**:
       - What did the Market Opportunity Analyst find?
       - What market conditions were identified?
       - What analysis was completed vs incomplete?
       - What was the final market assessment?
       - **Premium Selling Assessment**: Was there a premium selling opportunity? (IV rank, volatility regime, market conditions)
    
    2. **Capital & Position Analysis**:
       - What did the Capital Availability Checker determine?
       - What are the current position counts and capital status?
       - What constraints were identified?
       - What was the capital availability decision?
    
    3. **Execution Decision Analysis**:
       - What did the Trade Executor decide?
       - What specific reasons were given for execution or waiting?
       - What strategy details were validated?
       - What was the final execution decision?
    
    4. **COMPREHENSIVE REASONING**:
       - **DECISION**: What is the final decision? (EXECUTE or WAIT)
       - **PRIMARY REASON**: What is the main reason for this decision?
       - **SUPPORTING FACTORS**: List 3-5 key factors that support this decision
       - **RISK ASSESSMENT**: What risks were considered?
       - **OPPORTUNITY QUALITY**: How profitable was the opportunity (if any)?
       - **CAPITAL PRESERVATION**: How does this decision align with capital preservation goals?
       - **MARKET CONDITIONS**: How do current market conditions support this decision?
       - **TIMING FACTORS**: What timing considerations influenced the decision?
    
    5. **DETAILED EXPLANATION**:
       - Explain the decision-making process step by step
       - Connect the dots between market analysis, capital availability, and execution decision
       - Provide specific data points that led to the final decision
       - Explain why this decision preserves capital or why the opportunity was profitable enough to execute
    
    **OUTPUT FORMAT**:
    
    # OPPORTUNITY HUNTER FINAL DECISION - {current_datetime}
    
    ## DECISION: [EXECUTE/WAIT]
    
    ## PRIMARY REASONING
    [Clear, concise explanation of the main reason for the decision]
    
    ## FAST-TRACK ANALYSIS (if NO_OPPORTUNITY)
    - **Fast-Track Decision**: [What fast-track determined]
    - **Reason for Skipping**: [Why detailed analysis was skipped]
    - **Capital Preservation**: [How this decision preserves capital]
    - **Market Conditions**: [Brief market condition summary from fast-track]
    
    ## DETAILED ANALYSIS (only if opportunity exists)
    
    ### Market Analysis Summary
    - **Market Conditions**: [What was found]
    - **Analysis Completeness**: [What was analyzed vs missing]
    - **Market Assessment**: [Final market evaluation]
    
    ### Capital & Position Analysis
    - **Position Status**: [Current positions and constraints]
    - **Capital Availability**: [Capital status and limitations]
    - **Capital Decision**: [What the capital checker determined]
    
    ### Execution Analysis
    - **Strategy Validation**: [What was validated]
    - **Execution Decision**: [What the executor decided]
    - **Execution Reasoning**: [Why execution was approved/rejected]
    
    ## SUPPORTING FACTORS
    1. [Factor 1]
    2. [Factor 2]
    3. [Factor 3]
    4. [Factor 4]
    5. [Factor 5]
    
    ## RISK ASSESSMENT
    - **Identified Risks**: [What risks were considered]
    - **Risk Mitigation**: [How risks were addressed]
    - **Risk-Reward Ratio**: [If applicable]
    
    ## PREMIUM SELLING ASSESSMENT
    - **Premium Selling Opportunity**: [Was there a premium selling opportunity? IV rank, volatility conditions]
    - **Strategy Type Selected**: [Why this strategy type was chosen over alternatives]
    - **Time Decay Advantage**: [How the strategy benefits from theta decay]
    - **IV Rank Analysis**: [How IV rank influenced strategy selection]
    
    ## CAPITAL PRESERVATION ALIGNMENT
    - **How This Decision Preserves Capital**: [Explanation]
    - **Opportunity Quality Assessment**: [If applicable]
    - **Patience vs Action Balance**: [How the decision balances patience with opportunity]
    
    ## CONCLUSION
    [Final summary of why this decision is the right one for capital preservation and trading success, emphasizing premium selling benefits]
    
    **CRITICAL**: 
    - If fast-track said "NO_OPPORTUNITY": Focus on fast-track reasoning and capital preservation
    - If opportunity exists: Provide comprehensive analysis
    - NEVER run additional tools if fast-track already determined no opportunity
    - The reasoning should be detailed enough that someone reading it understands the complete thought process
    """,
    agent=market_analyst,  # Use the market analyst for final reasoning
    expected_output="Comprehensive final reasoning that explains the complete decision-making process and why the final decision was made.",
    context=[opportunity_analysis_task, capital_check_task, trade_execution_task]
)

# ============================================================================
# ENHANCED CREW WITH FINAL REASONING
# ============================================================================

opportunity_hunter_crew = Crew(
    agents=[market_analyst, capital_checker, trade_executor],
    tasks=[opportunity_analysis_task, capital_check_task, trade_execution_task, final_reasoning_task],
    verbose=True,
    process="sequential",
    max_rpm=20,  # Reduced API calls
    planning=True,  # Enable better coordination between agents
    **llm_kwargs
)

# ============================================================================
# THREE-STAGE EXECUTION FUNCTIONS
# ============================================================================

def run_stage_1_fast_track_only():
    """
    Stage 1: Run only fast-track analysis to determine execution path
    """
    print("\n" + "="*80)
    print("🎯 STAGE 1: FAST-TRACK ANALYSIS ONLY")
    print(f"📅 Current Time: {current_datetime}")
    print("🔄 Mission: Quick assessment to determine if full analysis is needed")
    print("="*80)
    
    try:
        # Run only the fast-track analysis
        fast_track_result = optimized_market_analysis()
        print(f"Fast-track result: {fast_track_result}")
        
        return fast_track_result
        
    except Exception as e:
        print(f"❌ Error in Stage 1: {e}")
        return {'decision': 'ERROR', 'reason': f'Stage 1 failed: {str(e)}'}

def run_stage_3_full_analysis():
    """
    Stage 3: Run full crew analysis when normal opportunity exists
    """
    print("\n" + "="*80)
    print("🎯 STAGE 3: FULL OPPORTUNITY ANALYSIS")
    print(f"📅 Current Time: {current_datetime}")
    print("🔄 Mission: Comprehensive analysis and execution for identified opportunity")
    print("="*80)
    
    try:
        # Run the full crew
        result = opportunity_hunter_crew.kickoff()
        return result
        
    except Exception as e:
        print(f"❌ Error in Stage 2: {e}")
        return f"Stage 3 failed: {str(e)}"

def create_emergency_summary(emergency_result):
    """
    Create a summary for emergency execution results
    """
    print("\n" + "="*80)
    print("🚨 EMERGENCY EXECUTION SUMMARY")
    print("="*80)
    
    # Check if execution was successful (handle multiple success statuses)
    execution_result = emergency_result.get('execution_result', {})
    execution_status = execution_result.get('status', 'UNKNOWN')
    success_statuses = ['SUCCESS', 'BASKET_SUCCESS', 'PARTIAL_SUCCESS', 'EXECUTED']
    is_successful = execution_status in success_statuses
    
    # Determine the type of opportunity based on the reason
    reason = emergency_result.get('reason', 'No reason provided')
    if 'HIGH_IV' in reason:
        opportunity_type = "high IV opportunity"
        benefit_description = "maximize premium selling benefits while managing volatility risk"
    else:
        opportunity_type = "time-sensitive opportunity"
        benefit_description = "maximize time decay benefits while minimizing risk"
    
    # Create appropriate summary based on execution status
    if is_successful:
        summary = f"""
# EMERGENCY EXECUTION COMPLETED - {current_datetime}

## DECISION: EMERGENCY EXECUTION

## EXECUTION DETAILS
- **Strategy**: {emergency_result.get('strategy', 'Unknown')}
- **Emergency Level**: {emergency_result.get('emergency_level', 'Unknown')}
- **Reason**: {emergency_result.get('reason', 'No reason provided')}
- **Token Efficiency**: {emergency_result.get('token_efficiency', 'Unknown')}

## EXECUTION RESULT
{emergency_result.get('execution_result', {})}

## EMERGENCY EXECUTION BENEFITS
1. **Speed**: Immediate execution for time-sensitive opportunities
2. **Efficiency**: Minimal token usage (~300 tokens)
3. **Safety**: Conservative parameters reduce risk
4. **Capital Preservation**: Quick action on high-probability setups

## CONCLUSION
Emergency execution completed successfully. The system identified a {opportunity_type} and executed immediately with conservative parameters to {benefit_description}.

**Token Efficiency**: This execution used only ~300 tokens compared to ~7,000-11,000 for full analysis.
"""
    else:
        # Get failure reason
        failure_reason = execution_result.get('message', 'Unknown error')
        summary = f"""
# EMERGENCY EXECUTION FAILED - {current_datetime}

## DECISION: EMERGENCY EXECUTION ATTEMPTED

## EXECUTION DETAILS
- **Strategy**: {emergency_result.get('strategy', 'Unknown')}
- **Emergency Level**: {emergency_result.get('emergency_level', 'Unknown')}
- **Reason**: {emergency_result.get('reason', 'No reason provided')}
- **Token Efficiency**: {emergency_result.get('token_efficiency', 'Unknown')}

## EXECUTION RESULT
{emergency_result.get('execution_result', {})}

## FAILURE ANALYSIS
- **Status**: {execution_status}
- **Failure Reason**: {failure_reason}
- **Impact**: No trade executed due to execution failure

## EMERGENCY EXECUTION ATTEMPT BENEFITS
1. **Speed**: Immediate attempt for time-sensitive opportunities
2. **Efficiency**: Minimal token usage (~300 tokens)
3. **Safety**: Conservative parameters reduce risk
4. **Capital Preservation**: No capital risked due to execution failure

## CONCLUSION
Emergency execution was attempted but failed. The system identified a {opportunity_type} but could not execute due to: {failure_reason}. No capital was risked.

**Token Efficiency**: This attempt used only ~300 tokens compared to ~7,000-11,000 for full analysis.
"""
    
    print(summary)
    return summary

def create_simple_summary(fast_track_result):
    """
    Create a simple summary when no opportunity exists
    """
    print("\n" + "="*80)
    print("📋 SIMPLE SUMMARY: NO OPPORTUNITY DETECTED")
    print("="*80)
    
    summary = f"""
# OPPORTUNITY HUNTER FINAL DECISION - {current_datetime}

## DECISION: WAIT

## PRIMARY REASONING
Fast-track analysis determined no profitable opportunity exists in current market conditions.

## FAST-TRACK ANALYSIS
- **Fast-Track Decision**: {fast_track_result.get('decision', 'UNKNOWN')}
- **Reason for Skipping**: {fast_track_result.get('reason', 'No reason provided')}
- **Capital Preservation**: Decision to wait preserves capital for better opportunities
- **Market Conditions**: Current conditions do not support profitable premium selling strategies

## SUPPORTING FACTORS
1. **Fast-Track Assessment**: Quick analysis identified unfavorable conditions
2. **Capital Preservation**: Avoiding trades in suboptimal conditions
3. **Risk Management**: No compelling risk-reward setup identified
4. **Market Timing**: Current market environment not conducive to profitable strategies
5. **Patience Discipline**: Better to wait for clear opportunities

## CONCLUSION
The decision to wait is optimal for capital preservation. Fast-track analysis efficiently identified that current market conditions do not support profitable trading opportunities. This approach saves significant analysis time and computational resources while maintaining capital for future favorable conditions.

**Token Efficiency**: This decision saved ~7,000-11,000 tokens by avoiding unnecessary detailed analysis.
"""
    
    print(summary)
    return summary

# ============================================================================
# MAIN EXECUTION - THREE-STAGE APPROACH
# ============================================================================

def run_enhanced_three_stage_opportunity_hunter():
    """
    Main function to run the enhanced three-stage opportunity hunter
    This can be called from the crew driver
    """
    print("🚀 Starting THREE-STAGE Opportunity Hunter...")
    print("📋 Stage 1: Fast-track analysis only")
    print("📋 Stage 2: Emergency execution (if high IV/time-sensitive)")
    print("📋 Stage 3: Full crew analysis (if normal opportunity)")
    print("💰 Strategy Priority: PREMIUM SELLING FIRST")
    print("⚡ Emergency Mode: Immediate execution for high IV")
    print("⏱️  Token Efficiency: Optimal for all scenarios")
    print("-" * 50)

    # === PROFIT TARGET CHECK: BLOCK NEW TRADES IF TARGET REACHED ===
    import os
    profit_target_file = "/tmp/algotrade_no_more_trades_today"
    if os.path.exists(profit_target_file):
        print(f"🚫 PROFIT TARGET BLOCK: Day's profit target (₹5,000) already reached. No new trades allowed.")
        print(f"📁 Marker file exists: {profit_target_file}")
        return {
            'decision': 'PROFIT_TARGET_BLOCK', 
            'reason': 'Day profit target (₹5,000) reached - no new trades allowed',
            'timestamp': datetime.now(IST_TIMEZONE).isoformat()
        }

    # Check live balance before running
    try:
        from core_tools.execution_portfolio_tools import get_account_margins
        margins = get_account_margins()
        if margins.get('status') == 'SUCCESS':
            live_balance = margins['equity'].get('live_balance', 0)
            if live_balance < 50000:
                print(f"Live balance is too low (₹{live_balance}). No trades will be executed.")
                return {'decision': 'INSUFFICIENT_BALANCE', 'reason': f'Live balance too low: ₹{live_balance}'}
        else:
            print("Could not fetch account margins. Proceeding with caution.")
    except Exception as e:
        print(f"Error checking live balance: {e}. Proceeding with caution.")

    # Check adjusted day profit before running any opportunity analysis
    try:
        from core_tools.execution_portfolio_tools import get_daily_trading_summary
        summary_result = get_daily_trading_summary()
        if summary_result.get('status') in ['SUCCESS', 'PARTIAL_SUCCESS']:
            summary = summary_result.get('summary', {})
            total_pnl = summary.get('total_pnl', 0)
            total_positions = summary.get('total_positions', 0)
            # For simplicity, treat all positions as open (if you have closed_positions, use that count)
            # Adjusted profit: total_pnl - (total_positions * 2) * 30
            adjusted_profit = total_pnl - (total_positions * 2 * 30)
            if adjusted_profit > 5000:
                print(f"🚫 BLOCKING NEW TRADES: Adjusted day profit (₹{adjusted_profit:.2f}) > ₹5,000. No further trades will be taken today.")
                return {'decision': 'PROFIT_TARGET_BLOCK', 'reason': f'Adjusted day profit (₹{adjusted_profit:.2f}) > ₹5,000. No further trades today.'}
            
            # Log summary status for debugging
            if summary_result.get('status') == 'PARTIAL_SUCCESS':
                print(f"⚠️  Daily summary generated with partial data: {summary_result.get('message', 'Unknown issues')}")
        else:
                print(f"✅ Daily summary generated successfully")
        else:
            print(f"⚠️  Could not fetch daily trading summary: {summary_result.get('message', 'Unknown error')}")
            if summary_result.get('errors'):
                print(f"   Errors: {', '.join(summary_result.get('errors', [])[:3])}")
    except Exception as e:
        print(f"Error checking adjusted day profit: {e}. Proceeding with caution.")

    # STAGE 1: Fast-track analysis only
    fast_track_result = run_stage_1_fast_track_only()
    
    # THREE-STAGE DECISION LOGIC
    if fast_track_result.get('decision') == 'EMERGENCY_EXECUTION':
        # Get the actual emergency signal for accurate messaging
        emergency_signal = fast_track_result.get('emergency_signal', {})
        signal_type = emergency_signal.get('emergency_signal', 'UNKNOWN')
        
        if 'HIGH_IV' in signal_type:
            print("\n🚨 EMERGENCY EXECUTION: High IV opportunity detected")
        else:
            print("\n🚨 EMERGENCY EXECUTION: Time-sensitive opportunity detected")
            
        print("🔄 Proceeding to Stage 2 (emergency execution)")
        print("⚡ Speed: Immediate execution with conservative parameters")
        
        # STAGE 2: Emergency execution
        emergency_result = run_emergency_execution(emergency_signal)
        
        # Create emergency summary for both success and failure cases
        if emergency_result.get('decision') in ['EMERGENCY_EXECUTION_COMPLETE', 'EMERGENCY_EXECUTION_FAILED']:
            result = create_emergency_summary(emergency_result)
        else:
            result = emergency_result
        
    elif fast_track_result.get('decision') in ['NO_OPPORTUNITY', 'SKIP_DETAILED_ANALYSIS']:
        print("\n✅ FAST-TRACK: No opportunity detected")
        print("🔄 Skipping Stage 2 & 3 (no analysis needed)")
        print("💰 Token savings: ~7,000-11,000 tokens")
        
        # Create simple summary
        result = create_simple_summary(fast_track_result)
        
    elif fast_track_result.get('decision') in ['PROCEED_TO_DETAILED', 'DETAILED_ANALYSIS_COMPLETE']:
        print("\n✅ FAST-TRACK: Normal opportunity detected")
        print("🔄 Proceeding to Stage 3 (full crew analysis)")
        
        # STAGE 3: Full crew analysis
        result = run_stage_3_full_analysis()
        
    else:
        print(f"\n⚠️  FAST-TRACK: Unexpected result - {fast_track_result}")
        print("🔄 Proceeding to Stage 3 for safety")
        result = run_stage_3_full_analysis()
    
    print("\n" + "="*80)
    print(f"🏁 THREE-STAGE OPPORTUNITY HUNTER COMPLETED - {current_datetime}")
    print("="*80)
    print("💡 Remember: WAIT = Good outcome (capital preserved)")
    print("🚨 EMERGENCY = Fast execution for high IV opportunities")
    print("⚡ EXECUTE = Only when profitable opportunity exists")
    print("💰 Token Efficiency: Optimal for all scenarios")
    print("="*80)
    
    return result

if __name__ == "__main__":
    # Run the enhanced three-stage opportunity hunter
    result = run_enhanced_three_stage_opportunity_hunter()