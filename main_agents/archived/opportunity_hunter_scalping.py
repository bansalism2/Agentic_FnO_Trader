#!/usr/bin/env python3
"""
AlgoTrade - INTRADAY SCALPING & MOMENTUM OPPORTUNITY HUNTER
=================================================================================

INTRADAY SCALPING & MOMENTUM STRATEGIES:
========================================

1. **MOMENTUM BREAKOUT TRADES** ✅ IMPLEMENTED
   - IDENTIFIES: Strong momentum breakouts with volume confirmation
   - EXECUTES: Directional calls/puts on momentum continuation
   - TIMING: Intraday momentum signals with tight stops
   - RESULT: Captures short-term momentum moves

2. **SCALPING OPPORTUNITIES** ✅ IMPLEMENTED
   - DETECTS: High-frequency price movements and volatility spikes
   - EXECUTES: Quick in-and-out trades with minimal holding time
   - FOCUS: Near-the-money options with high liquidity
   - RESULT: Small but consistent profits from market inefficiencies

3. **VOLATILITY BREAKOUTS** ✅ IMPLEMENTED
   - MONITORS: Sudden volatility expansions and IV spikes
   - EXECUTES: Straddles/strangles on volatility breakouts
   - TIMING: Intraday volatility regime changes
   - RESULT: Profits from volatility expansion

4. **TECHNICAL MOMENTUM** ✅ IMPLEMENTED
   - ANALYZES: RSI, MACD, SuperTrend momentum signals
   - EXECUTES: Directional trades on strong technical momentum
   - CONFIRMATION: Volume and price action validation
   - RESULT: Technical momentum-based profits

5. **LIQUIDITY-BASED SCALPING** ✅ IMPLEMENTED
   - IDENTIFIES: High-liquidity options with tight spreads
   - EXECUTES: Quick scalping trades on liquid instruments
   - FOCUS: ATM and near-ATM options with high volume
   - RESULT: Low-slippage intraday trading

INTRADAY SAFETY FEATURES:
========================
- Strict intraday-only execution (no overnight positions)
- Tight stop-losses and profit targets
- High-frequency monitoring and quick exits
- Volume and liquidity validation for all trades
- Momentum confirmation before entry

RISK MANAGEMENT:
===============
- Maximum 2-3% risk per trade
- Intraday position sizing based on volatility
- Quick exit on momentum reversal
- No overnight exposure
- Strict time-based exits (before 3:15 PM)

Specialized agent for INTRADAY SCALPING and MOMENTUM trading opportunities ONLY.
This agent focuses on quick, high-frequency trades with minimal holding time.

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
    INTRADAY MOMENTUM & SCALPING STRATEGY SELECTION
    """
    try:
        print(f"🎯 INTRADAY MOMENTUM STRATEGY SELECTION:")
        print(f"   Emergency Signal: {emergency_signal_type}")
        
        if 'STRONG_BULL_MOMENTUM' in emergency_signal_type:
            strategy = 'Long Call Scalp'
            reasoning = "Strong bullish momentum = Quick call scalping"
            
        elif 'STRONG_BEAR_MOMENTUM' in emergency_signal_type:
            strategy = 'Long Put Scalp'
            reasoning = "Strong bearish momentum = Quick put scalping"
            
        elif 'VOLATILITY_BREAKOUT' in emergency_signal_type:
            strategy = 'Long Straddle Scalp'
            reasoning = "Volatility breakout = Straddle scalping"
            
        elif 'POOR_LIQUIDITY' in emergency_signal_type:
            strategy = 'WAIT'
            reasoning = "Poor liquidity = Wait for better conditions"
            
        else:
            # Unknown signal = Conservative default
            strategy = 'ATM Scalp'
            reasoning = "Unknown momentum = Conservative ATM scalping"
            
        print(f"   ✅ INTRADAY Strategy: {strategy}")
            print(f"   💡 Reasoning: {reasoning}")
        
        return {
            'strategy': strategy,
            'reasoning': reasoning,
            'signal_type': emergency_signal_type
        }
        
    except Exception as e:
        print(f"❌ Error in intraday strategy selection: {e}")
        # Safe fallback
        return {
            'strategy': 'WAIT',
            'reasoning': 'Error occurred - wait for better conditions',
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
    INTRADAY MOMENTUM & SCALPING EMERGENCY FAST-TRACK
    Detects high-probability intraday momentum and scalping opportunities
    """
    try:
        # Import required modules for timezone handling
        from datetime import datetime
        import pytz
        IST_TIMEZONE = pytz.timezone('Asia/Kolkata')
        
        print("🚨 INTRADAY MOMENTUM & SCALPING FAST-TRACK:")
        
        # Get basic market data
        spot_price = get_nifty_spot_price_safe()
        if not spot_price:
            return {'status': 'ERROR', 'message': 'Cannot get spot price'}
        
        options_chain = get_options_chain_safe()
        if not options_chain:
            return {'status': 'ERROR', 'message': 'Cannot get options chain'}
        
        # Get technical analysis for momentum signals
        technical_analysis = get_nifty_technical_analysis_tool()
        rsi = technical_analysis.get('rsi', 50)
        macd_signal = technical_analysis.get('macd_signal', 'NEUTRAL')
        supertrend_signal = technical_analysis.get('supertrend_signal', 'NEUTRAL')
        
        # Get volume analysis
        volume_analysis = analyze_options_liquidity()
        volume_ratio = volume_analysis.get('volume_metrics', {}).get('volume_ratio', 1.0) if volume_analysis.get('status') == 'SUCCESS' else 1.0
        
        # Get PCR analysis for sentiment
        pcr_analysis = calculate_pcr_technical_analysis_wrapper()
        pcr_signal = pcr_analysis.get('entry_signal', 'NEUTRAL')
        
        print(f"📊 MOMENTUM ANALYSIS:")
        print(f"   RSI: {rsi:.1f}")
        print(f"   MACD Signal: {macd_signal}")
        print(f"   SuperTrend: {supertrend_signal}")
        print(f"   Volume Ratio: {volume_ratio:.2f}")
        print(f"   PCR Signal: {pcr_signal}")
        
        # MOMENTUM BREAKOUT DETECTION
        emergency_signal = 'NO_EMERGENCY'
        
        # Strong momentum breakout with volume
        if volume_ratio > 1.5:
            if rsi > 70 and macd_signal == 'BUY' and supertrend_signal == 'BUY':
                emergency_signal = 'STRONG_BULL_MOMENTUM'
                print(f"🚨 STRONG BULL MOMENTUM: RSI={rsi:.1f}, Volume={volume_ratio:.2f}")
            elif rsi < 30 and macd_signal == 'SELL' and supertrend_signal == 'SELL':
                emergency_signal = 'STRONG_BEAR_MOMENTUM'
                print(f"🚨 STRONG BEAR MOMENTUM: RSI={rsi:.1f}, Volume={volume_ratio:.2f}")
        
        # Volatility breakout detection
        iv_analysis = calculate_iv_rank_analysis_wrapper()
        if iv_analysis.get('status') == 'SUCCESS':
            current_iv = iv_analysis.get('current_iv', 0)
            iv_percentile = iv_analysis.get('iv_percentile', 0)
            
            if iv_percentile > 80 and volume_ratio > 2.0:
                emergency_signal = 'VOLATILITY_BREAKOUT'
                print(f"🚨 VOLATILITY BREAKOUT: IV={current_iv:.4f}, Volume={volume_ratio:.2f}")
        
        # Technical momentum confirmation
        if emergency_signal != 'NO_EMERGENCY':
            # Additional confirmation checks
            if abs(rsi - 50) < 10:  # RSI too neutral
                emergency_signal = 'NO_EMERGENCY'
                print(f"❌ MOMENTUM BLOCKED: RSI too neutral ({rsi:.1f})")
            
            elif volume_ratio < 1.2:  # Insufficient volume
                emergency_signal = 'NO_EMERGENCY'
                print(f"❌ MOMENTUM BLOCKED: Insufficient volume ({volume_ratio:.2f})")
            
            elif pcr_signal == 'NEUTRAL' and abs(rsi - 50) < 15:  # Weak signals
            emergency_signal = 'NO_EMERGENCY'
                print(f"❌ MOMENTUM BLOCKED: Weak technical signals")
                    
        # LIQUIDITY CHECK
        liquidity_analysis = iv_analysis.get('liquidity_analysis', {}) if iv_analysis.get('status') == 'SUCCESS' else {}
            if liquidity_analysis.get('status') == 'SUCCESS':
                liquidity_score = liquidity_analysis.get('liquidity_metrics', {}).get('overall_liquidity_score', 0)
            if liquidity_score < 50:  # Poor liquidity for scalping
                    if emergency_signal != 'NO_EMERGENCY':
                        emergency_signal = f"{emergency_signal}_POOR_LIQUIDITY"
                        print(f"⚠️  POOR LIQUIDITY: Score={liquidity_score}/100")
                    else:
                        emergency_signal = 'POOR_LIQUIDITY'
                        print(f"⚠️  POOR LIQUIDITY ONLY: Score={liquidity_score}/100")
            
            return {
                'status': 'SUCCESS',
                'emergency_signal': emergency_signal,
            'rsi': rsi,
            'macd_signal': macd_signal,
            'supertrend_signal': supertrend_signal,
            'volume_ratio': volume_ratio,
            'pcr_signal': pcr_signal,
            'current_iv': iv_analysis.get('current_iv', 0) if iv_analysis.get('status') == 'SUCCESS' else 0,
            'iv_percentile': iv_analysis.get('iv_percentile', 0) if iv_analysis.get('status') == 'SUCCESS' else 0,
                'liquidity_score': liquidity_analysis.get('liquidity_metrics', {}).get('overall_liquidity_score', 0) if liquidity_analysis.get('status') == 'SUCCESS' else None,
                'spot_price': spot_price,
                'timestamp': datetime.now(IST_TIMEZONE).isoformat()
            }
            
    except Exception as e:
        print(f"❌ Intraday momentum fast-track failed: {str(e)}")
        return {'status': 'ERROR', 'message': f'Intraday momentum fast-track failed: {str(e)}'}

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
        
        # SAFE EMERGENCY EXECUTION with additional safety checks
        print(f"🔒 PERFORMING FINAL SAFETY CHECKS BEFORE EXECUTION:")
        
        # Get additional data for safety checks
        try:
            # Get market regime and technical analysis
            regime_analysis = detect_market_regime_wrapper()
            technical_analysis = get_nifty_technical_analysis_tool()
            
            # Prepare data for safety check
            market_structure = {
                'regime': regime_analysis.get('classification', 'UNKNOWN'),
                'trend_strength': abs(technical_analysis.get('rsi', 50) - 50) / 50,
                'current_price': spot_price,
                'support_levels': [spot_price * 0.98, spot_price * 0.96],
                'resistance_levels': [spot_price * 1.02, spot_price * 1.04]
            }
            
            iv_conditions = {
                'iv_percentile': emergency_signal.get('iv_percentile', 0),
                'iv_ratio': emergency_signal.get('current_iv', 0) / emergency_signal.get('realized_volatility', 1)
            }
            
            # Final safety check
            final_safety = safe_emergency_check(iv_conditions, market_structure, technical_analysis)
            
            if not final_safety.get('emergency_allowed', False):
                print(f"❌ FINAL SAFETY CHECK FAILED: {final_safety.get('reason', 'Unknown')}")
                return {
                    'decision': 'EMERGENCY_EXECUTION_BLOCKED',
                    'reason': final_safety.get('reason', 'Safety check failed'),
                    'risk_level': final_safety.get('risk_level', 'HIGH'),
                    'fallback': 'WAIT - Safety check failed'
                }
            
            print(f"✅ FINAL SAFETY CHECK PASSED: Proceeding with execution")
            
        except Exception as e:
            print(f"⚠️  Final safety check failed: {e}")
            return {
                'decision': 'EMERGENCY_EXECUTION_FAILED',
                'reason': f'Safety check error: {str(e)}',
                'fallback': 'WAIT - Safety check error'
            }
        
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
    """INTRADAY MOMENTUM & SCALPING STRATEGY DETERMINATION"""
    try:
        print(f"🎯 INTRADAY MOMENTUM & SCALPING ANALYSIS:")
        
        iv_percentile = iv_rank.get('iv_percentile', 0)
        regime_class = regime.get('classification', 'UNKNOWN')
        intraday_signal = intraday_tech.get('signal', 'NEUTRAL')
        daily_signal = daily_tech.get('signal', 'NEUTRAL')
        rsi = intraday_tech.get('rsi', 50)
        macd_signal = intraday_tech.get('macd_signal', 'NEUTRAL')
        volume_ratio = comprehensive.get('volume_analysis', {}).get('volume_ratio', 1.0)
        
        print(f"   IV Percentile: {iv_percentile:.1%}")
        print(f"   Market Regime: {regime_class}")
        print(f"   RSI: {rsi:.1f}")
        print(f"   MACD Signal: {macd_signal}")
        print(f"   Volume Ratio: {volume_ratio:.2f}")
        print(f"   Intraday Signal: {intraday_signal}")
        print(f"   Daily Signal: {daily_signal}")
        
        # MOMENTUM BREAKOUT STRATEGIES
        if intraday_signal in ['STRONG_BUY', 'STRONG_SELL'] and volume_ratio > 1.2:
            if intraday_signal == 'STRONG_BUY' and rsi < 70:
                print(f"   ✅ MOMENTUM BREAKOUT: Strong bullish momentum with volume")
                return 'Long Call Scalp'
            elif intraday_signal == 'STRONG_SELL' and rsi > 30:
                print(f"   ✅ MOMENTUM BREAKOUT: Strong bearish momentum with volume")
                return 'Long Put Scalp'
        
        # VOLATILITY BREAKOUT STRATEGIES
        if iv_percentile > 70 and volume_ratio > 1.5:
            print(f"   ✅ VOLATILITY BREAKOUT: High IV with volume spike")
            return 'Long Straddle Scalp'
        
        # TECHNICAL MOMENTUM STRATEGIES
        if abs(rsi - 50) > 15:  # Strong momentum
            if rsi > 65 and macd_signal == 'BUY':
                print(f"   ✅ TECHNICAL MOMENTUM: Strong bullish technical signals")
                return 'Long Call Momentum'
            elif rsi < 35 and macd_signal == 'SELL':
                print(f"   ✅ TECHNICAL MOMENTUM: Strong bearish technical signals")
                return 'Long Put Momentum'
        
        # LIQUIDITY-BASED SCALPING
        if volume_ratio > 2.0 and iv_percentile < 50:
            print(f"   ✅ LIQUIDITY SCALPING: High volume with low IV")
            if intraday_signal in ['BUY', 'STRONG_BUY']:
                return 'Call Scalp'
            elif intraday_signal in ['SELL', 'STRONG_SELL']:
                return 'Put Scalp'
            else:
                return 'ATM Scalp'  # Neutral scalping
        
        # RANGE-BOUND SCALPING
        if regime_class in ['RANGING', 'COMPRESSED'] and 40 <= rsi <= 60:
            if volume_ratio > 1.3:
                print(f"   ✅ RANGE SCALPING: Range-bound with good volume")
                return 'Range Scalp'
        
        # TREND FOLLOWING (for strong trends)
        if regime_class in ['TRENDING_BULL', 'TRENDING_BEAR']:
            if regime_class == 'TRENDING_BULL' and rsi < 75:
                print(f"   ✅ TREND FOLLOWING: Bullish trend continuation")
                return 'Trend Call'
            elif regime_class == 'TRENDING_BEAR' and rsi > 25:
                print(f"   ✅ TREND FOLLOWING: Bearish trend continuation")
                return 'Trend Put'
        
        print(f"   ❌ NO INTRADAY OPPORTUNITY: Insufficient momentum or volume")
        return 'WAIT - No intraday opportunity'
        
    except Exception as e:
        print(f"❌ Intraday strategy determination failed: {e}")
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
# SAFETY FUNCTIONS (MUST BE DEFINED BEFORE AGENTS)
# ============================================================================

def safe_emergency_check(iv_conditions, market_structure, technical_analysis):
    """
    SAFE EMERGENCY CHECK: Multi-factor analysis instead of IV-only logic
    
    CRITICAL: This replaces the dangerous IV-only emergency execution logic
    """
    try:
        print(f"🔒 SAFE EMERGENCY CHECK:")
        print(f"   IV Percentile: {iv_conditions.get('iv_percentile', 0):.1%}")
        print(f"   IV vs Realized: {iv_conditions.get('iv_ratio', 0):.2f}")
        print(f"   Market Regime: {market_structure.get('regime', 'UNKNOWN')}")
        print(f"   Trend Strength: {market_structure.get('trend_strength', 0):.2f}")
        
        # CRITICAL CHECK 1: Market Regime Analysis
        regime = market_structure.get('regime', 'UNKNOWN')
        if regime in ['TRENDING_BULL', 'TRENDING_BEAR']:
            print(f"❌ EMERGENCY BLOCKED: Strong trend detected ({regime}) - dangerous for premium selling")
            return {
                'emergency_allowed': False,
                'reason': f'Strong trend detected: {regime}',
                'risk_level': 'HIGH'
            }
        
        # CRITICAL CHECK 2: Trend Strength Analysis
        trend_strength = market_structure.get('trend_strength', 0)
        if trend_strength > 0.6:
            print(f"❌ EMERGENCY BLOCKED: High trend strength ({trend_strength:.2f}) - dangerous for premium selling")
            return {
                'emergency_allowed': False,
                'reason': f'High trend strength: {trend_strength:.2f}',
                'risk_level': 'HIGH'
            }
        
        # CRITICAL CHECK 3: IV vs Realized Volatility
        iv_ratio = iv_conditions.get('iv_ratio', 0)
        if iv_ratio < 1.3:
            print(f"❌ EMERGENCY BLOCKED: IV not significantly overpriced (ratio: {iv_ratio:.2f})")
            return {
                'emergency_allowed': False,
                'reason': f'IV not overpriced vs realized volatility',
                'risk_level': 'MEDIUM'
            }
        
        # CRITICAL CHECK 4: Technical Analysis Integration
        technical_signal = technical_analysis.get('signal', 'NEUTRAL')
        if technical_signal in ['STRONG_BUY', 'STRONG_SELL']:
            print(f"❌ EMERGENCY BLOCKED: Strong technical signal ({technical_signal}) - potential breakout")
            return {
                'emergency_allowed': False,
                'reason': f'Strong technical signal: {technical_signal}',
                'risk_level': 'HIGH'
            }
        
        # CRITICAL CHECK 5: Support/Resistance Levels
        current_price = market_structure.get('current_price', 0)
        support_levels = market_structure.get('support_levels', [])
        resistance_levels = market_structure.get('resistance_levels', [])
        
        if support_levels and current_price < support_levels[0]:
            print(f"❌ EMERGENCY BLOCKED: Price below key support ({current_price} < {support_levels[0]})")
            return {
                'emergency_allowed': False,
                'reason': 'Price below key support level',
                'risk_level': 'HIGH'
            }
        
        if resistance_levels and current_price > resistance_levels[-1]:
            print(f"❌ EMERGENCY BLOCKED: Price above key resistance ({current_price} > {resistance_levels[-1]})")
            return {
                'emergency_allowed': False,
                'reason': 'Price above key resistance level',
                'risk_level': 'HIGH'
            }
        
        # If all checks pass, emergency execution is allowed
        print(f"✅ EMERGENCY ALLOWED: All safety checks passed")
        return {
            'emergency_allowed': True,
            'reason': 'All safety checks passed',
            'risk_level': 'LOW',
            'recommended_strategy': 'Iron Condor' if regime == 'RANGING' else 'Short Strangle'
        }
        
    except Exception as e:
        print(f"❌ Safe emergency check failed: {e}")
        return {
            'emergency_allowed': False,
            'reason': f'Safety check failed: {str(e)}',
            'risk_level': 'HIGH'
        }

def analyze_volatility_term_structure():
    """
    Analyze IV across different expiries to identify term structure opportunities
    
    CRITICAL: This helps identify calendar spread opportunities and optimal expiry selection
    """
    try:
        print(f"📊 ANALYZING VOLATILITY TERM STRUCTURE:")
        
        # Get available expiries
        expiry_analysis = get_available_expiry_dates_with_analysis()
        if not expiry_analysis or 'available_expiries' not in expiry_analysis:
            return {'status': 'ERROR', 'message': 'Could not get expiry analysis'}
        
        expiries = expiry_analysis['available_expiries']
        if len(expiries) < 2:
            return {'status': 'ERROR', 'message': 'Need at least 2 expiries for term structure analysis'}
        
        # Get IV for different expiries
        term_structure_data = {}
        
        for expiry_info in expiries[:3]:  # Analyze first 3 expiries
            expiry_date = expiry_info.get('date')
            category = expiry_info.get('category', 'UNKNOWN')
            
            if expiry_date:
                # Get options chain for this expiry
                options_chain = get_options_chain_safe(expiry_date=expiry_date)
                if options_chain and options_chain.get('status') == 'SUCCESS':
                    # Calculate ATM IV for this expiry
                    spot_price = options_chain.get('spot_price', 0)
                    atm_strike = options_chain.get('atm_strike', 0)
                    
                    if spot_price and atm_strike:
                        # Find ATM options
                        chain_data = options_chain.get('options_chain', [])
                        atm_option = None
                        for option in chain_data:
                            if abs(option.get('strike', 0) - atm_strike) < 50:  # Within 50 points
                                atm_option = option
                                break
                        
                        if atm_option:
                            # Calculate IV (simplified - using average of CE and PE)
                            ce_price = atm_option.get('CE_ltp', 0)
                            pe_price = atm_option.get('PE_ltp', 0)
                            
                            if ce_price > 0 and pe_price > 0:
                                # Simplified IV calculation (in real implementation, use proper IV calculation)
                                avg_price = (ce_price + pe_price) / 2
                                # Rough IV estimate (this should be replaced with proper IV calculation)
                                rough_iv = (avg_price / spot_price) * 100
                                
                                term_structure_data[expiry_date] = {
                                    'category': category,
                                    'rough_iv': rough_iv,
                                    'ce_price': ce_price,
                                    'pe_price': pe_price,
                                    'strike': atm_option.get('strike', 0)
                                }
        
        if len(term_structure_data) < 2:
            return {'status': 'ERROR', 'message': 'Insufficient data for term structure analysis'}
        
        # Analyze term structure
        expiry_dates = list(term_structure_data.keys())
        expiry_dates.sort()  # Sort by date
        
        if len(expiry_dates) >= 2:
            front_month = expiry_dates[0]
            back_month = expiry_dates[1]
            
            front_iv = term_structure_data[front_month]['rough_iv']
            back_iv = term_structure_data[back_month]['rough_iv']
            
            iv_ratio = front_iv / back_iv if back_iv > 0 else 0
            
            print(f"   Front Month ({front_month}): {front_iv:.2f}%")
            print(f"   Back Month ({back_month}): {back_iv:.2f}%")
            print(f"   IV Ratio: {iv_ratio:.2f}")
            
            # Determine term structure condition
            if iv_ratio > 1.3:
                term_structure = 'BACKWARDATION'
                opportunity = 'SELL_FRONT_BUY_BACK'
                print(f"   📈 BACKWARDATION: Front month expensive - Calendar spread opportunity")
            elif iv_ratio < 0.7:
                term_structure = 'CONTANGO'
                opportunity = 'SELL_BACK_BUY_FRONT'
                print(f"   📉 CONTANGO: Back month expensive - Reverse calendar opportunity")
            else:
                term_structure = 'NORMAL'
                opportunity = 'NO_TERM_STRUCTURE_OPPORTUNITY'
                print(f"   ➡️  NORMAL: Standard term structure")
            
            return {
                'status': 'SUCCESS',
                'term_structure': term_structure,
                'opportunity': opportunity,
                'front_month_iv': front_iv,
                'back_month_iv': back_iv,
                'iv_ratio': iv_ratio,
                'front_month_date': front_month,
                'back_month_date': back_month,
                'data': term_structure_data
            }
        else:
            return {'status': 'ERROR', 'message': 'Insufficient expiry data'}
        
    except Exception as e:
        print(f"❌ Volatility term structure analysis failed: {e}")
        return {'status': 'ERROR', 'message': f'Analysis failed: {str(e)}'}

def integrate_technical_with_options(technical_analysis, options_strategy, current_price):
    """
    Integrate technical analysis with options strategy selection
    
    CRITICAL: This ensures strikes are selected based on support/resistance levels
    """
    try:
        print(f"🔗 INTEGRATING TECHNICAL ANALYSIS WITH OPTIONS STRATEGY:")
        print(f"   Strategy: {options_strategy}")
        print(f"   Current Price: {current_price}")
        
        # Extract technical levels
        rsi = technical_analysis.get('rsi', 50)
        signal = technical_analysis.get('signal', 'NEUTRAL')
        
        # Calculate support and resistance levels based on technical analysis
        support_levels = []
        resistance_levels = []
        
        # Simple support/resistance calculation based on RSI
        if rsi < 30:  # Oversold
            support_levels = [current_price * 0.98, current_price * 0.96]
            resistance_levels = [current_price * 1.02, current_price * 1.04]
        elif rsi > 70:  # Overbought
            support_levels = [current_price * 0.96, current_price * 0.94]
            resistance_levels = [current_price * 1.04, current_price * 1.06]
        else:  # Neutral
            support_levels = [current_price * 0.97, current_price * 0.95]
            resistance_levels = [current_price * 1.03, current_price * 1.05]
        
        # Strategy-specific strike selection
        if options_strategy == 'Short Strangle':
            # Sell OTM options based on support/resistance
            put_strike = support_levels[0] if support_levels else current_price * 0.97
            call_strike = resistance_levels[0] if resistance_levels else current_price * 1.03
            
            print(f"   📊 Short Strangle Strikes:")
            print(f"      Put Strike: {put_strike:.0f} (near support)")
            print(f"      Call Strike: {call_strike:.0f} (near resistance)")
            
            return {
                'status': 'SUCCESS',
                'strategy': 'Short Strangle',
                'put_strike': put_strike,
                'call_strike': call_strike,
                'support_levels': support_levels,
                'resistance_levels': resistance_levels,
                'technical_signal': signal
            }
            
        elif options_strategy == 'Iron Condor':
            # Sell spreads within support/resistance range
            put_spread_lower = support_levels[0] if support_levels else current_price * 0.97
            put_spread_upper = support_levels[1] if len(support_levels) > 1 else current_price * 0.98
            call_spread_lower = resistance_levels[0] if resistance_levels else current_price * 1.02
            call_spread_upper = resistance_levels[1] if len(resistance_levels) > 1 else current_price * 1.03
            
            print(f"   📊 Iron Condor Strikes:")
            print(f"      Put Spread: {put_spread_lower:.0f} / {put_spread_upper:.0f}")
            print(f"      Call Spread: {call_spread_lower:.0f} / {call_spread_upper:.0f}")
            
            return {
                'status': 'SUCCESS',
                'strategy': 'Iron Condor',
                'put_spread_lower': put_spread_lower,
                'put_spread_upper': put_spread_upper,
                'call_spread_lower': call_spread_lower,
                'call_spread_upper': call_spread_upper,
                'support_levels': support_levels,
                'resistance_levels': resistance_levels,
                'technical_signal': signal
            }
            
        else:
            # Generic strike selection
            print(f"   📊 Generic Strike Selection:")
            print(f"      Support Levels: {[f'{s:.0f}' for s in support_levels]}")
            print(f"      Resistance Levels: {[f'{r:.0f}' for r in resistance_levels]}")
            
            return {
                'status': 'SUCCESS',
                'strategy': options_strategy,
                'support_levels': support_levels,
                'resistance_levels': resistance_levels,
                'technical_signal': signal,
                'recommended_strikes': {
                    'conservative': {
                        'put_strike': support_levels[0] if support_levels else current_price * 0.97,
                        'call_strike': resistance_levels[0] if resistance_levels else current_price * 1.03
                    },
                    'aggressive': {
                        'put_strike': support_levels[1] if len(support_levels) > 1 else current_price * 0.95,
                        'call_strike': resistance_levels[1] if len(resistance_levels) > 1 else current_price * 1.05
                    }
                }
            }
        
    except Exception as e:
        print(f"❌ Technical integration failed: {e}")
        return {'status': 'ERROR', 'message': f'Integration failed: {str(e)}'}

# ============================================================================
# SIMPLIFIED AGENTS - FOCUS ON NEW OPPORTUNITIES ONLY
# ============================================================================

# Market Opportunity Analyst (with Intraday-Only Constraints)
market_analyst = Agent(
    role="INTRADAY MOMENTUM & SCALPING ANALYST",
    goal="Identify high-probability intraday momentum and scalping opportunities in NIFTY F&O markets while preserving capital through quick, disciplined trades.",
    backstory=f"""You are an INTRADAY MOMENTUM & SCALPING specialist focused ONLY on finding profitable short-term trading setups.
    You assume the portfolio is already clean and managed by another system.
    
    CURRENT DATE AND TIME: {current_datetime}
    
    **Your Core Philosophy**: Quick profits through momentum capture. You believe that intraday momentum 
    and scalping opportunities provide the best risk-reward for short-term trading. Your success comes 
    from identifying strong momentum moves early and capturing small but consistent profits through 
    quick in-and-out trades. You understand that timing is everything in scalping.
    
    **MOMENTUM & SCALPING FIRST APPROACH**: You prioritize momentum-based strategies because:
    - Momentum trades capture quick directional moves
    - Scalping provides consistent small profits
    - Intraday trades avoid overnight risk
    - High-frequency trading capitalizes on market inefficiencies
    - Volume confirmation validates momentum signals
    
    Your job is simple:
    1. Analyze current market conditions for momentum signals
    2. Identify ONLY profitable, high-probability intraday opportunities  
    3. **ALWAYS prefer momentum and scalping strategies** (Long Call/Put Scalp, Straddle Scalp, ATM Scalp)
    4. Recommend directional trades when strong momentum signals exist
    
    You do NOT manage existing positions - that's handled elsewhere.
    You focus purely on: "Is there a profitable momentum/scalping opportunity right now?"
    
    **CRITICAL: MOMENTUM ANALYSIS**
    - **Technical Momentum**: RSI, MACD, SuperTrend for directional signals
    - **Volume Analysis**: High volume confirms momentum strength
    - **Volatility Breakouts**: IV spikes create scalping opportunities
    - **Liquidity Check**: High liquidity ensures quick entry/exit
    - **Time Management**: Avoid trades too close to market close
    
    **SCALPING DISCIPLINE**: Quick profits, tight stops, no overnight exposure.
    
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
        # === NEW SAFETY TOOLS ===
        tool("Safe Emergency Check with Multi-Factor Analysis")(safe_emergency_check),
        tool("Volatility Term Structure Analysis")(analyze_volatility_term_structure),
        tool("Technical Integration with Options Strategy")(integrate_technical_with_options),
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
    role="INTRADAY CAPITAL & POSITION CHECKER", 
    goal="Verify if sufficient capital exists for new intraday trades and check basic portfolio constraints.",
    backstory=f"""You are a simple INTRADAY capital checker. Your job is straightforward:
    
    1. Check available capital for intraday trades
    2. Count existing positions  
    3. Give go/no-go for new intraday trades
    
    SIMPLE RULES:
    - 0-5 positions: OK to trade (if capital available)
    - 5+ positions: NO new trades (avoid overexposure)
    - After 14:30: NO new trades (insufficient time for intraday)
    - Insufficient capital: NO trades
    - Poor liquidity: NO trades (critical for scalping)
    
    You don't manage existing positions - just check if new intraday trades are allowed.
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
    role="INTRADAY MOMENTUM & SCALPING EXECUTOR",
    goal="Execute NEW intraday momentum and scalping trades only when all conditions are perfect. Default is to WAIT and preserve capital.",
    backstory=f"""You are a disciplined INTRADAY MOMENTUM & SCALPING executor focused ONLY on profitable short-term opportunities.
    
    CURRENT DATE AND TIME: {current_datetime}
    
    **Your Core Philosophy**: Quick profits through momentum capture. You believe that intraday momentum 
    and scalping opportunities provide the best risk-reward for short-term trading. Your success comes 
    from identifying strong momentum moves early and capturing small but consistent profits through 
    quick in-and-out trades. You understand that timing is everything in scalping.
    
    **MOMENTUM & SCALPING FIRST APPROACH**: You prioritize momentum-based strategies because:
    - Momentum trades capture quick directional moves
    - Scalping provides consistent small profits
    - Intraday trades avoid overnight risk
    - High-frequency trading capitalizes on market inefficiencies
    - Volume confirmation validates momentum signals
    
    Your philosophy: "Quick profits, tight stops, no overnight exposure."
    
    **MANDATORY EXECUTION PROCESS:**
    You ALWAYS follow a two-step process:
    1. **CREATE**: Use strategy creation tools to build strategy structure (no execution)
    2. **EXECUTE**: Use execute_and_store_strategy() to execute and store the trade
    
    **STRATEGY PRIORITY ORDER:**
    1. **Long Call Scalp** (bullish momentum scalping)
    2. **Long Put Scalp** (bearish momentum scalping)
    3. **Long Straddle Scalp** (volatility breakout scalping)
    4. **Call Scalp** / **Put Scalp** (directional scalping)
    5. **ATM Scalp** (neutral scalping)
    6. **Range Scalp** (range-bound scalping)
    7. **Trend Call** / **Trend Put** (trend following)
    
    You execute trades ONLY when:
    1. Market analysis shows STRONG MOMENTUM setup (not just good, but strong momentum)
    2. Capital checker confirms availability  
    3. Time is before 14:30
    4. Risk-reward is compelling (>1.5:1 for scalping)
    5. All momentum and volume conditions are perfectly aligned
    6. **Momentum opportunity exists** (strong technical signals + volume confirmation)
    7. No doubts or uncertainties exist
    
    You do NOT:
    - Manage existing positions
    - Close conflicting positions  
    - Do complex portfolio analysis
    - Use strategy creation tools for execution (they only create structures)
    - Force trades in uncertain conditions
    
    You simply ask: "Is this a profitable momentum/scalping opportunity with clear high probability?"
    If yes: Create strategy → Execute and store. If no: Wait and preserve capital.
    
    **SCALPING DISCIPLINE**: Quick profits, tight stops, no overnight exposure.
    
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
        # === NEW SAFETY TOOLS ===
        tool("Safe Emergency Check with Multi-Factor Analysis")(safe_emergency_check),
        tool("Volatility Term Structure Analysis")(analyze_volatility_term_structure),
        tool("Technical Integration with Options Strategy")(integrate_technical_with_options),
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
    Find profitable INTRADAY MOMENTUM & SCALPING opportunities for NEW positions while preserving capital:
    
    CURRENT DATE AND TIME: {current_datetime}
    
    **CRITICAL: Your output determines if subsequent tasks run.**
    - If you find NO_OPPORTUNITY, capital checker and trade executor will be SKIPPED
    - This saves significant token usage and execution time
    - Only proceed if you find PROFITABLE opportunities
    
    **MOMENTUM & SCALPING FIRST MANDATE**: Your primary goal is to find momentum and scalping opportunities that benefit from quick directional moves. Focus on strategies that capture short-term momentum with tight risk management.
    
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
       
       **STRATEGY SELECTION LOGIC (MOMENTUM & SCALPING PRIORITY):**
       - **MOMENTUM & SCALPING FIRST APPROACH**: Always prefer momentum-based strategies to benefit from quick directional moves
         * **Primary Preference**: Long Call Scalp, Long Put Scalp, Long Straddle Scalp, ATM Scalp
         * **Secondary Preference**: Range Scalp, Trend Call/Put for range-bound and trend-following
         * **Momentum Advantage**: Capturing quick directional moves with tight risk management
       
       - **Momentum-Based Strategy Selection**:
         * **Strong Bullish Momentum**: Long Call Scalp, Call Scalp (RSI >70, MACD BUY, Volume >1.5x)
         * **Strong Bearish Momentum**: Long Put Scalp, Put Scalp (RSI <30, MACD SELL, Volume >1.5x)
         * **Volatility Breakout**: Long Straddle Scalp (IV >80%, Volume >2.0x)
         * **Range-Bound**: Range Scalp (RSI 40-60, Volume >1.3x)
       
       - **Technical-Based Strategy Selection**:
         * **Strong Technical Signals**: Directional scalping based on RSI, MACD, SuperTrend alignment
         * **Volume Confirmation**: High volume validates momentum signals
         * **Liquidity Check**: High liquidity ensures quick entry/exit for scalping
         * **Time Management**: Avoid trades too close to market close
       
       - **Final Decision Logic**:
         * **DEFAULT**: Always look for momentum and scalping opportunities first
         * **Momentum Criteria**: Strong technical signals + volume confirmation + liquidity
         * **Scalping Criteria**: Quick profit potential + tight risk management + intraday timeframe
         * If Comprehensive Analysis shows NEUTRAL: Use ATM Scalp or Range Scalp
         * If Comprehensive Analysis shows WAIT: No trade (preserve capital)
         * **SCALPING RULE**: If ANY doubt exists about the setup: WAIT and preserve capital
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
    Execute NEW INTRADAY MOMENTUM & SCALPING trades ONLY if profitable opportunity exists with capital preservation focus:
    
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
    
    **YOUR JOB**: Analyze the results from all previous agents and provide a clear, detailed explanation of why the final decision was made, with emphasis on momentum and scalping opportunities.
    
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
       - **Momentum & Scalping Assessment**: Was there a momentum or scalping opportunity? (technical signals, volume, liquidity)
    
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
    Main function to run the INTRADAY SCALPING & MOMENTUM HUNTER
    This can be called from the crew driver
    """
    print("🚀 Starting INTRADAY SCALPING & MOMENTUM HUNTER...")
    print("📋 Stage 1: Fast-track momentum analysis only")
    print("📋 Stage 2: Emergency momentum execution (if strong breakout)")
    print("📋 Stage 3: Full crew analysis (if normal opportunity)")
    print("💰 Strategy Priority: MOMENTUM & SCALPING FIRST")
    print("⚡ Emergency Mode: Immediate execution for momentum breakouts")
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
        
        if 'MOMENTUM' in signal_type:
            print("\n🚨 EMERGENCY EXECUTION: Strong momentum opportunity detected")
        else:
            print("\n🚨 EMERGENCY EXECUTION: Time-sensitive opportunity detected")
            
        print("🔄 Proceeding to Stage 2 (emergency momentum execution)")
        print("⚡ Speed: Immediate execution with tight stops")
        
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
    print(f"🏁 INTRADAY SCALPING & MOMENTUM HUNTER COMPLETED - {current_datetime}")
    print("="*80)
    print("💡 Remember: WAIT = Good outcome (capital preserved)")
    print("🚨 MOMENTUM = Fast execution for momentum breakouts")
    print("⚡ SCALP = Quick in-and-out trades for small profits")
    print("💰 Token Efficiency: Optimal for all scenarios")
    print("="*80)
    
    return result

def safe_emergency_check(iv_conditions, market_structure, technical_analysis):
    """
    SAFE EMERGENCY CHECK: Multi-factor analysis instead of IV-only logic
    
    CRITICAL: This replaces the dangerous IV-only emergency execution logic
    """
    try:
        print(f"🔒 SAFE EMERGENCY CHECK:")
        print(f"   IV Percentile: {iv_conditions.get('iv_percentile', 0):.1%}")
        print(f"   IV vs Realized: {iv_conditions.get('iv_ratio', 0):.2f}")
        print(f"   Market Regime: {market_structure.get('regime', 'UNKNOWN')}")
        print(f"   Trend Strength: {market_structure.get('trend_strength', 0):.2f}")
        
        # CRITICAL CHECK 1: Market Regime Analysis
        regime = market_structure.get('regime', 'UNKNOWN')
        if regime in ['TRENDING_BULL', 'TRENDING_BEAR']:
            print(f"❌ EMERGENCY BLOCKED: Strong trend detected ({regime}) - dangerous for premium selling")
            return {
                'emergency_allowed': False,
                'reason': f'Strong trend detected: {regime}',
                'risk_level': 'HIGH'
            }
        
        # CRITICAL CHECK 2: Trend Strength Analysis
        trend_strength = market_structure.get('trend_strength', 0)
        if trend_strength > 0.6:
            print(f"❌ EMERGENCY BLOCKED: High trend strength ({trend_strength:.2f}) - dangerous for premium selling")
            return {
                'emergency_allowed': False,
                'reason': f'High trend strength: {trend_strength:.2f}',
                'risk_level': 'HIGH'
            }
        
        # CRITICAL CHECK 3: IV vs Realized Volatility
        iv_ratio = iv_conditions.get('iv_ratio', 0)
        if iv_ratio < 1.3:
            print(f"❌ EMERGENCY BLOCKED: IV not significantly overpriced (ratio: {iv_ratio:.2f})")
            return {
                'emergency_allowed': False,
                'reason': f'IV not overpriced vs realized volatility',
                'risk_level': 'MEDIUM'
            }
        
        # CRITICAL CHECK 4: Technical Analysis Integration
        technical_signal = technical_analysis.get('signal', 'NEUTRAL')
        if technical_signal in ['STRONG_BUY', 'STRONG_SELL']:
            print(f"❌ EMERGENCY BLOCKED: Strong technical signal ({technical_signal}) - potential breakout")
            return {
                'emergency_allowed': False,
                'reason': f'Strong technical signal: {technical_signal}',
                'risk_level': 'HIGH'
            }
        
        # CRITICAL CHECK 5: Support/Resistance Levels
        current_price = market_structure.get('current_price', 0)
        support_levels = market_structure.get('support_levels', [])
        resistance_levels = market_structure.get('resistance_levels', [])
        
        if support_levels and current_price < support_levels[0]:
            print(f"❌ EMERGENCY BLOCKED: Price below key support ({current_price} < {support_levels[0]})")
            return {
                'emergency_allowed': False,
                'reason': 'Price below key support level',
                'risk_level': 'HIGH'
            }
        
        if resistance_levels and current_price > resistance_levels[-1]:
            print(f"❌ EMERGENCY BLOCKED: Price above key resistance ({current_price} > {resistance_levels[-1]})")
            return {
                'emergency_allowed': False,
                'reason': 'Price above key resistance level',
                'risk_level': 'HIGH'
            }
        
        # If all checks pass, emergency execution is allowed
        print(f"✅ EMERGENCY ALLOWED: All safety checks passed")
        return {
            'emergency_allowed': True,
            'reason': 'All safety checks passed',
            'risk_level': 'LOW',
            'recommended_strategy': 'Iron Condor' if regime == 'RANGING' else 'Short Strangle'
        }
        
    except Exception as e:
        print(f"❌ Safe emergency check failed: {e}")
        return {
            'emergency_allowed': False,
            'reason': f'Safety check failed: {str(e)}',
            'risk_level': 'HIGH'
        }

def analyze_volatility_term_structure():
    """
    Analyze IV across different expiries to identify term structure opportunities
    
    CRITICAL: This helps identify calendar spread opportunities and optimal expiry selection
    """
    try:
        print(f"📊 ANALYZING VOLATILITY TERM STRUCTURE:")
        
        # Get available expiries
        expiry_analysis = get_available_expiry_dates_with_analysis()
        if not expiry_analysis or 'available_expiries' not in expiry_analysis:
            return {'status': 'ERROR', 'message': 'Could not get expiry analysis'}
        
        expiries = expiry_analysis['available_expiries']
        if len(expiries) < 2:
            return {'status': 'ERROR', 'message': 'Need at least 2 expiries for term structure analysis'}
        
        # Get IV for different expiries
        term_structure_data = {}
        
        for expiry_info in expiries[:3]:  # Analyze first 3 expiries
            expiry_date = expiry_info.get('date')
            category = expiry_info.get('category', 'UNKNOWN')
            
            if expiry_date:
                # Get options chain for this expiry
                options_chain = get_options_chain_safe(expiry_date=expiry_date)
                if options_chain and options_chain.get('status') == 'SUCCESS':
                    # Calculate ATM IV for this expiry
                    spot_price = options_chain.get('spot_price', 0)
                    atm_strike = options_chain.get('atm_strike', 0)
                    
                    if spot_price and atm_strike:
                        # Find ATM options
                        chain_data = options_chain.get('options_chain', [])
                        atm_option = None
                        for option in chain_data:
                            if abs(option.get('strike', 0) - atm_strike) < 50:  # Within 50 points
                                atm_option = option
                                break
                        
                        if atm_option:
                            # Calculate IV (simplified - using average of CE and PE)
                            ce_price = atm_option.get('CE_ltp', 0)
                            pe_price = atm_option.get('PE_ltp', 0)
                            
                            if ce_price > 0 and pe_price > 0:
                                # Simplified IV calculation (in real implementation, use proper IV calculation)
                                avg_price = (ce_price + pe_price) / 2
                                # Rough IV estimate (this should be replaced with proper IV calculation)
                                rough_iv = (avg_price / spot_price) * 100
                                
                                term_structure_data[expiry_date] = {
                                    'category': category,
                                    'rough_iv': rough_iv,
                                    'ce_price': ce_price,
                                    'pe_price': pe_price,
                                    'strike': atm_option.get('strike', 0)
                                }
        
        if len(term_structure_data) < 2:
            return {'status': 'ERROR', 'message': 'Insufficient data for term structure analysis'}
        
        # Analyze term structure
        expiry_dates = list(term_structure_data.keys())
        expiry_dates.sort()  # Sort by date
        
        if len(expiry_dates) >= 2:
            front_month = expiry_dates[0]
            back_month = expiry_dates[1]
            
            front_iv = term_structure_data[front_month]['rough_iv']
            back_iv = term_structure_data[back_month]['rough_iv']
            
            iv_ratio = front_iv / back_iv if back_iv > 0 else 0
            
            print(f"   Front Month ({front_month}): {front_iv:.2f}%")
            print(f"   Back Month ({back_month}): {back_iv:.2f}%")
            print(f"   IV Ratio: {iv_ratio:.2f}")
            
            # Determine term structure condition
            if iv_ratio > 1.3:
                term_structure = 'BACKWARDATION'
                opportunity = 'SELL_FRONT_BUY_BACK'
                print(f"   📈 BACKWARDATION: Front month expensive - Calendar spread opportunity")
            elif iv_ratio < 0.7:
                term_structure = 'CONTANGO'
                opportunity = 'SELL_BACK_BUY_FRONT'
                print(f"   📉 CONTANGO: Back month expensive - Reverse calendar opportunity")
            else:
                term_structure = 'NORMAL'
                opportunity = 'NO_TERM_STRUCTURE_OPPORTUNITY'
                print(f"   ➡️  NORMAL: Standard term structure")
            
            return {
                'status': 'SUCCESS',
                'term_structure': term_structure,
                'opportunity': opportunity,
                'front_month_iv': front_iv,
                'back_month_iv': back_iv,
                'iv_ratio': iv_ratio,
                'front_month_date': front_month,
                'back_month_date': back_month,
                'data': term_structure_data
            }
        else:
            return {'status': 'ERROR', 'message': 'Insufficient expiry data'}
            
    except Exception as e:
        print(f"❌ Term structure analysis failed: {e}")
        return {'status': 'ERROR', 'message': f'Term structure analysis failed: {str(e)}'}

def integrate_technical_with_options(technical_analysis, options_strategy, current_price):
    """
    Integrate technical analysis with options strategy selection
    
    CRITICAL: This ensures strikes are selected based on support/resistance levels
    """
    try:
        print(f"🔗 INTEGRATING TECHNICAL ANALYSIS WITH OPTIONS STRATEGY:")
        print(f"   Strategy: {options_strategy}")
        print(f"   Current Price: {current_price}")
        
        # Extract technical levels
        rsi = technical_analysis.get('rsi', 50)
        signal = technical_analysis.get('signal', 'NEUTRAL')
        
        # Calculate support and resistance levels based on technical analysis
        support_levels = []
        resistance_levels = []
        
        # Simple support/resistance calculation based on RSI
        if rsi < 30:  # Oversold
            support_levels = [current_price * 0.98, current_price * 0.96]
            resistance_levels = [current_price * 1.02, current_price * 1.04]
        elif rsi > 70:  # Overbought
            support_levels = [current_price * 0.96, current_price * 0.94]
            resistance_levels = [current_price * 1.04, current_price * 1.06]
        else:  # Neutral
            support_levels = [current_price * 0.97, current_price * 0.95]
            resistance_levels = [current_price * 1.03, current_price * 1.05]
        
        # Strategy-specific strike selection
        if options_strategy == 'Short Strangle':
            # Sell OTM options based on support/resistance
            put_strike = support_levels[0] if support_levels else current_price * 0.97
            call_strike = resistance_levels[0] if resistance_levels else current_price * 1.03
            
            print(f"   📊 Short Strangle Strikes:")
            print(f"      Put Strike: {put_strike:.0f} (near support)")
            print(f"      Call Strike: {call_strike:.0f} (near resistance)")
            
            return {
                'status': 'SUCCESS',
                'strategy': 'Short Strangle',
                'put_strike': put_strike,
                'call_strike': call_strike,
                'support_levels': support_levels,
                'resistance_levels': resistance_levels,
                'technical_signal': signal
            }
            
        elif options_strategy == 'Iron Condor':
            # Sell spreads within support/resistance range
            put_spread_lower = support_levels[0] if support_levels else current_price * 0.97
            put_spread_upper = support_levels[1] if len(support_levels) > 1 else current_price * 0.98
            call_spread_lower = resistance_levels[0] if resistance_levels else current_price * 1.02
            call_spread_upper = resistance_levels[1] if len(resistance_levels) > 1 else current_price * 1.03
            
            print(f"   📊 Iron Condor Strikes:")
            print(f"      Put Spread: {put_spread_lower:.0f} / {put_spread_upper:.0f}")
            print(f"      Call Spread: {call_spread_lower:.0f} / {call_spread_upper:.0f}")
            
            return {
                'status': 'SUCCESS',
                'strategy': 'Iron Condor',
                'put_spread_lower': put_spread_lower,
                'put_spread_upper': put_spread_upper,
                'call_spread_lower': call_spread_lower,
                'call_spread_upper': call_spread_upper,
                'support_levels': support_levels,
                'resistance_levels': resistance_levels,
                'technical_signal': signal
            }
            
        else:
            # Generic strike selection
            print(f"   📊 Generic Strike Selection:")
            print(f"      Support Levels: {[f'{s:.0f}' for s in support_levels]}")
            print(f"      Resistance Levels: {[f'{r:.0f}' for r in resistance_levels]}")
            
            return {
                'status': 'SUCCESS',
                'strategy': options_strategy,
                'support_levels': support_levels,
                'resistance_levels': resistance_levels,
                'technical_signal': signal,
                'recommended_strikes': {
                    'conservative': {
                        'put_strike': support_levels[0] if support_levels else current_price * 0.97,
                        'call_strike': resistance_levels[0] if resistance_levels else current_price * 1.03
                    },
                    'aggressive': {
                        'put_strike': support_levels[1] if len(support_levels) > 1 else current_price * 0.95,
                        'call_strike': resistance_levels[1] if len(resistance_levels) > 1 else current_price * 1.05
                    }
                }
            }
        
    except Exception as e:
        print(f"❌ Technical integration failed: {e}")
        return {
            'strategy': options_strategy,
            'error': f'Technical integration failed: {str(e)}'
        }

if __name__ == "__main__":
    # Run the enhanced three-stage opportunity hunter
    result = run_enhanced_three_stage_opportunity_hunter()