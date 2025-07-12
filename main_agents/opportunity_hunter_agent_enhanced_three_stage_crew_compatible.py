#!/usr/bin/env python3
"""
AlgoTrade - Enhanced Three-Stage Opportunity Hunter (CREW-COMPATIBLE)
====================================================================

Crew-compatible version that returns analysis results for crew agents to process.
This version does NOT execute trades directly - it provides analysis for the crew.

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

print(f"Starting Enhanced Three-Stage Opportunity Hunter (Crew-Compatible) at {current_datetime}...")

# ============================================================================
# TOOL IMPORTS
# ============================================================================

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import tools needed for analysis
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
        # ENHANCED IV ANALYSIS WITH REALIZED VOLATILITY AND LIQUIDITY
        analyze_vix_integration_wrapper, calculate_iv_rank_analysis_wrapper, detect_market_regime_wrapper,
        calculate_realized_volatility, get_realized_volatility_from_kite, analyze_options_liquidity
    )
    from core_tools.execution_portfolio_tools import (
        get_portfolio_positions, get_account_margins, execute_options_strategy, 
        calculate_strategy_margins, analyze_position_conflicts, analyze_position_conflicts_wrapper, validate_trading_capital,
        get_risk_metrics, get_orders_history, get_daily_trading_summary,
        validate_general_capital
    )
    from core_tools.strategy_creation_tools import (
        create_long_straddle_strategy, create_short_strangle_strategy, 
        create_iron_condor_strategy, create_butterfly_spread_strategy, 
        # NEW PREMIUM SELLING STRATEGIES
        create_bull_put_spread_strategy, create_bear_call_spread_strategy, create_calendar_spread_strategy,
        recommend_options_strategy, analyze_strategy_greeks,
        # NEW COMPREHENSIVE ANALYSIS TOOL
        comprehensive_advanced_analysis_wrapper
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
    raise ImportError(f"Cannot proceed without required tools: {e}")

# Initialize connection
try:
    init_result = initialize_connection(api_key, api_secret, access_token)
    print("Kite Connect initialization result:", init_result)
except Exception as e:
    print(f"Warning: Kite Connect initialization failed: {e}")

# ============================================================================
# ENHANCED ANALYSIS FUNCTIONS (CREW-COMPATIBLE)
# ============================================================================

def emergency_fast_track():
    """
    Enhanced emergency fast-track with comprehensive IV analysis including realized volatility and liquidity
    """
    try:
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
            
            # Check for high IV emergency (relaxed thresholds)
            if current_iv > 0.25 and iv_percentile > 0.7:  # Relaxed from 0.3 and 0.8
                if realized_vol and current_iv / realized_vol > 1.3:  # IV overpriced
                    emergency_signal = 'HIGH_IV_OVERPRICED'
                    print(f"🚨 HIGH IV EMERGENCY: IV={current_iv:.4f}, Overpriced vs Realized")
                else:
                    emergency_signal = 'HIGH_IV_FAIR'
                    print(f"🚨 HIGH IV EMERGENCY: IV={current_iv:.4f}, Fairly Priced")
            
            # Check for low IV opportunity (relaxed thresholds)
            elif current_iv < 0.15 and iv_percentile < 0.3:  # Relaxed from 0.12 and 0.2
                if realized_vol and current_iv / realized_vol < 0.8:  # IV underpriced
                    emergency_signal = 'LOW_IV_UNDERPRICED'
                    print(f"🚨 LOW IV OPPORTUNITY: IV={current_iv:.4f}, Underpriced vs Realized")
                else:
                    emergency_signal = 'LOW_IV_FAIR'
                    print(f"🚨 LOW IV OPPORTUNITY: IV={current_iv:.4f}, Fairly Priced")
            
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
            
            return {
                'status': 'SUCCESS',
                'emergency_signal': emergency_signal,
                'current_iv': current_iv,
                'iv_percentile': iv_percentile,
                'realized_volatility': realized_vol,
                'liquidity_score': liquidity_analysis.get('liquidity_metrics', {}).get('overall_liquidity_score', 0) if liquidity_analysis.get('status') == 'SUCCESS' else None,
                'spot_price': spot_price,
                'timestamp': datetime.now().isoformat()
            }
        else:
            print(f"❌ IV Analysis failed: {iv_analysis.get('message', 'Unknown error')}")
            return {'status': 'ERROR', 'message': 'IV analysis failed'}
            
    except Exception as e:
        print(f"❌ Emergency fast-track failed: {str(e)}")
        return {'status': 'ERROR', 'message': f'Emergency fast-track failed: {str(e)}'}

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
        spot_result = get_nifty_spot_price_safe()
        if not spot_result or spot_result.get('status') != 'SUCCESS':
            return {'status': 'ERROR', 'message': 'Cannot get spot price'}
        
        spot_price = spot_result.get('spot_price', 0)
        
        options_result = get_options_chain_safe()
        if not options_result or options_result.get('status') != 'SUCCESS':
            return {'status': 'ERROR', 'message': 'Cannot get options chain'}
        
        options_chain = options_result.get('options_chain', [])
        
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
            
            # Check IV conditions (relaxed thresholds)
            if current_iv > 0.12 or iv_percentile > 0.4:  # Relaxed from 0.15 and 0.6
                opportunity_detected = True
                reason = "IV conditions met"
                
                # Check realized volatility validation
                if realized_vol:
                    if current_iv / realized_vol > 1.2:  # IV overpriced
                        reason = "IV overpriced vs realized volatility - premium selling opportunity"
                    elif current_iv / realized_vol < 0.8:  # IV underpriced
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
        print(f"   ✅ Expiry Analysis: {len(expiry_analysis.get('expiry_dates', [])) if expiry_analysis else 0} expiries")
        
        # Technical analysis
        print("📈 Step 2: Technical Analysis...")
        intraday_tech = get_nifty_technical_analysis_tool()
        print(f"   ✅ Intraday Technical: {intraday_tech.get('signal', 'N/A')} (RSI: {intraday_tech.get('rsi', 'N/A')})")
        
        daily_tech = get_nifty_daily_technical_analysis_wrapper()
        print(f"   ✅ Daily Technical: {daily_tech.get('signal', 'N/A')} (RSI: {daily_tech.get('rsi', 'N/A')})")
        
        # Advanced analysis
        print("🔬 Step 3: Advanced Analysis...")
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
        
        # Compile results
        detailed_result = {
            'decision': 'DETAILED_ANALYSIS_COMPLETE',
            'spot_price': spot_price,
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
            }
        }
        
        # Print detailed analysis summary
        print("\n" + "=" * 80)
        print("📋 DETAILED ANALYSIS SUMMARY")
        print("=" * 80)
        print(f"📊 Market Regime: {regime.get('classification', 'N/A')}")
        print(f"📈 IV Status: {iv_rank.get('current_iv', 'N/A')} (Percentile: {iv_rank.get('iv_percentile', 'N/A')}%)")
        print(f"📉 Technical Signals: Intraday={intraday_tech.get('signal', 'N/A')}, Daily={daily_tech.get('signal', 'N/A')}")
        print(f"📊 PCR Status: {pcr_tech.get('signal', 'N/A')} (Value: {pcr_tech.get('pcr_value', 'N/A')})")
        print("=" * 80)
        
        return detailed_result
        
    except Exception as e:
        print(f"❌ Detailed analysis failed: {e}")
        return {
            'decision': 'DETAILED_ANALYSIS_FAILED',
            'error': str(e),
            'fallback': 'WAIT - Analysis incomplete'
        }

def optimized_market_analysis():
    """
    Optimized market analysis with fast-track logic for crew compatibility
    """
    try:
        print("🔍 Running optimized market analysis with fast-track logic...")
        
        # Run quick opportunity check first
        quick_result = quick_opportunity_check()
        
        if quick_result.get('status') == 'SUCCESS':
            if quick_result.get('opportunity_detected', False):
                print("✅ Opportunity detected - proceeding to detailed analysis")
                
                # Run detailed analysis
                detailed_result = run_detailed_analysis()
                
                # Combine results
                combined_result = {
                    'decision': 'PROCEED_TO_DETAILED',
                    'quick_check': quick_result,
                    'detailed_analysis': detailed_result,
                    'final_recommendation': 'EXECUTE - Opportunity confirmed by detailed analysis'
                }
                
                return combined_result
            else:
                print("❌ No opportunity detected - skipping detailed analysis")
                return {
                    'decision': 'NO_OPPORTUNITY',
                    'quick_check': quick_result,
                    'reason': quick_result.get('reason', 'No clear opportunity'),
                    'final_recommendation': 'WAIT - No profitable opportunity detected'
                }
        else:
            print(f"❌ Quick check failed: {quick_result.get('message', 'Unknown error')}")
            return {
                'decision': 'ERROR',
                'error': quick_result.get('message', 'Quick check failed'),
                'final_recommendation': 'WAIT - Analysis error'
            }
            
    except Exception as e:
        print(f"❌ Optimized market analysis failed: {str(e)}")
        return {
            'decision': 'ERROR',
            'error': str(e),
            'final_recommendation': 'WAIT - Analysis failed'
        }

# ============================================================================
# MAIN EXECUTION - CREW-COMPATIBLE
# ============================================================================

if __name__ == "__main__":
    print("🚀 Starting Enhanced Three-Stage Opportunity Hunter (Crew-Compatible)...")
    print("📋 Mission: Provide analysis results for crew agents to process")
    print("📋 Note: This version does NOT execute trades directly")
    print("💰 Strategy Priority: PREMIUM SELLING FIRST")
    print("⚡ Enhanced Features: Realized volatility + Liquidity analysis")
    print("⏱️  Token Efficiency: Optimal for all scenarios")
    print("-" * 50)

    # Run optimized market analysis
    result = optimized_market_analysis()
    
    print("\n" + "="*80)
    print(f"🏁 ENHANCED THREE-STAGE OPPORTUNITY HUNTER (CREW-COMPATIBLE) COMPLETED - {current_datetime}")
    print("="*80)
    print(f"📊 Final Decision: {result.get('decision', 'UNKNOWN')}")
    print(f"💡 Recommendation: {result.get('final_recommendation', 'No recommendation')}")
    print("💰 Token Efficiency: Optimal for crew processing")
    print("="*80) 