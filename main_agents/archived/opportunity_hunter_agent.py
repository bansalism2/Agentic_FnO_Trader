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
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

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
        fetch_historical_data
    )
    from core_tools.calculate_analysis_tools import (
        calculate_option_greeks, calculate_implied_volatility, calculate_strategy_pnl, 
        find_arbitrage_opportunities, calculate_probability_of_profit,
        calculate_portfolio_greeks, calculate_volatility_surface,
        # NEW ADVANCED ANALYSIS TOOLS
        analyze_vix_integration_wrapper, calculate_iv_rank_analysis_wrapper, detect_market_regime_wrapper
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
        recommend_options_strategy, analyze_strategy_greeks,
        # NEW COMPREHENSIVE ANALYSIS TOOL
        comprehensive_advanced_analysis_wrapper,
        # CREWAI-COMPATIBLE WRAPPER FUNCTIONS
        create_long_straddle_wrapper, create_short_strangle_wrapper, 
        create_iron_condor_wrapper, create_butterfly_spread_wrapper
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
# SIMPLIFIED AGENTS - FOCUS ON NEW OPPORTUNITIES ONLY
# ============================================================================

# Market Opportunity Analyst (with Intraday-Only Constraints)
market_analyst = Agent(
    role="Market Opportunity Analyst",
    goal="Identify high-probability trading opportunities in NIFTY F&O markets while preserving capital through patience and discipline.",
    backstory=f"""You are a market opportunity specialist focused ONLY on finding exceptional trading setups.
    You assume the portfolio is already clean and managed by another system.
    
    CURRENT DATE AND TIME: {current_datetime}
    
    **Your Core Philosophy**: Capital preservation is paramount. You believe that patience and discipline 
    are more valuable than aggressive trading. You wait for clear, high-probability opportunities rather 
    than forcing trades in uncertain conditions. Your success comes from being selective and only taking 
    calculated risks when the odds are clearly in your favor. You understand that missing a trade is 
    better than losing money on a poor setup.
    
    Your job is simple:
    1. Analyze current market conditions thoroughly
    2. Identify ONLY exceptional, high-probability trading opportunities  
    3. Recommend specific strategies ONLY when conditions are perfect
    
    You do NOT manage existing positions - that's handled elsewhere.
    You focus purely on: "Is there an exceptional trade setup right now?"
    
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
    backstory=f"""You are a disciplined trade executor focused ONLY on exceptional new opportunities.
    
    CURRENT DATE AND TIME: {current_datetime}
    
    **Your Core Philosophy**: Capital preservation is paramount. You believe that patience and discipline 
    are more valuable than aggressive trading. You wait for clear, high-probability opportunities rather 
    than forcing trades in uncertain conditions. Your success comes from being selective and only taking 
    calculated risks when the odds are clearly in your favor. You understand that missing a trade is 
    better than losing money on a poor setup.
    
    Your philosophy: "When in doubt, don't trade. Preserve capital for better opportunities."
    
    **MANDATORY EXECUTION PROCESS:**
    You ALWAYS follow a two-step process:
    1. **CREATE**: Use strategy creation tools to build strategy structure (no execution)
    2. **EXECUTE**: Use execute_and_store_strategy() to execute and store the trade
    
    You execute trades ONLY when:
    1. Market analysis shows EXCEPTIONAL setup (not just good, but exceptional)
    2. Capital checker confirms availability  
    3. Time is before 14:30
    4. Risk-reward is compelling (>2:1)
    5. All volatility and movement conditions are perfectly aligned
    6. No doubts or uncertainties exist
    
    You do NOT:
    - Manage existing positions
    - Close conflicting positions  
    - Do complex portfolio analysis
    - Use strategy creation tools for execution (they only create structures)
    - Force trades in uncertain conditions
    
    You simply ask: "Is this an exceptional new trade opportunity with clear high probability?"
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
    Find exceptional NIFTY F&O trading opportunities for NEW positions while preserving capital:
    
    CURRENT DATE AND TIME: {current_datetime}
    
    **Your ONLY job: Identify EXCEPTIONAL trading opportunities using ADVANCED ANALYSIS**
    
    **CAPITAL PRESERVATION MANDATE**: Your primary goal is to preserve capital. Only recommend trades 
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
    
    1. **MANDATORY Advanced Market Analysis:**
       - **FIRST**: Get global market conditions using get_global_market_conditions() (CRITICAL for pre-market context)
       - **SECOND**: Get correct instrument symbols using get_nifty_instruments()
       - **THIRD**: Get current NIFTY spot price using get_nifty_spot_price_safe()
       - **FOURTH**: Get available NIFTY expiry dates using get_nifty_expiry_dates()
       - **FIFTH**: Get complete options chain using get_options_chain_safe()
       - **SIXTH**: Get NIFTY technical analysis using get_nifty_technical_analysis_tool() (intraday data)
       - **SEVENTH**: Get NIFTY daily technical analysis using get_nifty_daily_technical_analysis_wrapper() (longer-term trend)
       - **EIGHTH**: If needed, fetch historical data using fetch_historical_data() for specific analysis
       
       **USE THESE ADVANCED ANALYSIS TOOLS IN SEQUENCE:**
       
       a) **VIX Integration & Volatility Regime Analysis:**
          - Use analyze_vix_integration_wrapper() to get objective volatility metrics
          - Provides VIX-like measure, volatility regime (HIGH_STRESS, ELEVATED, NORMAL, COMPRESSED)
          - Provides regime description and volatility ratio vs historical
          - Use these metrics to assess current market volatility environment
       
       b) **IV Rank Analysis for Premium Decisions:**
          - Use calculate_iv_rank_analysis_wrapper() to get objective IV metrics
          - Provides current IV, IV rank, and IV percentile relative to historical ranges
          - Use these metrics to assess if current IV is high/low compared to history
          - High IV percentile (>80%) suggests expensive options, low (<20%) suggests cheap options
       
       c) **PCR + Technical Analysis for Entry Timing:**
          - Use calculate_pcr_technical_analysis_wrapper() to combine PCR with technical indicators
          - This provides entry timing signals (STRONG_BUY, BUY, NEUTRAL, SELL, STRONG_SELL)
          - Use this for precise entry timing and strategy direction
       
       d) **PCR Extremes Analysis for Contrarian Opportunities:**
          - Use analyze_pcr_extremes_wrapper() to identify extreme sentiment levels
          - Extreme PCR levels often signal contrarian opportunities
          - This can identify high-probability reversal trades
       
       e) **Market Regime Detection for Strategy Selection:**
          - Use detect_market_regime_wrapper() to get objective market regime metrics
          - Provides regime classification (TRENDING_BULL, TRENDING_BEAR, RANGING, VOLATILE, COMPRESSED)
          - Provides regime scores, technical indicators, and market sentiment
          - Use these metrics to understand current market environment and conditions
       
       f) **Multi-Timeframe Analysis:**
          - **Intraday Analysis (15min/5min)**: For short-term momentum and entry timing
          - **Daily Analysis (60+ days)**: For medium-term trend direction and support/resistance
          - **Trend Alignment**: Ensure intraday and daily trends are aligned for higher probability trades
          - **Divergence Detection**: Identify when short-term and long-term trends diverge (caution signal)
          - **Data Quality Check**: Verify both analyses have sufficient data for reliable signals
       
       g) **Volatility & Movement Analysis:**
          - **Current Volatility Assessment**: Analyze recent NIFTY movement patterns and volatility
          - **Strategy Suitability Check**: Match strategy requirements with current market conditions
          - **Movement Requirements**:
            * **Long Straddle**: Requires significant movement (>2% daily range) to be profitable
            * **Short Strangle**: Works best in low volatility, range-bound markets
            * **Iron Condor**: Ideal for sideways markets with low volatility
            * **Directional Strategies**: Need clear trend direction and momentum
          - **Volatility Regime Matching**: Ensure strategy matches current volatility environment
       
       h) **Comprehensive Advanced Analysis (All-in-One):**
          - Use comprehensive_advanced_analysis_wrapper() for final integrated assessment
          - This combines all analyses into one overall recommendation
          - Provides final GO/NO-GO decision with specific strategy recommendations
    
    2. **Expiry Selection Strategy:**
       - **CRITICAL**: First get available expiry dates using get_nifty_expiry_dates()
       - **SAFETY RULE**: NEVER use expiries with ≤3 days to expiry (too risky)
       - **Weekly expiry (4-7 days)**: High theta decay, high gamma risk, use for quick directional plays
         * Best for: High-conviction directional trades, momentum plays, event-driven trades
         * Risk: High time decay, sensitive to small price movements
         * Avoid if: Low conviction, high volatility uncertainty, or if holding overnight
         * **MINIMUM**: 4 days to expiry for any weekly strategy
       - **Monthly expiry (8-30 days)**: Balanced theta/gamma, preferred for most strategies
         * Best for: Most strategies including straddles, strangles, iron condors
         * Optimal range: 15-25 days to expiry for balanced risk/reward
         * Provides time for strategy to work while managing time decay
       - **Next month expiry (30+ days)**: Lower theta decay, use for longer-term positions
         * Best for: Longer-term directional views, low volatility strategies
         * Risk: Lower gamma, less responsive to short-term moves
       - **DEFAULT**: Use monthly expiry (15-25 days to expiry) for most strategies
       - **EXPIRY ANALYSIS**: Consider liquidity, open interest, and bid-ask spreads across different expiries
    
    3. **Multi-Timeframe Strategy Recommendations:**
       - **Daily Trend Analysis**: Use daily data to determine overall market direction and key levels
       - **Intraday Momentum**: Use intraday data for precise entry timing and momentum confirmation
       - **Trend Alignment Check**: Ensure intraday momentum aligns with daily trend direction
       - **Volatility Regime + IV Rank**: Determine premium buying vs selling approach
       - **PCR + Technical**: Determine entry timing and directional bias
       - **Market Regime**: Select appropriate strategy types
       - **Comprehensive Analysis**: Final strategy recommendation with risk management
       
       **STRATEGY SELECTION LOGIC:**
       - **Volatility-Based Strategy Selection**:
         * **High Volatility (>2% daily range)**: Long Straddle, Long Strangle, Directional strategies
         * **Low Volatility (<1% daily range)**: Short Strangle, Iron Condor, Calendar spreads
         * **Medium Volatility (1-2% daily range)**: Balanced strategies based on trend direction
       - **Movement-Based Strategy Selection**:
         * **Strong Trend + High Volatility**: Directional strategies (Long Calls/Puts, Debit spreads)
         * **Sideways + Low Volatility**: Premium selling strategies (Short Strangle, Iron Condor)
         * **Mixed Signals**: Conservative strategies or WAIT for clearer setup
       - **Final Decision Logic**:
         * If Comprehensive Analysis shows STRONG_BUY/STRONG_SELL + High Volatility: Execute directional strategies
         * If Comprehensive Analysis shows NEUTRAL + Low Volatility: Use range-bound strategies
         * If Comprehensive Analysis shows WAIT: No trade (preserve capital)
         * If Volatility doesn't match strategy requirements: WAIT for better conditions
         * **PATIENCE RULE**: If ANY doubt exists about the setup: WAIT and preserve capital
         * **EXCEPTIONAL OPPORTUNITY RULE**: Only execute if the opportunity is truly exceptional, not just good
       
       - Specify exact strikes, expiry, and quantity (minimum 75 lots)
       - Include risk management guidelines from the analysis
       - Expected profit potential and risk-reward ratio
    
    **CRITICAL CONSTRAINTS:**
    - **CAPITAL PRESERVATION FIRST**: Preserve capital is the primary goal - only take calculated risks
    - **PATIENCE DISCIPLINE**: Wait for CLEAR, HIGH-PROBABILITY opportunities - better to miss a trade than lose money
    - **CONSERVATIVE APPROACH**: When in doubt, WAIT - there will always be another opportunity
    - Focus ONLY on NEW opportunities
    - **MULTI-TIMEFRAME ANALYSIS**: Use both intraday (15min/5min) and daily (60+ days) data
    - **TREND ALIGNMENT**: Prefer trades where intraday and daily trends align
    - **DIVERGENCE CAUTION**: Be extra cautious when short-term and long-term trends diverge
    - **VOLATILITY MATCHING**: Ensure strategy requirements match current market volatility
    - **MOVEMENT ANALYSIS**: Verify NIFTY movement patterns support the chosen strategy
    - Ignore existing portfolio - assume it's clean
    - Use ALL advanced analysis tools for comprehensive assessment
    - Only recommend trades when Comprehensive Analysis gives clear signal
    - **SAFETY RULE**: NEVER use expiries with ≤3 days to expiry (too risky)
    - **DEFAULT BEHAVIOR**: WAIT unless exceptional opportunity exists - patience is profitable
    
    Output: Detailed opportunity assessment using advanced analysis with specific strategy recommendation or "NO CLEAR OPPORTUNITY - WAIT TO PRESERVE CAPITAL"
    """,
    agent=market_analyst,
    expected_output="Market opportunity analysis with specific new trade recommendations or clear statement that no opportunities exist."
)

# Simple Capital Check (with Intraday-Only Constraints)
capital_check_task = Task(
    description=f"""
    Simple capital and position count check for new trades with capital preservation focus:
    
    CURRENT DATE AND TIME: {current_datetime}
    
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
    Execute new trade ONLY if exceptional opportunity exists with capital preservation focus:
    
    CURRENT DATE AND TIME: {current_datetime}
    
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
       - Is the setup truly exceptional? (High probability, good risk-reward >2:1)
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
    expected_output="Clear EXECUTE (with actual trade execution) or WAIT decision with specific reasoning. Default should be WAIT unless exceptional opportunity.",
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
    max_rpm=20,  # Reduced API calls
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
    
    **YOUR JOB**: Analyze the results from all previous agents and provide a clear, detailed explanation of why the final decision was made.
    
    **REQUIRED ANALYSIS**:
    
    1. **Market Analysis Summary**:
       - What did the Market Opportunity Analyst find?
       - What market conditions were identified?
       - What analysis was completed vs incomplete?
       - What was the final market assessment?
    
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
       - **OPPORTUNITY QUALITY**: How exceptional was the opportunity (if any)?
       - **CAPITAL PRESERVATION**: How does this decision align with capital preservation goals?
       - **MARKET CONDITIONS**: How do current market conditions support this decision?
       - **TIMING FACTORS**: What timing considerations influenced the decision?
    
    5. **DETAILED EXPLANATION**:
       - Explain the decision-making process step by step
       - Connect the dots between market analysis, capital availability, and execution decision
       - Provide specific data points that led to the final decision
       - Explain why this decision preserves capital or why the opportunity was exceptional enough to execute
    
    **OUTPUT FORMAT**:
    
    # OPPORTUNITY HUNTER FINAL DECISION - {current_datetime}
    
    ## DECISION: [EXECUTE/WAIT]
    
    ## PRIMARY REASONING
    [Clear, concise explanation of the main reason for the decision]
    
    ## DETAILED ANALYSIS
    
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
    
    ## CAPITAL PRESERVATION ALIGNMENT
    - **How This Decision Preserves Capital**: [Explanation]
    - **Opportunity Quality Assessment**: [If applicable]
    - **Patience vs Action Balance**: [How the decision balances patience with opportunity]
    
    ## CONCLUSION
    [Final summary of why this decision is the right one for capital preservation and trading success]
    
    **CRITICAL**: This reasoning must be comprehensive and explain exactly why the decision was made, whether it's to execute or wait. The reasoning should be detailed enough that someone reading it understands the complete thought process.
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
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("Starting ENHANCED Opportunity Hunter...")
    print("Focus: NEW trading opportunities ONLY")
    print("Assumption: Portfolio is already managed by Position Manager")
    print("Default behavior: WAIT (most runs should not execute trades)")
    print("Enhanced: Comprehensive reasoning in final output")
    print("-" * 50)

    # Check live balance before running the crew
    try:
        from core_tools.execution_portfolio_tools import get_account_margins
        margins = get_account_margins()
        if margins.get('status') == 'SUCCESS':
            live_balance = margins['equity'].get('live_balance', 0)
            if live_balance < 50000:
                print(f"Live balance is too low (₹{live_balance}). No trades will be executed.")
                exit(0)
        else:
            print("Could not fetch account margins. Proceeding with caution.")
    except Exception as e:
        print(f"Error checking live balance: {e}. Proceeding with caution.")

    result = opportunity_hunter_crew.kickoff()
    
    print("\n" + "="*50)
    print(f"OPPORTUNITY HUNTER RESULT - {current_datetime}")
    print("="*50)
    print(result)
    print("\n" + "="*50)
    print("Remember: WAIT = Good outcome (capital preserved)")
    print("EXECUTE = Only when exceptional opportunity exists")
    print("="*50)