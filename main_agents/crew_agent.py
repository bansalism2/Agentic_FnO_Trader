#!/usr/bin/env python3
"""
AlgoTrade - Main Trading Agent
=============================

Main CrewAI agent file for NIFTY F&O trading.
Contains only essential components: credentials, imports, agents, tasks, crew, and execution.

Author: AlgoTrade Team
"""

# ============================================================================
# CREDENTIALS AND CONFIG
# ============================================================================

from dotenv import load_dotenv
import os
from datetime import datetime

# Load credentials
load_dotenv(dotenv_path='./.env')
api_key = os.getenv("kite_api_key")
api_secret = os.getenv("kite_api_secret")
access_token = None
try:
    with open("data/access_token.txt", "r") as f:
        access_token = f.read().strip()
except Exception as e:
    print("Could not read data/access_token.txt:", e)

# Set OpenAI API key
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Get current date and time
current_date = datetime.now().strftime('%Y-%m-%d')
current_time = datetime.now().strftime('%H:%M:%S')
current_datetime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

print(f"Starting F&O Trading Analysis at {current_datetime}...")
print("OPENAI_API_KEY loaded:", os.environ.get("OPENAI_API_KEY"))
print("Kite credentials loaded:", api_key, api_secret, access_token)

# ============================================================================
# CLEAN TOOL IMPORTS
# ============================================================================

from crewai import Agent, Task, Crew
from crewai.tools import tool

# Import all required tools from their appropriate modules
try:
    # Core connection and data tools
    from core_tools.connect_data_tools import (
        get_nifty_spot_price, get_nifty_expiry_dates, get_options_chain, 
        get_historical_volatility, analyze_options_flow, initialize_connection, fetch_historical_data,
        get_nifty_instruments, get_nifty_spot_price_safe, debug_kite_instruments,
        get_options_chain_safe, analyze_options_flow_safe, get_global_market_conditions
    )
except ImportError as e:
    print(f"Warning: Could not import connect_data_tools: {e}")

try:
    # Calculation and analysis tools
    from core_tools.calculate_analysis_tools import (
        calculate_option_greeks, calculate_implied_volatility, calculate_strategy_pnl, 
        find_arbitrage_opportunities, calculate_portfolio_greeks, calculate_volatility_surface, 
        calculate_probability_of_profit
    )
except ImportError as e:
    print(f"Warning: Could not import calculate_analysis_tools: {e}")

try:
    # Execution and portfolio tools
    from core_tools.execution_portfolio_tools import (
        get_portfolio_positions, get_account_margins, get_orders_history,
        get_daily_trading_summary, get_risk_metrics, execute_options_strategy, calculate_strategy_margins,
        validate_trading_capital, calculate_realistic_pricing, analyze_position_conflicts
    )
except ImportError as e:
    print(f"Warning: Could not import execution_portfolio_tools: {e}")

try:
    # Strategy creation tools
    from core_tools.strategy_creation_tools import (
        create_long_straddle_strategy, create_short_strangle_strategy, create_iron_condor_strategy, 
        create_butterfly_spread_strategy, create_ratio_spread_strategy, recommend_options_strategy, 
        analyze_strategy_greeks
    )
except ImportError as e:
    print(f"Warning: Could not import strategy_creation_tools: {e}")

try:
    # Technical analysis tools
    from core_tools.master_indicators import get_nifty_technical_analysis_tool
except ImportError as e:
    print(f"Warning: Could not import master_indicators: {e}")

# Initialize Kite Connect session
try:
    init_result = initialize_connection(api_key, api_secret, access_token)
    print("Kite Connect initialization result:", init_result)
except Exception as e:
    print(f"Warning: Kite Connect initialization failed: {e}")
    init_result = {'status': 'ERROR', 'message': str(e)}

# ============================================================================
# AGENT DEFINITIONS
# ============================================================================

# Market Analyst Agent
market_analyst = Agent(
    role="Short-Term Market Analyst",
    goal="Analyze short-term market dynamics, volatility patterns, and immediate opportunities in NIFTY F&O markets.",
    backstory=f"""You are an expert in short-term market analysis with deep knowledge of intraday 
    patterns, options flow, volatility dynamics, and F&O market microstructure. You understand 
    that F&O markets are driven by short-term catalysts, volatility changes, and immediate 
    sentiment rather than long-term trends. You excel at identifying actionable patterns 
    within 1-5 day timeframes and understand the unique dynamics of Indian options markets.
    
    CURRENT DATE AND TIME: {current_datetime}
    You are analyzing markets as of {current_date} at {current_time}. Use this as your reference point for all analysis.
    - Identify and assess the best expiry date since the nearest is not always the best
    - All date references should be relative to {current_date}
    
    **GLOBAL MARKET ANALYSIS GUIDANCE:**
    - Use get_global_market_conditions() tool to assess overnight global market movements
    - **TIME RESTRICTION**: This tool is only available before 9:30 AM IST (market opening)
    - After 9:30 AM, the tool will return a restriction message and you should focus on real-time NIFTY data
    - This data is MOST USEFUL for market opening predictions and early trading decisions
    - The tool provides NIFTY gap predictions, global sentiment, and trading signals
    - Use this information to adjust your opening strategy and risk management
    - Note: This data becomes less relevant as the day progresses - focus on current market conditions for intraday decisions
    
    **IMPORTANT DISCLAIMER - GLOBAL MARKET DATA:**
    - DO NOT rely solely on global market data predictions for trading decisions
    - These are indicators that MAY or MAY NOT impact NIFTY - they are not guarantees
    - Global markets can be misleading and NIFTY often moves independently
    - Always combine global data with local technical analysis and market conditions
    - Use global data as one of many inputs, not as the primary decision factor
    - Past correlations do not guarantee future behavior
    - Exercise caution and maintain proper risk management regardless of global predictions
    
    DATA INTERVAL LIMITS: When fetching historical data or technical analysis, use these valid intervals and maximum days:
    - minute: maximum 60 days
    - 3minute: maximum 100 days  
    - 5minute: maximum 100 days
    - 10minute: maximum 100 days
    - 15minute: maximum 200 days
    - 30minute: maximum 200 days
    - 60minute: maximum 400 days
    - day: maximum 2000 days
    
    **CRITICAL: INTRADAY DATA REQUIREMENT DURING MARKET HOURS**
    - During market hours (9:30 AM - 3:30 PM), ALWAYS fetch intraday data (5minute, 10minute, or 15minute intervals)
    - Do NOT rely solely on daily historical data during market hours as it becomes stale quickly
    - Use daily data only for pre-market analysis or overnight gap analysis
    - For real-time trading decisions, use 5-15 minute intervals to capture current market dynamics
    - This ensures you're analyzing fresh, relevant data that reflects current market conditions
    
    Note that if intraday_payin is there then it means that the money has been added today and might not reflect in available_cash
    and it will be reflected in the next day's available_cash but its available for trading today. Always include intraday_payin
    in your capital calculations when validating trading strategies.
    Note that minimum quantity to trade a Nifty option is 75""",
    tools=[
        tool("Get correct instrument symbols for NIFTY trading")(get_nifty_instruments),
        tool("Get NIFTY spot price with error handling")(get_nifty_spot_price_safe),
        tool("Debug Kite Connect instruments and symbols")(debug_kite_instruments),
        tool("Get safe options chain data with fallback parameters")(get_options_chain_safe),
        tool("Analyze options flow with safe parameter handling")(analyze_options_flow_safe),
        tool("Get global market conditions with NIFTY gap predictions (cached, most useful for market opening, RESTRICTED after 9:30 AM)")(get_global_market_conditions),
        tool("Run full NIFTY technical analysis (OHLCV, indicators, signals) for a given days/interval.")(get_nifty_technical_analysis_tool),
        tool("Get available NIFTY expiry dates.")(get_nifty_expiry_dates),
        tool("Get historical volatility for NIFTY.")(get_historical_volatility),
        tool("Fetch historical OHLCV data for a symbol and date range (RESTRICTED: max 5 days for non-daily intervals).")(fetch_historical_data),
    ],
    verbose=True,
    max_iter=3
)

# Risk Manager Agent
risk_manager = Agent(
    role="Risk Management Specialist",
    goal="Assess and manage portfolio risk, position sizing, and capital preservation with focus on F&O-specific risks.",
    backstory=f"""You are a seasoned risk management professional with expertise in options 
    risk metrics, portfolio optimization, and capital allocation. You understand F&O-specific 
    risks like time decay, volatility risk, and liquidity constraints. You prioritize capital 
    preservation while maximizing risk-adjusted returns in volatile markets.
    
    CURRENT DATE AND TIME: {current_datetime}
    You are managing risk as of {current_date} at {current_time}. Use this as your reference point for all risk assessments.
    - All date references should be relative to {current_date}
    
    DATA INTERVAL LIMITS: When fetching historical data or technical analysis, use these valid intervals and maximum days:
    - minute: maximum 60 days
    - 3minute: maximum 100 days  
    - 5minute: maximum 100 days
    - 10minute: maximum 100 days
    - 15minute: maximum 200 days
    - 30minute: maximum 200 days
    - 60minute: maximum 400 days
    - day: maximum 2000 days
    
    **CRITICAL: INTRADAY DATA REQUIREMENT DURING MARKET HOURS**
    - During market hours (9:30 AM - 3:30 PM), ALWAYS fetch intraday data (5minute, 10minute, or 15minute intervals)
    - Do NOT rely solely on daily historical data during market hours as it becomes stale quickly
    - Use daily data only for pre-market analysis or overnight gap analysis
    - For real-time trading decisions, use 5-15 minute intervals to capture current market dynamics
    - This ensures you're analyzing fresh, relevant data that reflects current market conditions
    
    **POSITION SIZING GUIDELINES:**
    - High-conviction trades: Up to 80% of capital at risk
    - Medium-conviction trades: 60% of capital at risk  
    - Speculative trades: 30% of capital at risk
    - Never risk more than 80% on any single position""",
    tools=[
        tool("Validate if sufficient capital is available for trading strategy")(validate_trading_capital),
        tool("Analyze potential conflicts between existing positions and proposed new strategy")(analyze_position_conflicts),
        tool("Get all open NIFTY positions and their P&L")(get_portfolio_positions),
        tool("Get account margin and cash details.")(get_account_margins),
        tool("Get risk metrics for the account.")(get_risk_metrics),
        tool("Calculate margin requirement for a strategy.")(calculate_strategy_margins),
    ],
    verbose=True,
    max_iter=2
)

# Strategy Executor Agent
strategy_executor = Agent(
    role="F&O Strategy Execution Specialist", 
    goal="Execute optimal F&O strategies ONLY when high-conviction opportunities exist. Master of timing entries and exits.",
    backstory=f"""You are an experienced F&O trader who understands that timing is everything 
    in options trading. You have the discipline to wait for perfect setups and the agility 
    to act quickly when opportunities arise. You know that most F&O profits come from a few 
    exceptional trades rather than constant activity. You're expert at reading market 
    microstructure and understanding when liquidity and timing align perfectly.
    
    CURRENT DATE AND TIME: {current_datetime}
    You are evaluating trading opportunities as of {current_date} at {current_time}. Use this as your reference point for all strategy decisions.
    - Identify and assess the best expiry date since the nearest is not always the best
    - All date references should be relative to {current_date}
    
    **GLOBAL MARKET INTEGRATION:**
    - Use get_global_market_conditions() tool to assess overnight global market movements
    - **TIME RESTRICTION**: This tool is only available before 9:30 AM IST (market opening)
    - After 9:30 AM, the tool will return a restriction message and you should focus on real-time NIFTY data
    - This data is MOST USEFUL for market opening predictions and early trading decisions
    - Consider global sentiment and NIFTY gap predictions when planning opening strategies
    - Adjust position sizing and risk management based on global market conditions
    - Note: This data becomes less relevant as the day progresses - focus on current market conditions for intraday decisions
    
    **IMPORTANT DISCLAIMER - GLOBAL MARKET DATA:**
    - DO NOT rely solely on global market data predictions for trading decisions
    - These are indicators that MAY or MAY NOT impact NIFTY - they are not guarantees
    - Global markets can be misleading and NIFTY often moves independently
    - Always combine global data with local technical analysis and market conditions
    - Use global data as one of many inputs, not as the primary decision factor
    - Past correlations do not guarantee future behavior
    - Exercise caution and maintain proper risk management regardless of global predictions
    
    **CRITICAL TIME-BASED TRADING RULES:**
    - If current time is AFTER 14:30 (2:30 PM), DO NOT take any new trades
    - After 14:30, focus ONLY on analyzing existing positions and closing them if required
    - This is because market liquidity decreases significantly in the last 1 hours
    - New positions opened after 14:30 have higher execution risk and wider spreads
    - Use the last 1 hours for position management, not new entries
    
    DATA INTERVAL LIMITS: When fetching historical data or technical analysis, use these valid intervals and maximum days:
    - minute: maximum 60 days
    - 3minute: maximum 100 days  
    - 5minute: maximum 100 days
    - 10minute: maximum 100 days
    - 15minute: maximum 200 days
    - 30minute: maximum 200 days
    - 60minute: maximum 400 days
    - day: maximum 2000 days
    
    **CRITICAL: INTRADAY DATA REQUIREMENT DURING MARKET HOURS**
    - During market hours (9:30 AM - 3:30 PM), ALWAYS fetch intraday data (5minute, 10minute, or 15minute intervals)
    - Do NOT rely solely on daily historical data during market hours as it becomes stale quickly
    - Use daily data only for pre-market analysis or overnight gap analysis
    - For real-time trading decisions, use 5-15 minute intervals to capture current market dynamics
    - This ensures you're analyzing fresh, relevant data that reflects current market conditions
    
    Note that if intraday_payin is there then it means that the money has been added today and might not reflect in available_cash
    and it will be reflected in the next day's available_cash but its available for trading today. Always include intraday_payin
    in your capital calculations when validating trading strategies.
    Note that minimum quantity to trade a Nifty option is 75
    Remember: Professional traders spend most of their time waiting for the right setup.
    Your job is to PROTECT capital first, GROW it second.   
    **CRITICAL: When you recommend EXECUTE, you MUST actually execute the trade using execute_options_strategy() tool.**
    Checking option chain data is important. You should use 'NIFTY 50' as the symbol for all NIFTY instrument historical data.
    All date references should be relative to {current_date}.
    """,
    tools=[
        tool("Execute a multi-leg options strategy.")(execute_options_strategy),
        tool("Create a long straddle options strategy.")(create_long_straddle_strategy),
        tool("Create a short strangle options strategy.")(create_short_strangle_strategy),
        tool("Create an iron condor options strategy.")(create_iron_condor_strategy),
        tool("Create a butterfly spread options strategy.")(create_butterfly_spread_strategy),
        tool("Create a ratio spread options strategy.")(create_ratio_spread_strategy),
        tool("Recommend an options strategy based on market outlook.")(recommend_options_strategy),
        tool("Analyze Greeks for a given options strategy.")(analyze_strategy_greeks),
        tool("Calculate option Greeks using Black-Scholes.")(calculate_option_greeks),
        tool("Calculate implied volatility for an option.")(calculate_implied_volatility),
        tool("Calculate P&L for a multi-leg options strategy.")(calculate_strategy_pnl),
        tool("Find arbitrage opportunities in the options chain.")(find_arbitrage_opportunities),
        tool("Calculate portfolio Greeks for open positions.")(calculate_portfolio_greeks),
        tool("Calculate the volatility surface for options data.")(calculate_volatility_surface),
        tool("Calculate probability of profit for a strategy.")(calculate_probability_of_profit),
        tool("Get order history for the account.")(get_orders_history),
        tool("Get daily trading summary for the account.")(get_daily_trading_summary),
    ],
    verbose=True,
    max_iter=2
)

# ============================================================================
# TASK DEFINITIONS
# ============================================================================

# Portfolio Review Task
portfolio_review_task = Task(
    description=f"""
    PRIORITY TASK: Review and manage all existing NIFTY positions before any new analysis:
    
    CURRENT DATE AND TIME: {current_datetime}
    You are reviewing positions as of {current_date} at {current_time}. Use this as your reference point.
    
    0. **First, assess global market conditions (if early in the day):**
       - Use get_global_market_conditions() to understand overnight global market movements
       - **TIME RESTRICTION**: This tool is only available before 9:30 AM IST (market opening)
       - After 9:30 AM, the tool will return a restriction message and you should focus on real-time NIFTY data
       - This data is most useful for market opening predictions and early trading decisions
       - Consider global sentiment and NIFTY gap predictions when managing positions
       - Note: This becomes less relevant as the day progresses
       - **DISCLAIMER**: Global market data are indicators only - they may or may not impact NIFTY
       - Always combine with local analysis and maintain proper risk management
    
    1. **Resolve any symbol/instrument issues:**
       - Use debug_kite_instruments() and get_nifty_instruments() if needed
       - Verify correct symbol formats for NIFTY instruments
       - Handle any API connection or symbol resolution errors gracefully
    
    2. **Get all current NIFTY positions with their P&L status:**
       - Use get_portfolio_positions() to fetch all open positions
       - **CRITICAL**: If there are 0 open positions, document this clearly and skip to step 6
       - If 0 positions exist, no action is needed as positions have already been closed
       - Only proceed with detailed analysis if there are >0 open positions
    
    3. **For positions >0, conduct thorough position assessment focusing on:**
       
       **A. Days to Expiry Analysis (PRIORITY):**
       - Calculate exact days remaining to expiry for each position
       - **CRITICAL THRESHOLDS:**
         * <3 days to expiry: IMMEDIATE ACTION REQUIRED - close or roll positions
         * 3-7 days to expiry: HIGH PRIORITY - assess for closure or rolling
         * 7-14 days to expiry: MODERATE PRIORITY - monitor closely
         * >14 days to expiry: NORMAL MONITORING
       - For positions <7 days to expiry, analyze current market conditions for optimal exit timing
       - Consider rolling positions to next expiry if market conditions are favorable
       
       **B. Current Day's Market Data Analysis:**
       - Get current NIFTY spot price and intraday movement
       - Fetch current day's technical analysis (5minute or 15minute intervals)
       - Analyze how current market movement affects each position:
         * For long positions: Is the move favorable or adverse?
         * For short positions: Is the move favorable or adverse?
         * Calculate current P&L impact of today's movement
       - Assess if current market conditions warrant immediate action
       
       **C. Options Chain Analysis for Each Position:**
       - For each position, fetch the relevant options chain data
       - Analyze current bid-ask spreads for exit liquidity
       - Check open interest and volume for the specific strikes
       - Assess implied volatility changes since position entry
       - Calculate current Greeks (delta, gamma, theta, vega) if tools available
       - Determine if current market conditions are optimal for exit
    
    4. **Position-Specific Risk Assessment:**
       - **Time Decay Impact**: Calculate theta decay for each position
       - **Delta Exposure**: Assess directional risk and current market alignment
       - **Volatility Exposure**: Check if IV changes favor or hurt the position
       - **Liquidity Assessment**: Verify exit liquidity for each strike
       - **Greeks Management**: Identify positions with excessive risk exposure
    
    5. **Take Immediate Corrective Actions Based on F&O-Specific Criteria:**
       - **Expiry-Based Actions:**
         * <3 days to expiry: CLOSE IMMEDIATELY unless deep ITM with high probability of assignment
         * 3-7 days to expiry: Close if P&L > 25% profit or loss > 50% of premium
         * 7-14 days to expiry: Close if P&L > 40% profit or loss > 75% of premium
       - **Market Condition Actions:**
         * Close long volatility positions if IV drops >20% from entry
         * Close short volatility positions if IV spikes >30% from entry
         * Exit positions in strikes with low volume/OI (<100 contracts)
         * Close positions with bid-ask spread >10% of option price
       - **Risk Management Actions:**
         * Reduce delta exposure if >50% of capital in directional risk
         * Reduce gamma exposure if >30% of capital in gamma risk
         * Close positions if total portfolio risk exceeds 80% of capital
       - **Execution Strategy:**
         * Use realistic exit pricing with 2-3% slippage buffer for urgent exits
         * For profitable positions: Use limit orders at favorable prices
         * For loss-making positions: Use market orders if urgent, limit orders if time permits
    
    6. **Document Position Status and Actions:**
       - If 0 positions: Document "No open positions - no action required"
       - If >0 positions: Document each position's status and actions taken
       - Include reasoning for each decision (expiry, market conditions, risk management)
       - Document any positions that were closed and their final P&L
    
    **CRITICAL: INTRADAY DATA REQUIREMENT DURING MARKET HOURS**
    - During market hours (9:30 AM - 3:30 PM), ALWAYS fetch intraday data (5minute, 10minute, or 15minute intervals)
    - Do NOT rely solely on daily historical data during market hours as it becomes stale quickly
    - Use daily data only for pre-market analysis or overnight gap analysis
    - For real-time trading decisions, use 5-15 minute intervals to capture current market dynamics
    - This ensures you're analyzing fresh, relevant data that reflects current market conditions
    
    This task has execution authority - close positions if risk management dictates.
    Only after portfolio cleanup should new opportunities be considered.
    """,
    agent=risk_manager,
    expected_output="Portfolio review report with position count, detailed analysis for >0 positions, and all corrective actions taken. If 0 positions, clearly state no action required. Include any symbol/API resolution steps."
)

# Market Analysis Task
market_analysis_task = Task(
    description=f"""
    Conduct SHORT-TERM focused market analysis for NIFTY F&O trading:
    
    CURRENT DATE AND TIME: {current_datetime}
    You are analyzing markets as of {current_date} at {current_time}. Use this as your reference point.
    
    1. **Short-term price action (last 5-10 trading days from {current_date}):**
       - Recent support and resistance levels
       - Intraday volatility patterns and ranges
       - Key price levels from last 3-5 sessions
       - Short-term momentum and trend strength
    
    2. **Volatility Analysis (immediate focus):**
       - Current implied volatility vs recent realized volatility
       - IV rank and percentile (30-day lookback from {current_date})
       - Volatility skew analysis across strikes
       - Expected moves for current and next expiry
    
    3. **Options-specific metrics:**
       - Put-call ratio and sentiment indicators
       - Identify which option is in the money and which is out of the money
       - Identify and assess the best expiry date since the nearest is not always the best
       - Max pain levels for current expiry
       - Open interest patterns across strikes
       - Unusual options activity or flow
       - Time decay environment (days to expiry impact)
    
    4. **Intraday dynamics:**
       - Current session's price action and volume
       - Key support/resistance for today's session
       - Market opening patterns and momentum
       - FII/DII activity in index futures
    
    5. **Near-term catalysts (1-5 days from {current_date}):**
       - Upcoming economic data releases
       - Corporate earnings or events
       - RBI announcements or policy events
       - Global market cues affecting NIFTY
    
    6. **Short-term technical levels:**
       - 15-min, 1-hour, and daily chart patterns
       - Recent breakouts or breakdowns
       - Key pivot points and Fibonacci levels
       - Volume-weighted average price (VWAP) analysis
    
    **CRITICAL: INTRADAY DATA REQUIREMENT DURING MARKET HOURS**
    - During market hours (9:30 AM - 3:30 PM), ALWAYS fetch intraday data (5minute, 10minute, or 15minute intervals)
    - Do NOT rely solely on daily historical data during market hours as it becomes stale quickly
    - Use daily data only for pre-market analysis or overnight gap analysis
    - For real-time trading decisions, use 5-15 minute intervals to capture current market dynamics
    - This ensures you're analyzing fresh, relevant data that reflects current market conditions
    
    **IMPORTANT - Data Interval Limits:**
    When fetching historical data or technical analysis, use these valid intervals and maximum days:
    - minute: maximum 60 days
    - 3minute: maximum 100 days  
    - 5minute: maximum 100 days
    - 10minute: maximum 100 days
    - 15minute: maximum 200 days
    - 30minute: maximum 200 days
    - 60minute: maximum 400 days
    - day: maximum 2000 days
    
    Focus on actionable insights for the current trading session and next 2-3 days.
    Ignore long-term trends beyond 2 weeks as they're less relevant for F&O positions.
    Checking option chain data is important.
    You should use 'NIFTY 50' as the symbol for all NIFTY instrument historical data.
    All date references should be relative to {current_date}.
    """,
    agent=market_analyst,
    expected_output="Short-term focused market analysis with immediate actionable insights for F&O trading.",
    context=[portfolio_review_task]
)

# Risk Assessment Task
risk_assessment_task = Task(
    description=f"""
    Assess current risk profile and available capital after portfolio actions:
    
    CURRENT DATE AND TIME: {current_datetime}
    You are assessing risk as of {current_date} at {current_time}. Use this as your reference point.
    
    1. **Review current portfolio status:**
       - Get current positions using get_portfolio_positions()
       - **POSITION COUNT IMPACT:**
         * 0 positions: Full capital available for new trades
         * 1-2 positions: Assess remaining risk capacity
         * 3+ positions: **NO NEW POSITIONS** - focus on managing existing ones
       - Document current position count and its impact on new trade decisions
    
    2. **Review account margins and available capital post-adjustments:**
       - Use get_account_margins() to get current capital status
       - Calculate remaining capital after existing positions
       - Assess intraday_payin availability for trading today
       - Document capital constraints for new positions
    
    3. **Analyze current portfolio risk exposure:**
       - **If positions exist**: Analyze portfolio Greeks and risk exposure
       - **If no positions**: Document clean slate for new opportunities
       - Assess recent trading performance and patterns
       - Calculate risk budget remaining for new positions
       - Identify any concentration risks in existing positions
    
    4. **Position-Specific Risk Assessment (if positions exist):**
       - **Days to Expiry Risk**: Assess time decay impact on existing positions
       - **Directional Risk**: Calculate current delta exposure
       - **Volatility Risk**: Assess IV exposure and changes
       - **Liquidity Risk**: Check exit liquidity for existing positions
       - **Concentration Risk**: Identify if positions are too concentrated in specific strikes/expiries
    
    5. **New Position Risk Guidelines:**
       Based on current portfolio state and market analysis:
       - **If 0 positions**: Full risk budget available (up to 80% of capital)
       - **If 1 position**: Moderate risk budget (up to 60% of remaining capital)
       - **If 2 positions**: Conservative risk budget (up to 40% of remaining capital)
       - **If 3+ positions**: **NO NEW POSITIONS** - manage existing ones
       - Determine appropriate position sizing for new trades
       - Set risk parameters for new positions
       - Identify portfolio needs (hedging, income, speculation)
       - Assess if new positions are advisable given current exposure
    
    6. **Position Conflict Analysis:**
       - If existing positions exist, assess potential conflicts with new trades:
         * Directional conflicts (long vs short bias)
         * Volatility conflicts (long vs short volatility)
         * Expiry conflicts (same expiry concentration)
         * Strike conflicts (overlapping strikes)
       - Document any position closures required before new trades
       - Provide guidelines for position management priority
    
    **CRITICAL: INTRADAY DATA REQUIREMENT DURING MARKET HOURS**
    - During market hours (9:30 AM - 3:30 PM), ALWAYS fetch intraday data (5minute, 10minute, or 15minute intervals)
    - Do NOT rely solely on daily historical data during market hours as it becomes stale quickly
    - Use daily data only for pre-market analysis or overnight gap analysis
    - For real-time trading decisions, use 5-15 minute intervals to capture current market dynamics
    - This ensures you're analyzing fresh, relevant data that reflects current market conditions
    
    Provide clear guidelines for new position initiation based on current portfolio state.
    Checking option chain data is important.
    All date references should be relative to {current_date}.
    """,
    agent=risk_manager,
    expected_output="Updated risk assessment with position count analysis, remaining risk capacity, and guidelines for new position sizing and risk management. Include position conflict analysis if existing positions exist.",
    context=[portfolio_review_task, market_analysis_task]
)

# Strategy Execution Task
strategy_execution_task = Task(
    description=f"""
    Evaluate whether new trading opportunities warrant execution. Remember: NOT trading is often the best decision.
    
    CURRENT DATE AND TIME: {current_datetime}
    You are evaluating opportunities as of {current_date} at {current_time}. Use this as your reference point.
    
    **CRITICAL TIME-BASED TRADING RESTRICTIONS:**
    - If current time is AFTER 14:30 (2:30 PM), DO NOT take any new trades
    - After 14:30, focus ONLY on analyzing existing positions and closing them if required
    - This is because market liquidity decreases significantly in the last 1.5 hours
    - New positions opened after 14:30 have higher execution risk and wider spreads
    - Use the last 1 hours for position management, not new entries
    - If it's after 14:30, your task is to review existing positions and close any that need immediate attention
    
    1. **First, check the current time:**
       - If current time is AFTER 14:30, skip new trade evaluation entirely
       - Focus only on existing position management and closures
       - Document why no new trades are being considered (time restriction)
       - **Note**: Global market data is restricted after 9:30 AM - focus on real-time NIFTY data during market hours
    
    2. **CRITICAL: Assess current positions before considering new trades:**
       - Get current portfolio positions using get_portfolio_positions()
       - **POSITION COUNT ANALYSIS:**
         * If 0 positions: Proceed with new trade evaluation (clean slate)
         * If 1-2 positions: Assess if new trade would create portfolio conflicts
         * If 3+ positions: **DO NOT take new trades** - manage existing positions first
       - **POSITION CONFLICT ANALYSIS (if 1-2 positions exist):**
         * Analyze existing positions for potential conflicts with new trade:
           - Directional conflicts (long vs short bias)
           - Volatility conflicts (long vs short volatility)
           - Expiry conflicts (same expiry creating concentration risk)
           - Strike conflicts (overlapping strikes creating gamma risk)
         * If conflicts exist, consider squaring off existing positions before new trade
         * Document any position closures required for new trade execution
    
    3. **If before 14:30 AND position analysis allows, determine if trading is advisable:**
       - Is the market environment conducive to options trading?
       - Are there clear, high-conviction opportunities?
       - Is the risk-reward profile compelling (minimum 2:1 or better)?
       - Do we have adequate capital without over-leveraging?
       - Are we emotionally and analytically prepared for the trade?
       - **NEW**: Does the new trade complement or conflict with existing positions?
    
    4. **If trading conditions are NOT favorable, explicitly state:**
       - Why no trades should be executed
       - What specific conditions you're waiting for
       - What market scenarios would trigger action
       - How to monitor for better opportunities
       - **NEW**: Any position conflicts that need resolution
    
    5. **Only if exceptional SHORT-TERM opportunities exist (AND before 14:30 AND position count <3):**
       - Identify immediate, high-conviction setups (1-5 day horizon from {current_date})
       - Focus on volatility plays, momentum trades, or mean reversion
       - Consider upcoming events, expiry effects, or sentiment shifts
       - Calculate comprehensive metrics for short-term outlook
       - Ensure the strategy capitalizes on immediate market dynamics
       - Verify position can be managed actively if needed
       - **NEW**: Ensure new trade doesn't create excessive portfolio risk
    
    6. **F&O Trading Standards (ALL must be met AND before 14:30):**
       - Probability of profit > 55% OR risk-reward ratio > 1.5:1 (adjusted for F&O volatility)
       - Clear short-term catalyst or technical setup
       - Adequate liquidity: minimum 500 OI and 50+ daily volume in chosen strikes
       - Bid-ask spread < 8% of option price (reject if spread too wide)
       - **POSITION COUNT LIMIT**: Maximum 3 active F&O positions at any time
       - **POSITION CONFLICT CHECK**: New trade must not create excessive risk with existing positions
       - **SUFFICIENT CAPITAL VALIDATION:**
         * Calculate exact margin requirement using calculate_strategy_margins() tool
         * Ensure total available cash (including intraday_payin) > (margin required + 20% buffer)
         * Verify total position value doesn't exceed 80% of account value
         * Check that premium cost for debit strategies is affordable
       - Position size based on strategy type and conviction:
         * High-conviction trades: Up to 80% of capital at risk
         * Medium-conviction trades: 60% of capital at risk  
         * Speculative trades: 30% of capital at risk
         * Never risk more than 80% on any single position
       - Clear intraday or short-term exit strategy defined
       - Trade only during high-liquidity hours (9:30-11:30 AM, 1:30-3:15 PM)
       - No trades in last 30 minutes of session unless closing positions
       - Factor in realistic slippage (2-5%) when calculating expected returns
    
    7. **Default Recommendation: DO NOT TRADE**
       Unless all criteria are exceptionally met, recommend waiting.
       It's better to miss an opportunity than to force a bad trade.
    
    8. **Before executing any trade, MANDATORY VALIDATIONS:**
       
       **A. Capital Validation:**
       - Use get_account_margins() to check current available cash and intraday_payin
       - Use calculate_strategy_margins() to calculate exact margin requirement
       - Verify: total_available_cash (including intraday_payin) >= (margin required + 20% safety buffer)
       - For debit strategies: ensure total_available_cash >= (total_premium_cost + margin)
       - For credit strategies: ensure margin available for worst-case scenario
       - If insufficient funds, reject trade immediately with specific reason
       
       **B. Position Conflict Resolution:**
       - If existing positions conflict with new trade, square off conflicting positions first
       - Document any position closures required for new trade execution
       - Ensure total position count will not exceed 3 after new trade
       
       **C. Execution Pricing Validation:**
       - Always check current bid-ask spread before placing orders
       - For buying options: Use ASK price + 2-5% buffer (or ₹0.05-0.25 whichever is higher)
       - For selling options: Use BID price - 2-5% buffer (or ₹0.05-0.25 whichever is lower)
       - For spreads: Account for slippage on BOTH legs
       - Use LIMIT orders with realistic pricing, avoid MARKET orders
       - Adjust buffer based on:
         * Market volatility (higher volatility = larger buffer)
         * Liquidity (low OI/volume = larger buffer)
         * Time of day (opening/closing hours = larger buffer)
         * Strike distance from ATM (OTM strikes = larger buffer)
       - Verify pricing is still profitable after applying buffers
       - Cancel and re-evaluate if spread becomes too wide (>8-10% of option price)
    
    9. **IF RECOMMENDING EXECUTION, ACTUALLY EXECUTE THE TRADE:**
       - If you recommend EXECUTE, you MUST use the execute_options_strategy() tool
       - **FIRST**: Square off any conflicting existing positions if required
       - Create the strategy legs with exact symbols, quantities, and prices
       - Use the strategy creation tools (create_long_straddle_strategy, etc.) to build the strategy
       - Execute the trade with proper risk management and position sizing
       - Document the execution details and order confirmation
       - If execution fails, provide specific error details and alternative recommendations
    
    **CRITICAL: INTRADAY DATA REQUIREMENT DURING MARKET HOURS**
    - During market hours (9:30 AM - 3:30 PM), ALWAYS fetch intraday data (5minute, 10minute, or 15minute intervals)
    - Do NOT rely solely on daily historical data during market hours as it becomes stale quickly
    - Use daily data only for pre-market analysis or overnight gap analysis
    - For real-time trading decisions, use 5-15 minute intervals to capture current market dynamics
    - This ensures you're analyzing fresh, relevant data that reflects current market conditions
    
    **IMPORTANT - Data Interval Limits:**
    When fetching historical data or technical analysis, use these valid intervals and maximum days:
    - minute: maximum 60 days
    - 3minute: maximum 100 days  
    - 5minute: maximum 100 days
    - 10minute: maximum 100 days
    - 15minute: maximum 200 days
    - 30minute: maximum 200 days
    - 60minute: maximum 400 days
    - day: maximum 2000 days
    
    Note that minimum quantity to trade a Nifty option is 75
    Remember: Professional traders spend most of their time waiting for the right setup.
    Your job is to PROTECT capital first, GROW it second.   
    Checking option chain data is important. You should use 'NIFTY 50' as the symbol for all NIFTY instrument historical data.
    All date references should be relative to {current_date}.
    """,
    agent=strategy_executor,
    expected_output="Trade evaluation report with position count analysis, conflict assessment, and clear recommendation to either EXECUTE (with actual trade execution using execute_options_strategy tool) or WAIT (with specific conditions to monitor). If recommending EXECUTE, you MUST actually execute the trade. If after 14:30, focus only on position management and closures. Default should be to recommend waiting unless exceptional opportunity exists.",
    context=[portfolio_review_task, market_analysis_task, risk_assessment_task]
)

# ============================================================================
# CREW DEFINITION
# ============================================================================

trading_crew = Crew(
    agents=[market_analyst, risk_manager, strategy_executor],
    tasks=[portfolio_review_task, market_analysis_task, risk_assessment_task, strategy_execution_task],
    verbose=True,
    process="sequential",
    max_rpm=30,  # Limit API calls for cost control
    planning=True  # Enable better task coordination
)

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("Note: The agent will only execute trades if exceptional opportunities exist.")
    print("Most runs should result in 'WAIT' recommendations - this is normal and profitable behavior.")
    print("-" * 70)
    
    result = trading_crew.kickoff()
    
    print("\n" + "="*70)
    print(f"TRADING CREW FINAL RESULT - {current_datetime}")
    print("="*70)
    print(result)
    print("\n" + "="*70)
    print("REMEMBER: No trades executed = Capital preserved = Good outcome")
    print("="*70)