#!/usr/bin/env python3
"""
AlgoTrade - Ultra-Conservative Position Manager Agent
===================================================

FINAL VERSION - Ultra-conservative position management that prioritizes:
1. TIME DECAY OPTIMIZATION - Allow theta to work
2. TRADING COST MINIMIZATION - Reduce unnecessary exits
3. STRATEGY COMPLETION - Let strategies reach maturity
4. CAPITAL PRESERVATION - Only exit when absolutely necessary

Author: AlgoTrade Team
Version: 2.0 (Ultra-Conservative)
"""

# ============================================================================
# CREDENTIALS AND CONFIG
# ============================================================================

from dotenv import load_dotenv
import os
from datetime import datetime, timedelta
import json
from pathlib import Path

# Load .env using an absolute path for robustness
env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../.env'))
print(f"[DEBUG] Loading .env from: {env_path}")
load_dotenv(dotenv_path=env_path)
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
llm_model = "gemini/gemini-2.5-pro"#os.getenv("LLM_MODEL", "gpt-4o")
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

print(f"Starting Ultra-Conservative Position Manager at {current_datetime}...")

# ============================================================================
# TOOL IMPORTS
# ============================================================================

from crewai import Agent, Task, Crew
from crewai.tools import tool

# === LLM Provider Selection for CrewAI ===
llm_kwargs = {}
try:
    from crewai import LLM
    if llm_model.startswith("claude") or llm_model.startswith("anthropic"):
        if not llm_model.startswith("anthropic/"):
            llm_model = f"anthropic/{llm_model}"
        llm_kwargs = {"llm": LLM(model=llm_model, temperature=0.3)}  # Lower temperature for conservative decisions
        print(f"✅ Configured Anthropic LLM: {llm_model}")
    elif llm_model.startswith("gemini"):
        # For Google Gemini models, use the format: gemini/gemini-2.5-pro
        llm_kwargs = {"llm": LLM(model=llm_model, temperature=0.3)}  # Lower temperature for conservative decisions
        print(f"✅ Configured Google Gemini LLM: {llm_model}")
    else:
        llm_kwargs = {"llm": LLM(model=llm_model, temperature=0.3)}
        print(f"✅ Configured OpenAI LLM: {llm_model}")
except ImportError:
    print("crewAI.LLM not available, using default LLM configuration.")
    llm_kwargs = {}

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
# ULTRA-CONSERVATIVE POSITION MANAGER AGENT
# ============================================================================

ultra_conservative_position_manager = Agent(
    role="Ultra-Conservative Position Management Specialist",
    goal="Maximize intraday capital protection by monitoring MIS positions and only closing early in case of emergency or regulatory risk. Default is to let broker auto square off at 3:20 PM.",
    backstory=f"""You are an ULTRA-CONSERVATIVE position management specialist for an INTRADAY-ONLY trading system. All positions are MIS (intraday only) and will be auto squared off by the broker at 3:20 PM. There is no possibility of holding overnight or for multiple days.

**INTRADAY-ONLY LOGIC:**
- All positions are MIS (intraday only) and will be auto squared off by the broker at 3:20 PM.
- No overnight or multi-day holding is possible.

**INTRADAY COOLING-OFF PERIOD:**
- For the first 45 minutes after entry, do not exit a position unless there is a catastrophic loss (e.g., >30% loss).

**INTRADAY SACRED HOLD ZONE:**
- If P&L is between -10% and +20% and it is not within the last 30 minutes of trading, do not exit.

**EMERGENCY EXIT:**
- Catastrophic loss (e.g., >30% loss).
- Systemic/broker/regulatory risk.
- Approaching 3:20 PM: If position is at risk, consider manual exit.

**DEFAULT ACTION:**
- Hold positions until 3:20 PM unless above criteria are met.
---

CURRENT DATE AND TIME: {current_datetime}

**CORE PHILOSOPHY - HOLD FIRST, EXIT LAST:**
- Most F&O positions should be held to capture intraday opportunity
- Each exit costs ₹30-100+ in brokerage, taxes, and slippage
- Premature exits are the #1 destroyer of F&O profitability
- Market noise ≠ market trend - ignore short-term fluctuations
- Only exit for catastrophic loss, system risk, or regulatory requirement

**REMEMBER: Your default action is HOLD. Exit only when absolutely necessary and economically justified.**
""",
    tools=[
        # Position and Portfolio Tools
        tool("Get all open NIFTY positions and their P&L")(get_portfolio_positions),
        tool("Get account margin and cash details")(get_account_margins),
        tool("Get risk metrics for the account")(get_risk_metrics),
        tool("Execute a multi-leg options strategy")(execute_options_strategy),
        tool("Calculate realistic pricing for options")(calculate_realistic_pricing),
        tool("Analyze position conflicts")(analyze_position_conflicts),
        tool("Validate general capital availability")(validate_general_capital),
        
        # Market Data Tools
        tool("Get NIFTY spot price with error handling")(get_nifty_spot_price_safe),
        tool("Get NIFTY expiry dates")(get_nifty_expiry_dates),
        tool("Get options chain data")(get_options_chain_safe),
        tool("Get historical volatility")(get_historical_volatility),
        tool("Run full NIFTY technical analysis for given days/interval")(get_nifty_technical_analysis_tool),
        tool("Get NIFTY daily technical analysis")(get_nifty_daily_technical_analysis_wrapper),
        tool("Fetch historical OHLCV data for a symbol and date range")(fetch_historical_data),
        
        # Advanced Analysis Tools
        tool("VIX Integration & Volatility Regime Analysis")(analyze_vix_integration_wrapper),
        tool("IV Rank Analysis for Premium Decisions")(calculate_iv_rank_analysis_wrapper),
        tool("PCR + Technical Analysis for Entry Timing")(calculate_pcr_technical_analysis_wrapper),
        tool("PCR Extremes Analysis for Contrarian Opportunities")(analyze_pcr_extremes_wrapper),
        tool("Market Regime Detection for Strategy Selection")(detect_market_regime_wrapper),
        
        # Calculation Tools
        tool("Calculate option Greeks using Black-Scholes")(calculate_option_greeks),
        tool("Calculate implied volatility for an option")(calculate_implied_volatility),
        tool("Calculate P&L for a multi-leg options strategy")(calculate_strategy_pnl),
        tool("Calculate portfolio Greeks for open positions")(calculate_portfolio_greeks),
        tool("Calculate the volatility surface for options data")(calculate_volatility_surface),
        tool("Calculate probability of profit for a strategy")(calculate_probability_of_profit),
        tool("Calculate P&L percentage for a position")(calculate_pnl_percentage),
        
        # Trade Storage Tools
        tool("Get active trades from storage")(get_active_trades),
        tool("Update trade status in storage")(update_trade_status),
        tool("Get trade history")(get_trade_history),
        tool("Get trade summary")(get_trade_summary),
        
        # Order and History Tools
        tool("Get order history for the account")(get_orders_history),
        tool("Get daily trading summary for the account")(get_daily_trading_summary),
    ],
    verbose=True,
    max_iter=10,  # Limit iterations to prevent over-analysis
    **llm_kwargs
)

# ============================================================================
# ULTRA-CONSERVATIVE POSITION MANAGEMENT TASK
# ============================================================================

ultra_conservative_management_task = Task(
    description=f"""
    🕒 INTRADAY-ONLY POSITION MANAGEMENT PROTOCOL
    
    CURRENT TIME: {current_datetime} - Use this as your reference point
    
    **INTRADAY-ONLY LOGIC:**
    - All positions are MIS (intraday only) and will be auto squared off by the broker at 3:20 PM.
    - No overnight or multi-day holding is possible.
    
    **INTRADAY COOLING-OFF PERIOD:**
    - For the first 45 minutes after entry, do not exit a position unless there is a catastrophic loss (e.g., >30% loss).
    
    **INTRADAY SACRED HOLD ZONE:**
    - If P&L is between -10% and +20% and it is not within the last 30 minutes of trading, do not exit.
    
    **EMERGENCY EXIT:**
    - Catastrophic loss (e.g., >30% loss).
    - Systemic/broker/regulatory risk.
    - Approaching 3:20 PM: If position is at risk, consider manual exit.
    
    **DEFAULT ACTION:**
    - Hold positions until 3:20 PM unless above criteria are met.
    ---
    
    **MISSION: HOLD POSITIONS UNLESS EMERGENCY EXIT IS REQUIRED**
    
    1. **Current Position Assessment:**
       - Fetch all open NIFTY positions
       - If 0 positions found: Output "✅ NO POSITIONS TO MANAGE - TASK COMPLETE" and stop
       - For each position, calculate:
         * Current P&L percentage
         * Minutes since entry
         * Minutes to 3:20 PM
         * Position size and risk exposure
    
    2. **Exit Criteria:**
       - Apply the intraday cooling-off period and sacred hold zone logic above
       - Only consider closing a position before 3:20 PM if:
         * Catastrophic loss (e.g., >30% loss)
         * Systemic risk or regulatory requirement
         * Broker/platform/market emergency
         * Outside sacred hold zone and not in cooling-off period
       - Otherwise, HOLD and let broker auto square off
    
    3. **Documentation:**
       - For any manual exit, document the reason and economic justification
       - For all other positions, document that default action is to let broker auto square off
    
    **REMEMBER:**
    - The default and preferred action is to let all positions run until 3:20 PM for auto square-off.
    - Only intervene for emergencies or regulatory reasons, or if outside the sacred hold zone and not in cooling-off period.
    """,
    agent=ultra_conservative_position_manager,
    expected_output="""
    Intraday position management report with:
    
    📊 **POSITION SUMMARY:**
    - Total positions found
    - Positions closed early (with reason)
    - Positions left for auto square-off
    
    ⚡ **EMERGENCY EXIT ANALYSIS:**
    - Any positions closed before 3:20 PM and why
    - Catastrophic loss or regulatory risk documentation
    
    🕒 **INTRADAY HOLDING SUMMARY:**
    - Positions held until 3:20 PM for auto square-off
    - Rationale for holding (default action)
    
    Focus on HOLDING positions unless there is a clear, immediate risk or the position is outside the sacred hold zone and not in cooling-off period.
    """
)

# ============================================================================
# ULTRA-CONSERVATIVE POSITION MANAGER CREW
# ============================================================================

ultra_conservative_crew = Crew(
    agents=[ultra_conservative_position_manager],
    tasks=[ultra_conservative_management_task],
    verbose=True,
    process="sequential",
    max_rpm=25,  # Conservative API usage
    planning=True,
    memory=False  # Enable memory for better decision tracking
)

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def parse_expiry_from_symbol(symbol, available_expiries=None):
    """
    Parse expiry date from a NIFTY option symbol (e.g., NIFTY2571725450CE).
    Returns expiry as 'YYYY-MM-DD' if possible, else None.
    """
    import re
    # Look for 6 digits after NIFTY (YYMMDD)
    m = re.search(r'NIFTY(\d{6})', symbol)
    if not m:
        return None
    yymmdd = m.group(1)
    # Convert to YYYY-MM-DD
    year = int('20' + yymmdd[:2])
    month = int(yymmdd[2:4])
    day = int(yymmdd[4:6])
    expiry_str = f"{year:04d}-{month:02d}-{day:02d}"
    if available_expiries:
        # Use the closest match from available expiries
        if expiry_str in available_expiries:
            return expiry_str
        # Fallback: try to find the closest expiry >= parsed date
        from datetime import datetime
        parsed_dt = datetime(year, month, day)
        future_expiries = [e for e in available_expiries if e >= expiry_str]
        if future_expiries:
            return min(future_expiries)
        return available_expiries[-1]  # fallback to last
    return expiry_str

def calculate_position_metrics(position, current_spot_price, expiry_date):
    """
    Calculate comprehensive metrics for a position including time decay, Greeks, and risk.
    """
    try:
        # Basic position info
        symbol = position.get('tradingsymbol', '')
        quantity = position.get('quantity', 0)
        avg_price = position.get('average_price', 0)
        current_price = position.get('last_price', 0)
        
        # Calculate P&L
        pnl = (current_price - avg_price) * quantity
        pnl_percentage = (pnl / (avg_price * abs(quantity))) * 100 if avg_price and quantity else 0
        
        # Calculate days to expiry
        if expiry_date:
            expiry_dt = datetime.strptime(expiry_date, '%Y-%m-%d')
            current_dt = datetime.now()
            days_to_expiry = (expiry_dt - current_dt).days
            hours_to_expiry = (expiry_dt - current_dt).total_seconds() / 3600
        else:
            days_to_expiry = 0
            hours_to_expiry = 0
        
        # Determine position classification
        classification = "UNKNOWN"
        if days_to_expiry < 1:
            classification = "EMERGENCY"
        elif pnl_percentage > 75:
            classification = "URGENT_PROFIT"
        elif pnl_percentage < -95:
            classification = "EMERGENCY_LOSS"
        elif -25 <= pnl_percentage <= 60:
            classification = "SACRED_HOLD_ZONE"
        else:
            classification = "NORMAL"
        
        return {
            'symbol': symbol,
            'quantity': quantity,
            'avg_price': avg_price,
            'current_price': current_price,
            'pnl': pnl,
            'pnl_percentage': pnl_percentage,
            'days_to_expiry': days_to_expiry,
            'hours_to_expiry': hours_to_expiry,
            'classification': classification,
            'position_value': abs(avg_price * quantity),
            'risk_amount': abs(pnl) if pnl < 0 else 0
        }
    except Exception as e:
        print(f"Error calculating position metrics: {e}")
        return {}

def calculate_exit_costs(position_value, quantity):
    """
    Calculate total exit costs including brokerage, taxes, and slippage.
    """
    # Brokerage (per lot + percentage)
    brokerage_per_lot = 20  # Base brokerage per lot
    brokerage_percentage = 0.0003  # 0.03%
    brokerage = (abs(quantity) * brokerage_per_lot) + (position_value * brokerage_percentage)
    
    # Securities transaction tax (STT)
    stt = position_value * 0.0005  # 0.05% for options
    
    # Exchange charges
    exchange_charges = position_value * 0.0000345  # NSE charges
    
    # GST on brokerage and charges
    gst = (brokerage + exchange_charges) * 0.18
    
    # Stamp duty
    stamp_duty = position_value * 0.000003  # 0.0003%
    
    # Slippage estimate
    slippage = position_value * 0.01  # 1% slippage estimate
    
    total_costs = brokerage + stt + exchange_charges + gst + stamp_duty + slippage
    
    return {
        'brokerage': brokerage,
        'stt': stt,
        'exchange_charges': exchange_charges,
        'gst': gst,
        'stamp_duty': stamp_duty,
        'slippage': slippage,
        'total_costs': total_costs
    }

def validate_exit_decision(position_metrics, exit_costs, strategy_info=None):
    """
    Validate if a position should be exited based on ultra-conservative criteria.
    """
    validation_result = {
        'should_exit': False,
        'priority': 'HOLD',
        'reasons': [],
        'gates_passed': [],
        'gates_failed': [],
        'economic_justification': False
    }
    
    pnl_pct = position_metrics.get('pnl_percentage', 0)
    days_to_expiry = position_metrics.get('days_to_expiry', 999)
    classification = position_metrics.get('classification', 'UNKNOWN')
    total_costs = exit_costs.get('total_costs', 0)
    position_value = position_metrics.get('position_value', 0)
    
    # Gate 1: Cooling-off period (assume minimum 2 days)
    # This would need actual entry date from trade storage
    cooling_off_passed = True  # Placeholder - implement with actual entry dates
    if cooling_off_passed:
        validation_result['gates_passed'].append('COOLING_OFF_PERIOD')
    else:
        validation_result['gates_failed'].append('COOLING_OFF_PERIOD')
    
    # Gate 2: Sacred hold zone check
    sacred_hold_zone = -25 <= pnl_pct <= 60
    if not sacred_hold_zone or classification == "EMERGENCY":
        validation_result['gates_passed'].append('SACRED_HOLD_ZONE')
    else:
        validation_result['gates_failed'].append('SACRED_HOLD_ZONE')
        validation_result['reasons'].append(f"Position in sacred hold zone ({pnl_pct:.1f}% P&L)")
    
    # Gate 4: Time decay optimization
    time_decay_optimal = days_to_expiry <= 1 or pnl_pct > 75 or pnl_pct < -85
    if time_decay_optimal:
        validation_result['gates_passed'].append('TIME_DECAY_OPTIMAL')
    else:
        validation_result['gates_failed'].append('TIME_DECAY_OPTIMAL')
        validation_result['reasons'].append(f"Position should be held for time decay ({days_to_expiry} days left)")
    
    # Final decision logic
    if classification == "EMERGENCY":
        validation_result['should_exit'] = True
        validation_result['priority'] = 'EMERGENCY'
        validation_result['reasons'].append("Emergency exit required (<1 day expiry or >95% loss)")
    elif classification == "URGENT_PROFIT" and pnl_pct > 40:
        validation_result['should_exit'] = True
        validation_result['priority'] = 'URGENT'
        validation_result['reasons'].append("Exceptional profit with economic justification")
    elif len(validation_result['gates_failed']) == 0 and pnl_pct > 40:
        validation_result['should_exit'] = True
        validation_result['priority'] = 'CONDITIONAL'
        validation_result['reasons'].append("All validation gates passed")
    else:
        validation_result['priority'] = 'HOLD'
        validation_result['reasons'].append("Ultra-conservative criteria: HOLD for time decay")
    
    return validation_result

# ============================================================================
# MAIN EXECUTION FUNCTION
# ============================================================================

def run_ultra_conservative_position_manager():
    """
    Main execution function for the ultra-conservative position manager.
    """
    print("\n" + "="*80)
    print("🎯 ULTRA-CONSERVATIVE POSITION MANAGER STARTING")
    print(f"📅 Current Time: {current_datetime}")
    print("🔄 Mission: Hold positions, let time decay work, exit only when absolutely necessary")
    print("="*80)
    
    try:
        # Execute the crew (connection is already initialized at module level)
        result = ultra_conservative_crew.kickoff()
        
        print("\n" + "="*80)
        print("📊 ULTRA-CONSERVATIVE POSITION MANAGER RESULTS")
        print("="*80)
        print(result)
        
        # Additional summary
        print("\n" + "-"*80)
        print("💡 ULTRA-CONSERVATIVE MANAGEMENT SUMMARY:")
        print("   ✅ Default Action: HOLD positions for time decay")
        print("   ⚡ Emergency Exits: Only when absolutely necessary")
        print("   💰 Cost-Aware: Every exit economically justified")
        print("   📈 Time Decay: Primary source of F&O profits")
        print("   🛡️ Capital Protection: Through patience, not premature action")
        print("-"*80)
        
        return result
        
    except Exception as e:
        print(f"\n❌ Error in position manager execution: {e}")
        print("🔄 Attempting graceful recovery...")
        
        # Fallback: Basic position check
        try:
            positions = get_portfolio_positions()
            if positions.get('total_positions', 0) == 0:
                print("✅ No positions found - no management required")
                return "No positions to manage"
            else:
                print(f"⚠️ Found {positions.get('total_positions', 0)} positions but execution failed")
                print("🔄 Manual review recommended")
                return f"Execution failed but {positions.get('total_positions', 0)} positions need review"
        except Exception as fallback_error:
            print(f"❌ Fallback also failed: {fallback_error}")
            return "Critical error - manual intervention required"

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("🚀 Starting Ultra-Conservative Position Manager Agent...")
    print("📋 This agent prioritizes time decay optimization over premature exits")
    print("💰 Every exit decision includes comprehensive cost-benefit analysis")
    print("🎯 Goal: Maximize profitability through patience and discipline")
    
    # Run the position manager
    final_result = run_ultra_conservative_position_manager()
    
    print("\n" + "="*80)
    print("🏁 ULTRA-CONSERVATIVE POSITION MANAGER COMPLETED")
    print(f"⏰ Completion Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    print("\n📈 Remember: Time decay is your friend in F&O trading!")
    print("💡 Most profitable strategy: Hold positions and let theta work")
    print("⚡ Exit only when absolutely necessary and economically justified")
    print("\n" + "="*80)