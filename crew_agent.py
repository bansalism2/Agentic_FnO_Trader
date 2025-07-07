from dotenv import load_dotenv
import os
from datetime import datetime

def load_kite_credentials():
    load_dotenv(dotenv_path='./.env')
    api_key = os.getenv("kite_api_key")
    api_secret = os.getenv("kite_api_secret")
    access_token = None
    try:
        with open("access_token.txt", "r") as f:
            access_token = f.read().strip()
    except Exception as e:
        print("Could not read access_token.txt:", e)
    return api_key, api_secret, access_token

api_key, api_secret, access_token = load_kite_credentials()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
print("OPENAI_API_KEY loaded:", os.environ.get("OPENAI_API_KEY"))
print("Kite credentials loaded:", api_key, api_secret, access_token)

from crewai import Agent, Task, Crew
from crewai.tools import tool

# Import your functions - avoid circular imports
try:
    from connect_data_tools import (
        get_nifty_spot_price, get_nifty_expiry_dates, get_options_chain, 
        get_historical_volatility, analyze_options_flow, initialize_connection, fetch_historical_data
    )
except ImportError as e:
    print(f"Warning: Could not import connect_data_tools: {e}")

try:
    from calculate_analysis_tools import (
        calculate_option_greeks, calculate_implied_volatility, calculate_strategy_pnl, 
        find_arbitrage_opportunities, calculate_portfolio_greeks, calculate_volatility_surface, 
        calculate_probability_of_profit
    )
except ImportError as e:
    print(f"Warning: Could not import calculate_analysis_tools: {e}")

# Import technical analysis tool
try:
    from master_indicators import get_nifty_technical_analysis
except ImportError as e:
    print(f"Warning: Could not import get_nifty_technical_analysis from master_indicators: {e}")

# Import pre-market data tool with caching
try:
    from pre_market_data import main as fetch_pre_market_data
except ImportError as e:
    print(f"Warning: Could not import pre_market_data: {e}")

# Global cache for pre-market data
pre_market_cache = {
    'data': None,
    'date': None,
    'timestamp': None
}

# Import execution tools with error handling to avoid circular imports
def safe_import_execution_tools():
    """Safely import execution tools to avoid circular imports"""
    execution_tools = {}
    try:
        import execution_portfolio_tools as ept
        execution_tools = {
            'get_portfolio_positions': getattr(ept, 'get_portfolio_positions', None),
            'get_account_margins': getattr(ept, 'get_account_margins', None),
            'get_orders_history': getattr(ept, 'get_orders_history', None),
            'get_daily_trading_summary': getattr(ept, 'get_daily_trading_summary', None),
            'get_risk_metrics': getattr(ept, 'get_risk_metrics', None),
            'execute_options_strategy': getattr(ept, 'execute_options_strategy', None),
            'calculate_strategy_margins': getattr(ept, 'calculate_strategy_margins', None)
        }
    except ImportError as e:
        print(f"Warning: Could not import execution_portfolio_tools: {e}")
        # Create dummy functions to prevent errors
        def dummy_function(*args, **kwargs):
            return {'status': 'ERROR', 'message': 'Function not available due to import error'}
        
        execution_tools = {
            'get_portfolio_positions': dummy_function,
            'get_account_margins': dummy_function,
            'get_orders_history': dummy_function,
            'get_daily_trading_summary': dummy_function,
            'get_risk_metrics': dummy_function,
            'execute_options_strategy': dummy_function,
            'calculate_strategy_margins': dummy_function
        }
    return execution_tools

# Get execution tools
execution_tools = safe_import_execution_tools()
get_portfolio_positions = execution_tools['get_portfolio_positions']
get_account_margins = execution_tools['get_account_margins']
get_orders_history = execution_tools['get_orders_history']
get_daily_trading_summary = execution_tools['get_daily_trading_summary']
get_risk_metrics = execution_tools['get_risk_metrics']
execute_options_strategy = execution_tools['execute_options_strategy']
calculate_strategy_margins = execution_tools['calculate_strategy_margins']

try:
    from strategy_creation_tools import (
        create_long_straddle_strategy, create_short_strangle_strategy, create_iron_condor_strategy, 
        create_butterfly_spread_strategy, create_ratio_spread_strategy, recommend_options_strategy, 
        analyze_strategy_greeks
    )
except ImportError as e:
    print(f"Warning: Could not import strategy_creation_tools: {e}")

# Initialize Kite Connect session at startup
try:
    init_result = initialize_connection(api_key, api_secret, access_token)
    print("Kite Connect initialization result:", init_result)
except NameError:
    print("Warning: initialize_connection not available - Kite Connect tools may not work")
    init_result = {'status': 'ERROR', 'message': 'initialize_connection not available'}
except Exception as e:
    print(f"Warning: Kite Connect initialization failed: {e}")
    init_result = {'status': 'ERROR', 'message': str(e)}

# Define pricing validation function
def calculate_realistic_pricing(strike_price, option_type, expiry, transaction_type, current_bid, current_ask, market_volatility="normal"):
    """
    Calculate realistic execution prices accounting for bid-ask spreads and slippage.
    
    Args:
        strike_price: Option strike price
        option_type: 'CE' or 'PE'
        expiry: Expiry date
        transaction_type: 'buy' or 'sell'
        current_bid: Current bid price
        current_ask: Current ask price
        market_volatility: 'low', 'normal', 'high' for buffer adjustment
    
    Returns:
        Recommended order price with buffer, spread analysis, and execution probability
    """
    try:
        bid = float(current_bid)
        ask = float(current_ask)
        mid_price = (bid + ask) / 2
        spread_pct = ((ask - bid) / mid_price) * 100 if mid_price > 0 else 0
        
        # Determine buffer based on market conditions
        if market_volatility == "high":
            buffer_pct = 5
            buffer_absolute = max(0.25, mid_price * 0.05)
        elif market_volatility == "low":
            buffer_pct = 2
            buffer_absolute = max(0.05, mid_price * 0.02)
        else:  # normal
            buffer_pct = 3
            buffer_absolute = max(0.10, mid_price * 0.03)
        
        if transaction_type.lower() == "buy":
            recommended_price = ask + buffer_absolute
            execution_probability = "High" if spread_pct < 5 else "Medium" if spread_pct < 10 else "Low"
        else:  # sell
            recommended_price = bid - buffer_absolute
            execution_probability = "High" if spread_pct < 5 else "Medium" if spread_pct < 10 else "Low"
        
        # Round to nearest 0.05 (NSE tick size)
        recommended_price = round(recommended_price * 20) / 20
        
        analysis = {
            "recommended_price": recommended_price,
            "current_bid": bid,
            "current_ask": ask,
            "spread_percentage": round(spread_pct, 2),
            "buffer_applied": round(buffer_absolute, 2),
            "execution_probability": execution_probability,
            "warning": "Reject trade" if spread_pct > 8 else "Proceed with caution" if spread_pct > 5 else "Good liquidity"
        }
        
        return analysis
        
    except Exception as e:
        return {"error": f"Pricing calculation failed: {str(e)}"}

# Define capital validation function
def validate_trading_capital(strategy_legs=None, risk_percentage=5.0):
    """
    Validate if sufficient capital is available for a trading strategy.
    
    IMPORTANT: This function includes intraday_payin amounts in the available cash calculation
    since these funds are available for trading today even though they may not reflect in
    available_cash until tomorrow.
    
    Args:
        strategy_legs: List of strategy legs with order details (can be None for general validation)
        risk_percentage: Maximum percentage of capital to risk (default 5%)
    
    Returns:
        Validation result with detailed capital analysis including intraday_payin consideration
    """
    # Handle case where strategy_legs is not provided
    if strategy_legs is None:
        strategy_legs = []
    try:
        # Get current account margins
        margins_result = get_account_margins()
        if margins_result.get('status') != 'SUCCESS':
            return {
                'status': 'ERROR',
                'message': 'Could not fetch account margins',
                'validation_passed': False
            }
        
        available_cash = margins_result['equity']['available_cash']
        live_balance = margins_result['equity']['live_balance']
        intraday_payin = margins_result['equity'].get('intraday_payin', 0)
        
        # Add intraday_payin to available cash since it's available for trading today
        # even though it might not reflect in available_cash until tomorrow
        total_available_cash = available_cash + intraday_payin
        
        # Calculate strategy margin requirement
        if strategy_legs:
            margin_result = calculate_strategy_margins(strategy_legs)
            if margin_result.get('status') != 'SUCCESS':
                # Fallback: estimate margin as 20% of option premium for simple strategies
                estimated_margin = 0
                for leg in strategy_legs:
                    if 'price' in leg and 'quantity' in leg:
                        estimated_margin += leg['price'] * leg['quantity'] * 0.20
                
                margin_result = {
                    'status': 'SUCCESS',
                    'total_margin_required': estimated_margin,
                    'note': 'Estimated margin calculation used'
                }
            
            required_margin = margin_result['total_margin_required']
            
            # Calculate premium costs for debit strategies
            total_premium_cost = 0
            for leg in strategy_legs:
                if leg['action'] == 'BUY' and 'price' in leg:
                    total_premium_cost += leg['price'] * leg['quantity']
        else:
            # General capital validation without specific strategy
            required_margin = 0
            total_premium_cost = 0
            margin_result = {
                'status': 'SUCCESS',
                'total_margin_required': 0,
                'note': 'General capital validation - no specific strategy'
            }
        
        # Calculate total capital requirement
        safety_buffer = required_margin * 0.20  # 20% safety buffer
        total_requirement = required_margin + safety_buffer + total_premium_cost
        
        # Risk validation
        max_risk_amount = live_balance * (risk_percentage / 100)
        risk_amount = max(total_premium_cost, required_margin * 0.5)  # Estimate potential loss
        
        # Validation checks
        validations = {
            'sufficient_cash': total_available_cash >= total_requirement,
            'within_risk_limit': risk_amount <= max_risk_amount,
            'reasonable_margin_usage': required_margin <= (total_available_cash * 0.8),
            'emergency_buffer': (total_available_cash - total_requirement) >= (live_balance * 0.1)
        }
        
        validation_passed = all(validations.values())
        
        # Generate warnings and recommendations
        warnings = []
        if not validations['sufficient_cash']:
            warnings.append(f"Insufficient cash: Need ₹{total_requirement:.2f}, have ₹{total_available_cash:.2f} (₹{available_cash:.2f} + ₹{intraday_payin:.2f} intraday_payin)")
        if not validations['within_risk_limit']:
            warnings.append(f"Exceeds risk limit: Risk ₹{risk_amount:.2f} > Limit ₹{max_risk_amount:.2f}")
        if not validations['reasonable_margin_usage']:
            warnings.append(f"High margin usage: ₹{required_margin:.2f} > 80% of total available cash")
        if not validations['emergency_buffer']:
            warnings.append("Insufficient emergency buffer after trade")
        
        return {
            'status': 'SUCCESS',
            'validation_passed': validation_passed,
            'capital_analysis': {
                'available_cash': available_cash,
                'intraday_payin': intraday_payin,
                'total_available_cash': total_available_cash,
                'live_balance': live_balance,
                'required_margin': required_margin,
                'premium_cost': total_premium_cost,
                'safety_buffer': safety_buffer,
                'total_requirement': total_requirement,
                'remaining_cash': total_available_cash - total_requirement,
                'risk_amount': risk_amount,
                'max_risk_allowed': max_risk_amount,
                'margin_utilization_pct': (required_margin / total_available_cash) * 100 if total_available_cash > 0 else 0
            },
            'validation_checks': validations,
            'warnings': warnings,
            'recommendations': [
                'Maintain at least 10% emergency cash buffer',
                'Keep margin utilization below 80% of total available cash (including intraday_payin)',
                f'Risk per trade should not exceed {risk_percentage}% of total capital',
                'Consider paper trading if capital is very limited',
                'Intraday_payin amounts are included in available cash for trading today'
            ],
            'margin_calculation_note': margin_result.get('note', 'Exact margin calculation used'),
            'validation_type': 'strategy_specific' if strategy_legs else 'general_capital'
        }
        
    except Exception as e:
        return {
            'status': 'ERROR',
            'message': f'Capital validation failed: {str(e)}',
            'validation_passed': False
        }

# Fix the calculate_realistic_pricing function to work with CrewAI
def calculate_realistic_pricing(strike_price, option_type, expiry, transaction_type, current_bid, current_ask, market_volatility="normal"):
    """
    Calculate realistic execution prices accounting for bid-ask spreads and slippage.
    
    Args:
        strike_price: Option strike price
        option_type: 'CE' or 'PE'
        expiry: Expiry date
        transaction_type: 'buy' or 'sell'
        current_bid: Current bid price
        current_ask: Current ask price
        market_volatility: 'low', 'normal', 'high' for buffer adjustment
    
    Returns:
        Recommended order price with buffer, spread analysis, and execution probability
    """
    try:
        bid = float(current_bid)
        ask = float(current_ask)
        mid_price = (bid + ask) / 2
        spread_pct = ((ask - bid) / mid_price) * 100 if mid_price > 0 else 0
        
        # Determine buffer based on market conditions
        if market_volatility == "high":
            buffer_pct = 5
            buffer_absolute = max(0.25, mid_price * 0.05)
        elif market_volatility == "low":
            buffer_pct = 2
            buffer_absolute = max(0.05, mid_price * 0.02)
        else:  # normal
            buffer_pct = 3
            buffer_absolute = max(0.10, mid_price * 0.03)
        
        if transaction_type.lower() == "buy":
            recommended_price = ask + buffer_absolute
            execution_probability = "High" if spread_pct < 5 else "Medium" if spread_pct < 10 else "Low"
        else:  # sell
            recommended_price = bid - buffer_absolute
            execution_probability = "High" if spread_pct < 5 else "Medium" if spread_pct < 10 else "Low"
        
        # Round to nearest 0.05 (NSE tick size)
        recommended_price = round(recommended_price * 20) / 20
        
        analysis = {
            "recommended_price": recommended_price,
            "current_bid": bid,
            "current_ask": ask,
            "spread_percentage": round(spread_pct, 2),
            "buffer_applied": round(buffer_absolute, 2),
            "execution_probability": execution_probability,
            "warning": "Reject trade" if spread_pct > 8 else "Proceed with caution" if spread_pct > 5 else "Good liquidity"
        }
        
        return analysis
        
    except Exception as e:
        return {"error": f"Pricing calculation failed: {str(e)}"}

def get_nifty_instruments():
    """
    Get the correct instrument symbols and tokens for NIFTY trading.
    Returns mapping of common names to actual API symbols.
    """
    try:
        # Common NIFTY instrument mappings for Kite Connect
        nifty_symbols = {
            "NIFTY_INDEX": "NIFTY 50",  # For spot price
            "NIFTY_FUTURES": "NIFTY25JANFUT",  # Current month futures
            "NIFTY_OPTIONS": "NIFTY",  # For options chain
            "BANKNIFTY_INDEX": "NIFTY BANK",
            "BANKNIFTY_FUTURES": "BANKNIFTY25JANFUT",
            "BANKNIFTY_OPTIONS": "BANKNIFTY"
        }
        
        return {
            "symbols": nifty_symbols,
            "notes": {
                "options_format": "NIFTY[EXPIRY][STRIKE][CE/PE]",
                "futures_format": "NIFTY[EXPIRY]FUT",
                "index_format": "NIFTY 50",
                "example_option": "NIFTY25JAN25000CE",
                "example_future": "NIFTY25JANFUT"
            }
        }
    except Exception as e:
        return {"error": f"Could not fetch instrument symbols: {str(e)}"}

def get_nifty_spot_price_safe():
    """
    Get NIFTY spot price with proper error handling and symbol resolution.
    """
    try:
        # Try different symbol formats that Kite might expect
        possible_symbols = ["NIFTY 50"]
        
        for symbol in possible_symbols:
            try:
                result = get_nifty_spot_price()  # Your original function
                if result and 'error' not in str(result).lower():
                    return result
            except Exception as e:
                continue
        
        # If all symbols fail, return structured error
        return {
            "error": "Could not fetch NIFTY spot price",
            "tried_symbols": possible_symbols,
            "suggestion": "Check Kite Connect instrument list or use get_nifty_instruments() tool"
        }
        
    except Exception as e:
        return {"error": f"NIFTY spot price fetch failed: {str(e)}"}

def debug_kite_instruments():
    """
    Debug function to check available instruments and find correct NIFTY symbols.
    """
    try:
        debug_info = {
            "common_issues": [
                "Symbol format incorrect (try 'NIFTY 50' instead of 'NIFTY')",
                "Instrument token required instead of symbol",
                "Exchange prefix needed (NIFTY)",
                "Case sensitivity (NIFTY vs nifty)"
            ],
            "solutions": [
                "Use kite.instruments() to get all available symbols",
                "Search for instruments containing 'NIFTY'",
                "Use instrument tokens instead of symbols",
                "Check exchange segment (NSE vs NFO)"
            ]
        }
        
        return debug_info
        
    except Exception as e:
        return {"error": f"Debug failed: {str(e)}"}

# Simplified options chain wrapper to handle parameter issues
def get_options_chain_safe(expiry_date=None, strike_range=10):
    """
    Safe wrapper for options chain with fallback parameters
    """
    try:
        if expiry_date is None:
            # Get nearest expiry first
            expiry_result = get_nifty_expiry_dates('nearest')
            if expiry_result.get('status') == 'SUCCESS':
                expiry_date = expiry_result.get('nearest_expiry', '2025-07-10')
            else:
                expiry_date = '2025-07-10'  # Fallback
        
        # Call the actual options chain function
        result = get_options_chain(expiry_date=expiry_date, strike_range=strike_range)
        return result
        
    except Exception as e:
        return {
            'status': 'ERROR',
            'message': f'Options chain fetch failed: {str(e)}',
            'expiry_date': expiry_date
        }

# Safe options flow analysis
def analyze_options_flow_safe(expiry_date=None):
    """
    Safe wrapper for options flow analysis
    """
    try:
        if expiry_date is None:
            # Get nearest expiry first
            expiry_result = get_nifty_expiry_dates('nearest')
            if expiry_result.get('status') == 'SUCCESS':
                expiry_date = expiry_result.get('nearest_expiry', '2025-07-10')
            else:
                expiry_date = '2025-07-10'  # Fallback
        
        # Call the actual function with expiry_date
        result = analyze_options_flow(expiry_date)
        return result
        
    except Exception as e:
        return {
            'status': 'ERROR',
            'message': f'Options flow analysis failed: {str(e)}',
            'expiry_date': expiry_date
        }

# Import technical analysis tool
try:
    from master_indicators import get_nifty_technical_analysis
except ImportError as e:
    print(f"Warning: Could not import get_nifty_technical_analysis from master_indicators: {e}")

def get_nifty_technical_analysis_tool(days: int = 2, interval: str = "15minute"):
    """
    Run full NIFTY technical analysis (OHLCV, indicators, signals) for the given days and interval.
    
    Valid intervals and maximum days:
    - minute: 60 days
    - 3minute: 100 days  
    - 5minute: 100 days
    - 10minute: 100 days
    - 15minute: 200 days
    - 30minute: 200 days
    - 60minute: 400 days
    - day: 2000 days
    
    Args:
        days (int): Number of days of historical data (limited by interval)
        interval (str): Data interval from the valid list above
    
    Returns:
        dict: Complete NIFTY technical analysis with indicators and signals
        
    Example usage: get_nifty_technical_analysis_tool(days=4, interval="15minute")
    """
    return get_nifty_technical_analysis(days=days, interval=interval)

def analyze_position_conflicts(existing_positions, proposed_strategy):
    """
    Analyze potential conflicts between existing positions and a proposed new strategy.
    
    Args:
        existing_positions: List of current portfolio positions
        proposed_strategy: Dictionary containing proposed strategy details
    
    Returns:
        Conflict analysis with recommendations for position management
    """
    try:
        conflicts = {
            'directional_conflicts': [],
            'volatility_conflicts': [],
            'expiry_conflicts': [],
            'strike_conflicts': [],
            'risk_concentration': [],
            'recommendations': []
        }
        
        if not existing_positions or len(existing_positions) == 0:
            return {
                'status': 'SUCCESS',
                'conflicts_found': False,
                'message': 'No existing positions - clean slate for new trade',
                'analysis': conflicts
            }
        
        # Analyze each existing position against proposed strategy
        for position in existing_positions:
            # Directional conflict analysis
            if 'direction' in position and 'direction' in proposed_strategy:
                if position['direction'] != proposed_strategy['direction']:
                    conflicts['directional_conflicts'].append({
                        'position': position,
                        'conflict_type': 'opposite_direction',
                        'risk_level': 'high'
                    })
            
            # Expiry conflict analysis
            if 'expiry' in position and 'expiry' in proposed_strategy:
                if position['expiry'] == proposed_strategy['expiry']:
                    conflicts['expiry_conflicts'].append({
                        'position': position,
                        'conflict_type': 'same_expiry',
                        'risk_level': 'medium'
                    })
            
            # Strike conflict analysis
            if 'strike' in position and 'strike' in proposed_strategy:
                strike_diff = abs(position['strike'] - proposed_strategy['strike'])
                if strike_diff < 100:  # Close strikes create gamma risk
                    conflicts['strike_conflicts'].append({
                        'position': position,
                        'conflict_type': 'close_strikes',
                        'risk_level': 'medium',
                        'strike_distance': strike_diff
                    })
        
        # Risk concentration analysis
        total_positions = len(existing_positions) + 1  # Including proposed
        if total_positions > 3:
            conflicts['risk_concentration'].append({
                'issue': 'too_many_positions',
                'current_count': len(existing_positions),
                'proposed_count': total_positions,
                'risk_level': 'high'
            })
        
        # Generate recommendations
        if conflicts['directional_conflicts']:
            conflicts['recommendations'].append({
                'action': 'close_opposite_positions',
                'priority': 'high',
                'reason': 'Directional conflicts create hedging inefficiency'
            })
        
        if conflicts['expiry_conflicts']:
            conflicts['recommendations'].append({
                'action': 'consider_different_expiry',
                'priority': 'medium',
                'reason': 'Same expiry creates concentration risk'
            })
        
        if conflicts['strike_conflicts']:
            conflicts['recommendations'].append({
                'action': 'avoid_close_strikes',
                'priority': 'medium',
                'reason': 'Close strikes create excessive gamma risk'
            })
        
        if conflicts['risk_concentration']:
            conflicts['recommendations'].append({
                'action': 'manage_existing_positions',
                'priority': 'high',
                'reason': 'Too many positions create management complexity'
            })
        
        conflicts_found = any([
            conflicts['directional_conflicts'],
            conflicts['volatility_conflicts'],
            conflicts['expiry_conflicts'],
            conflicts['strike_conflicts'],
            conflicts['risk_concentration']
        ])
        
        return {
            'status': 'SUCCESS',
            'conflicts_found': conflicts_found,
            'total_conflicts': sum(len(conflicts[key]) for key in conflicts if key != 'recommendations'),
            'analysis': conflicts,
            'recommendation': 'Proceed with caution' if conflicts_found else 'No conflicts detected'
        }
        
    except Exception as e:
        return {
            'status': 'ERROR',
            'message': f'Position conflict analysis failed: {str(e)}',
            'conflicts_found': True,  # Assume conflict if analysis fails
            'recommendation': 'Manual review required'
        }

def get_global_market_conditions(current_date: str = None):
    """
    Get global market conditions with caching to avoid repeated API calls.
    This tool is most useful for market opening predictions and early trading decisions.
    
    IMPORTANT DISCLAIMER: These global market data points are indicators only and may or may not impact NIFTY.
    Do not rely solely on these predictions for trading decisions. Always combine with local technical analysis
    and maintain proper risk management regardless of global predictions.
    
    **TIME RESTRICTION**: This tool is only available before 9:30 AM IST (market opening).
    After 9:30 AM, focus on real-time NIFTY data instead of global indicators.
    
    Args:
        current_date: Date in YYYY-MM-DD format. If None, uses today's date.
    
    Returns:
        Cached global market data with NIFTY gap predictions and trading insights.
        Note: This data is most relevant for market opening analysis, not for 
        intraday trading decisions later in the day.
    """
    try:
        from datetime import datetime
        import pytz
        
        # Check current time - restrict access after 9:30 AM
        ist_timezone = pytz.timezone('Asia/Kolkata')
        current_time = datetime.now(ist_timezone)
        current_time_str = current_time.strftime('%H:%M')
        
        # If time is after 9:30 AM, return restriction message
        if current_time.hour > 9 or (current_time.hour == 9 and current_time.minute >= 30):
            return {
                'status': 'RESTRICTED',
                'message': 'Instead of global indicators now you should focus on real nifty data for the day since trading has already started',
                'current_time': current_time_str,
                'restriction_reason': 'Market hours - use real-time NIFTY data instead of global indicators',
                'recommendation': 'Use get_nifty_spot_price_safe(), get_nifty_technical_analysis_tool(), or get_safe_options_chain_data() for current market analysis'
            }
        
        # Use current date if not provided
        if current_date is None:
            current_date = current_time.strftime('%Y-%m-%d')
        
        # Check cache first
        if (pre_market_cache['data'] is not None and 
            pre_market_cache['date'] == current_date):
            print(f"Using cached pre-market data for {current_date}")
            return {
                'status': 'SUCCESS',
                'source': 'CACHE',
                'date': current_date,
                'current_time': current_time_str,
                'data': pre_market_cache['data'],
                'note': 'This data is most useful for market opening predictions. For intraday decisions, consider current market conditions.'
            }
        
        # Fetch fresh data
        print(f"Fetching fresh pre-market data for {current_date}")
        
        # Check if fetch_pre_market_data is available
        if 'fetch_pre_market_data' not in globals():
            return {
                'status': 'ERROR',
                'message': 'Pre-market data module not available',
                'date': current_date,
                'current_time': current_time_str
            }
        
        data = fetch_pre_market_data()
        
        if data:
            # Update cache
            pre_market_cache['data'] = data
            pre_market_cache['date'] = current_date
            pre_market_cache['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            return {
                'status': 'SUCCESS',
                'source': 'FRESH_FETCH',
                'date': current_date,
                'current_time': current_time_str,
                'data': data,
                'note': 'This data is most useful for market opening predictions. For intraday decisions, consider current market conditions.'
            }
        else:
            return {
                'status': 'ERROR',
                'message': 'Failed to fetch pre-market data',
                'date': current_date,
                'current_time': current_time_str
            }
            
    except ImportError:
        return {
            'status': 'ERROR',
            'message': 'Pre-market data module not available',
            'date': current_date
        }
    except Exception as e:
        return {
            'status': 'ERROR',
            'message': f'Error fetching global market conditions: {str(e)}',
            'date': current_date
        }

# Update tools list to include new error-handling tools
def create_tools_list():
    """Create tools list with error handling for missing imports"""
    tools = []
    
    # Always available tools (defined in this file)
    tools.extend([
        tool("Get correct instrument symbols for NIFTY trading")(get_nifty_instruments),
        tool("Get NIFTY spot price with error handling")(get_nifty_spot_price_safe),
        tool("Debug Kite Connect instruments and symbols")(debug_kite_instruments),
        tool("Validate if sufficient capital is available for trading strategy")(validate_trading_capital),
        tool("Get safe options chain data with fallback parameters")(get_options_chain_safe),
        tool("Analyze options flow with safe parameter handling")(analyze_options_flow_safe),
        tool("Get global market conditions with NIFTY gap predictions (cached, most useful for market opening, RESTRICTED after 9:30 AM)")(get_global_market_conditions),
        tool("Analyze potential conflicts between existing positions and proposed new strategy")(analyze_position_conflicts),
    ])
    
    # Connection tools (if available)
    try:
        tools.extend([
            tool("Get available NIFTY expiry dates.")(get_nifty_expiry_dates),
            tool("Get historical volatility for NIFTY.")(get_historical_volatility),
            tool("Fetch historical OHLCV data for a symbol and date range (RESTRICTED: max 5 days for non-daily intervals).")(fetch_historical_data),
            tool("Run full NIFTY technical analysis (OHLCV, indicators, signals) for a given days/interval.")(get_nifty_technical_analysis_tool),
        ])
    except NameError:
        print("Warning: Some connection tools not available")
    
    # Execution tools (if available)
    tools.extend([
        tool("Get all open NIFTY positions and their P&L")(get_portfolio_positions),
        tool("Get account margin and cash details.")(get_account_margins),
        tool("Get order history for the account.")(get_orders_history),
        tool("Get daily trading summary for the account.")(get_daily_trading_summary),
        tool("Get risk metrics for the account.")(get_risk_metrics),
        tool("Execute a multi-leg options strategy.")(execute_options_strategy),
    ])
    
    # Add margin calculation if available
    try:
        tools.append(tool("Calculate margin requirement for a strategy.")(calculate_strategy_margins))
    except NameError:
        print("Warning: calculate_strategy_margins not available")
    
    # Calculation tools (if available)
    try:
        tools.extend([
            tool("Calculate option Greeks using Black-Scholes.")(calculate_option_greeks),
            tool("Calculate implied volatility for an option.")(calculate_implied_volatility),
            tool("Calculate P&L for a multi-leg options strategy.")(calculate_strategy_pnl),
            tool("Find arbitrage opportunities in the options chain.")(find_arbitrage_opportunities),
            tool("Calculate portfolio Greeks for open positions.")(calculate_portfolio_greeks),
            tool("Calculate the volatility surface for options data.")(calculate_volatility_surface),
            tool("Calculate probability of profit for a strategy.")(calculate_probability_of_profit),
        ])
    except NameError:
        print("Warning: Calculation tools not available")
    
    # Strategy tools (if available)
    try:
        tools.extend([
            tool("Create a long straddle options strategy.")(create_long_straddle_strategy),
            tool("Create a short strangle options strategy.")(create_short_strangle_strategy),
            tool("Create an iron condor options strategy.")(create_iron_condor_strategy),
            tool("Create a butterfly spread options strategy.")(create_butterfly_spread_strategy),
            tool("Create a ratio spread options strategy.")(create_ratio_spread_strategy),
            tool("Recommend an options strategy based on market outlook.")(recommend_options_strategy),
            tool("Analyze Greeks for a given options strategy.")(analyze_strategy_greeks),
        ])
    except NameError:
        print("Warning: Strategy tools not available")
    
    return tools

tools = create_tools_list()

# Get current date and time for context
current_date = datetime.now().strftime('%Y-%m-%d')
current_time = datetime.now().strftime('%H:%M:%S')
current_datetime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

# Create specialized agents for different aspects of trading
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
    tools=tools,
    verbose=True,
    max_iter=3,  # Limit iterations to prevent over-analysis
    #memory=True   # Remember recent analysis patterns
)

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
    tools=tools,
    verbose=True,
    max_iter=2,  # Quick risk decisions
    #memory=True
)

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
    tools=tools,
    verbose=True,
    max_iter=2,  # Quick execution decisions
    #memory=True
)

# --- TASKS ---
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

# --- CREW ---
trading_crew = Crew(
    agents=[market_analyst, risk_manager, strategy_executor],
    tasks=[portfolio_review_task, market_analysis_task, risk_assessment_task, strategy_execution_task],
    verbose=True,
    process="sequential",
    max_rpm=30,  # Limit API calls for cost control
    #memory=True,  # Remember patterns across sessions
    planning=True  # Enable better task coordination
)

if __name__ == "__main__":
    print(f"Starting F&O Trading Analysis at {current_datetime}...")
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