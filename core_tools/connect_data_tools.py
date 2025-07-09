#!/usr/bin/env python3
"""
NIFTY Options - Connection & Data Tools (File 1 of 4)
====================================================

Connection management, authentication, and market data tools for NIFTY options trading.
This file contains the foundational tools for connecting to Zerodha and fetching market data.

Dependencies:
pip install kiteconnect pandas numpy scipy

Author: AI Assistant
"""

import os
import json
import logging
import datetime as dt
from typing import Dict, List, Optional, Union, Any
import pandas as pd
import numpy as np
from scipy.stats import norm

try:
    from kiteconnect import KiteConnect, KiteTicker
    from kiteconnect import exceptions as kite_exceptions
except ImportError:
    print("Please install kiteconnect: pip install kiteconnect")
    raise

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global connection instance
_kite_instance = None
_connection_config = {}


def initialize_connection(api_key: str, api_secret: str = None, access_token: str = None) -> Dict[str, Any]:
    """
    Initialize Zerodha connection for NIFTY options trading.
    
    Args:
        api_key: Zerodha API key
        api_secret: Zerodha API secret (optional if access_token provided)
        access_token: Access token (optional if api_secret provided)
    
    Returns:
        Dict with connection status and user info
    """
    global _kite_instance, _connection_config
    
    try:
        _kite_instance = KiteConnect(api_key=api_key)
        _connection_config = {
            'api_key': api_key,
            'api_secret': api_secret,
            'access_token': access_token
        }
        
        if access_token:
            _kite_instance.set_access_token(access_token)
            profile = _kite_instance.profile()
            
            return {
                'status': 'SUCCESS',
                'message': 'Connection established',
                'user_name': profile.get('user_name'),
                'user_id': profile.get('user_id'),
                'email': profile.get('email')
            }
        else:
            login_url = _kite_instance.login_url()
            return {
                'status': 'LOGIN_REQUIRED',
                'message': 'Please authenticate',
                'login_url': login_url
            }
            
    except Exception as e:
        return {
            'status': 'ERROR',
            'message': f'Connection failed: {str(e)}'
        }


def authenticate_session(request_token: str) -> Dict[str, Any]:
    """
    Authenticate session using request token.
    
    Args:
        request_token: Request token from login redirect
    
    Returns:
        Dict with authentication status and access token
    """
    global _kite_instance, _connection_config
    
    if not _kite_instance or not _connection_config.get('api_secret'):
        return {
            'status': 'ERROR',
            'message': 'Connection not initialized or API secret missing'
        }
    
    try:
        data = _kite_instance.generate_session(request_token, _connection_config['api_secret'])
        _connection_config['access_token'] = data['access_token']
        
        return {
            'status': 'SUCCESS',
            'message': 'Authentication successful',
            'access_token': data['access_token'],
            'user_name': data.get('user_name'),
            'login_time': str(data.get('login_time'))
        }
        
    except Exception as e:
        return {
            'status': 'ERROR',
            'message': f'Authentication failed: {str(e)}'
        }


def get_nifty_spot_price() -> Dict[str, Any]:
    """
    Get current NIFTY spot price.
    
    Returns:
        Dict with spot price and timestamp
    """
    if not _kite_instance:
        return {'status': 'ERROR', 'message': 'Connection not initialized'}
    
    try:
        # Try NIFTY 50 first
        quote = _kite_instance.ltp(['NSE:NIFTY 50'])
        if quote and 'NSE:NIFTY 50' in quote:
            spot_price = quote['NSE:NIFTY 50']['last_price']
        else:
            # Fallback to NIFTY
            quote = _kite_instance.ltp(['NSE:NIFTY'])
            spot_price = quote['NSE:NIFTY']['last_price']
        
        return {
            'status': 'SUCCESS',
            'spot_price': spot_price,
            'timestamp': dt.datetime.now().isoformat(),
            'symbol': 'NIFTY'
        }
        
    except Exception as e:
        return {
            'status': 'ERROR',
            'message': f'Failed to get spot price: {str(e)}'
        }


def get_nifty_expiry_dates(expiry_type: str = 'all') -> dict:
    """
    Get NIFTY options expiry dates.
    Args:
        expiry_type (str, optional): 'all', 'weekly', 'monthly', or 'nearest'. Defaults to 'all'.
            - 'all': All future expiry dates
            - 'weekly': Only weekly expiries
            - 'monthly': Only monthly expiries
            - 'nearest': The next upcoming expiry
    Returns:
        dict: Dict with status, list of expiry dates (as strings), and type information.
    """
    if not _kite_instance:
        return {'status': 'ERROR', 'message': 'Connection not initialized'}
    
    try:
        # Get NFO instruments
        instruments = _kite_instance.instruments('NFO')
        nifty_options = [inst for inst in instruments 
                        if inst['name'] == 'NIFTY' and inst['instrument_type'] in ['CE', 'PE']]
        
        if not nifty_options:
            return {'status': 'ERROR', 'message': 'No NIFTY options found'}
        
        # Extract expiry dates
        all_expiries = sorted(list(set(opt['expiry'] for opt in nifty_options)))
        
        def is_last_thursday_of_month(date):
            if date.weekday() != 3:  # Not Thursday
                return False
            next_thursday = date + dt.timedelta(days=7)
            return next_thursday.month != date.month
        
        # Classify expiries
        monthly_expiries = [exp for exp in all_expiries if is_last_thursday_of_month(exp)]
        weekly_expiries = [exp for exp in all_expiries if exp not in monthly_expiries]
        
        today = dt.date.today()
        future_expiries = [exp for exp in all_expiries if exp >= today]
        nearest_expiry = min(future_expiries) if future_expiries else None
        
        # Format dates as strings
        result = {
            'status': 'SUCCESS',
            'current_date': today.isoformat(),
            'nearest_expiry': nearest_expiry.isoformat() if nearest_expiry else None
        }
        
        if expiry_type == 'all':
            result['expiries'] = [exp.isoformat() for exp in all_expiries]
        elif expiry_type == 'weekly':
            result['expiries'] = [exp.isoformat() for exp in weekly_expiries if exp >= today]
        elif expiry_type == 'monthly':
            result['expiries'] = [exp.isoformat() for exp in monthly_expiries if exp >= today]
        elif expiry_type == 'nearest':
            result['expiries'] = [nearest_expiry.isoformat()] if nearest_expiry else []
        
        result['total_count'] = len(result['expiries'])
        result['expiry_type_requested'] = expiry_type
        
        return result
        
    except Exception as e:
        return {
            'status': 'ERROR',
            'message': f'Failed to get expiry dates: {str(e)}'
        }


def get_options_chain(expiry_date: str = None, expiry_type: str = None, strike_range: int = 10) -> dict:
    """
    Get NIFTY options chain with live data.
    Args:
        expiry_date (str, optional): Specific expiry date (YYYY-MM-DD). REQUIRED if expiry_type is not provided.
        expiry_type (str, optional): 'weekly' or 'monthly' if expiry_date is not provided. If None or not a string, will be ignored. Defaults to 'weekly'.
        strike_range (int, optional): Number of strikes above/below ATM. Defaults to 10.
    Returns:
        dict: Dict with status, options chain data, ATM strike, and error message if any.
    Note:
        If expiry_date is not provided, expiry_type must be specified and the nearest expiry will be used.
        If expiry_type is None or not a string, it will be ignored.
    """
    if not _kite_instance:
        return {'status': 'ERROR', 'message': 'Connection not initialized'}
    
    try:
        # Only use expiry_type if it's a valid string
        if expiry_type is not None and not isinstance(expiry_type, str):
            expiry_type = None
        
        # Get expiry date if not provided
        if not expiry_date:
            expiry_result = get_nifty_expiry_dates(expiry_type)
            if expiry_result['status'] != 'SUCCESS' or not expiry_result['expiries']:
                return {'status': 'ERROR', 'message': 'No expiry dates available'}
            expiry_date = expiry_result['expiries'][0]
        
        # Get spot price
        spot_result = get_nifty_spot_price()
        if spot_result['status'] != 'SUCCESS':
            return spot_result
        
        spot_price = spot_result['spot_price']
        
        # Get NIFTY options for the expiry
        instruments = _kite_instance.instruments('NFO')
        expiry_dt = dt.datetime.strptime(expiry_date, '%Y-%m-%d').date()
        
        options_for_expiry = [
            inst for inst in instruments 
            if (inst['name'] == 'NIFTY' and 
                inst['instrument_type'] in ['CE', 'PE'] and 
                inst['expiry'] == expiry_dt)
        ]
        
        if not options_for_expiry:
            return {'status': 'ERROR', 'message': f'No options found for expiry {expiry_date}'}
        
        # Get available strikes and find ATM
        strikes = sorted(list(set(opt['strike'] for opt in options_for_expiry)))
        atm_strike = min(strikes, key=lambda x: abs(x - spot_price))
        atm_index = strikes.index(atm_strike)
        
        # Select strikes around ATM
        start_index = max(0, atm_index - strike_range)
        end_index = min(len(strikes), atm_index + strike_range + 1)
        selected_strikes = strikes[start_index:end_index]
        
        # Build options chain
        chain_data = []
        symbols_to_quote = []
        
        for strike in selected_strikes:
            row = {'strike': strike}
            
            for option_type in ['CE', 'PE']:
                option = next((opt for opt in options_for_expiry 
                             if opt['strike'] == strike and opt['instrument_type'] == option_type), None)
                if option:
                    symbol = option['tradingsymbol']
                    row[f'{option_type}_symbol'] = symbol
                    row[f'{option_type}_token'] = option['instrument_token']
                    symbols_to_quote.append(f"NFO:{symbol}")
            
            chain_data.append(row)
        
        # Get live quotes
        if symbols_to_quote:
            try:
                quotes = _kite_instance.quote(symbols_to_quote)
                
                for row in chain_data:
                    for option_type in ['CE', 'PE']:
                        if f'{option_type}_symbol' in row:
                            symbol_key = f"NFO:{row[f'{option_type}_symbol']}"
                            if symbol_key in quotes:
                                quote_data = quotes[symbol_key]
                                row[f'{option_type}_ltp'] = quote_data.get('last_price', 0)
                                row[f'{option_type}_volume'] = quote_data.get('volume', 0)
                                row[f'{option_type}_oi'] = quote_data.get('oi', 0)
                                
                                # Get bid/ask from depth
                                depth = quote_data.get('depth', {})
                                buy_orders = depth.get('buy', [])
                                sell_orders = depth.get('sell', [])
                                
                                row[f'{option_type}_bid'] = buy_orders[0]['price'] if buy_orders else 0
                                row[f'{option_type}_ask'] = sell_orders[0]['price'] if sell_orders else 0
                            else:
                                row[f'{option_type}_ltp'] = 0
                                row[f'{option_type}_volume'] = 0
                                row[f'{option_type}_oi'] = 0
                                row[f'{option_type}_bid'] = 0
                                row[f'{option_type}_ask'] = 0
            except Exception as e:
                logger.warning(f"Failed to get quotes: {e}")
                # Set default values
                for row in chain_data:
                    for option_type in ['CE', 'PE']:
                        if f'{option_type}_symbol' in row:
                            row[f'{option_type}_ltp'] = 0
                            row[f'{option_type}_volume'] = 0
                            row[f'{option_type}_oi'] = 0
                            row[f'{option_type}_bid'] = 0
                            row[f'{option_type}_ask'] = 0
        
        return {
            'status': 'SUCCESS',
            'expiry_date': expiry_date,
            'spot_price': spot_price,
            'atm_strike': atm_strike,
            'total_strikes': len(chain_data),
            'strike_range': strike_range,
            'options_chain': chain_data,
            'timestamp': dt.datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            'status': 'ERROR',
            'message': f'Failed to get options chain: {str(e)}'
        }


def get_historical_volatility(days: int = 30) -> dict:
    # days=max(days, 31)
    """
    Calculate NIFTY historical volatility.
    Args:
        days (int, optional): Number of days for volatility calculation. Defaults to 30.
    Returns:
        dict: Dict with status, volatility metrics, and error message if any.
    """
    if not _kite_instance:
        return {'status': 'ERROR', 'message': 'Connection not initialized'}
    
    try:
        # Get NIFTY instrument
        instruments = _kite_instance.instruments('NSE')
        nifty_inst = next((inst for inst in instruments if inst['tradingsymbol'] == 'NIFTY 50'), None)
        
        if not nifty_inst:
            return {'status': 'ERROR', 'message': 'NIFTY 50 instrument not found'}
        
        # Get historical data
        end_date = dt.datetime.now().strftime('%Y-%m-%d')
        start_date = (dt.datetime.now() - dt.timedelta(days=days + 10)).strftime('%Y-%m-%d')
        
        historical_data = _kite_instance.historical_data(
            instrument_token=nifty_inst['instrument_token'],
            from_date=start_date,
            to_date=end_date,
            interval='day'
        )
        
        # if len(historical_data) < days:
        #     return {'status': 'ERROR', 'message': f'Insufficient data. Got {len(historical_data)} days, need {days}'}
        
        # Calculate returns and volatility
        prices = [candle['close'] for candle in historical_data]
        returns = [np.log(prices[i] / prices[i-1]) for i in range(1, len(prices))]
        
        # Annualized volatility (252 trading days)
        volatility = np.std(returns) * np.sqrt(252)
        
        # Recent volatility (last 10 days)
        recent_returns = returns[-10:] if len(returns) >= 10 else returns
        recent_volatility = np.std(recent_returns) * np.sqrt(252) if recent_returns else 0
        
        return {
            'status': 'SUCCESS',
            'period_days': len(returns),
            'historical_volatility': round(volatility, 4),
            'recent_volatility_10d': round(recent_volatility, 4),
            'volatility_percentile': round(volatility * 100, 2),  # As percentage
            'price_range': {
                'high': max(prices),
                'low': min(prices),
                'current': prices[-1]
            },
            'calculation_period': f'{start_date} to {end_date}',
            'timestamp': dt.datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            'status': 'ERROR',
            'message': f'Historical volatility calculation failed: {str(e)}'
        }


def analyze_options_flow(expiry_date: str) -> dict:
    """
    Analyze NIFTY options flow and sentiment indicators.
    Args:
        expiry_date (str): Expiry date to analyze (YYYY-MM-DD). REQUIRED.
    Returns:
        dict: Dict with status, options flow analysis, and error message if any.
    Raises:
        Returns an error dict if expiry_date is missing or invalid.
    """
    if not expiry_date:
        return {'status': 'ERROR', 'message': 'expiry_date is required (YYYY-MM-DD)'}
    try:
        # Get options chain
        chain_result = get_options_chain(expiry_date, strike_range=15)
        if chain_result['status'] != 'SUCCESS':
            return chain_result
        
        chain_data = chain_result['options_chain']
        spot_price = chain_result['spot_price']
        
        # Calculate Put-Call Ratio
        total_ce_oi = sum(row.get('CE_oi', 0) for row in chain_data)
        total_pe_oi = sum(row.get('PE_oi', 0) for row in chain_data)
        pcr = total_pe_oi / total_ce_oi if total_ce_oi > 0 else 0
        
        # Find Max Pain
        max_pain_data = []
        for row in chain_data:
            strike = row['strike']
            total_loss = 0
            
            # Loss for call writers if price > strike
            ce_oi = row.get('CE_oi', 0)
            if ce_oi > 0 and spot_price > strike:
                total_loss += (spot_price - strike) * ce_oi * 25  # Lot size
            
            # Loss for put writers if price < strike
            pe_oi = row.get('PE_oi', 0)
            if pe_oi > 0 and spot_price < strike:
                total_loss += (strike - spot_price) * pe_oi * 25
            
            max_pain_data.append({'strike': strike, 'total_loss': total_loss})
        
        max_pain_strike = max(max_pain_data, key=lambda x: x['total_loss'])['strike'] if max_pain_data else 0
        
        # High OI strikes
        ce_strikes = sorted(chain_data, key=lambda x: x.get('CE_oi', 0), reverse=True)[:5]
        pe_strikes = sorted(chain_data, key=lambda x: x.get('PE_oi', 0), reverse=True)[:5]
        
        return {
            'status': 'SUCCESS',
            'expiry_date': expiry_date,
            'spot_price': spot_price,
            'put_call_ratio': round(pcr, 3),
            'max_pain_strike': max_pain_strike,
            'total_ce_oi': total_ce_oi,
            'total_pe_oi': total_pe_oi,
            'high_oi_strikes': {
                'calls': [{'strike': row['strike'], 'oi': row.get('CE_oi', 0)} for row in ce_strikes],
                'puts': [{'strike': row['strike'], 'oi': row.get('PE_oi', 0)} for row in pe_strikes]
            },
            'market_sentiment': {
                'interpretation': 'Bullish' if pcr < 0.7 else 'Bearish' if pcr > 1.3 else 'Neutral',
                'pcr_level': 'Low' if pcr < 0.7 else 'High' if pcr > 1.3 else 'Normal'
            },
            'timestamp': dt.datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            'status': 'ERROR',
            'message': f'Options flow analysis failed: {str(e)}'
        }


def fetch_historical_data(symbol: str, from_date: str, to_date: str, interval: str = "day") -> dict:
    """
    Fetch historical OHLCV data for a given symbol and date range.
    
    **IMPORTANT RESTRICTION**: For intervals other than 'day', the duration is automatically 
    restricted to 5 days maximum to prevent excessive data requests and API rate limiting.
    
    Args:
        symbol (str): Trading symbol (e.g., 'NIFTY', 'RELIANCE', etc.). REQUIRED.
        from_date (str): Start date (YYYY-MM-DD). REQUIRED.
        to_date (str): End date (YYYY-MM-DD). REQUIRED.
        interval (str, optional): Data interval ('day', '5minute', '15minute', etc.). Defaults to 'day'.
    Returns:
        dict: Dict with status, data (list of bars), and error message if any.
    Raises:
        Returns an error dict if any required argument is missing or invalid.
    """
    if not symbol or not from_date or not to_date:
        return {'status': 'ERROR', 'message': 'symbol, from_date, and to_date are required'}
    
    global _kite_instance
    if not _kite_instance:
        return {'status': 'ERROR', 'message': 'Connection not initialized'}
    
    try:
        # Apply duration restriction for non-daily intervals
        adjusted_from_date = from_date
        adjusted_to_date = to_date
        duration_restricted = False
        
        if interval != "day":
            # Parse dates to calculate duration
            from_dt = dt.datetime.strptime(from_date, '%Y-%m-%d')
            to_dt = dt.datetime.strptime(to_date, '%Y-%m-%d')
            duration_days = (to_dt - from_dt).days
            
            # If duration > 5 days, restrict to last 5 days
            if duration_days > 5:
                adjusted_from_date = (to_dt - dt.timedelta(days=5)).strftime('%Y-%m-%d')
                duration_restricted = True
        
        # Try NFO first
        if not hasattr(fetch_historical_data, '_nfo_instruments'):
            fetch_historical_data._nfo_instruments = _kite_instance.instruments('NFO')
        nfo_instruments = fetch_historical_data._nfo_instruments
        instrument = next((inst for inst in nfo_instruments if inst['tradingsymbol'] == symbol), None)
        
        # If not found in NFO, try NSE
        if not instrument:
            if not hasattr(fetch_historical_data, '_nse_instruments'):
                fetch_historical_data._nse_instruments = _kite_instance.instruments('NSE')
            nse_instruments = fetch_historical_data._nse_instruments
            instrument = next((inst for inst in nse_instruments if inst['tradingsymbol'] == symbol), None)
        
        if not instrument:
            return {'status': 'ERROR', 'message': f'Instrument not found for symbol: {symbol}'}
        
        instrument_token = instrument['instrument_token']
        
        # Fetch historical data with adjusted dates
        data = _kite_instance.historical_data(
            instrument_token=instrument_token,
            from_date=adjusted_from_date,
            to_date=adjusted_to_date,
            interval=interval
        )
        
        response = {
            'status': 'SUCCESS', 
            'symbol': symbol, 
            'interval': interval, 
            'from_date': adjusted_from_date, 
            'to_date': adjusted_to_date, 
            'data': data
        }
        
        # Add warning if duration was restricted
        if duration_restricted:
            response['warning'] = f'Duration automatically restricted to 5 days for {interval} interval (original request: {from_date} to {to_date})'
            response['original_request'] = {
                'from_date': from_date,
                'to_date': to_date
            }
        
        return response
        
    except Exception as e:
        return {'status': 'ERROR', 'message': f'Failed to fetch historical data: {str(e)}'}





if __name__ == "__main__":
    from dotenv import load_dotenv
    import os
    
    # Try multiple possible paths for .env file
    env_paths = [
        './.env',  # Current directory
        '../.env',  # Relative to agent_tools
        '../../.env',  # Relative to core_tools
        './.env'  # Alternative relative path
    ]
    
    env_loaded = False
    for env_path in env_paths:
        try:
            load_dotenv(dotenv_path=env_path)
            print(f"✅ Successfully loaded .env from: {env_path}")
            env_loaded = True
            break
        except Exception:
            continue
            
    if not env_loaded:
        print("❌ Could not find .env file in any expected location")
    
    api_key = os.getenv("kite_api_key")
    api_secret = os.getenv("kite_api_secret")
    
    # Try multiple possible paths for access token
    access_token_paths = [
        "access_token.txt",  # Current directory
        "../data/access_token.txt",  # Relative to agent_tools
        "../../data/access_token.txt",  # Relative to core_tools
        "./data/access_token.txt"  # Alternative relative path
    ]
    
    access_token = None
    for path in access_token_paths:
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    access_token = f.read().strip()
                    print(f"✅ Successfully loaded access token from: {path}")
                    break
            except Exception as e:
                print(f"❌ Error reading access token from {path}: {e}")
                continue
                
    if access_token is None:
        print("❌ Could not find access_token.txt in any expected location")
    print("Script started!")
    print("API Key:", api_key.strip() if api_key else None)
    print("API Secret:", api_secret.strip() if api_secret else None)
    print("Access Token:", access_token if access_token else None)
    if not api_key or not api_secret:
        print("API key or secret not found. Check your .env file and variable names.")
    else:
        print("Environment variables loaded successfully.")

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
            # Get appropriate expiry (not too close to expiry)
            expiry_date = get_appropriate_expiry_date()
        
        # Call the actual options chain function
        result = get_options_chain(expiry_date=expiry_date, strike_range=strike_range)
        return result
        
    except Exception as e:
        return {
            'status': 'ERROR',
            'message': f'Options chain fetch failed: {str(e)}',
            'expiry_date': expiry_date
        }


def get_appropriate_expiry_date(min_days_to_expiry: int = 4) -> str:
    """
    Get an appropriate expiry date that's not too close to expiry.
    
    Args:
        min_days_to_expiry: Minimum days to expiry required (default: 4)
    
    Returns:
        Appropriate expiry date string (YYYY-MM-DD)
    """
    try:
        # Get all future expiry dates
        expiry_result = get_nifty_expiry_dates('all')
        if expiry_result.get('status') != 'SUCCESS':
            return '2025-07-17'  # Fallback to a safe date
        
        today = dt.date.today()
        future_expiries = []
        
        for expiry_str in expiry_result.get('expiries', []):
            try:
                expiry_date = dt.datetime.strptime(expiry_str, '%Y-%m-%d').date()
                days_to_expiry = (expiry_date - today).days
                
                if days_to_expiry >= min_days_to_expiry:
                    future_expiries.append({
                        'date': expiry_str,
                        'days_to_expiry': days_to_expiry
                    })
            except:
                continue
        
        if not future_expiries:
            return '2025-07-17'  # Fallback
        
        # Sort by days to expiry and return the first appropriate one
        future_expiries.sort(key=lambda x: x['days_to_expiry'])
        return future_expiries[0]['date']
        
    except Exception as e:
        return '2025-07-17'  # Fallback


def get_available_expiry_dates_with_analysis() -> Dict[str, Any]:
    """
    Get all available expiry dates with analysis for agent decision making.
    
    Returns:
        Dict with available expiry dates, analysis, and recommendations
    """
    try:
        # Get all future expiry dates
        expiry_result = get_nifty_expiry_dates('all')
        if expiry_result.get('status') != 'SUCCESS':
            return {
                'status': 'ERROR',
                'message': 'Failed to get expiry dates'
            }
        
        today = dt.date.today()
        available_expiries = []
        
        for expiry_str in expiry_result.get('expiries', []):
            try:
                expiry_date = dt.datetime.strptime(expiry_str, '%Y-%m-%d').date()
                days_to_expiry = (expiry_date - today).days
                
                # Categorize expiry based on days to expiry
                if days_to_expiry <= 3:
                    category = 'TOO_CLOSE'
                    recommendation = 'Avoid - too close to expiry'
                    risk_level = 'HIGH'
                elif days_to_expiry <= 10:
                    category = 'SHORT_TERM'
                    recommendation = 'Suitable for short-term strategies'
                    risk_level = 'MEDIUM'
                elif days_to_expiry <= 14:
                    category = 'MEDIUM_TERM'
                    recommendation = 'Good for medium-term strategies'
                    risk_level = 'LOW'
                else:
                    category = 'LONG_TERM'
                    recommendation = 'Suitable for longer-term strategies'
                    risk_level = 'LOW'
                
                available_expiries.append({
                    'date': expiry_str,
                    'days_to_expiry': days_to_expiry,
                    'category': category,
                    'recommendation': recommendation,
                    'risk_level': risk_level,
                    'suitable_for': {
                        'TOO_CLOSE': [],
                        'SHORT_TERM': ['Day trading', 'Scalping', 'Quick momentum plays'],
                        'MEDIUM_TERM': ['Swing trading', 'Mean reversion', 'Volatility strategies'],
                        'LONG_TERM': ['Positional trading', 'Theta decay strategies', 'Directional bets']
                    }.get(category, [])
                })
            except:
                continue
        
        # Sort by days to expiry
        available_expiries.sort(key=lambda x: x['days_to_expiry'])
        
        # Find recommended expiries
        recommended_expiries = [exp for exp in available_expiries if exp['category'] != 'TOO_CLOSE']
        
        return {
            'status': 'SUCCESS',
            'current_date': today.isoformat(),
            'total_available': len(available_expiries),
            'recommended_count': len(recommended_expiries),
            'available_expiries': available_expiries,
            'recommended_expiries': recommended_expiries,
            'analysis': {
                'short_term_options': len([exp for exp in recommended_expiries if exp['category'] == 'SHORT_TERM']),
                'medium_term_options': len([exp for exp in recommended_expiries if exp['category'] == 'MEDIUM_TERM']),
                'long_term_options': len([exp for exp in recommended_expiries if exp['category'] == 'LONG_TERM']),
                'best_for_day_trading': [exp for exp in recommended_expiries if exp['category'] == 'SHORT_TERM'][:2],
                'best_for_swing_trading': [exp for exp in recommended_expiries if exp['category'] == 'MEDIUM_TERM'][:2],
                'best_for_positional': [exp for exp in recommended_expiries if exp['category'] == 'LONG_TERM'][:2]
            },
            'recommendations': {
                'conservative': recommended_expiries[0]['date'] if recommended_expiries else None,
                'balanced': recommended_expiries[len(recommended_expiries)//2]['date'] if len(recommended_expiries) > 1 else None,
                'aggressive': recommended_expiries[-1]['date'] if recommended_expiries else None
            }
        }
        
    except Exception as e:
        return {
            'status': 'ERROR',
            'message': f'Failed to analyze expiry dates: {str(e)}'
        }


# Safe options flow analysis
def analyze_options_flow_safe(expiry_date=None):
    """
    Safe wrapper for options flow analysis
    """
    try:
        if expiry_date is None:
            # Get appropriate expiry (not too close to expiry)
            expiry_date = get_appropriate_expiry_date()
        
        # Call the actual function with expiry_date
        result = analyze_options_flow(expiry_date)
        return result
        
    except Exception as e:
        return {
            'status': 'ERROR',
            'message': f'Options flow analysis failed: {str(e)}',
            'expiry_date': expiry_date
        }


# Global cache for pre-market data
pre_market_cache = {
    'data': None,
    'date': None,
    'timestamp': None
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
        try:
            from pre_market_data import main as fetch_pre_market_data
            data = fetch_pre_market_data()
        except ImportError:
            return {
                'status': 'ERROR',
                'message': 'Pre-market data module not available',
                'date': current_date,
                'current_time': current_time_str
            }
        
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


# Connection and Data Tools Registry
CONNECTION_DATA_TOOLS = {
    'initialize_connection': initialize_connection,
    'authenticate_session': authenticate_session,
    'get_nifty_spot_price': get_nifty_spot_price,
    'get_nifty_expiry_dates': get_nifty_expiry_dates,
    'get_available_expiry_dates_with_analysis': get_available_expiry_dates_with_analysis,
    'get_options_chain': get_options_chain,
    'get_historical_volatility': get_historical_volatility,
    'analyze_options_flow': analyze_options_flow,
    'fetch_historical_data': fetch_historical_data
}