# -*- coding: utf-8 -*-
"""
Master Technical Indicators - Combines all technical indicators from AlgoTradeActive/technicalIndicators

This module provides a comprehensive function to calculate multiple technical indicators
for a given OHLCV dataframe, returning all indicators in a single dictionary.
Also includes Kite Connect API integration to fetch NIFTY data automatically.

@author: AlgoTrade Team
"""

import pandas as pd
import numpy as np
import datetime as dt
import os
import sys
import warnings
from typing import Dict, Any, Optional, List

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Add the parent directory to path to import Kite Connect
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from kiteconnect import KiteConnect
from dotenv import load_dotenv


def calculate_rsi(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """Calculate Relative Strength Index (RSI)"""
    delta = df["close"].diff().dropna()
    u = delta * 0
    d = u.copy()
    u[delta > 0] = delta[delta > 0]
    d[delta < 0] = -delta[delta < 0]
    u[u.index[n-1]] = np.mean(u[:n])  # first value is average of gains
    u = u.drop(u.index[:(n-1)])
    d[d.index[n-1]] = np.mean(d[:n])  # first value is average of losses
    d = d.drop(d.index[:(n-1)])
    rs = u.ewm(com=n, min_periods=n).mean() / d.ewm(com=n, min_periods=n).mean()
    return 100 - 100 / (1 + rs)


def calculate_macd(df: pd.DataFrame, a: int = 12, b: int = 26, c: int = 9) -> Dict[str, pd.Series]:
    """Calculate MACD (Moving Average Convergence Divergence)"""
    df_copy = df.copy()
    df_copy["MA_Fast"] = df_copy["close"].ewm(span=a, min_periods=a).mean()
    df_copy["MA_Slow"] = df_copy["close"].ewm(span=b, min_periods=b).mean()
    df_copy["MACD"] = df_copy["MA_Fast"] - df_copy["MA_Slow"]
    df_copy["Signal"] = df_copy["MACD"].ewm(span=c, min_periods=c).mean()
    df_copy.dropna(inplace=True)
    
    return {
        "macd": df_copy["MACD"],
        "signal": df_copy["Signal"],
        "histogram": df_copy["MACD"] - df_copy["Signal"]
    }


def calculate_bollinger_bands(df: pd.DataFrame, n: int = 20) -> Dict[str, pd.Series]:
    """Calculate Bollinger Bands"""
    df_copy = df.copy()
    df_copy["MA"] = df_copy['close'].rolling(n).mean()
    df_copy["BB_up"] = df_copy["MA"] + 2 * df_copy['close'].rolling(n).std(ddof=0)
    df_copy["BB_dn"] = df_copy["MA"] - 2 * df_copy['close'].rolling(n).std(ddof=0)
    df_copy["BB_width"] = df_copy["BB_up"] - df_copy["BB_dn"]
    df_copy.dropna(inplace=True)
    
    return {
        "bb_upper": df_copy["BB_up"],
        "bb_middle": df_copy["MA"],
        "bb_lower": df_copy["BB_dn"],
        "bb_width": df_copy["BB_width"]
    }


def calculate_adx(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """Calculate Average Directional Index (ADX)"""
    df2 = df.copy()
    df2['H-L'] = abs(df2['high'] - df2['low'])
    df2['H-PC'] = abs(df2['high'] - df2['close'].shift(1))
    df2['L-PC'] = abs(df2['low'] - df2['close'].shift(1))
    df2['TR'] = df2[['H-L', 'H-PC', 'L-PC']].max(axis=1, skipna=False)
    df2['DMplus'] = np.where((df2['high'] - df2['high'].shift(1)) > (df2['low'].shift(1) - df2['low']), 
                            df2['high'] - df2['high'].shift(1), 0)
    df2['DMplus'] = np.where(df2['DMplus'] < 0, 0, df2['DMplus'])
    df2['DMminus'] = np.where((df2['low'].shift(1) - df2['low']) > (df2['high'] - df2['high'].shift(1)), 
                             df2['low'].shift(1) - df2['low'], 0)
    df2['DMminus'] = np.where(df2['DMminus'] < 0, 0, df2['DMminus'])
    
    TRn = []
    DMplusN = []
    DMminusN = []
    TR = df2['TR'].tolist()
    DMplus = df2['DMplus'].tolist()
    DMminus = df2['DMminus'].tolist()
    
    for i in range(len(df2)):
        if i < n:
            TRn.append(np.nan)
            DMplusN.append(np.nan)
            DMminusN.append(np.nan)
        elif i == n:
            TRn.append(df2['TR'].rolling(n).sum().tolist()[n])
            DMplusN.append(df2['DMplus'].rolling(n).sum().tolist()[n])
            DMminusN.append(df2['DMminus'].rolling(n).sum().tolist()[n])
        elif i > n:
            TRn.append(TRn[i-1] - (TRn[i-1]/n) + TR[i])
            DMplusN.append(DMplusN[i-1] - (DMplusN[i-1]/n) + DMplus[i])
            DMminusN.append(DMminusN[i-1] - (DMminusN[i-1]/n) + DMminus[i])
    
    df2['TRn'] = np.array(TRn)
    df2['DMplusN'] = np.array(DMplusN)
    df2['DMminusN'] = np.array(DMminusN)
    df2['DIplusN'] = 100 * (df2['DMplusN'] / df2['TRn'])
    df2['DIminusN'] = 100 * (df2['DMminusN'] / df2['TRn'])
    df2['DIdiff'] = abs(df2['DIplusN'] - df2['DIminusN'])
    df2['DIsum'] = df2['DIplusN'] + df2['DIminusN']
    df2['DX'] = 100 * (df2['DIdiff'] / df2['DIsum'])
    
    ADX = []
    DX = df2['DX'].tolist()
    for j in range(len(df2)):
        if j < 2*n-1:
            ADX.append(np.nan)
        elif j == 2*n-1:
            ADX.append(df2['DX'][j-n+1:j+1].mean())
        elif j > 2*n-1:
            ADX.append(((n-1) * ADX[j-1] + DX[j]) / n)
    
    return pd.Series(ADX, index=df2.index)


def calculate_atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """Calculate Average True Range (ATR)"""
    df_copy = df.copy()
    df_copy['H-L'] = abs(df_copy['high'] - df_copy['low'])
    df_copy['H-PC'] = abs(df_copy['high'] - df_copy['close'].shift(1))
    df_copy['L-PC'] = abs(df_copy['low'] - df_copy['close'].shift(1))
    df_copy['TR'] = df_copy[['H-L', 'H-PC', 'L-PC']].max(axis=1, skipna=False)
    df_copy['ATR'] = df_copy['TR'].ewm(com=n, min_periods=n).mean()
    return df_copy['ATR']


def calculate_supertrend(df: pd.DataFrame, n: int = 7, m: float = 3) -> pd.Series:
    """Calculate SuperTrend indicator"""
    df_copy = df.copy()
    df_copy['ATR'] = calculate_atr(df_copy, n)
    df_copy["B-U"] = ((df_copy['high'] + df_copy['low']) / 2) + m * df_copy['ATR']
    df_copy["B-L"] = ((df_copy['high'] + df_copy['low']) / 2) - m * df_copy['ATR']
    df_copy["U-B"] = df_copy["B-U"]
    df_copy["L-B"] = df_copy["B-L"]
    
    ind = df_copy.index
    for i in range(n, len(df_copy)):
        if df_copy['close'][i-1] <= df_copy['U-B'][i-1]:
            df_copy.loc[ind[i], 'U-B'] = min(df_copy['B-U'][i], df_copy['U-B'][i-1])
        else:
            df_copy.loc[ind[i], 'U-B'] = df_copy['B-U'][i]
    
    for i in range(n, len(df_copy)):
        if df_copy['close'][i-1] >= df_copy['L-B'][i-1]:
            df_copy.loc[ind[i], 'L-B'] = max(df_copy['B-L'][i], df_copy['L-B'][i-1])
        else:
            df_copy.loc[ind[i], 'L-B'] = df_copy['B-L'][i]
    
    df_copy['Strend'] = np.nan
    for test in range(n, len(df_copy)):
        if df_copy['close'][test-1] <= df_copy['U-B'][test-1] and df_copy['close'][test] > df_copy['U-B'][test]:
            df_copy.loc[ind[test], 'Strend'] = df_copy['L-B'][test]
            break
        if df_copy['close'][test-1] >= df_copy['L-B'][test-1] and df_copy['close'][test] < df_copy['L-B'][test]:
            df_copy.loc[ind[test], 'Strend'] = df_copy['U-B'][test]
            break
    
    for i in range(test+1, len(df_copy)):
        if df_copy['Strend'][i-1] == df_copy['U-B'][i-1] and df_copy['close'][i] <= df_copy['U-B'][i]:
            df_copy.loc[ind[i], 'Strend'] = df_copy['U-B'][i]
        elif df_copy['Strend'][i-1] == df_copy['U-B'][i-1] and df_copy['close'][i] >= df_copy['U-B'][i]:
            df_copy.loc[ind[i], 'Strend'] = df_copy['L-B'][i]
        elif df_copy['Strend'][i-1] == df_copy['L-B'][i-1] and df_copy['close'][i] >= df_copy['L-B'][i]:
            df_copy.loc[ind[i], 'Strend'] = df_copy['L-B'][i]
        elif df_copy['Strend'][i-1] == df_copy['L-B'][i-1] and df_copy['close'][i] <= df_copy['L-B'][i]:
            df_copy.loc[ind[i], 'Strend'] = df_copy['U-B'][i]
    
    return df_copy['Strend']


def calculate_pivot_points(df: pd.DataFrame) -> Dict[str, float]:
    """Calculate Pivot Points"""
    if len(df) == 0:
        return {}
    
    latest = df.iloc[-1]
    high = latest['high']
    low = latest['low']
    close = latest['close']
    
    pivot = (high + low + close) / 3
    r1 = 2 * pivot - low
    s1 = 2 * pivot - high
    r2 = pivot + (high - low)
    s2 = pivot - (high - low)
    r3 = high + 2 * (pivot - low)
    s3 = low - 2 * (high - pivot)
    
    return {
        "pivot": pivot,
        "r1": r1, "r2": r2, "r3": r3,
        "s1": s1, "s2": s2, "s3": s3
    }


def calculate_slope(df: pd.DataFrame, column: str = 'close', period: int = 14) -> pd.Series:
    """Calculate slope of a given column over a period"""
    df_copy = df.copy()
    df_copy[f'{column}_slope'] = df_copy[column].diff(period) / period
    return df_copy[f'{column}_slope']


def detect_candlestick_patterns(df: pd.DataFrame) -> Dict[str, bool]:
    """Detect common candlestick patterns"""
    if len(df) < 2:
        return {}
    
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    
    # Doji pattern
    body_size = abs(latest['close'] - latest['open'])
    total_range = latest['high'] - latest['low']
    doji = body_size <= (total_range * 0.1)  # Body is less than 10% of total range
    
    # Hammer pattern
    body_size = abs(latest['close'] - latest['open'])
    lower_shadow = min(latest['open'], latest['close']) - latest['low']
    upper_shadow = latest['high'] - max(latest['open'], latest['close'])
    hammer = (lower_shadow > 2 * body_size) and (upper_shadow < body_size)
    
    # Shooting Star pattern
    shooting_star = (upper_shadow > 2 * body_size) and (lower_shadow < body_size)
    
    # Maru Bozu (Strong bullish/bearish candle)
    maru_bozu_bullish = (latest['close'] > latest['open']) and (body_size > 0.8 * total_range)
    maru_bozu_bearish = (latest['close'] < latest['open']) and (body_size > 0.8 * total_range)
    
    return {
        "doji": doji,
        "hammer": hammer,
        "shooting_star": shooting_star,
        "maru_bozu_bullish": maru_bozu_bullish,
        "maru_bozu_bearish": maru_bozu_bearish
    }


def calculate_all_indicators(df: pd.DataFrame, 
                           rsi_period: int = 14,
                           macd_fast: int = 12, macd_slow: int = 26, macd_signal: int = 9,
                           bb_period: int = 20,
                           adx_period: int = 14,
                           atr_period: int = 14,
                           supertrend_atr: int = 7, supertrend_multiplier: float = 3,
                           slope_period: int = 14) -> Dict[str, Any]:
    """
    Calculate all technical indicators for a given OHLCV dataframe.
    
    Args:
        df: DataFrame with columns ['open', 'high', 'low', 'close', 'volume']
        rsi_period: Period for RSI calculation (default: 14)
        macd_fast: Fast EMA period for MACD (default: 12)
        macd_slow: Slow EMA period for MACD (default: 26)
        macd_signal: Signal line period for MACD (default: 9)
        bb_period: Period for Bollinger Bands (default: 20)
        adx_period: Period for ADX (default: 14)
        atr_period: Period for ATR (default: 14)
        supertrend_atr: ATR period for SuperTrend (default: 7)
        supertrend_multiplier: Multiplier for SuperTrend (default: 3)
        slope_period: Period for slope calculation (default: 14)
    
    Returns:
        Dictionary containing all calculated indicators
    """
    
    if df.empty or len(df) < max(rsi_period, macd_slow, bb_period, adx_period, atr_period, supertrend_atr, slope_period):
        return {"error": "Insufficient data for indicator calculation"}
    
    try:
        indicators = {}
        
        # Volume Ratio Calculation
        df['volume_avg'] = df['volume'].rolling(window=20, min_periods=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_avg']
        # CRITICAL FIX: Handle NaN or infinity values in volume_ratio
        df['volume_ratio'].replace([np.inf, -np.inf], np.nan, inplace=True)
        df['volume_ratio'].fillna(1.0, inplace=True)
        indicators["volume_ratio"] = df['volume_ratio']

        # Momentum Indicators
        indicators["rsi"] = calculate_rsi(df, rsi_period)
        indicators["macd"] = calculate_macd(df, macd_fast, macd_slow, macd_signal)
        
        # Volatility Indicators
        indicators["bollinger_bands"] = calculate_bollinger_bands(df, bb_period)
        indicators["atr"] = calculate_atr(df, atr_period)
        
        # Trend Indicators
        indicators["adx"] = calculate_adx(df, adx_period)
        indicators["supertrend"] = calculate_supertrend(df, supertrend_atr, supertrend_multiplier)
        indicators["close_slope"] = calculate_slope(df, 'close', slope_period)
        
        # Support/Resistance
        indicators["pivot_points"] = calculate_pivot_points(df)
        
        # Candlestick Patterns
        indicators["patterns"] = detect_candlestick_patterns(df)
        
        # Latest values for easy access
        indicators["latest_values"] = {
            "rsi": indicators["rsi"].iloc[-1] if not indicators["rsi"].empty else None,
            "volume_ratio": indicators["volume_ratio"].iloc[-1] if not indicators["volume_ratio"].empty else 1.0,
            "macd": indicators["macd"]["macd"].iloc[-1] if not indicators["macd"]["macd"].empty else None,
            "macd_signal": indicators["macd"]["signal"].iloc[-1] if not indicators["macd"]["signal"].empty else None,
            "bb_upper": indicators["bollinger_bands"]["bb_upper"].iloc[-1] if not indicators["bollinger_bands"]["bb_upper"].empty else None,
            "bb_lower": indicators["bollinger_bands"]["bb_lower"].iloc[-1] if not indicators["bollinger_bands"]["bb_lower"].empty else None,
            "adx": indicators["adx"].iloc[-1] if not indicators["adx"].empty else None,
            "atr": indicators["atr"].iloc[-1] if not indicators["atr"].empty else None,
            "supertrend": indicators["supertrend"].iloc[-1] if not indicators["supertrend"].empty else None,
            "close_slope": indicators["close_slope"].iloc[-1] if not indicators["close_slope"].empty else None
        }
        
        # Replace full Series with just latest values to reduce output size
        indicators["rsi"] = indicators["latest_values"]["rsi"]
        indicators["volume_ratio"] = indicators["latest_values"]["volume_ratio"]
        indicators["macd"] = {
            "macd": indicators["latest_values"]["macd"],
            "signal": indicators["latest_values"]["macd_signal"],
            "histogram": indicators["latest_values"]["macd"] - indicators["latest_values"]["macd_signal"] if indicators["latest_values"]["macd"] is not None and indicators["latest_values"]["macd_signal"] is not None else None
        }
        indicators["bollinger_bands"] = {
            "bb_upper": indicators["latest_values"]["bb_upper"],
            "bb_middle": indicators["bollinger_bands"]["bb_middle"].iloc[-1] if not indicators["bollinger_bands"]["bb_middle"].empty else None,
            "bb_lower": indicators["latest_values"]["bb_lower"],
            "bb_width": indicators["bollinger_bands"]["bb_width"].iloc[-1] if not indicators["bollinger_bands"]["bb_width"].empty else None
        }
        indicators["adx"] = indicators["latest_values"]["adx"]
        indicators["atr"] = indicators["latest_values"]["atr"]
        indicators["supertrend"] = indicators["latest_values"]["supertrend"]
        indicators["close_slope"] = indicators["latest_values"]["close_slope"]
        
        return indicators
        
    except Exception as e:
        return {"error": f"Error calculating indicators: {str(e)}"}


def get_trading_signals(indicators: Dict[str, Any], current_price: float) -> Dict[str, str]:
    """
    Generate trading signals based on technical indicators.
    
    Args:
        indicators: Dictionary containing all calculated indicators
        current_price: Current market price
    
    Returns:
        Dictionary containing trading signals
    """
    signals = {}
    
    try:
        latest = indicators.get("latest_values", {})
        
        # RSI signals
        rsi = latest.get("rsi")
        if rsi is not None:
            if rsi > 70:
                signals["rsi"] = "SELL"
            elif rsi < 30:
                signals["rsi"] = "BUY"
            else:
                signals["rsi"] = "NEUTRAL"
        
        # MACD signals
        macd = latest.get("macd")
        macd_signal = latest.get("macd_signal")
        if macd is not None and macd_signal is not None:
            if macd > macd_signal:
                signals["macd"] = "BUY"
            else:
                signals["macd"] = "SELL"
        
        # Bollinger Bands signals
        bb_upper = latest.get("bb_upper")
        bb_lower = latest.get("bb_lower")
        if bb_upper is not None and bb_lower is not None:
            if current_price > bb_upper:
                signals["bollinger_bands"] = "SELL"
            elif current_price < bb_lower:
                signals["bollinger_bands"] = "BUY"
            else:
                signals["bollinger_bands"] = "NEUTRAL"
        
        # SuperTrend signals
        supertrend = latest.get("supertrend")
        if supertrend is not None:
            if current_price > supertrend:
                signals["supertrend"] = "BUY"
            else:
                signals["supertrend"] = "SELL"
        
        # ADX signals (trend strength)
        adx = latest.get("adx")
        if adx is not None:
            if adx > 25:
                signals["adx"] = "STRONG_TREND"
            else:
                signals["adx"] = "WEAK_TREND"
        
        # # Overall signal
        # buy_signals = sum(1 for signal in signals.values() if signal == "BUY")
        # sell_signals = sum(1 for signal in signals.values() if signal == "SELL")
        
        # if buy_signals > sell_signals:
        #     signals["overall"] = "BUY"
        # elif sell_signals > buy_signals:
        #     signals["overall"] = "SELL"
        # else:
        #     signals["overall"] = "NEUTRAL"
        
        return signals
        
    except Exception as e:
        return {"error": f"Error generating signals: {str(e)}"}


def get_kite_connection() -> KiteConnect:
    """
    Establish connection to Kite Connect API using stored credentials.
    
    Returns:
        KiteConnect instance with valid access token
    """
    try:
        # Load environment variables from .env file
        # CRITICAL: Adjust path for robustness when called from different locations
        dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
        if not os.path.exists(dotenv_path):
            dotenv_path = './.env' # Fallback for direct script execution
        
        load_dotenv(dotenv_path=dotenv_path)
        
        api_key = os.getenv("kite_api_key")
        api_secret = os.getenv("kite_api_secret")
        
        if not api_key or not api_secret:
            raise ValueError("API key or secret not found in .env file")
        
        # Read access token from access_token.txt - try multiple paths
        access_token = None
        # Robust path finding for access_token.txt
        token_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'access_token.txt')
        if not os.path.exists(token_path):
            token_path = "../data/access_token.txt" # Fallback
            
        if os.path.exists(token_path):
            try:
                with open(token_path, "r") as f:
                    access_token = f.read().strip()
                if not access_token:
                    raise ValueError("Access token file is empty.")
            except Exception as e:
                raise ValueError(f"Could not read access token from {token_path}: {e}")
        else:
            raise ValueError(f"Access token file not found at {token_path}")
        
        # Initialize Kite Connect
        kite = KiteConnect(api_key=api_key)
        kite.set_access_token(access_token)
        
        # Verify connection by fetching profile
        kite.profile()
        
        return kite
        
    except Exception as e:
        # Provide a much more detailed error message
        error_message = f"CRITICAL: Failed to connect to Kite Connect. Reason: {str(e)}. "
        error_message += f"Please check API keys in .env and ensure access_token.txt is valid and accessible."
        print(error_message) # Print for immediate visibility
        raise Exception(error_message)


def get_nifty_instrument_token(kite: KiteConnect) -> int:
    """
    Get the instrument token for NIFTY 50.
    
    Args:
        kite: KiteConnect instance
    
    Returns:
        Instrument token for NIFTY 50
    """
    try:
        # Get all NSE instruments
        instruments = kite.instruments("NSE")
        instrument_df = pd.DataFrame(instruments)
        
        # Look for NIFTY 50
        nifty_row = instrument_df[instrument_df['tradingsymbol'] == 'NIFTY 50']
        
        if nifty_row.empty:
            # Try alternative names
            nifty_row = instrument_df[instrument_df['tradingsymbol'] == 'NIFTY']
        
        if nifty_row.empty:
            # Try NSE:NIFTY 50 format
            nifty_row = instrument_df[instrument_df['tradingsymbol'] == 'NSE:NIFTY 50']
        
        if nifty_row.empty:
            raise ValueError("NIFTY 50 instrument not found in NSE instruments")
        
        return nifty_row.iloc[0]['instrument_token']
        
    except Exception as e:
        raise Exception(f"Failed to get NIFTY instrument token: {str(e)}")


def fetch_nifty_ohlcv_data(days: int = 2, interval: str = "15minute") -> pd.DataFrame:
    """
    Fetch NIFTY 50 OHLCV data from Kite Connect API.
    
    Args:
        days: Number of days of historical data to fetch (default: 2)
        interval: Data interval (default: "15minute")
    
    Returns:
        DataFrame with OHLCV data
    """
    try:
        # Get Kite Connect connection
        kite = get_kite_connection()
        
        # Get NIFTY instrument token
        instrument_token = get_nifty_instrument_token(kite)
        
        # Calculate date range
        end_date = dt.date.today()
        start_date = end_date - dt.timedelta(days=days)
        
        # Fetch historical data
        data = kite.historical_data(
            instrument_token=instrument_token,
            from_date=start_date,
            to_date=end_date,
            interval=interval
        )
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        if df.empty:
            raise ValueError("No data received from Kite Connect API. This could be due to a holiday or market closure.")
        
        # Set date as index
        df.set_index("date", inplace=True)
        
        # Ensure all required columns are present
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        return df
        
    except Exception as e:
        # Improved error message for data fetching failure
        error_message = f"Failed to fetch NIFTY OHLCV data. Reason: {str(e)}. "
        error_message += "This could be due to an invalid access token, network issues, or API limits."
        print(error_message)
        raise Exception(error_message)


def get_nifty_technical_analysis(days: int = 2, interval: str = "15minute") -> Dict[str, Any]:
    """
    Complete NIFTY technical analysis function.
    Fetches data from Kite Connect and calculates all technical indicators.
    
    Args:
        days: Number of days of historical data to fetch (default: 2)
        interval: Data interval (default: "15minute")
    
    Returns:
        Dictionary containing:
        - OHLCV data
        - All technical indicators
        - Trading signals
        - Current price
        - Analysis timestamp
    """
    try:
        # Fetch NIFTY data
        df = fetch_nifty_ohlcv_data(days, interval)
        
        if df.empty:
            return {"error": "No data available for analysis"}
        
        # Get current price (latest close price)
        current_price = df['close'].iloc[-1]
        
        # Calculate all indicators
        indicators = calculate_all_indicators(df)
        
        if "error" in indicators:
            # Propagate error with more context
            error_message = f"Indicator calculation failed: {indicators['error']}"
            return {"status": "ERROR", "error": error_message}
        
        # Generate trading signals
        signals = get_trading_signals(indicators, current_price)
        
        # Prepare final result - return what we print
        result = {
            "status": "SUCCESS",
            "current_price": current_price,
            "data_points": len(df),
            "date_range": {
                "start": df.index[0].strftime('%Y-%m-%d %H:%M:%S'),
                "end": df.index[-1].strftime('%Y-%m-%d %H:%M:%S')
            },
            "analysis_timestamp": pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            "trading_signals": signals,
            "latest_indicator_values": indicators["latest_values"],
            "candlestick_patterns": indicators["patterns"],
            "recent_ohlcv_data": df.tail(10).to_dict('records')
        }
        
        return result
        
    except Exception as e:
        # Centralized error handling for the entire analysis function
        error_message = f"Technical analysis failed: {str(e)}"
        return {"status": "ERROR", "error": error_message}


# Example usage function
def analyze_market_data(df: pd.DataFrame, current_price: float) -> Dict[str, Any]:
    """
    Complete market analysis function that calculates all indicators and generates signals.
    
    Args:
        df: OHLCV DataFrame
        current_price: Current market price
    
    Returns:
        Dictionary containing indicators and trading signals
    """
    indicators = calculate_all_indicators(df)
    
    if "error" in indicators:
        return indicators
    
    signals = get_trading_signals(indicators, current_price)
    
    return {
        "indicators": indicators,
        "signals": signals,
        "analysis_timestamp": pd.Timestamp.now()
    }


# Main function to run NIFTY analysis
if __name__ == "__main__":
    try:
        print("Fetching NIFTY technical analysis...")
        analysis = get_nifty_technical_analysis(days=4, interval="15minute")
        
        if "error" in analysis:
            print(f"Error: {analysis['error']}")
        else:
            print("=== NIFTY 50 Technical Analysis ===")
            print(f"Current Price: {analysis['current_price']}")
            print(f"Data Points: {analysis['data_points']}")
            print(f"Date Range: {analysis['date_range']['start']} to {analysis['date_range']['end']}")
            print(f"Analysis Time: {analysis['analysis_timestamp']}")
            
            print("\n=== Trading Signals ===")
            for signal_name, signal_value in analysis['trading_signals'].items():
                print(f"{signal_name}: {signal_value}")
            
            print("\n=== Latest Indicator Values ===")
            latest = analysis['latest_indicator_values']
            for indicator_name, value in latest.items():
                if value is not None:
                    print(f"{indicator_name}: {value:.4f}")
            
            print("\n=== Candlestick Patterns ===")
            patterns = analysis['candlestick_patterns']
            for pattern_name, detected in patterns.items():
                if detected:
                    print(f"{pattern_name}: DETECTED")
            
    except Exception as e:
        print(f"Error running analysis: {str(e)}")


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


def calculate_pcr_technical_analysis_wrapper() -> Dict[str, Any]:
    """
    Wrapper function for PCR + Technical analysis that automatically fetches required data.
    
    Returns:
        Dict with PCR + Technical analysis for entry timing
    """
    print("[PCR DEBUG] calculate_pcr_technical_analysis_wrapper() called")
    try:
        # Import required functions - handle different import contexts
        try:
            from connect_data_tools import get_nifty_spot_price_safe, get_options_chain_safe
            print("[PCR DEBUG] Imported from connect_data_tools")
        except ImportError:
            from core_tools.connect_data_tools import get_nifty_spot_price_safe, get_options_chain_safe
            print("[PCR DEBUG] Imported from core_tools.connect_data_tools")
        
        # Get current spot price
        print("[PCR DEBUG] Getting spot price...")
        spot_result = get_nifty_spot_price_safe()
        print(f"[PCR DEBUG] get_nifty_spot_price_safe() returned: {spot_result}")
        if spot_result.get('status') != 'SUCCESS':
            print(f"[PCR DEBUG] Spot price failed: {spot_result}")
            return {
                'status': 'ERROR',
                'message': f'Failed to get spot price: {spot_result.get("message", spot_result.get("error", "Unknown error"))}'
            }
        
        spot_price = spot_result.get('spot_price', 0)
        if spot_price <= 0:
            print(f"[PCR DEBUG] Invalid spot price: {spot_price}")
            return {
                'status': 'ERROR',
                'message': 'Invalid spot price received'
            }
        
        # Get options chain
        print("[PCR DEBUG] Getting options chain...")
        options_result = get_options_chain_safe()
        print(f"[PCR DEBUG] get_options_chain_safe() returned: {options_result}")
        if options_result.get('status') != 'SUCCESS':
            print(f"[PCR DEBUG] Options chain failed: {options_result}")
            return {
                'status': 'ERROR',
                'message': f'Failed to get options chain: {options_result.get("message", "Unknown error")}'
            }
        
        options_chain = options_result.get('options_chain', [])
        if not options_chain:
            print("[PCR DEBUG] Empty options chain")
            return {
                'status': 'ERROR',
                'message': 'Empty options chain received'
            }
        
        # Get technical indicators
        print("[PCR DEBUG] Getting technical analysis...")
        tech_result = get_nifty_technical_analysis_tool(days=5, interval="15minute")
        print(f"[PCR DEBUG] get_nifty_technical_analysis_tool() returned: {tech_result}")
        if 'error' in tech_result:
            print(f"[PCR DEBUG] Technical analysis failed: {tech_result}")
            return {
                'status': 'ERROR',
                'message': f'Failed to get technical analysis: {tech_result.get("error", "Unknown error")}'
            }
        
        technical_indicators = tech_result
        
        # Call the main PCR + Technical analysis function
        print("[PCR DEBUG] Calling calculate_pcr_technical_analysis...")
        result = calculate_pcr_technical_analysis(options_chain, technical_indicators, spot_price)
        print(f"[PCR DEBUG] calculate_pcr_technical_analysis() returned: {result}")
        return result
        
    except Exception as e:
        print(f"[PCR DEBUG] Exception in calculate_pcr_technical_analysis_wrapper: {e}")
        import traceback
        traceback.print_exc()
        return {
            'status': 'ERROR',
            'message': f'PCR + Technical analysis wrapper failed: {str(e)}'
        }


def analyze_pcr_extremes_wrapper() -> Dict[str, Any]:
    """
    Wrapper function for PCR extremes analysis that automatically fetches required data.
    
    Returns:
        Dict with PCR extremes analysis for contrarian opportunities
    """
    try:
        # Import required functions - handle different import contexts
        try:
            from connect_data_tools import get_options_chain_safe
        except ImportError:
            from core_tools.connect_data_tools import get_options_chain_safe
        
        # Get options chain
        options_result = get_options_chain_safe()
        if options_result.get('status') != 'SUCCESS':
            return {
                'status': 'ERROR',
                'message': f'Failed to get options chain: {options_result.get("message", "Unknown error")}'
            }
        
        options_chain = options_result.get('options_chain', [])
        if not options_chain:
            return {
                'status': 'ERROR',
                'message': 'Empty options chain received'
            }
        
        # Call the main PCR extremes analysis function
        return analyze_pcr_extremes(options_chain, lookback_days=30)
        
    except Exception as e:
        return {
            'status': 'ERROR',
            'message': f'PCR extremes analysis wrapper failed: {str(e)}'
        }


def calculate_pcr_technical_analysis(options_chain: List[Dict[str, Any]], 
                                   technical_indicators: Dict[str, Any],
                                   spot_price: float) -> Dict[str, Any]:
    """
    Combine Put-Call Ratio analysis with technical indicators for entry timing.
    
    Args:
        options_chain: Options chain data with OI information
        technical_indicators: Technical analysis results
        spot_price: Current spot price
    
    Returns:
        Dict with PCR + Technical analysis for entry timing
    """
    try:
        # Calculate Put-Call Ratio
        total_call_oi = sum(opt.get('CE_oi', 0) for opt in options_chain)
        total_put_oi = sum(opt.get('PE_oi', 0) for opt in options_chain)
        pcr = total_put_oi / total_call_oi if total_call_oi > 0 else 1.0
        
        # Extract technical indicators
        rsi = technical_indicators.get('latest_indicator_values', {}).get('rsi', 50)
        adx = technical_indicators.get('latest_indicator_values', {}).get('adx', 20)
        macd_signal = technical_indicators.get('trading_signals', {}).get('macd', 'NEUTRAL')
        supertrend_signal = technical_indicators.get('trading_signals', {}).get('supertrend', 'NEUTRAL')
        bb_signal = technical_indicators.get('trading_signals', {}).get('bollinger_bands', 'NEUTRAL')
        
        # PCR Sentiment Analysis
        if pcr > 1.5:
            pcr_sentiment = "EXTREME_BEARISH"
            pcr_strength = "Very Strong"
        elif pcr > 1.2:
            pcr_sentiment = "BEARISH"
            pcr_strength = "Strong"
        elif pcr > 0.9:
            pcr_sentiment = "SLIGHTLY_BEARISH"
            pcr_strength = "Moderate"
        elif pcr > 0.7:
            pcr_sentiment = "SLIGHTLY_BULLISH"
            pcr_strength = "Moderate"
        elif pcr > 0.5:
            pcr_sentiment = "BULLISH"
            pcr_strength = "Strong"
        else:
            pcr_sentiment = "EXTREME_BULLISH"
            pcr_strength = "Very Strong"
        
        # Technical Analysis Scoring
        technical_score = 0
        technical_signals = []
        
        # RSI Analysis
        if rsi < 30:
            technical_score += 2
            technical_signals.append("RSI oversold - bullish signal")
        elif rsi > 70:
            technical_score -= 2
            technical_signals.append("RSI overbought - bearish signal")
        elif 40 <= rsi <= 60:
            technical_score += 0
            technical_signals.append("RSI neutral")
        
        # ADX Analysis (trend strength)
        if adx > 25:
            technical_signals.append("Strong trend detected")
            if macd_signal == 'BUY' and supertrend_signal == 'BUY':
                technical_score += 3
                technical_signals.append("Strong bullish trend")
            elif macd_signal == 'SELL' and supertrend_signal == 'SELL':
                technical_score -= 3
                technical_signals.append("Strong bearish trend")
        else:
            technical_signals.append("Weak trend - ranging market")
        
        # MACD Analysis
        if macd_signal == 'BUY':
            technical_score += 1
            technical_signals.append("MACD bullish")
        elif macd_signal == 'SELL':
            technical_score -= 1
            technical_signals.append("MACD bearish")
        
        # SuperTrend Analysis
        if supertrend_signal == 'BUY':
            technical_score += 1
            technical_signals.append("SuperTrend bullish")
        elif supertrend_signal == 'SELL':
            technical_score -= 1
            technical_signals.append("SuperTrend bearish")
        
        # Bollinger Bands Analysis
        if bb_signal == 'BUY':
            technical_score += 1
            technical_signals.append("Price near BB lower band - potential bounce")
        elif bb_signal == 'SELL':
            technical_score -= 1
            technical_signals.append("Price near BB upper band - potential reversal")
        
        # Combined Analysis for Entry Timing
        entry_signals = []
        entry_confidence = 0
        
        # PCR + Technical Convergence Analysis
        if pcr_sentiment in ["EXTREME_BULLISH", "BULLISH"] and technical_score >= 2:
            entry_signals.append("STRONG_BUY")
            entry_confidence = 8
            reasoning = "PCR shows extreme bullish sentiment + strong technical bullish signals"
        elif pcr_sentiment in ["EXTREME_BEARISH", "BEARISH"] and technical_score <= -2:
            entry_signals.append("STRONG_SELL")
            entry_confidence = 8
            reasoning = "PCR shows extreme bearish sentiment + strong technical bearish signals"
        elif pcr_sentiment in ["SLIGHTLY_BULLISH", "BULLISH"] and technical_score >= 1:
            entry_signals.append("BUY")
            entry_confidence = 6
            reasoning = "PCR shows bullish sentiment + technical bullish signals"
        elif pcr_sentiment in ["SLIGHTLY_BEARISH", "BEARISH"] and technical_score <= -1:
            entry_signals.append("SELL")
            entry_confidence = 6
            reasoning = "PCR shows bearish sentiment + technical bearish signals"
        elif abs(technical_score) <= 1 and pcr_sentiment in ["SLIGHTLY_BULLISH", "SLIGHTLY_BEARISH"]:
            entry_signals.append("NEUTRAL")
            entry_confidence = 4
            reasoning = "Mixed signals - wait for clearer confirmation"
        else:
            entry_signals.append("WAIT")
            entry_confidence = 3
            reasoning = "Conflicting signals - PCR and technical analysis diverge"
        
        # Entry Timing Recommendations
        timing_recommendations = {
            "STRONG_BUY": {
                "action": "Immediate entry recommended",
                "strategy": "Long calls, call debit spreads, bull put spreads",
                "risk_level": "Moderate",
                "stop_loss": "Below recent support levels"
            },
            "BUY": {
                "action": "Consider entry on pullbacks",
                "strategy": "Long calls, call debit spreads",
                "risk_level": "Conservative",
                "stop_loss": "Below entry price - 2%"
            },
            "STRONG_SELL": {
                "action": "Immediate entry recommended",
                "strategy": "Long puts, put debit spreads, bear call spreads",
                "risk_level": "Moderate",
                "stop_loss": "Above recent resistance levels"
            },
            "SELL": {
                "action": "Consider entry on rallies",
                "strategy": "Long puts, put debit spreads",
                "risk_level": "Conservative",
                "stop_loss": "Above entry price + 2%"
            },
            "NEUTRAL": {
                "action": "Wait for clearer signals",
                "strategy": "Iron condor, calendar spreads",
                "risk_level": "Conservative",
                "stop_loss": "N/A - wait for entry"
            },
            "WAIT": {
                "action": "Avoid new positions",
                "strategy": "Manage existing positions only",
                "risk_level": "Very Conservative",
                "stop_loss": "N/A - no new entries"
            }
        }
        
        return {
            'status': 'SUCCESS',
            'put_call_ratio': round(pcr, 3),
            'pcr_sentiment': pcr_sentiment,
            'pcr_strength': pcr_strength,
            'technical_score': technical_score,
            'technical_signals': technical_signals,
            'entry_signal': entry_signals[0],
            'entry_confidence': entry_confidence,
            'reasoning': reasoning,
            'timing_recommendations': timing_recommendations[entry_signals[0]],
            'key_indicators': {
                'rsi': round(rsi, 2),
                'adx': round(adx, 2),
                'macd_signal': macd_signal,
                'supertrend_signal': supertrend_signal,
                'bb_signal': bb_signal
            },
            'timestamp': dt.datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            'status': 'ERROR',
            'message': f'PCR + Technical analysis failed: {str(e)}'
        }


def analyze_pcr_extremes(options_chain: List[Dict[str, Any]], 
                        lookback_days: int = 30) -> Dict[str, Any]:
    """
    Analyze PCR extremes for contrarian trading opportunities.
    
    Args:
        options_chain: Current options chain data
        lookback_days: Number of days for historical comparison
    
    Returns:
        Dict with PCR extreme analysis
    """
    try:
        # Calculate current PCR
        total_call_oi = sum(opt.get('CE_oi', 0) for opt in options_chain)
        total_put_oi = sum(opt.get('PE_oi', 0) for opt in options_chain)
        current_pcr = total_put_oi / total_call_oi if total_call_oi > 0 else 1.0
        
        # Simulate historical PCR data (in real implementation, this would come from database)
        # For now, using typical PCR ranges for NIFTY
        typical_pcr_range = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
        
        # Calculate PCR percentile
        pcr_percentile = len([x for x in typical_pcr_range if x <= current_pcr]) / len(typical_pcr_range)
        
        # Extreme PCR Analysis
        if pcr_percentile >= 0.90:
            extreme_type = "EXTREME_BEARISH"
            contrarian_signal = "STRONG_BUY"
            reasoning = "PCR at extreme bearish levels - contrarian bullish opportunity"
            strategy = "Long calls, call debit spreads, bull put spreads"
        elif pcr_percentile >= 0.80:
            extreme_type = "BEARISH"
            contrarian_signal = "BUY"
            reasoning = "PCR at bearish levels - potential bullish reversal"
            strategy = "Long calls, call debit spreads"
        elif pcr_percentile <= 0.10:
            extreme_type = "EXTREME_BULLISH"
            contrarian_signal = "STRONG_SELL"
            reasoning = "PCR at extreme bullish levels - contrarian bearish opportunity"
            strategy = "Long puts, put debit spreads, bear call spreads"
        elif pcr_percentile <= 0.20:
            extreme_type = "BULLISH"
            contrarian_signal = "SELL"
            reasoning = "PCR at bullish levels - potential bearish reversal"
            strategy = "Long puts, put debit spreads"
        else:
            extreme_type = "NORMAL"
            contrarian_signal = "NEUTRAL"
            reasoning = "PCR within normal range - no extreme signals"
            strategy = "Standard strategies based on other indicators"
        
        # Risk management for extreme PCR trades
        risk_management = {
            "EXTREME_BEARISH": {
                "position_size": "Aggressive (70-80% of capital)",
                "stop_loss": "Wide (15-20% of premium)",
                "profit_target": "200-300% of premium",
                "timeframe": "1-3 days for extreme moves"
            },
            "BEARISH": {
                "position_size": "Moderate (50-60% of capital)",
                "stop_loss": "Standard (10-15% of premium)",
                "profit_target": "100-150% of premium",
                "timeframe": "3-7 days"
            },
            "EXTREME_BULLISH": {
                "position_size": "Aggressive (70-80% of capital)",
                "stop_loss": "Wide (15-20% of premium)",
                "profit_target": "200-300% of premium",
                "timeframe": "1-3 days for extreme moves"
            },
            "BULLISH": {
                "position_size": "Moderate (50-60% of capital)",
                "stop_loss": "Standard (10-15% of premium)",
                "profit_target": "100-150% of premium",
                "timeframe": "3-7 days"
            },
            "NORMAL": {
                "position_size": "Standard (40-50% of capital)",
                "stop_loss": "Standard (8-12% of premium)",
                "profit_target": "50-100% of premium",
                "timeframe": "5-10 days"
            }
        }
        
        return {
            'status': 'SUCCESS',
            'current_pcr': round(current_pcr, 3),
            'pcr_percentile': round(pcr_percentile, 3),
            'extreme_type': extreme_type,
            'contrarian_signal': contrarian_signal,
            'reasoning': reasoning,
            'recommended_strategy': strategy,
            'risk_management': risk_management[extreme_type],
            'lookback_period': f"{lookback_days} days",
            'timestamp': dt.datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            'status': 'ERROR',
            'message': f'PCR extremes analysis failed: {str(e)}'
        }


# Add new functions to the registry
MASTER_INDICATORS_TOOLS = {
    'get_nifty_technical_analysis_tool': get_nifty_technical_analysis_tool,
    'calculate_pcr_technical_analysis': calculate_pcr_technical_analysis_wrapper,
    'analyze_pcr_extremes': analyze_pcr_extremes_wrapper
}


def get_nifty_daily_technical_analysis_wrapper(days: int = 90) -> Dict[str, Any]:
    """
    Wrapper function for daily NIFTY technical analysis with robust error handling.
    
    Args:
        days: Number of days of historical data (default: 90 for sufficient indicator calculation)
    
    Returns:
        Dict with daily technical analysis or fallback analysis
    """
    try:
        # Try with the requested number of days
        result = get_nifty_technical_analysis(days=days, interval="day")
        
        if 'error' in result:
            # If insufficient data, try with more days
            if 'Insufficient data' in result.get('error', ''):
                print(f"Warning: Insufficient data for {days} days, trying with 90 days...")
                result = get_nifty_technical_analysis(days=90, interval="day")
                
                if 'error' in result:
                    # If still insufficient, try with 120 days
                    if 'Insufficient data' in result.get('error', ''):
                        print(f"Warning: Still insufficient data, trying with 120 days...")
                        result = get_nifty_technical_analysis(days=120, interval="day")
                        
                        if 'error' in result:
                            # Final fallback: return basic analysis with available data
                            return {
                                'status': 'PARTIAL_SUCCESS',
                                'message': f'Limited daily data available: {result.get("error", "Unknown error")}',
                                'analysis_type': 'daily_fallback',
                                'recommendation': 'Use intraday analysis for current market conditions',
                                'data_quality': 'limited'
                            }
        
        # If successful, add metadata
        if 'error' not in result:
            result['analysis_type'] = 'daily_complete'
            result['data_quality'] = 'full'
            result['days_analyzed'] = days
        
        return result
        
    except Exception as e:
        return {
            'status': 'ERROR',
            'message': f'Daily technical analysis failed: {str(e)}',
            'analysis_type': 'daily_error',
            'recommendation': 'Fall back to intraday analysis only',
            'data_quality': 'none'
        } 