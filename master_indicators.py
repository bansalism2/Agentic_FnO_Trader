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
from typing import Dict, Any, Optional

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
        load_dotenv(dotenv_path='./.env')
        
        # Get API credentials from environment variables
        api_key = os.getenv("kite_api_key")
        api_secret = os.getenv("kite_api_secret")
        
        if not api_key or not api_secret:
            raise ValueError("API key or secret not found in .env file")
        
        # Read access token from access_token.txt
        access_token = None
        if os.path.exists("access_token.txt"):
            with open("access_token.txt", "r") as f:
                access_token = f.read().strip()
        
        if not access_token:
            raise ValueError("Access token not found in access_token.txt")
        
        # Initialize Kite Connect
        kite = KiteConnect(api_key=api_key)
        kite.set_access_token(access_token)
        
        return kite
        
    except Exception as e:
        raise Exception(f"Failed to connect to Kite Connect: {str(e)}")


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
            raise ValueError("No data received from Kite Connect API")
        
        # Set date as index
        df.set_index("date", inplace=True)
        
        # Ensure all required columns are present
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        return df
        
    except Exception as e:
        raise Exception(f"Failed to fetch NIFTY data: {str(e)}")


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
            return indicators
        
        # Generate trading signals
        signals = get_trading_signals(indicators, current_price)
        
        # Prepare final result - return what we print
        result = {
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
        return {"error": f"Technical analysis failed: {str(e)}"}


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