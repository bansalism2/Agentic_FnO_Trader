#!/usr/bin/env python3
"""
Pre-Market Global Data Fetcher for NIFTY F&O Trading

Fetches global market data before Indian market opens and generates
comprehensive JSON output with trading insights and improved gap predictions.

Usage: python premarket_fetcher.py
Output: Returns JSON string with analysis

@author: AlgoTrade Team
"""

import yfinance as yf
import pandas as pd
import json
import logging
import numpy as np
from datetime import datetime, timedelta
import pytz
from typing import Dict, List, Optional, Tuple
import time
from functools import wraps

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global market tickers most relevant for NIFTY
PRIMARY_TICKERS = [
    # US Markets - Strongest correlation with Nifty
    '^GSPC',      # S&P 500 - Most important
    '^IXIC',      # Nasdaq - Tech correlation
    '^DJI',       # Dow Jones
    
    # Asian Markets - Regional influence
    '^N225',      # Nikkei - Strong Asian correlation
    '^HSI',       # Hang Seng - China impact
    '^KS11',      # KOSPI - Similar market cap
    
    # Commodities - India is major importer
    'CL=F',       # Crude Oil - Major impact on INR/inflation
    'GC=F',       # Gold - Safe haven, INR hedge
    
    # Currency
    'USDINR=X',   # USD/INR - Direct impact
    
    # Fear Index
    '^VIX'        # Global risk sentiment
]

def retry_on_failure(max_retries=3, delay=1):
    """Retry decorator for API calls"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        logger.error(f"Failed after {max_retries} attempts: {str(e)}")
                        raise e
                    logger.warning(f"Attempt {attempt + 1} failed: {str(e)}. Retrying...")
                    time.sleep(delay * (2 ** attempt))  # Exponential backoff
            return None
        return wrapper
    return decorator

@retry_on_failure(max_retries=3, delay=2)
def fetch_ticker_data(ticker: str, period: str = None, interval: str = "1d") -> Optional[Dict]:
    """
    Fetch data for a single ticker with error handling and retry logic
    """
    try:
        # Use dynamic period if not specified
        if period is None:
            period = get_optimal_data_period()
        
        logger.info(f"Fetching data for {ticker} (period: {period})...")
        
        # Get ticker info
        ticker_obj = yf.Ticker(ticker)
        info = ticker_obj.info
        
        # Get historical data
        hist = ticker_obj.history(period=period, interval=interval)
        
        if hist.empty or len(hist) < 2:
            logger.warning(f"Insufficient data for {ticker}")
            return None
        
        # Calculate key metrics
        current_price = hist['Close'].iloc[-1]
        prev_close = hist['Close'].iloc[-2]
        change = current_price - prev_close
        change_pct = (change / prev_close) * 100 if prev_close != 0 else 0
        
        # Calculate volatility (standard deviation of returns)
        returns = hist['Close'].pct_change().dropna()
        volatility = returns.std() * 100 if len(returns) > 0 else 0
        
        # Get volume
        volume = hist['Volume'].iloc[-1] if 'Volume' in hist.columns else 0
        
        # Get high/low for the session
        high = hist['High'].iloc[-1]
        low = hist['Low'].iloc[-1]
        open_price = hist['Open'].iloc[-1]
        
        return {
            'ticker': ticker,
            'name': info.get('longName', info.get('shortName', ticker)),
            'current_price': round(float(current_price), 4),
            'previous_close': round(float(prev_close), 4),
            'change': round(float(change), 4),
            'change_pct': round(float(change_pct), 2),
            'volume': int(volume) if volume and not pd.isna(volume) else 0,
            'volatility': round(float(volatility), 2),
            'high': round(float(high), 4),
            'low': round(float(low), 4),
            'open': round(float(open_price), 4),
            'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
    except Exception as e:
        logger.error(f"Error fetching data for {ticker}: {str(e)}")
        return None

def get_optimal_data_period() -> str:
    """
    Determine optimal data period based on current day to handle weekends
    """
    ist_timezone = pytz.timezone('Asia/Kolkata')
    current_time = datetime.now(ist_timezone)
    weekday = current_time.weekday()  # Monday=0, Sunday=6
    
    # If it's Monday, we need more days to get Friday's data
    if weekday == 0:  # Monday
        return "5d"  # Get Friday's data
    elif weekday == 6:  # Sunday
        return "4d"  # Get Friday's data
    elif weekday == 5:  # Saturday
        return "3d"  # Get Friday's data
    else:
        return "3d"  # Regular weekday, 3 days should be sufficient

def get_market_status() -> Dict[str, str]:
    """
    Get current market status for major exchanges
    """
    ist_timezone = pytz.timezone('Asia/Kolkata')
    current_time = datetime.now(ist_timezone)
    
    # Market hours in their local time zones
    markets = {
        'US_Markets': {
            'open_time': 9.5,   # 9:30 AM EST
            'close_time': 16.0, # 4:00 PM EST
            'timezone': 'US/Eastern'
        },
        'Japan_Nikkei': {
            'open_time': 9.0,   # 9:00 AM JST
            'close_time': 15.5, # 3:30 PM JST
            'timezone': 'Asia/Tokyo'
        },
        'Hong_Kong': {
            'open_time': 9.5,   # 9:30 AM HKT
            'close_time': 16.0, # 4:00 PM HKT
            'timezone': 'Asia/Hong_Kong'
        },
        'South_Korea': {
            'open_time': 9.0,   # 9:00 AM KST
            'close_time': 15.5, # 3:30 PM KST
            'timezone': 'Asia/Seoul'
        }
    }
    
    status = {}
    for market, info in markets.items():
        try:
            tz = pytz.timezone(info['timezone'])
            market_time = current_time.astimezone(tz)
            current_hour = market_time.hour + market_time.minute / 60.0
            
            # Check if it's a weekday
            is_weekday = market_time.weekday() < 5
            
            if is_weekday and info['open_time'] <= current_hour <= info['close_time']:
                status[market] = 'OPEN'
            else:
                status[market] = 'CLOSED'
        except Exception as e:
            logger.warning(f"Error getting status for {market}: {str(e)}")
            status[market] = 'UNKNOWN'
    
    return status

def calculate_dynamic_correlations(market_data: Dict) -> Dict:
    """
    Calculate dynamic correlations based on market regime
    """
    # Base correlations (conservative estimates)
    base_correlations = {
        '^GSPC': 0.55,    # Reduced from 0.70
        '^IXIC': 0.50,    # Reduced from 0.65
        '^DJI': 0.45,     # Reduced from 0.60
        '^N225': 0.35,    # Reduced from 0.50
        '^HSI': 0.30,     # Reduced from 0.45
        '^KS11': 0.25,    # Reduced from 0.40
        'CL=F': -0.15,    # Reduced from -0.25
        'GC=F': -0.08,    # Reduced from -0.15
        'USDINR=X': -0.20, # Reduced from -0.30
        '^VIX': -0.35,    # Reduced from -0.50
    }
    
    # Adjust correlations based on VIX (market stress)
    vix_level = 20  # Default
    if '^VIX' in market_data and 'current_price' in market_data['^VIX']:
        vix_level = market_data['^VIX']['current_price']
    
    # During high stress (VIX > 25), correlations increase
    if vix_level > 25:
        stress_multiplier = min(1.4, 1 + (vix_level - 25) / 50)  # Cap at 1.4x
        for ticker in ['^GSPC', '^IXIC', '^DJI']:
            if ticker in base_correlations:
                base_correlations[ticker] *= stress_multiplier
    
    # During low stress (VIX < 15), correlations decrease
    elif vix_level < 15:
        calm_multiplier = max(0.7, 1 - (15 - vix_level) / 30)  # Floor at 0.7x
        for ticker in ['^GSPC', '^IXIC', '^DJI']:
            if ticker in base_correlations:
                base_correlations[ticker] *= calm_multiplier
    
    return base_correlations

def apply_dampening_function(change_pct: float) -> float:
    """
    Apply non-linear dampening to extreme moves
    Large moves often don't translate proportionally
    """
    if abs(change_pct) <= 1.0:
        return change_pct  # No dampening for small moves
    elif abs(change_pct) <= 3.0:
        # Moderate dampening for medium moves
        sign = 1 if change_pct > 0 else -1
        dampened = sign * (1.0 + 0.7 * (abs(change_pct) - 1.0))
        return dampened
    else:
        # Strong dampening for extreme moves
        sign = 1 if change_pct > 0 else -1
        dampened = sign * (2.4 + 0.4 * (abs(change_pct) - 3.0))
        return dampened

def calculate_prediction_confidence(market_data: Dict, prediction: float) -> float:
    """
    Calculate realistic confidence based on multiple factors
    """
    confidence_factors = []
    
    # 1. Data quality factor (30% weight)
    available_tickers = len([t for t in market_data if 'change_pct' in market_data[t]])
    total_tickers = 11  # PRIMARY_TICKERS count
    data_quality = (available_tickers / total_tickers) * 30
    confidence_factors.append(data_quality)
    
    # 2. Market consensus factor (25% weight)
    # Check if major markets are moving in same direction
    us_moves = []
    for ticker in ['^GSPC', '^IXIC', '^DJI']:
        if ticker in market_data and 'change_pct' in market_data[ticker]:
            us_moves.append(market_data[ticker]['change_pct'])
    
    consensus_score = 0
    if len(us_moves) >= 2:
        # Check if moves are in same direction
        positive_moves = sum(1 for move in us_moves if move > 0)
        negative_moves = sum(1 for move in us_moves if move < 0)
        
        if positive_moves >= 2 or negative_moves >= 2:
            consensus_score = 25  # Strong consensus
        elif len(us_moves) == 3 and (positive_moves == 2 or negative_moves == 2):
            consensus_score = 20  # Moderate consensus
        else:
            consensus_score = 10  # Weak consensus
    
    confidence_factors.append(consensus_score)
    
    # 3. VIX stability factor (20% weight)
    vix_factor = 20  # Default
    if '^VIX' in market_data and 'current_price' in market_data['^VIX']:
        vix_level = market_data['^VIX']['current_price']
        if 15 <= vix_level <= 25:
            vix_factor = 20  # Normal volatility = higher confidence
        elif vix_level > 30:
            vix_factor = 5   # Extreme volatility = low confidence
        elif vix_level > 25:
            vix_factor = 12  # High volatility = reduced confidence
        else:
            vix_factor = 15  # Very low volatility = uncertain
    
    confidence_factors.append(vix_factor)
    
    # 4. Prediction magnitude factor (15% weight)
    # Extreme predictions should have lower confidence
    magnitude_factor = 15
    if abs(prediction) > 2.0:
        magnitude_factor = 5   # Very low confidence for extreme predictions
    elif abs(prediction) > 1.5:
        magnitude_factor = 8   # Low confidence
    elif abs(prediction) > 1.0:
        magnitude_factor = 12  # Moderate confidence
    else:
        magnitude_factor = 15  # Normal confidence for small predictions
    
    confidence_factors.append(magnitude_factor)
    
    # 5. Time-based factor (10% weight)
    # Lower confidence on weekends/holidays when data might be stale
    ist_now = datetime.now(pytz.timezone('Asia/Kolkata'))
    weekday = ist_now.weekday()
    hour = ist_now.hour
    
    time_factor = 10
    if weekday >= 5:  # Weekend
        time_factor = 6
    elif hour < 6 or hour > 21:  # Very early/late hours
        time_factor = 7
    
    confidence_factors.append(time_factor)
    
    # Calculate final confidence (max 100%)
    total_confidence = min(100, sum(confidence_factors))
    
    return round(total_confidence, 1)

def predict_nifty_gap_improved(market_data: Dict) -> Dict:
    """
    Improved NIFTY gap prediction with realistic confidence
    """
    # Get dynamic correlations based on market conditions
    correlations = calculate_dynamic_correlations(market_data)
    
    gap_impacts = {
        'us_markets_impact': 0,
        'asian_markets_impact': 0,
        'commodities_impact': 0,
        'currency_impact': 0,
        'sentiment_impact': 0
    }
    
    total_weighted_impact = 0
    contributing_tickers = []
    
    # Calculate dampened impacts for each ticker
    for ticker, ticker_data in market_data.items():
        if ticker in correlations and 'change_pct' in ticker_data:
            raw_change = ticker_data['change_pct']
            
            # Apply dampening to extreme moves
            dampened_change = apply_dampening_function(raw_change)
            
            # Calculate impact using dynamic correlation
            correlation = correlations[ticker]
            impact = correlation * dampened_change
            
            total_weighted_impact += impact
            contributing_tickers.append({
                'ticker': ticker,
                'raw_change': raw_change,
                'dampened_change': round(dampened_change, 2),
                'correlation': round(correlation, 2),
                'impact': round(impact, 2)
            })
            
            # Categorize impacts
            if ticker in ['^GSPC', '^IXIC', '^DJI']:
                gap_impacts['us_markets_impact'] += impact
            elif ticker in ['^N225', '^HSI', '^KS11']:
                gap_impacts['asian_markets_impact'] += impact
            elif ticker in ['CL=F', 'GC=F']:
                gap_impacts['commodities_impact'] += impact
            elif ticker == 'USDINR=X':
                gap_impacts['currency_impact'] += impact
            elif ticker == '^VIX':
                gap_impacts['sentiment_impact'] += impact
    
    # Calculate final prediction
    expected_gap = total_weighted_impact
    
    # Determine gap type and strength with more conservative thresholds
    if expected_gap > 0.75:
        gap_type = 'GAP_UP'
        gap_strength = 'STRONG' if expected_gap > 1.5 else 'MODERATE'
    elif expected_gap < -0.75:
        gap_type = 'GAP_DOWN'
        gap_strength = 'STRONG' if expected_gap < -1.5 else 'MODERATE'
    elif abs(expected_gap) > 0.3:
        gap_type = 'GAP_UP' if expected_gap > 0 else 'GAP_DOWN'
        gap_strength = 'WEAK'
    else:
        gap_type = 'FLAT_OPENING'
        gap_strength = 'NEUTRAL'
    
    # Calculate realistic confidence
    confidence = calculate_prediction_confidence(market_data, expected_gap)
    
    # Create uncertainty range based on confidence
    uncertainty_factor = (100 - confidence) / 100 * 0.5  # Max 0.5% uncertainty
    range_low = expected_gap - uncertainty_factor
    range_high = expected_gap + uncertainty_factor
    
    return {
        'expected_gap_percent': round(expected_gap, 2),
        'gap_type': gap_type,
        'gap_strength': gap_strength,
        'confidence_score': confidence,
        'gap_range_low': round(range_low, 2),
        'gap_range_high': round(range_high, 2),
        'contributing_factors': {k: round(v, 2) for k, v in gap_impacts.items()},
        'total_weighted_impact': round(total_weighted_impact, 2),
        'contributing_tickers': contributing_tickers,
        'methodology': {
            'dampening_applied': True,
            'dynamic_correlations': True,
            'confidence_factors': [
                'data_quality', 'market_consensus', 'vix_stability', 
                'prediction_magnitude', 'timing'
            ]
        }
    }

def analyze_market_sentiment(market_data: Dict) -> Dict:
    """
    Analyze overall market sentiment and risk levels
    """
    sentiment_factors = {
        'us_sentiment': 'NEUTRAL',
        'asian_sentiment': 'NEUTRAL',
        'risk_sentiment': 'NEUTRAL',
        'commodity_sentiment': 'NEUTRAL'
    }
    
    # Analyze US markets
    us_changes = []
    for ticker in ['^GSPC', '^IXIC', '^DJI']:
        if ticker in market_data and 'change_pct' in market_data[ticker]:
            us_changes.append(market_data[ticker]['change_pct'])
    
    if us_changes:
        avg_us_change = sum(us_changes) / len(us_changes)
        if avg_us_change > 0.5:
            sentiment_factors['us_sentiment'] = 'BULLISH'
        elif avg_us_change < -0.5:
            sentiment_factors['us_sentiment'] = 'BEARISH'
    
    # Analyze Asian markets
    asian_changes = []
    for ticker in ['^N225', '^HSI', '^KS11']:
        if ticker in market_data and 'change_pct' in market_data[ticker]:
            asian_changes.append(market_data[ticker]['change_pct'])
    
    if asian_changes:
        avg_asian_change = sum(asian_changes) / len(asian_changes)
        if avg_asian_change > 0.5:
            sentiment_factors['asian_sentiment'] = 'BULLISH'
        elif avg_asian_change < -0.5:
            sentiment_factors['asian_sentiment'] = 'BEARISH'
    
    # Analyze VIX for risk sentiment
    if '^VIX' in market_data and 'current_price' in market_data['^VIX']:
        vix_level = market_data['^VIX']['current_price']
        if vix_level > 25:
            sentiment_factors['risk_sentiment'] = 'HIGH_FEAR'
        elif vix_level < 15:
            sentiment_factors['risk_sentiment'] = 'COMPLACENCY'
        elif vix_level > 20:
            sentiment_factors['risk_sentiment'] = 'ELEVATED_FEAR'
    
    # Analyze commodities
    oil_impact = 0
    gold_impact = 0
    
    if 'CL=F' in market_data and 'change_pct' in market_data['CL=F']:
        oil_change = market_data['CL=F']['change_pct']
        oil_impact = oil_change
        
    if 'GC=F' in market_data and 'change_pct' in market_data['GC=F']:
        gold_change = market_data['GC=F']['change_pct']
        gold_impact = gold_change
    
    # Oil up and gold up = risk off
    if oil_impact > 2 and gold_impact > 1:
        sentiment_factors['commodity_sentiment'] = 'RISK_OFF'
    elif oil_impact < -2 and gold_impact < -1:
        sentiment_factors['commodity_sentiment'] = 'RISK_ON'
    
    # Determine overall sentiment
    bullish_count = sum(1 for s in sentiment_factors.values() if s == 'BULLISH')
    bearish_count = sum(1 for s in sentiment_factors.values() if s == 'BEARISH')
    
    if bullish_count > bearish_count:
        overall_sentiment = 'BULLISH'
    elif bearish_count > bullish_count:
        overall_sentiment = 'BEARISH'
    else:
        overall_sentiment = 'NEUTRAL'
    
    return {
        'overall_sentiment': overall_sentiment,
        'sentiment_breakdown': sentiment_factors,
        'sentiment_score': bullish_count - bearish_count,
        'market_fear_level': sentiment_factors['risk_sentiment']
    }

def generate_trading_signals(market_data: Dict, gap_prediction: Dict, sentiment: Dict) -> List[Dict]:
    """
    Generate actionable trading signals based on analysis
    """
    signals = []
    
    # Gap-based signals with improved confidence thresholds
    expected_gap = gap_prediction['expected_gap_percent']
    gap_strength = gap_prediction['gap_strength']
    confidence = gap_prediction['confidence_score']
    
    # Only generate signals for high confidence predictions
    if confidence > 60 and gap_strength in ['STRONG', 'MODERATE']:
        if expected_gap > 0.75:
            signals.append({
                'signal_type': 'GAP_UP_TRADE',
                'action': 'BUY_CALLS',
                'reason': f'Bullish gap expected: {expected_gap}% (Confidence: {confidence}%)',
                'confidence': confidence,
                'strategy': 'Consider ATM calls or bull call spread'
            })
        elif expected_gap < -0.75:
            signals.append({
                'signal_type': 'GAP_DOWN_TRADE',
                'action': 'BUY_PUTS',
                'reason': f'Bearish gap expected: {expected_gap}% (Confidence: {confidence}%)',
                'confidence': confidence,
                'strategy': 'Consider ATM puts or bear put spread'
            })
    
    # VIX-based volatility signals
    if '^VIX' in market_data and 'current_price' in market_data['^VIX']:
        vix_level = market_data['^VIX']['current_price']
        
        if vix_level > 25:
            signals.append({
                'signal_type': 'HIGH_VOLATILITY',
                'action': 'VOLATILITY_TRADE',
                'reason': f'High VIX at {vix_level:.1f} - elevated volatility expected',
                'confidence': 85,
                'strategy': 'Consider long straddle or strangle'
            })
        elif vix_level < 15:
            signals.append({
                'signal_type': 'LOW_VOLATILITY',
                'action': 'THETA_DECAY',
                'reason': f'Low VIX at {vix_level:.1f} - low volatility environment',
                'confidence': 75,
                'strategy': 'Consider iron condor or calendar spreads'
            })
    
    # Oil impact signals
    if 'CL=F' in market_data and 'change_pct' in market_data['CL=F']:
        oil_change = market_data['CL=F']['change_pct']
        
        if abs(oil_change) > 3:
            impact = 'NEGATIVE' if oil_change > 0 else 'POSITIVE'
            signals.append({
                'signal_type': 'OIL_IMPACT',
                'action': 'SECTOR_FOCUS',
                'reason': f'Major oil move: {oil_change:.1f}% - {impact} for Indian markets',
                'confidence': 70,
                'strategy': f'Focus on energy and auto stocks'
            })
    
    # Currency impact signals
    if 'USDINR=X' in market_data and 'change_pct' in market_data['USDINR=X']:
        usd_change = market_data['USDINR=X']['change_pct']
        
        if abs(usd_change) > 0.5:
            impact = 'NEGATIVE' if usd_change > 0 else 'POSITIVE'
            signals.append({
                'signal_type': 'CURRENCY_IMPACT',
                'action': 'FII_FLOW_TRADE',
                'reason': f'USD/INR move: {usd_change:.1f}% - {impact} for FII flows',
                'confidence': 65,
                'strategy': 'Monitor IT and pharma stocks'
            })
    
    return signals

def generate_key_alerts(market_data: Dict) -> List[str]:
    """
    Generate key alerts for major market moves
    """
    alerts = []
    
    # Major index moves
    for ticker in ['^GSPC', '^IXIC', '^DJI', '^N225']:
        if ticker in market_data and 'change_pct' in market_data[ticker]:
            change = market_data[ticker]['change_pct']
            if abs(change) > 2:
                direction = "📈" if change > 0 else "📉"
                alerts.append(f"{direction} MAJOR MOVE: {ticker} {change:+.1f}%")
    
    # VIX alerts
    if '^VIX' in market_data and 'current_price' in market_data['^VIX']:
        vix = market_data['^VIX']['current_price']
        if vix > 30:
            alerts.append(f"🚨 EXTREME FEAR: VIX at {vix:.1f}")
        elif vix > 25:
            alerts.append(f"⚠️ HIGH VOLATILITY: VIX at {vix:.1f}")
    
    # Commodity alerts
    if 'CL=F' in market_data and 'change_pct' in market_data['CL=F']:
        oil_change = market_data['CL=F']['change_pct']
        if abs(oil_change) > 4:
            alerts.append(f"🛢️ MAJOR OIL MOVE: {oil_change:+.1f}%")
    
    # Currency alerts
    if 'USDINR=X' in market_data and 'change_pct' in market_data['USDINR=X']:
        usd_change = market_data['USDINR=X']['change_pct']
        if abs(usd_change) > 0.7:
            alerts.append(f"💱 CURRENCY ALERT: USD/INR {usd_change:+.1f}%")
    
    return alerts

def fetch_pre_market_data() -> Dict:
    """
    Main function to fetch all pre-market data and generate comprehensive analysis
    """
    logger.info("Starting pre-market data fetch...")
    
    # Get market status and optimal data period
    market_status = get_market_status()
    data_period = get_optimal_data_period()
    logger.info(f"Using data period: {data_period} (current weekday: {datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%A')})")
    
    # Fetch data for all tickers
    market_data = {}
    successful_fetches = 0
    
    for ticker in PRIMARY_TICKERS:
        ticker_data = fetch_ticker_data(ticker, period=data_period)
        if ticker_data:
            market_data[ticker] = ticker_data
            successful_fetches += 1
        else:
            logger.warning(f"Failed to fetch data for {ticker}")
    
    # Generate analysis using improved prediction
    gap_prediction = predict_nifty_gap_improved(market_data)
    sentiment_analysis = analyze_market_sentiment(market_data)
    trading_signals = generate_trading_signals(market_data, gap_prediction, sentiment_analysis)
    key_alerts = generate_key_alerts(market_data)
    
    # Compile final output
    output = {
        'metadata': {
            'fetch_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'fetch_timestamp': int(datetime.now().timestamp()),
            'timezone': 'Asia/Kolkata',
            'total_tickers': len(PRIMARY_TICKERS),
            'successful_fetches': successful_fetches,
            'data_quality_score': round((successful_fetches / len(PRIMARY_TICKERS)) * 100, 1),
            'market_status': market_status,
            'prediction_version': 'improved_v2.0'
        },
        
        'nifty_prediction': gap_prediction,
        
        'market_sentiment': sentiment_analysis,
        
        'market_data': market_data,
        
        'trading_signals': trading_signals,
        
        'key_alerts': key_alerts,
        
        'summary': {
            'expected_nifty_move': f"{gap_prediction['expected_gap_percent']:+.1f}%",
            'gap_type': gap_prediction['gap_type'],
            'overall_sentiment': sentiment_analysis['overall_sentiment'],
            'confidence': gap_prediction['confidence_score'],
            'risk_level': sentiment_analysis['market_fear_level'],
            'recommendation': get_trading_recommendation(gap_prediction),
            'prediction_reliability': get_confidence_interpretation(gap_prediction['confidence_score'])
        }
    }
    
    logger.info(f"Pre-market analysis completed: {successful_fetches}/{len(PRIMARY_TICKERS)} tickers fetched")
    logger.info(f"Gap prediction: {gap_prediction['expected_gap_percent']:+.2f}% (Confidence: {gap_prediction['confidence_score']}%)")
    
    return output

def get_trading_recommendation(gap_prediction: Dict) -> str:
    """
    Generate trading recommendation based on gap prediction and confidence
    """
    gap = gap_prediction['expected_gap_percent']
    confidence = gap_prediction['confidence_score']
    
    # Only make strong recommendations for high confidence predictions
    if confidence < 50:
        return 'WAIT'  # Low confidence = wait for more data
    elif confidence >= 70:
        # High confidence recommendations
        if gap > 0.75:
            return 'BUY'
        elif gap < -0.75:
            return 'SELL'
        else:
            return 'NEUTRAL'
    else:
        # Medium confidence recommendations (more conservative)
        if gap > 1.0:
            return 'BUY'
        elif gap < -1.0:
            return 'SELL'
        else:
            return 'NEUTRAL'

def get_confidence_interpretation(confidence: float) -> str:
    """
    Interpret confidence score for users
    """
    if confidence >= 80:
        return 'HIGH'
    elif confidence >= 60:
        return 'MODERATE'
    elif confidence >= 40:
        return 'LOW'
    else:
        return 'VERY_LOW'

def get_json_output(data: Dict) -> str:
    """
    Convert the output data to JSON string
    """
    try:
        json_string = json.dumps(data, indent=2, ensure_ascii=False)
        logger.info("Data converted to JSON string successfully")
        return json_string
        
    except Exception as e:
        logger.error(f"Error converting to JSON: {str(e)}")
        return None

def print_summary(data: Dict):
    """
    Print a quick summary of the analysis with improved formatting
    """
    summary = data['summary']
    metadata = data['metadata']
    alerts = data['key_alerts']
    gap_data = data['nifty_prediction']
    
    print(f"""
🌍 GLOBAL MARKET OVERNIGHT SUMMARY
{'='*60}
⏰ Time: {metadata['fetch_time']} IST
📊 Data Quality: {metadata['data_quality_score']}%
🔧 Prediction Model: {metadata['prediction_version']}

🎯 NIFTY OPENING PREDICTION:
   Expected Move: {summary['expected_nifty_move']} ({summary['gap_type']})
   Confidence: {summary['confidence']}% ({summary['prediction_reliability']})
   Recommendation: {summary['recommendation']}
   Gap Range: {gap_data['gap_range_low']:+.2f}% to {gap_data['gap_range_high']:+.2f}%

💭 Market Sentiment: {summary['overall_sentiment']}
😰 Risk Level: {summary['risk_level']}

🚨 Key Alerts: {len(alerts)}
""")
    
    if alerts:
        for alert in alerts[:5]:  # Show top 5 alerts
            print(f"   {alert}")
    
    # Show top contributing factors
    contributing_tickers = gap_data.get('contributing_tickers', [])
    if contributing_tickers:
        print(f"\n📈 Top Contributing Factors:")
        sorted_contributors = sorted(contributing_tickers, key=lambda x: abs(x['impact']), reverse=True)
        for contrib in sorted_contributors[:3]:  # Top 3
            print(f"   • {contrib['ticker']}: {contrib['raw_change']:+.1f}% → {contrib['impact']:+.2f}% impact")
    
    print("="*60)

def generate_prediction_explanation(prediction_data: Dict) -> str:
    """
    Generate human-readable explanation of the prediction
    """
    gap = prediction_data['expected_gap_percent']
    confidence = prediction_data['confidence_score']
    
    explanation = f"""
📊 NIFTY Gap Prediction Analysis:

🎯 Expected Gap: {gap:+.2f}% ({prediction_data['gap_type']})
📈 Strength: {prediction_data['gap_strength']}
🎭 Confidence: {confidence}% ({get_confidence_interpretation(confidence)})

🔍 Key Contributing Factors:
"""
    
    # Sort contributing tickers by impact magnitude
    contributors = sorted(
        prediction_data['contributing_tickers'], 
        key=lambda x: abs(x['impact']), 
        reverse=True
    )
    
    for contrib in contributors[:5]:  # Top 5 contributors
        explanation += f"   • {contrib['ticker']}: {contrib['raw_change']:+.1f}% → {contrib['impact']:+.2f}% impact\n"
    
    explanation += f"""
💡 Methodology Applied:
   • Dampening function for extreme moves
   • Dynamic correlations based on VIX level
   • Multi-factor confidence scoring
   • Conservative thresholds

⚠️  Confidence Level Guide:
   • 80-100%: High confidence (trust the prediction)
   • 60-79%: Moderate confidence (proceed with caution)  
   • 40-59%: Low confidence (consider waiting)
   • <40%: Very uncertain (avoid trading on this signal)

📊 Current Analysis:
   Range: {prediction_data['gap_range_low']:+.2f}% to {prediction_data['gap_range_high']:+.2f}%
   Recommendation: {get_trading_recommendation(prediction_data)}
"""
    
    return explanation

def main():
    """
    Main execution function - returns JSON string
    """
    try:
        # Fetch and analyze data
        data = fetch_pre_market_data()
        
        # Convert to JSON string
        json_output = get_json_output(data)
        
        # Print summary for visual feedback
        print_summary(data)
        
        # Optionally print detailed explanation (uncomment if needed)
        # explanation = generate_prediction_explanation(data['nifty_prediction'])
        # print(explanation)
        
        # Return JSON string
        return json_output
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        return None

def test_with_sample_data():
    """
    Test function with sample data to validate the improved algorithm
    """
    sample_data = {
        '^GSPC': {'change_pct': -2.1, 'current_price': 4200},
        '^IXIC': {'change_pct': -2.8, 'current_price': 13000},
        '^DJI': {'change_pct': -1.9, 'current_price': 33000},
        '^N225': {'change_pct': -1.2, 'current_price': 28000},
        '^VIX': {'current_price': 22.5, 'change_pct': 15.2},
        'CL=F': {'change_pct': 3.1, 'current_price': 75.2},
        'USDINR=X': {'change_pct': 0.3, 'current_price': 83.15}
    }
    
    print("🧪 TESTING IMPROVED ALGORITHM:")
    print("="*50)
    
    # Test the improved prediction
    result = predict_nifty_gap_improved(sample_data)
    explanation = generate_prediction_explanation(result)
    
    print(f"Previous Algorithm: -2.6% (100% confidence) ❌")
    print(f"Improved Algorithm: {result['expected_gap_percent']:+.2f}% ({result['confidence_score']}% confidence) ✅")
    print(f"Gap Type: {result['gap_type']}")
    print(f"Recommendation: {get_trading_recommendation(result)}")
    print("\n" + explanation)
    
    return result

if __name__ == "__main__":
    # Run the main function
    result = main()
    print(result)
    
    # Uncomment to test with sample data:
    # test_with_sample_data()