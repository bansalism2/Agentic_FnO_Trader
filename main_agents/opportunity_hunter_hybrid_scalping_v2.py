#!/usr/bin/env python3
"""
AlgoTrade Hybrid Scalping Opportunity Hunter - Clear Mode Separation (v2)

This script implements a hybrid trading system with CLEAR SEPARATION between:
1. SCALPING MODE: Quick directional trades (Long Calls/Puts) with tight stops
2. PREMIUM SELLING MODE: Time-decay strategies (Iron Condor/Strangle) with IV-based entries

This version uses a configurable trade quantity for all strategies.
"""

import sys
import os
import json
import logging
from datetime import datetime, time as dt_time
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv
import numpy as np
import pandas as pd

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load credentials and initialize connection
load_dotenv(dotenv_path='../.env')
api_key = os.getenv("kite_api_key")
api_secret = os.getenv("kite_secret")
access_token = None
try:
    with open("../data/access_token.txt", "r") as f:
        access_token = f.read().strip()
except Exception as e:
    print("Could not read ../data/access_token.txt:", e)

from core_tools.calculate_analysis_tools import (
    calculate_iv_rank_analysis_wrapper, get_realized_volatility_from_kite
)
from core_tools.connect_data_tools import (
    get_nifty_spot_price, get_available_expiry_dates_with_analysis, initialize_connection, get_options_chain
)
from core_tools.execution_portfolio_tools import get_daily_trading_summary, get_portfolio_positions, execute_and_store_strategy, get_account_margins
from core_tools.strategy_creation_tools import (
    create_long_straddle_wrapper,
    create_short_strangle_wrapper, 
    create_iron_condor_wrapper,
    create_bull_put_spread_wrapper,
    create_bear_call_spread_wrapper
)
from core_tools.master_indicators import get_nifty_technical_analysis_tool, get_trading_signals
from core_tools.calculate_analysis_tools import detect_market_regime_wrapper
from core_tools.trade_storage import get_active_trades
# Removed crew_agent dependency - hybrid system is independent

# --- CONFIGURATION ---
DEFAULT_TRADE_QUANTITY = 75

# Initialize connection
try:
    init_result = initialize_connection(api_key, api_secret, access_token)
    print("Kite Connect initialization result:", init_result)
except Exception as e:
    print(f"Warning: Kite Connect initialization failed: {e}")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
MARKET_START_TIME = dt_time(9, 15)
MARKET_END_TIME = dt_time(15, 30)
IST_TIMEZONE = "Asia/Kolkata"

class HybridTradingMode:
    """Clear separation between scalping and premium selling modes"""
    
    @staticmethod
    def determine_trading_mode(market_conditions: Dict[str, Any]) -> Dict[str, Any]:
        """
        CLEAR SEPARATION: Scalping vs Premium Selling with realistic thresholds
        """
        rsi = market_conditions.get('rsi', 50)
        volume_ratio = market_conditions.get('volume_ratio', 1.0)
        iv_percentile = market_conditions.get('iv_percentile', 0)
        macd_signal = market_conditions.get('macd_signal', 'NEUTRAL')
        
        # Calculate momentum score (0-2 scale) - Fixed approach
        # RSI deviation: clamp to 0-1 range, differentiate bullish/bearish
        rsi_deviation = min(abs(rsi - 50) / 50, 1.0)  # Clamp to 0-1
        rsi_direction = 1 if rsi > 50 else -1  # Bullish vs bearish
        
        # MACD signal strength: differentiate BUY/SELL vs NEUTRAL
        macd_strength = 1 if macd_signal in ['BUY', 'SELL'] else 0
        
        momentum_score = rsi_deviation + macd_strength  # 0-2 scale
        
        # SCALPING MODE: Moderate to strong momentum + ADX confirmation
        adx = market_conditions.get('adx', 0)
        if momentum_score >= 1.0 and adx > 25:
            confidence = 'HIGH' if momentum_score >= 1.5 and adx > 30 else 'MEDIUM'
            return {
                'mode': 'SCALPING',
                'confidence': confidence,
                'score': momentum_score,
                'strategies': ['Long Call', 'Long Put'],
                'timeframe': 'minutes',
                'stop_loss': '3-5%',
                'profit_target': '8-15%',
                'reason': f'Momentum score: {momentum_score:.1f}, ADX: {adx:.1f}, RSI: {rsi:.1f}'
            }
        
        # PREMIUM SELLING MODE: High IV + low momentum
        elif iv_percentile >= 60 and momentum_score <= 1.0:
            confidence = 'HIGH' if iv_percentile >= 75 else 'MEDIUM'
            return {
                'mode': 'PREMIUM_SELLING',
                'confidence': confidence,
                'iv_percentile': iv_percentile,
                'strategies': ['Iron Condor', 'Short Strangle'],
                'timeframe': 'hours',
                'stop_loss': 'time-based',
                'profit_target': '20-40% of premium',
                'reason': f'High IV: {iv_percentile:.1f}%, Low momentum: {momentum_score:.1f}'
            }
        
        # WAIT: Unclear conditions
        else:
            return {
                'mode': 'WAIT', 
                'reason': f'No clear setup - Momentum: {momentum_score:.1f}, IV: {iv_percentile:.1f}%, ADX: {adx:.1f}'
            }
    
    @staticmethod
    def get_intraday_trading_mode(current_time: dt_time) -> str:
        """
        More practical time windows for intraday trading
        """
        if dt_time(9, 15) <= current_time < dt_time(9, 20):
            return 'PREMIUM_SELLING_ONLY'  # Early - set up time decay (optional)
        elif dt_time(9, 20) <= current_time < dt_time(14, 45):
            return 'BOTH'  # Main window - both modes allowed
        elif dt_time(14, 45) <= current_time < dt_time(15, 15):
            return 'SCALPING_ONLY'  # Only scalping allowed near close
        else:
            return 'NONE'  # Outside trading hours

class ScalpingEngine:
    """Pure scalping execution - single leg, quick exits with ACTUAL strategy execution"""
    
    def __init__(self):
        self.execute_strategy = execute_and_store_strategy
        self.get_margins = get_account_margins
        
        # Import actual strategy creation tools
        from core_tools.strategy_creation_tools import (
            create_long_call_strategy,
            create_long_put_strategy
        )
        self.create_long_call = create_long_call_strategy
        self.create_long_put = create_long_put_strategy
    
    def fetch_atm_option_price(self, option_type: str, expiry_date: str = None) -> float:
        """Fetch the LTP of the ATM NIFTY option (CE or PE) for the given expiry."""
        try:
            # Step 1: Get recommended expiry if not provided
            if not expiry_date:
                expiry_info = get_available_expiry_dates_with_analysis()
                if expiry_info.get('status') != 'SUCCESS' or not expiry_info.get('recommended_expiries'):
                    print("[ERROR] Could not fetch recommended expiry for option price lookup.")
                    return 0
                expiry_date = expiry_info['recommended_expiries'][0]['date']
            # Step 2: Get options chain for expiry
            chain_result = get_options_chain(expiry_date=expiry_date, strike_range=3)
            if chain_result.get('status') != 'SUCCESS':
                print(f"[ERROR] Could not fetch options chain for expiry {expiry_date}.")
                return 0
            atm_strike = chain_result['atm_strike']
            # Step 3: Find ATM row
            for row in chain_result['options_chain']:
                if row['strike'] == atm_strike:
                    ltp = row.get(f'{option_type}_ltp', 0)
                    return ltp
            print(f"[ERROR] ATM strike {atm_strike} not found in options chain.")
            return 0
        except Exception as e:
            print(f"[ERROR] Exception in fetch_atm_option_price: {e}")
            return 0

    def _get_optimal_expiry_for_mode(self, mode: str) -> str:
        """Get optimal expiry based on trading mode"""
        try:
            expiry_analysis = get_available_expiry_dates_with_analysis()
            if not expiry_analysis or 'available_expiries' not in expiry_analysis:
                return None
            
            expiries = expiry_analysis['available_expiries']
            
            if mode == 'SCALPING':
                # For scalping, prefer shorter expiries (more gamma, quicker response)
                for expiry in expiries:
                    if expiry.get('category') == 'SHORT_TERM':  # 4-7 days
                        return expiry.get('date')
                # Fallback to medium term
                for expiry in expiries:
                    if expiry.get('category') == 'MEDIUM_TERM':
                        return expiry.get('date')
            
            elif mode == 'PREMIUM_SELLING':
                # For premium selling, prefer longer expiries (more time decay)
                for expiry in expiries:
                    if expiry.get('category') == 'LONG_TERM':  # 15+ days
                        return expiry.get('date')
                # Fallback to medium term
                for expiry in expiries:
                    if expiry.get('category') == 'MEDIUM_TERM':
                        return expiry.get('date')
            
            # Last resort - first available expiry
            return expiries[0].get('date') if expiries else None
            
        except Exception as e:
            logger.error(f"Error getting expiry: {e}")
            return None
    
    def calculate_dynamic_position_size(self, account_balance: float, option_price: float, stop_loss_percent: float) -> int:
        """Calculate position size based on actual option price and stop loss"""
        # Risk amount per lot = option_price * stop_loss_percent * 75 (lot size)
        risk_per_lot = option_price * (stop_loss_percent / 100) * 75
        
        # Maximum risk (1-2% of account for scalping)
        max_risk = account_balance * 0.02
        
        # Calculate lots
        max_lots = int(max_risk / risk_per_lot) if risk_per_lot > 0 else 0
        
        # Cap for scalping
        return min(max_lots, 25)
    
    def execute(self, conditions: Dict[str, Any]) -> Dict[str, Any]:
        """
        SCALPING RULES with ACTUAL EXECUTION:
        - Single leg positions only
        - Moderate volume requirement (>1.3x average)
        - Quick exits (5-30 minutes max)
        - Directional bias required
        - Tight stops (3-5%)
        """
        rsi = conditions.get('rsi', 50)
        macd_signal = conditions.get('macd_signal', 'NEUTRAL')
        volume_ratio = conditions.get('volume_ratio', 1.0)
        adx = conditions.get('adx', 0)
        regime_not_ideal = conditions.get('regime_not_ideal', False)
        # Check account margins first
        try:
            margins = self.get_margins()
            if margins.get('status') != 'SUCCESS':
                return {'status': 'ERROR', 'reason': 'Cannot verify account margins'}
            available_capital = margins['equity'].get('live_balance', 0)
            if available_capital < 50000:
                return {'status': 'ERROR', 'reason': f'Insufficient capital: ₹{available_capital}'}
        except Exception as e:
            return {'status': 'ERROR', 'reason': f'Margin check failed: {str(e)}'}
        # Get optimal expiry
        expiry_date = self._get_optimal_expiry_for_mode('SCALPING')
        if not expiry_date:
            return {'status': 'ERROR', 'reason': 'No suitable expiry available'}
        # --- Risk-aware scalping logic ---
        # LOGIC FLOW: Priority-based execution (each section returns if trade is executed)
        
        # 1. PRIORITY 1: Extreme RSI Override (highest priority)
        # When RSI is extreme, trade regardless of ADX with reduced size for momentum reversal
        if rsi < 25 or rsi > 75:
            print(f"[PRIORITY 1 - Extreme RSI Override] RSI={rsi:.1f} is extreme. Using momentum reversal strategy with reduced size.")
            
            # Determine option type based on RSI extremes
            if rsi < 25:  # Oversold - expect bounce
                option_type = 'CE'  # Buy CALL for upside
                trade_direction = 'LONG_CALL'
            else:  # Overbought (rsi > 75) - expect reversal  
                option_type = 'PE'  # Buy PUT for downside
                trade_direction = 'LONG_PUT'
            
            option_price = self.fetch_atm_option_price(option_type, expiry_date)
            if not option_price or option_price <= 0:
                print(f"[ERROR] Could not fetch real ATM option price. Skipping trade.")
                return {'status': 'ERROR', 'reason': 'Failed to fetch ATM option price'}
            stop_loss_percent = 6  # Increased from 2% to 6% for options volatility
            # Calculate normal quantity, then halve it (minimum 75)
            normal_qty = self.calculate_dynamic_position_size(available_capital, option_price, stop_loss_percent)
            LOT_SIZE = 75
            quantity = max(LOT_SIZE, (normal_qty // LOT_SIZE) * LOT_SIZE // 2)
            override_conditions = dict(conditions)
            override_conditions['stop_loss_percent'] = stop_loss_percent
            override_conditions['quantity'] = quantity
            
            if trade_direction == 'LONG_CALL':
                print(f"[PRIORITY 1] Executing LONG CALL scalp (reduced size: {quantity}, tight stop: {stop_loss_percent}%)")
                return self._execute_long_call_scalp(expiry_date, available_capital, override_conditions)
            else:
                print(f"[PRIORITY 1] Executing LONG PUT scalp (reduced size: {quantity}, tight stop: {stop_loss_percent}%)")
                return self._execute_long_put_scalp(expiry_date, available_capital, override_conditions)
        
        # 2. PRIORITY 2: Reverse ADX Logic (only if RSI is NOT extreme)
        # NOTE: This section is intentionally unreachable when RSI < 25 or RSI > 75
        # because Priority 1 (Extreme RSI Override) returns before reaching here.
        # ADX < 20 (very weak trend) = Use momentum strategies (RSI + MACD)
        # ADX >= 30 (strong trend) = Use trend following (MACD only, ignore RSI extremes)
        # ADX 20-30 = Mixed conditions, use both approaches
        if adx < 20:
            # Very weak trend: Use momentum strategies (RSI + MACD alignment)
            if rsi > 55 and macd_signal == 'BUY':
                print(f"[PRIORITY 2 - Weak Trend] ADX={adx:.1f} < 20, RSI={rsi:.1f} > 55, MACD=BUY: Executing LONG CALL")
                return self._execute_long_call_scalp(expiry_date, available_capital, conditions)
            elif rsi < 45 and macd_signal == 'SELL':
                print(f"[PRIORITY 2 - Weak Trend] ADX={adx:.1f} < 20, RSI={rsi:.1f} < 45, MACD=SELL: Executing LONG PUT")
                return self._execute_long_put_scalp(expiry_date, available_capital, conditions)
        elif adx >= 30:
            # Strong trend: Use trend following (MACD only, ignore RSI extremes)
            if macd_signal == 'BUY':
                print(f"[PRIORITY 2 - Strong Trend] ADX={adx:.1f} >= 30, MACD=BUY: Executing LONG CALL (trend following)")
                return self._execute_long_call_scalp(expiry_date, available_capital, conditions)
            elif macd_signal == 'SELL':
                print(f"[PRIORITY 2 - Strong Trend] ADX={adx:.1f} >= 30, MACD=SELL: Executing LONG PUT (trend following)")
                return self._execute_long_put_scalp(expiry_date, available_capital, conditions)
        else:
            # Moderate trend (ADX 20-30): Use balanced approach (RSI + MACD, but less strict)
            if rsi > 50 and macd_signal == 'BUY':
                print(f"[PRIORITY 2 - Moderate Trend] ADX={adx:.1f} 20-30, RSI={rsi:.1f} > 50, MACD=BUY: Executing LONG CALL")
                return self._execute_long_call_scalp(expiry_date, available_capital, conditions)
            elif rsi < 50 and macd_signal == 'SELL':
                print(f"[PRIORITY 2 - Moderate Trend] ADX={adx:.1f} 20-30, RSI={rsi:.1f} < 50, MACD=SELL: Executing LONG PUT")
                return self._execute_long_put_scalp(expiry_date, available_capital, conditions)
        # 3. Regime not ideal: reduce size/tighten stop for normal trades
        if regime_not_ideal:
            print(f"[Regime Not Ideal] Reducing size and tightening stop for risk control.")
            option_price = self.fetch_atm_option_price('CE' if rsi > 50 else 'PE', expiry_date)
            if not option_price or option_price <= 0:
                print(f"[ERROR] Could not fetch real ATM option price. Skipping trade.")
                return {'status': 'ERROR', 'reason': 'Failed to fetch ATM option price'}
            stop_loss_percent = 6 # Increased from 2% to 6% for options volatility
            normal_qty = self.calculate_dynamic_position_size(available_capital, option_price, stop_loss_percent)
            LOT_SIZE = 75
            quantity = max(LOT_SIZE, (normal_qty // LOT_SIZE) * LOT_SIZE // 2)
            override_conditions = dict(conditions)
            override_conditions['stop_loss_percent'] = stop_loss_percent
            override_conditions['quantity'] = quantity
            if rsi > 55 and macd_signal == 'BUY':
                print(f"[Regime Not Ideal] Executing LONG scalp (reduced size: {quantity}, tight stop: {stop_loss_percent}%)")
                return self._execute_long_call_scalp(expiry_date, available_capital, override_conditions)
            elif rsi < 45 and macd_signal == 'SELL':
                print(f"[Regime Not Ideal] Executing SHORT scalp (reduced size: {quantity}, tight stop: {stop_loss_percent}%)")
                return self._execute_long_put_scalp(expiry_date, available_capital, override_conditions)
        # 4. Otherwise, no trade
        return {
            'status': 'ERROR',
            'reason': f'No clear scalping setup - RSI: {rsi:.1f}, MACD: {macd_signal}, ADX: {adx:.1f}'
        }
    
    def _execute_long_call_scalp(self, expiry_date: str, capital: float, conditions: Dict[str, Any]) -> Dict[str, Any]:
        """Execute long call scalping trade"""
        try:
            LOT_SIZE = 75
            # Fetch real ATM option price
            option_price = self.fetch_atm_option_price('CE', expiry_date)
            if not option_price or option_price <= 0:
                print(f"[ERROR] Could not fetch real ATM call option price. Skipping trade.")
                return {'status': 'ERROR', 'reason': 'Failed to fetch ATM call option price'}
            stop_loss_percent = conditions.get('stop_loss_percent', 6)  # Increased from 3% to 6% for options volatility
            quantity = conditions.get('quantity', None)
            if quantity is None:
                quantity = self.calculate_dynamic_position_size(capital, option_price, stop_loss_percent)
            # Ensure quantity is a valid multiple of lot size (minimum 75)
            quantity = max(LOT_SIZE, (quantity // LOT_SIZE) * LOT_SIZE)
            # 1. Create strategy
            strategy_result = self.create_long_call(
                expiry_date=expiry_date,
                strike_selection='OTM',
                quantity=quantity
            )
            if strategy_result.get('status') != 'SUCCESS':
                return {'status': 'ERROR', 'reason': f'Strategy creation failed: {strategy_result.get("message")}' }
            # 2. Execute strategy
            execution_result = self.execute_strategy(
                strategy_legs=strategy_result.get('legs', []),
                trade_metadata={
                    'strategy_name': 'LONG_CALL_SCALP',
                    'mode': 'SCALPING',
                    'entry_conditions': conditions,
                    'risk_management': {
                        'stop_loss': f'{stop_loss_percent}%',
                        'profit_target': '8%',  # Lower profit target for more frequent wins
                        'max_hold_time': '15 minutes',
                        'min_hold_time': '3 minutes'
                    },
                    'expiry_date': expiry_date
                }
            )
            return execution_result
        except Exception as e:
            return {'status': 'ERROR', 'reason': f'Execution failed: {str(e)}'}

    def _execute_long_put_scalp(self, expiry_date: str, capital: float, conditions: Dict[str, Any]) -> Dict[str, Any]:
        """Execute long put scalping trade"""
        try:
            LOT_SIZE = 75
            # Fetch real ATM option price
            option_price = self.fetch_atm_option_price('PE', expiry_date)
            if not option_price or option_price <= 0:
                print(f"[ERROR] Could not fetch real ATM put option price. Skipping trade.")
                return {'status': 'ERROR', 'reason': 'Failed to fetch ATM put option price'}
            stop_loss_percent = conditions.get('stop_loss_percent', 6)  # Increased from 3% to 6% for options volatility
            quantity = conditions.get('quantity', None)
            if quantity is None:
                quantity = self.calculate_dynamic_position_size(capital, option_price, stop_loss_percent)
            # Ensure quantity is a valid multiple of lot size (minimum 75)
            quantity = max(LOT_SIZE, (quantity // LOT_SIZE) * LOT_SIZE)
            # 1. Create strategy
            strategy_result = self.create_long_put(
                expiry_date=expiry_date,
                strike_selection='OTM',
                quantity=quantity
            )
            if strategy_result.get('status') != 'SUCCESS':
                return {'status': 'ERROR', 'reason': f'Strategy creation failed: {strategy_result.get("message")}' }
            # 2. Execute strategy
            execution_result = self.execute_strategy(
                strategy_legs=strategy_result.get('legs', []),
                trade_metadata={
                    'strategy_name': 'LONG_PUT_SCALP',
                    'mode': 'SCALPING',
                    'entry_conditions': conditions,
                    'risk_management': {
                        'stop_loss': f'{stop_loss_percent}%',
                        'profit_target': '8%',  # Lower profit target for more frequent wins
                        'max_hold_time': '15 minutes',
                        'min_hold_time': '3 minutes'
                    },
                    'expiry_date': expiry_date
                }
            )
            return execution_result
        except Exception as e:
            return {'status': 'ERROR', 'reason': f'Execution failed: {str(e)}'}

class PremiumSellingEngine:
    """Pure premium selling execution - multi-leg, time-based with ACTUAL strategy execution"""
    
    def __init__(self):
        self.execute_strategy = execute_and_store_strategy
        self.get_margins = get_account_margins
        
        # Import actual strategy creation tools
        self.create_short_strangle = create_short_strangle_wrapper
        self.create_iron_condor = create_iron_condor_wrapper
        self.create_bull_put_spread = create_bull_put_spread_wrapper
        self.create_bear_call_spread = create_bear_call_spread_wrapper
    
    def _get_optimal_expiry_for_mode(self, mode: str) -> str:
        """Get optimal expiry based on trading mode"""
        try:
            expiry_analysis = get_available_expiry_dates_with_analysis()
            if not expiry_analysis or 'available_expiries' not in expiry_analysis:
                return None
            
            expiries = expiry_analysis['available_expiries']
            
            if mode == 'PREMIUM_SELLING':
                # For premium selling, prefer longer expiries (more time decay)
                for expiry in expiries:
                    if expiry.get('category') == 'LONG_TERM':  # 15+ days
                        return expiry.get('date')
                # Fallback to medium term
                for expiry in expiries:
                    if expiry.get('category') == 'MEDIUM_TERM':
                        return expiry.get('date')
            
            # Last resort - first available expiry
            return expiries[0].get('date') if expiries else None
            
        except Exception as e:
            logger.error(f"Error getting expiry: {e}")
            return None
    
    def calculate_dynamic_position_size(self, account_balance: float, strategy_risk: float) -> int:
        """Calculate position size based on actual strategy risk"""
        # Risk 3-5% of account on premium selling
        max_risk = account_balance * 0.04
        position_size = int(max_risk / (strategy_risk * 200))  # Assuming ₹200 max loss per lot
        return min(position_size, 75)  # Cap at 75 lots for premium selling
    
    def execute(self, conditions: Dict[str, Any]) -> Dict[str, Any]:
        """
        PREMIUM SELLING RULES with ACTUAL EXECUTION:
        - Multi-leg positions
        - High IV requirement (>60th percentile)
        - Time-based exits (hours)
        - Range-bound market required
        - Greeks monitoring
        """
        iv_percentile = conditions.get('iv_percentile', 0)
        rsi = conditions.get('rsi', 50)
        market_regime = conditions.get('market_regime', 'UNKNOWN')
        
        # Check account margins first
        try:
            margins = self.get_margins()
            if margins.get('status') != 'SUCCESS':
                return {'status': 'ERROR', 'reason': 'Cannot verify account margins'}
            
            available_capital = margins['equity'].get('live_balance', 0)
            if available_capital < 100000:  # Higher capital requirement for premium selling
                return {'status': 'ERROR', 'reason': f'Insufficient capital for premium selling: ₹{available_capital}'}
        except Exception as e:
            return {'status': 'ERROR', 'reason': f'Margin check failed: {str(e)}'}
        
        # Get optimal expiry
        expiry_date = self._get_optimal_expiry_for_mode('PREMIUM_SELLING')
        if not expiry_date:
            return {'status': 'ERROR', 'reason': 'No suitable expiry available'}
        
        # Strategy selection and execution
        if iv_percentile > 75 and abs(rsi - 50) < 15:
            return self._execute_short_strangle(expiry_date, available_capital, conditions)
        elif iv_percentile > 60 and market_regime == 'RANGING':
            return self._execute_iron_condor(expiry_date, available_capital, conditions)
        else:
            return {
                'status': 'ERROR',
                'reason': f'No clear premium selling setup - IV: {iv_percentile:.1f}%, RSI: {rsi:.1f}, Regime: {market_regime}'
            }
    
    def _execute_short_strangle(self, expiry_date: str, capital: float, conditions: Dict[str, Any]) -> Dict[str, Any]:
        """Execute short strangle premium selling trade"""
        try:
            # 1. Create strategy
            strategy_result = self.create_short_strangle(
                expiry_date=expiry_date,
                quantity=DEFAULT_TRADE_QUANTITY
            )
            
            if strategy_result.get('status') != 'SUCCESS':
                return {'status': 'ERROR', 'reason': f'Strategy creation failed: {strategy_result.get("message")}'}
            
            # 2. Execute strategy
            execution_result = self.execute_strategy(
                strategy_legs=strategy_result.get('legs', []),
                trade_metadata={
                    'strategy_name': 'SHORT_STRANGLE',
                    'mode': 'PREMIUM_SELLING',
                    'entry_conditions': conditions,
                    'risk_management': {
                        'stop_loss': 'time-based',
                        'profit_target': '30% of premium',
                        'max_hold_time': '4 hours'
                    },
                    'expiry_date': expiry_date
                }
            )
            
            return execution_result
            
        except Exception as e:
            return {'status': 'ERROR', 'reason': f'Execution failed: {str(e)}'}
    
    def _execute_iron_condor(self, expiry_date: str, capital: float, conditions: Dict[str, Any]) -> Dict[str, Any]:
        """Execute iron condor premium selling trade"""
        try:
            # 1. Create strategy
            strategy_result = self.create_iron_condor(
                expiry_date=expiry_date,
                quantity=DEFAULT_TRADE_QUANTITY
            )
            
            if strategy_result.get('status') != 'SUCCESS':
                return {'status': 'ERROR', 'reason': f'Strategy creation failed: {strategy_result.get("message")}'}
            
            # 2. Execute strategy
            execution_result = self.execute_strategy(
                strategy_legs=strategy_result.get('legs', []),
                trade_metadata={
                    'strategy_name': 'IRON_CONDOR',
                    'mode': 'PREMIUM_SELLING',
                    'entry_conditions': conditions,
                    'risk_management': {
                        'stop_loss': 'time-based',
                        'profit_target': '25% of premium',
                        'max_hold_time': '6 hours'
                    },
                    'expiry_date': expiry_date
                }
            )
            
            return execution_result
            
        except Exception as e:
            return {'status': 'ERROR', 'reason': f'Execution failed: {str(e)}'}

class HybridTradingSystem:
    """Main hybrid trading system with clear mode separation"""
    
    def __init__(self):
        self.scalping_engine = ScalpingEngine()
        self.premium_selling_engine = PremiumSellingEngine()
    
    def execute_trade(self, market_conditions: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main execution with clear mode separation
        """
        # Determine trading mode
        mode = HybridTradingMode.determine_trading_mode(market_conditions)
        
        # Check time-based restrictions
        current_time = datetime.now().time()
        time_mode = HybridTradingMode.get_intraday_trading_mode(current_time)
        
        # Apply time restrictions
        if mode['mode'] == 'SCALPING' and time_mode == 'PREMIUM_SELLING_ONLY':
            return {'status': 'ERROR', 'reason': 'Scalping not allowed during mid-day calm period'}
        elif mode['mode'] == 'PREMIUM_SELLING' and time_mode == 'SCALPING_ONLY':
            return {'status': 'ERROR', 'reason': 'Premium selling not allowed during closing volatility'}
        elif time_mode == 'NONE':
            return {'status': 'ERROR', 'reason': 'Outside trading hours'}
        
        # Execute based on mode
        if mode['mode'] == 'SCALPING':
            result = self.scalping_engine.execute(market_conditions)
            result['mode'] = 'SCALPING'
            return result
        elif mode['mode'] == 'PREMIUM_SELLING':
            result = self.premium_selling_engine.execute(market_conditions)
            result['mode'] = 'PREMIUM_SELLING'
            return result
        else:
            return {'status': 'ERROR', 'reason': mode['reason'], 'mode': 'WAIT'}

def _calculate_market_conditions() -> Dict[str, Any]:
    """
    Fetches and calculates all necessary market conditions.
    This function isolates data fetching and calculation for clarity and robustness.
    """
    market_conditions = {
        'spot_price': 0, 'rsi': 50, 'macd_signal': 'NEUTRAL', 'volume_ratio': 1.0,
        'iv_percentile': 0, 'market_regime': 'UNKNOWN', 'realized_vol': 0, 'trend_strength': 0
    }
    
    try:
        spot_data = get_nifty_spot_price()
        if spot_data.get('status') == 'SUCCESS':
            market_conditions['spot_price'] = spot_data.get('spot_price', 0)
    except Exception as e:
        logger.warning(f"Failed to get spot price: {e}")
    
    try:
        technical_data = get_nifty_technical_analysis_tool()
        if technical_data.get('status') == 'SUCCESS':
            market_conditions.update({
                'spot_price': technical_data.get('spot_price', 0),
                'adx': technical_data.get('latest_indicator_values', {}).get('adx', 0),
                'trend_strength': technical_data.get('trend_strength', 0)
            })
    except Exception as e:
        logger.warning(f"Failed to get technical data: {e}")
    
    # Get market regime using the proper detection function
    try:
        regime_data = detect_market_regime_wrapper()
        if regime_data.get('status') == 'SUCCESS':
            market_conditions['market_regime'] = regime_data.get('primary_regime', 'UNKNOWN')
        else:
            market_conditions['market_regime'] = 'UNKNOWN'
    except Exception as e:
        logger.warning(f"Failed to get market regime: {e}")
        market_conditions['market_regime'] = 'UNKNOWN'
            
    market_conditions.update({
        'rsi': technical_data.get('latest_indicator_values', {}).get('rsi', 50),
        'macd_signal': technical_data.get('trading_signals', {}).get('macd', 'NEUTRAL'),
        'volume_ratio': technical_data.get('latest_indicator_values', {}).get('volume_ratio', 1.0)
    })
    
    try:
        iv_data = calculate_iv_rank_analysis_wrapper()
        if iv_data.get('status') == 'SUCCESS':
            market_conditions['iv_percentile'] = iv_data.get('iv_percentile', 0)
    except Exception as e:
        logger.warning(f"Failed to get IV data: {e}")
    
    try:
        realized_vol = get_realized_volatility_from_kite()
        if realized_vol and realized_vol > 0:
            market_conditions['realized_vol'] = realized_vol
    except Exception as e:
        logger.warning(f"Failed to get realized volatility: {e}")
        
    return {'status': 'SUCCESS', 'data': market_conditions}

def append_indicator_history(market_conditions, timestamp):
    """Append market conditions to indicator_history.jsonl as a JSON line."""
    log_entry = dict(market_conditions)
    log_entry['timestamp'] = str(timestamp)
    log_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'indicator_history.jsonl')
    with open(log_path, 'a') as f:
        f.write(json.dumps(log_entry) + '\n')

def load_indicator_history_today(path, lookback=3):
    today = datetime.now().date()
    if not os.path.exists(path):
        return []
    with open(path, 'r') as f:
        lines = f.readlines()
    # Parse and filter for today
    entries = [json.loads(line) for line in lines if 'timestamp' in line]
    today_entries = [e for e in entries if datetime.fromisoformat(e['timestamp']).date() == today]
    return today_entries[-lookback:]

def trend_following_signal(history, lookback=4):
    print(f"[DEBUG] trend_following_signal called with history length: {len(history)}")
    if len(history) < lookback:
        print(f"[DEBUG] Not enough history for lookback={lookback}. Returning WAIT.")
        return "WAIT"
    rsi_vals = [h.get('rsi', 0) for h in history[-lookback:]]
    adx_vals = [h.get('adx', 0) for h in history[-lookback:]]
    print(f"[DEBUG] RSI values (last {lookback}): {rsi_vals}")
    print(f"[DEBUG] ADX values (last {lookback}): {adx_vals}")
    
    # Get current market regime
    current_regime = history[-1].get('market_regime', 'UNKNOWN')
    print(f"[DEBUG] Current market regime: {current_regime}")
    
    # Momentum override (extreme conditions) - PRIORITY 1
    if rsi_vals[-1] < 25:
        print(f"[Momentum Override] RSI={rsi_vals[-1]:.2f} < 25: Triggering LONG signal (oversold reversal).")
        return "LONG"
    if rsi_vals[-1] > 75:
        print(f"[Momentum Override] RSI={rsi_vals[-1]:.2f} > 75: Triggering SHORT signal (overbought reversal).")
        return "SHORT"
    
    # Regime-based execution (PRIORITY 2) - captures obvious opportunities
    if current_regime == "TRENDING_BULL" and adx_vals[-1] > 35:
        if 40 <= rsi_vals[-1] <= 60:  # Neutral but favorable zone
            print(f"[Regime-Based] TRENDING_BULL regime + ADX={adx_vals[-1]:.2f} + RSI={rsi_vals[-1]:.2f}: Triggering LONG signal.")
            return "LONG"
    
    # Helper: count rising/flat and falling/flat bars
    def count_rising_or_flat(vals):
        return sum(vals[i] >= vals[i-1] for i in range(1, lookback))
    def count_falling_or_flat(vals):
        return sum(vals[i] <= vals[i-1] for i in range(1, lookback))
    
    # Trend continuation signals (PRIORITY 3) - ONLY if NO extreme RSI in current bar
    # This prevents conflict between momentum reversal and trend continuation
    current_rsi = rsi_vals[-1]
    
    # LONG: RSI > 60 for all bars, at least 2/3 rising/flat; ADX > 20 for all, at least 2/3 rising/flat
    # BUT only if current RSI is NOT extreme (not < 25 or > 75)
    if (all(r > 60 for r in rsi_vals) and all(a > 20 for a in adx_vals) and 
        25 <= current_rsi <= 75):  # Current RSI must be in normal range
        if count_rising_or_flat(rsi_vals) >= 2 and count_rising_or_flat(adx_vals) >= 2:
            print(f"[Trend Continuation] RSI > 60 trend + ADX > 20 + Normal RSI ({current_rsi:.1f}): Triggering LONG signal.")
            return "LONG"
    
    # SHORT: RSI < 40 for all bars, at least 2/3 falling/flat; ADX > 20 for all, at least 2/3 rising/flat
    # BUT only if current RSI is NOT extreme (not < 25 or > 75)
    if (all(r < 40 for r in rsi_vals) and all(a > 20 for a in adx_vals) and 
        25 <= current_rsi <= 75):  # Current RSI must be in normal range
        if count_falling_or_flat(rsi_vals) >= 2 and count_rising_or_flat(adx_vals) >= 2:
            print(f"[Trend Continuation] RSI < 40 trend + ADX > 20 + Normal RSI ({current_rsi:.1f}): Triggering SHORT signal.")
            return "SHORT"
    
    return "WAIT"

def volatility_regime_switch_signal(history):
    """
    If IV percentile > 0.7 and rising for 2+ bars, prefer premium selling.
    If IV percentile < 0.3 and realized vol is rising, prefer long options.
    Otherwise, WAIT.
    """
    if len(history) < 2:
        return "WAIT"
    iv_now = history[-1].get('iv_percentile', 0)
    iv_prev = history[-2].get('iv_percentile', 0)
    rv_now = history[-1].get('realized_vol', 0)
    rv_prev = history[-2].get('realized_vol', 0)
    if iv_now > 0.7 and iv_now > iv_prev:
        return "PREMIUM_SELLING"
    if iv_now < 0.3 and rv_now > rv_prev:
        return "LONG_OPTION"
    return "WAIT"

def log_intended_trade(symbol: str, action: str, reason: str, market_conditions: Dict[str, Any]) -> None:
    """
    Log what trade the opportunity hunter would have taken if no positions were open.
    This helps the position manager make intelligent exit decisions.
    """
    try:
        log_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'intended_trades.jsonl')
        intended_trade = {
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'action': action,  # 'LONG_CALL', 'LONG_PUT', etc.
            'reason': reason,
            'market_conditions': {
                'rsi': market_conditions.get('rsi', 0),
                'adx': market_conditions.get('adx', 0),
                'macd_signal': market_conditions.get('macd_signal', 'NEUTRAL'),
                'market_regime': market_conditions.get('market_regime', 'UNKNOWN'),
                'trend_signal': market_conditions.get('trend_signal', 'NEUTRAL')
            }
        }
        
        with open(log_file, 'a') as f:
            f.write(json.dumps(intended_trade) + '\n')
        
        print(f"📝 Logged intended trade: {action} on {symbol} (deferred due to existing position)")
        
    except Exception as e:
        print(f"⚠️ Failed to log intended trade: {e}")

def get_recent_intended_trades(symbol: str = None, minutes_back: int = 30) -> List[Dict[str, Any]]:
    """
    Get recent intended trades, optionally filtered by symbol.
    Used by position manager to check if opportunity hunter wants this position.
    """
    try:
        log_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'intended_trades.jsonl')
        if not os.path.exists(log_file):
            return []
        
        cutoff_time = datetime.now().timestamp() - (minutes_back * 60)
        recent_trades = []
        
        with open(log_file, 'r') as f:
            for line in f:
                try:
                    trade = json.loads(line.strip())
                    trade_time = datetime.fromisoformat(trade['timestamp']).timestamp()
                    
                    if trade_time >= cutoff_time:
                        if symbol is None or trade['symbol'] == symbol:
                            recent_trades.append(trade)
                except:
                    continue
        
        return recent_trades
        
    except Exception as e:
        print(f"⚠️ Failed to read intended trades: {e}")
        return []

def run_hybrid_scalping_opportunity_hunter() -> Dict[str, Any]:
    """
    Main entry for the hybrid scalping opportunity hunter.
    """
    # TIME CHECK: No trades after 3:00 PM
    current_time = datetime.now().time()
    if current_time.hour >= 15:  # 3:00 PM or later
        print("⏰ Market closing time reached (3:00 PM). No new trades allowed.")
        return {
            'status': 'TIME_RESTRICTED',
            'decision': 'WAIT',
            'reason': 'Trading stopped after 3:00 PM for risk management.',
            'timestamp': datetime.now().isoformat()
        }
    
    # We'll check for existing positions after determining what trade would be taken

    try:
        print("🚀 Starting HYBRID SCALPING & PREMIUM SELLING HUNTER...")
        print("📋 Mode: Clear separation between scalping and premium selling")
        print("📋 Scalping: Single-leg, quick exits, momentum-based")
        print("📋 Premium Selling: Multi-leg, time-based, IV-based")
        print("⚡ NO MORE CONTRADICTIONS - Each mode optimized for its conditions")
        print("⏱️  Token Efficiency: Optimal for all scenarios")
        print("-" * 50)
        
        # Get market data using the new, robust function
        market_data_result = _calculate_market_conditions()
        
        # Proper error handling
        if market_data_result.get('status') != 'SUCCESS':
            return {
                'status': 'ERROR',
                'decision': 'WAIT',
                'reason': market_data_result.get('message', 'Could not retrieve market conditions.'),
                'timestamp': datetime.now().isoformat()
            }
        
        market_conditions = market_data_result['data']
        
        # Always log the market conditions for trend analysis, even if no trade is taken
        append_indicator_history(market_conditions, datetime.now())

        # Load indicator history and apply trend-following filter (today only)
        log_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'indicator_history.jsonl')
        history = load_indicator_history_today(log_path, lookback=4)
        trend_signal = trend_following_signal(history, lookback=4)
        regime_signal = volatility_regime_switch_signal(history)
        print(f"[DEBUG] Trend signal: {trend_signal}, Regime signal: {regime_signal}")
        hybrid_system = HybridTradingSystem()
        
        # --- Premium selling remains regime-gated ---
        if regime_signal == "PREMIUM_SELLING":
            print("[Decision] Executing premium selling strategy (regime filter)")
            result = hybrid_system.premium_selling_engine.execute(market_conditions)
            # Check if execution was successful
            if result.get('status') == 'SUCCESS':
                result['mode'] = 'PREMIUM_SELLING'
                
                # Add analysis summary for successful premium selling
                result['analysis_summary'] = {
                    "mode_detection": "Detected PREMIUM_SELLING mode",
                    "strategy_selection": result.get('strategy', 'NONE'),
                    "risk_validation": "Appropriate for high IV regime",
                    "execution_plan": "Premium selling strategy executed"
                }
                
                return result
            else:
                # Execution failed, return the error
                return {
                    'status': 'EXECUTION_FAILED',
                    'execution_result': result,
                    'storage_result': result.get('storage_result'),
                    'message': result.get('message', 'Strategy execution failed'),
                    'mode': 'PREMIUM_SELLING',
                    'analysis_summary': {
                        "mode_detection": "Detected PREMIUM_SELLING mode",
                        "strategy_selection": "NONE",
                        "risk_validation": "Execution failed",
                        "execution_plan": "Strategy execution failed"
                    }
                }
        
        # --- Scalping: Regime is advisory, trend is main gate ---
        if trend_signal == "LONG":
            if regime_signal == "LONG_OPTION":
                print("[Decision] Executing scalping LONG (trend+regime aligned, normal size)")
                
                # CRITICAL SAFETY CHECK: Check if a position is already open before executing
                try:
                    positions_result = get_portfolio_positions()
                    if positions_result.get('status') == 'SUCCESS':
                        open_positions = [
                            p for p in positions_result.get('positions', [])
                            if 'NIFTY' in p.get('symbol', '')
                        ]
                        if open_positions:
                            print("🔵 Position already open. Logging intended trade and deferring to Position Manager.")
                            # Log what we would have done
                            log_intended_trade('NIFTY', 'LONG_CALL', f'Would have taken LONG CALL (trend+regime aligned)', market_conditions)
                            return {
                                'status': 'DEFERRED',
                                'decision': 'WAIT',
                                'reason': 'An existing NIFTY position is being managed.',
                                'intended_action': 'LONG_CALL'
                            }
                except Exception as e:
                    print(f"⚠️ Could not check for existing positions: {e}")
                
                result = hybrid_system.scalping_engine.execute(market_conditions)
                # Check if execution was successful
                if result.get('status') == 'SUCCESS':
                    result['mode'] = 'SCALPING'
                    result['direction'] = 'LONG'
                    
                    # Add analysis summary for successful scalping
                    result['analysis_summary'] = {
                        "mode_detection": "Detected SCALPING mode",
                        "strategy_selection": result.get('strategy', 'NONE'),
                        "risk_validation": "Appropriate for trend+regime aligned",
                        "execution_plan": "Scalping strategy executed (normal size)"
                    }
                    
                    return result
                else:
                    # Execution failed, return the error
                    return {
                        'status': 'EXECUTION_FAILED',
                        'execution_result': result,
                        'storage_result': result.get('storage_result'),
                        'message': result.get('message', 'Strategy execution failed'),
                        'mode': 'SCALPING',
                        'direction': 'LONG',
                        'analysis_summary': {
                            "mode_detection": "Detected SCALPING mode",
                            "strategy_selection": "NONE",
                            "risk_validation": "Execution failed",
                            "execution_plan": "Strategy execution failed"
                        }
                    }
            else:
                print("[Decision] Executing scalping LONG (trend only, regime not ideal: reduced size/tighter stop)")
                
                # CRITICAL SAFETY CHECK: Check if a position is already open before executing
                try:
                    positions_result = get_portfolio_positions()
                    if positions_result.get('status') == 'SUCCESS':
                        open_positions = [
                            p for p in positions_result.get('positions', [])
                            if 'NIFTY' in p.get('symbol', '')
                        ]
                        if open_positions:
                            print("🔵 Position already open. Logging intended trade and deferring to Position Manager.")
                            # Log what we would have done
                            log_intended_trade('NIFTY', 'LONG_CALL', f'Would have taken LONG CALL (trend only, regime not ideal)', market_conditions)
                            return {
                                'status': 'DEFERRED',
                                'decision': 'WAIT',
                                'reason': 'An existing NIFTY position is being managed.',
                                'intended_action': 'LONG_CALL'
                            }
                except Exception as e:
                    print(f"⚠️ Could not check for existing positions: {e}")
                
                # Patch market_conditions to signal reduced size/tighter stop
                patched = dict(market_conditions)
                patched['regime_not_ideal'] = True
                result = hybrid_system.scalping_engine.execute(patched)
                # Check if execution was successful
                if result.get('status') == 'SUCCESS':
                    result['mode'] = 'SCALPING'
                    result['direction'] = 'LONG'
                    result['note'] = 'Reduced size/tighter stop due to regime filter'
                    return result
                else:
                    # Execution failed, return the error
                    return {
                        'status': 'EXECUTION_FAILED',
                        'execution_result': result,
                        'storage_result': result.get('storage_result'),
                        'message': result.get('message', 'Strategy execution failed'),
                        'mode': 'SCALPING',
                        'direction': 'LONG',
                        'note': 'Reduced size/tighter stop due to regime filter'
                    }
        elif trend_signal == "SHORT":
            if regime_signal == "LONG_OPTION":
                print("[Decision] Executing scalping SHORT (trend+regime aligned, normal size)")
                
                # CRITICAL SAFETY CHECK: Check if a position is already open before executing
                try:
                    positions_result = get_portfolio_positions()
                    if positions_result.get('status') == 'SUCCESS':
                        open_positions = [
                            p for p in positions_result.get('positions', [])
                            if 'NIFTY' in p.get('symbol', '')
                        ]
                        if open_positions:
                            print("🔵 Position already open. Logging intended trade and deferring to Position Manager.")
                            # Log what we would have done
                            log_intended_trade('NIFTY', 'LONG_PUT', f'Would have taken LONG PUT (trend+regime aligned)', market_conditions)
                            return {
                                'status': 'DEFERRED',
                                'decision': 'WAIT',
                                'reason': 'An existing NIFTY position is being managed.',
                                'intended_action': 'LONG_PUT'
                            }
                except Exception as e:
                    print(f"⚠️ Could not check for existing positions: {e}")
                
                result = hybrid_system.scalping_engine.execute(market_conditions)
                # Check if execution was successful
                if result.get('status') == 'SUCCESS':
                    result['mode'] = 'SCALPING'
                    result['direction'] = 'SHORT'
                    return result
                else:
                    # Execution failed, return the error
                    return {
                        'status': 'EXECUTION_FAILED',
                        'execution_result': result,
                        'storage_result': result.get('storage_result'),
                        'message': result.get('message', 'Strategy execution failed'),
                        'mode': 'SCALPING',
                        'direction': 'SHORT'
                    }
            else:
                print("[Decision] Executing scalping SHORT (trend only, regime not ideal: reduced size/tighter stop)")
                
                # CRITICAL SAFETY CHECK: Check if a position is already open before executing
                try:
                    positions_result = get_portfolio_positions()
                    if positions_result.get('status') == 'SUCCESS':
                        open_positions = [
                            p for p in positions_result.get('positions', [])
                            if 'NIFTY' in p.get('symbol', '')
                        ]
                        if open_positions:
                            print("🔵 Position already open. Logging intended trade and deferring to Position Manager.")
                            # Log what we would have done
                            log_intended_trade('NIFTY', 'LONG_PUT', f'Would have taken LONG PUT (trend only, regime not ideal)', market_conditions)
                            return {
                                'status': 'DEFERRED',
                                'decision': 'WAIT',
                                'reason': 'An existing NIFTY position is being managed.',
                                'intended_action': 'LONG_PUT'
                            }
                except Exception as e:
                    print(f"⚠️ Could not check for existing positions: {e}")
                
                patched = dict(market_conditions)
                patched['regime_not_ideal'] = True
                result = hybrid_system.scalping_engine.execute(patched)
                # Check if execution was successful
                if result.get('status') == 'SUCCESS':
                    result['mode'] = 'SCALPING'
                    result['direction'] = 'SHORT'
                    result['note'] = 'Reduced size/tighter stop due to regime filter'
                    return result
                else:
                    # Execution failed, return the error
                    return {
                        'status': 'EXECUTION_FAILED',
                        'execution_result': result,
                        'storage_result': result.get('storage_result'),
                        'message': result.get('message', 'Strategy execution failed'),
                        'mode': 'SCALPING',
                        'direction': 'SHORT',
                        'note': 'Reduced size/tighter stop due to regime filter'
                    }
        else:
            print("[Decision] No trend-following signal for scalping. Skipping trade.")
            
            # Create analysis summary for WAIT case
            analysis_summary = {
                "mode_detection": "No mode detected (WAIT)",
                "strategy_selection": "NONE",
                "risk_validation": "No risk (no trade)",
                "execution_plan": "No execution needed"
            }
            
            return {
                'status': 'SUCCESS',
                'decision': 'WAIT',
                'reason': 'No trend-following signal',
                'market_conditions': market_conditions,
                'analysis_summary': analysis_summary,
                'timestamp': datetime.now().isoformat()
            }
        
        # This code is unreachable due to multiple return statements above
        # The analysis summary should be created within each execution path
        print("⚠️ WARNING: This code should never be reached due to multiple return statements above")
        return {
            'status': 'ERROR',
            'decision': 'WAIT',
            'reason': 'Logic error: Unreachable code reached',
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in hybrid scalping opportunity hunter: {str(e)}")
        return {'status': 'ERROR', 'message': str(e)}

if __name__ == "__main__":
    result = run_hybrid_scalping_opportunity_hunter()
    print(json.dumps(result, indent=2)) 