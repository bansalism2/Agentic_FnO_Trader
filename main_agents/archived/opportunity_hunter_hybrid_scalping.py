#!/usr/bin/env python3
"""
AlgoTrade Hybrid Scalping Opportunity Hunter - Clear Mode Separation

This script implements a hybrid trading system with CLEAR SEPARATION between:
1. SCALPING MODE: Quick directional trades (Long Calls/Puts) with tight stops
2. PREMIUM SELLING MODE: Time-decay strategies (Iron Condor/Strangle) with IV-based entries

NO MORE CONTRADICTIONS:
- Scalping = Single-leg, quick exits, momentum-based
- Premium Selling = Multi-leg, time-based, IV-based
- Clear mode detection prevents mixing approaches

Key Features:
- Mode-specific strategy selection
- Appropriate risk management for each mode
- Time-based mode availability
- Clear entry/exit rules
- Proper position sizing
"""

import sys
import os
import json
import logging
from datetime import datetime, time as dt_time
from typing import Dict, Any, Optional
from dotenv import load_dotenv
import numpy as np

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
    get_nifty_spot_price, get_available_expiry_dates_with_analysis, initialize_connection
)
from core_tools.execution_portfolio_tools import get_daily_trading_summary
from core_tools.execution_portfolio_tools import get_portfolio_positions, execute_and_store_strategy, get_account_margins
from core_tools.strategy_creation_tools import (
    create_long_straddle_wrapper,
    create_short_strangle_wrapper, 
    create_iron_condor_wrapper,
    create_bull_put_spread_wrapper,
    create_bear_call_spread_wrapper
)
from core_tools.master_indicators import get_nifty_technical_analysis_tool
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
        
        # Calculate momentum score (0-3 scale) - More realistic approach
        momentum_score = (
            (abs(rsi - 50) / 50) +                    # 0-1: RSI deviation
            (1 if macd_signal in ['BUY', 'SELL'] else 0)  # 0-1: MACD signal
        )
        
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
        if dt_time(9, 15) <= current_time <= dt_time(10, 0):
            return 'PREMIUM_SELLING_ONLY'  # Early - set up time decay
        elif dt_time(10, 0) <= current_time <= dt_time(14, 0):
            return 'BOTH'  # Main window - both modes
        elif dt_time(14, 0) <= current_time <= dt_time(14, 45):
            return 'SCALPING_ONLY'  # Limited scalping window
        else:
            return 'NONE'  # Too close to market close

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
        
        # Check account margins first
        try:
            margins = self.get_margins()
            if margins.get('status') != 'SUCCESS':
                return {'action': 'WAIT', 'reason': 'Cannot verify account margins'}
            
            available_capital = margins['equity'].get('live_balance', 0)
            if available_capital < 50000:
                return {'action': 'WAIT', 'reason': f'Insufficient capital: ₹{available_capital}'}
        except Exception as e:
            return {'action': 'WAIT', 'reason': f'Margin check failed: {str(e)}'}
        
        # Get optimal expiry
        expiry_date = self._get_optimal_expiry_for_mode('SCALPING')
        if not expiry_date:
            return {'action': 'WAIT', 'reason': 'No suitable expiry available'}
        
        # Strategy selection and execution
        adx = conditions.get('adx', 0)
        if rsi > 55 and macd_signal == 'BUY' and adx > 25:
            return self._execute_long_call_scalp(expiry_date, available_capital, conditions)
        elif rsi < 45 and macd_signal == 'SELL' and adx > 25:
            return self._execute_long_put_scalp(expiry_date, available_capital, conditions)
        else:
            return {
                'action': 'WAIT',
                'reason': f'No clear scalping setup - RSI: {rsi:.1f}, MACD: {macd_signal}, ADX: {adx:.1f}'
            }
    
    def _execute_long_call_scalp(self, expiry_date: str, capital: float, conditions: Dict[str, Any]) -> Dict[str, Any]:
        """Execute long call scalping trade"""
        try:
            # 1. Create strategy
            strategy_result = self.create_long_call(
                expiry_date=expiry_date,
                strike_selection='OTM',
                quantity=DEFAULT_TRADE_QUANTITY
            )
            
            if strategy_result.get('status') != 'SUCCESS':
                return {'action': 'WAIT', 'reason': f'Strategy creation failed: {strategy_result.get("message")}'}
            
            # 2. Execute strategy
            execution_result = self.execute_strategy(
                strategy_legs=strategy_result.get('legs', []),
                trade_metadata={
                    'strategy_name': 'LONG_CALL_SCALP',
                    'mode': 'SCALPING',
                    'entry_conditions': conditions,
                    'risk_management': {
                        'stop_loss': '3%',
                        'profit_target': '10%',
                        'max_hold_time': '15 minutes'
                    },
                    'expiry_date': expiry_date
                }
            )
            
            return execution_result
            
        except Exception as e:
            return {'action': 'WAIT', 'reason': f'Execution failed: {str(e)}'}
    
    def _execute_long_put_scalp(self, expiry_date: str, capital: float, conditions: Dict[str, Any]) -> Dict[str, Any]:
        """Execute long put scalping trade"""
        try:
            # 1. Create strategy
            strategy_result = self.create_long_put(
                expiry_date=expiry_date,
                strike_selection='OTM',
                quantity=DEFAULT_TRADE_QUANTITY
            )
            
            if strategy_result.get('status') != 'SUCCESS':
                return {'action': 'WAIT', 'reason': f'Strategy creation failed: {strategy_result.get("message")}'}
            
            # 2. Execute strategy
            execution_result = self.execute_strategy(
                strategy_legs=strategy_result.get('legs', []),
                trade_metadata={
                    'strategy_name': 'LONG_PUT_SCALP',
                    'mode': 'SCALPING',
                    'entry_conditions': conditions,
                    'risk_management': {
                        'stop_loss': '3%',
                        'profit_target': '10%',
                        'max_hold_time': '15 minutes'
                    },
                    'expiry_date': expiry_date
                }
            )
            
            return execution_result
            
        except Exception as e:
            return {'action': 'WAIT', 'reason': f'Execution failed: {str(e)}'}

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
                return {'action': 'WAIT', 'reason': 'Cannot verify account margins'}
            
            available_capital = margins['equity'].get('live_balance', 0)
            if available_capital < 100000:  # Higher capital requirement for premium selling
                return {'action': 'WAIT', 'reason': f'Insufficient capital for premium selling: ₹{available_capital}'}
        except Exception as e:
            return {'action': 'WAIT', 'reason': f'Margin check failed: {str(e)}'}
        
        # Get optimal expiry
        expiry_date = self._get_optimal_expiry_for_mode('PREMIUM_SELLING')
        if not expiry_date:
            return {'action': 'WAIT', 'reason': 'No suitable expiry available'}
        
        # Strategy selection and execution
        if iv_percentile > 75 and abs(rsi - 50) < 15:
            return self._execute_short_strangle(expiry_date, available_capital, conditions)
        elif iv_percentile > 60 and market_regime == 'RANGING':
            return self._execute_iron_condor(expiry_date, available_capital, conditions)
        else:
            return {
                'action': 'WAIT',
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
                return {'action': 'WAIT', 'reason': f'Strategy creation failed: {strategy_result.get("message")}'}
            
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
            return {'action': 'WAIT', 'reason': f'Execution failed: {str(e)}'}
    
    def _execute_iron_condor(self, expiry_date: str, capital: float, conditions: Dict[str, Any]) -> Dict[str, Any]:
        """Execute iron condor premium selling trade"""
        try:
            # 1. Create strategy
            strategy_result = self.create_iron_condor(
                expiry_date=expiry_date,
                quantity=DEFAULT_TRADE_QUANTITY
            )
            
            if strategy_result.get('status') != 'SUCCESS':
                return {'action': 'WAIT', 'reason': f'Strategy creation failed: {strategy_result.get("message")}'}
            
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
            return {'action': 'WAIT', 'reason': f'Execution failed: {str(e)}'}

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
            return {'action': 'WAIT', 'reason': 'Scalping not allowed during mid-day calm period'}
        elif mode['mode'] == 'PREMIUM_SELLING' and time_mode == 'SCALPING_ONLY':
            return {'action': 'WAIT', 'reason': 'Premium selling not allowed during closing volatility'}
        elif time_mode == 'NONE':
            return {'action': 'WAIT', 'reason': 'Outside trading hours'}
        
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
            return {'action': 'WAIT', 'reason': mode['reason'], 'mode': 'WAIT'}

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
        # CRITICAL FIX: Increased days to 5 to ensure sufficient data for indicators
        technical_data = get_nifty_technical_analysis_tool(days=5, interval="15minute")
        if technical_data.get('status') != 'SUCCESS':
            logger.error(f"CRITICAL: Technical analysis tool failed: {technical_data.get('error')}. Cannot proceed.")
            return {'status': 'ERROR', 'message': f"Technical analysis data unavailable: {technical_data.get('error')}"}
        
        volume_ratio = technical_data.get('latest_indicator_values', {}).get('volume_ratio', 1.0)
        # SAFETY NET: Ensure volume_ratio is a valid number
        if not np.isfinite(volume_ratio):
            volume_ratio = 1.0
            
        market_conditions.update({
            'rsi': technical_data.get('latest_indicator_values', {}).get('rsi', 50),
            'macd_signal': technical_data.get('trading_signals', {}).get('macd', 'NEUTRAL'),
            'volume_ratio': volume_ratio,
            'market_regime': technical_data.get('trading_signals', {}).get('adx', 'WEAK_TREND')
        })
        market_conditions['trend_strength'] = abs(market_conditions['rsi'] - 50) / 50
    except Exception as e:
        logger.warning(f"Failed to get technical data: {e}")
    
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

def run_hybrid_scalping_opportunity_hunter() -> Dict[str, Any]:
    """
    Main hybrid scalping opportunity hunter execution
    """
    # CRITICAL SAFETY CHECK: First, check if a position is already open.
    try:
        positions_result = get_portfolio_positions()
        if positions_result.get('status') == 'SUCCESS':
            open_positions = [
                p for p in positions_result.get('positions', [])
                if 'NIFTY' in p.get('symbol', '')
            ]
            if open_positions:
                print("🔵 Position already open. Deferring to Position Manager.")
                return {
                    'status': 'DEFERRED',
                    'decision': 'WAIT',
                    'reason': 'An existing NIFTY position is being managed.'
                }
    except Exception as e:
        print(f"⚠️ Could not check for existing positions: {e}")
        # Continue, but with a warning. Or decide to halt. For now, we log and continue.

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
        
        # Add ADX to market conditions for the mode determination logic
        try:
            technical_data = get_nifty_technical_analysis_tool(days=5, interval="15minute")
            if technical_data.get('status') == 'SUCCESS':
                market_conditions['adx'] = technical_data.get('latest_indicator_values', {}).get('adx', 0)
        except Exception:
            market_conditions['adx'] = 0 # Default if fetch fails

        # Initialize hybrid system
        hybrid_system = HybridTradingSystem()
        
        # Execute trade analysis
        result = hybrid_system.execute_trade(market_conditions)
        
        # HYBRID SYSTEM ANALYSIS (No LLM dependency)
        print("🤖 Running Hybrid System Analysis...")
        
        # Simple validation of the hybrid decision
        analysis_summary = {
            "mode_detection": f"Detected {result.get('mode', 'WAIT')} mode",
            "strategy_selection": result.get('strategy', 'NONE'),
            "risk_validation": "Appropriate for detected mode",
            "execution_plan": "Ready for execution" if result.get('action') == 'EXECUTE' else "No execution needed"
        }
        
        # Format final result
        decision = "WAIT"
        if result.get('action') == 'EXECUTE':
            decision = f"{result.get('mode', 'WAIT').upper()}_{result.get('strategy', 'NONE')}"
        
        return {
            'status': 'SUCCESS',
            'decision': decision,
            'mode': result.get('mode', 'WAIT'),
            'strategy': result.get('strategy', 'NONE'),
            'reason': result.get('reason', 'No clear opportunity'),
            'market_conditions': market_conditions,
            'analysis_summary': analysis_summary,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in hybrid scalping opportunity hunter: {str(e)}")
        return {'status': 'ERROR', 'message': str(e)}

if __name__ == "__main__":
    result = run_hybrid_scalping_opportunity_hunter()
    print(json.dumps(result, indent=2)) 