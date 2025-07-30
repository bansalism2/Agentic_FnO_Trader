#!/usr/bin/env python3
"""
AlgoTrade - INTRADAY SCALPING POSITION MANAGER (RULES-BASED)
============================================================

This is a high-speed, rules-based position manager designed specifically for
intraday scalping. It operates without any LLM dependency for maximum speed
and reliability.

CORE LOGIC:
1.  **Profit Target**: Take profits at a pre-defined percentage (e.g., +15%).
2.  **Stop-Loss**: Cut losses at a tight stop (e.g., -8%).
3.  **Momentum Reversal**: Exit positions if the underlying momentum fades
    (e.g., RSI crosses below 50 for a long position).
4.  **Forced Square-Off**: Automatically closes all positions at 3:15 PM.

This manager is designed to be fast, efficient, and perfectly aligned with the
logic of the scalping opportunity hunter.
"""

import os
import json
from datetime import datetime, time as dt_time, timedelta
import pytz
from dotenv import load_dotenv
from typing import Dict, Any, Optional

# Add parent directory to Python path for robust imports
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core_tools.connect_data_tools import initialize_connection
from core_tools.execution_portfolio_tools import get_portfolio_positions, execute_options_strategy
from core_tools.master_indicators import get_nifty_technical_analysis_tool
from core_tools.trade_storage import get_active_trades, update_trade_status
from core_tools.portfolio_history import record_portfolio_snapshot, detect_trend_patterns

# --- Configuration ---
IST_TIMEZONE = pytz.timezone('Asia/Kolkata')
PROFIT_TARGET_PCT = 15.0
STOP_LOSS_PCT = -8.0
RSI_EXIT_THRESHOLD_LONG = 50
RSI_EXIT_THRESHOLD_SHORT = 50
FORCED_SQUARE_OFF_TIME = dt_time(15, 15)

# --- Connection Setup ---
try:
    env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../.env'))
    load_dotenv(dotenv_path=env_path)
    api_key = os.getenv("kite_api_key")
    api_secret = os.getenv("kite_api_secret")
    with open("../data/access_token.txt", "r") as f:
        access_token = f.read().strip()
    initialize_connection(api_key, api_secret, access_token)
    print("✅ Position Manager: Kite Connect connection initialized.")
except Exception as e:
    print(f"❌ Position Manager: Failed to initialize Kite Connect: {e}")

class ScalpingPositionManager:
    """A high-speed, rules-based manager for scalping trades."""

    def __init__(self):
        self.active_trades = get_active_trades()
        self.recently_closed = {}  # symbol: datetime of last non-profit closure

    def get_market_conditions(self) -> Dict[str, Any]:
        """Fetches the latest technical indicators."""
        try:
            tech_data = get_nifty_technical_analysis_tool(days=5, interval="5minute")
            if tech_data.get('status') == 'SUCCESS':
                return {
                    'rsi': tech_data.get('latest_indicator_values', {}).get('rsi', 50)
        }
        except Exception as e:
            print(f"⚠️  Could not fetch market conditions: {e}")
        return {'rsi': 50} # Fail-safe default

    def manage_positions(self):
        """The core logic loop for managing open scalping positions."""
        print("\n" + "="*80)
        print("⚡ Running Scalping Position Manager...")
    
        positions_result = get_portfolio_positions()
        if positions_result.get('status') != 'SUCCESS':
            print("❌ Could not get portfolio positions. Aborting.")
            return {'status': 'ERROR', 'message': 'Could not get portfolio positions'}

        # CRITICAL FIX: The API returns 'symbol', not 'tradingsymbol'.
        open_positions = [
            p for p in positions_result.get('positions', [])
            if 'NIFTY' in p.get('symbol', '')
        ]

        if not open_positions:
            print("✅ No open NIFTY positions to manage.")
            return {'status': 'SUCCESS', 'decision': 'NO_POSITIONS'}
        
        print(f"🔎 Found {len(open_positions)} open NIFTY position(s).")
        market_conditions = self.get_market_conditions()
        current_rsi = market_conditions.get('rsi', 50)

        # --- Trend-Based Safety Check ---
        trend_issues = detect_trend_patterns()
        if trend_issues:
            print(f"⚠️ Trend Issues Detected: {trend_issues}")
            for symbol, issue in trend_issues.items():
                if any(keyword in issue for keyword in ["CATASTROPHIC_LOSS", "LOSS_ACCELERATION"]):
                    pos_to_exit = next((p for p in open_positions if p.get('symbol') == symbol), None)
                    if pos_to_exit:
                        print(f"🚨 EMERGENCY EXIT due to critical trend issue: {issue}")
                        trade_id = self.find_trade_id_for_symbol(symbol)
                        self.exit_position(pos_to_exit.get('symbol'), pos_to_exit.get('quantity'), trade_id, f"Critical Trend: {issue}")
                        # Remove from list to avoid double-processing
                        open_positions = [p for p in open_positions if p.get('symbol') != symbol]

        # --- P&L Logging & Standard Evaluation ---
            total_pnl = 0
            total_exposure = 0
            strategy_pnl_dict = {}

        for pos in open_positions:
            pnl, pnl_percentage, exposure = self.evaluate_position(pos, current_rsi)
            
            # For logging
            total_pnl += pnl
            total_exposure += exposure
            strategy_pnl_dict[pos.get('symbol', '')] = {'pnl_pct': pnl_percentage, 'strategy_name': 'SCALP'}

        # Record snapshot of portfolio P&L
        if total_exposure > 0:
            total_portfolio_pnl_pct = (total_pnl / total_exposure) * 100
            record_portfolio_snapshot(total_portfolio_pnl_pct, strategy_pnl_dict)
            print(f"💾 Portfolio snapshot recorded. Total P&L: {total_portfolio_pnl_pct:.2f}%")
            
        return {'status': 'SUCCESS', 'decision': 'MANAGEMENT_CYCLE_COMPLETE'}
            
    def evaluate_position(self, pos, rsi):
        """Evaluates and acts on a single open position."""
        # CRITICAL FIX: Use 'symbol' key to align with API response
        symbol = pos.get('symbol')
        quantity = pos.get('quantity')
        avg_price = pos.get('average_price')
        last_price = pos.get('last_price')

        pnl = (last_price - avg_price) * quantity
        
        # Calculate P&L percentage
        initial_investment = avg_price * abs(quantity)
        pnl_percentage = (pnl / initial_investment) * 100 if initial_investment > 0 else 0
        
        is_long = quantity > 0
        trade_id = self.find_trade_id_for_symbol(symbol)

        print(f"  - Evaluating {symbol} (Qty: {quantity}): P&L: {pnl_percentage:.2f}% | Absolute P&L: ₹{pnl:.2f}")

        # 0. Check for absolute profit target (₹500 per 75 lot)
        lot_size = 75
        target_per_lot = 500
        if abs(quantity) % lot_size == 0 and pnl >= target_per_lot * (abs(quantity) // lot_size):
            print(f"  💰 ABSOLUTE PROFIT TARGET HIT: Exiting {symbol} at ₹{pnl:.2f} profit (Qty: {quantity}).")
            self.exit_position(symbol, quantity, trade_id, f"Absolute Profit Target (₹{target_per_lot} per lot)")
            # Remove from recently_closed so profit exits are always allowed
            if symbol in self.recently_closed:
                del self.recently_closed[symbol]
            return pnl, pnl_percentage, initial_investment

        # 1. Check for Profit Target (percentage)
        if pnl_percentage >= PROFIT_TARGET_PCT:
            print(f"  ✅ PROFIT TARGET HIT: Exiting {symbol} at {pnl_percentage:.2f}% profit.")
            self.exit_position(symbol, quantity, trade_id, "Profit Target")
            if symbol in self.recently_closed:
                del self.recently_closed[symbol]
            return pnl, pnl_percentage, initial_investment

        # Buffer logic: skip non-profit exits if closed in last 30 minutes
        now = datetime.now()
        buffer_minutes = 30
        last_closed = self.recently_closed.get(symbol)
        if last_closed and (now - last_closed) < timedelta(minutes=buffer_minutes):
            print(f"  ⏳ BUFFER: Skipping non-profit exit for {symbol} (closed {int((now - last_closed).total_seconds()//60)} min ago)")
            return pnl, pnl_percentage, initial_investment

        # 2. Check for Stop-Loss
        if pnl_percentage <= STOP_LOSS_PCT:
            print(f"  ❌ STOP-LOSS HIT: Exiting {symbol} at {pnl_percentage:.2f}% loss.")
            self.exit_position(symbol, quantity, trade_id, "Stop-Loss")
            self.recently_closed[symbol] = now
            return pnl, pnl_percentage, initial_investment

        # 3. Check for Momentum Reversal
        if is_long and rsi < RSI_EXIT_THRESHOLD_LONG:
            print(f"  📉 MOMENTUM REVERSAL (LONG): RSI dropped to {rsi:.2f}. Exiting {symbol}.")
            self.exit_position(symbol, quantity, trade_id, "Momentum Reversal (RSI)")
            self.recently_closed[symbol] = now
            return pnl, pnl_percentage, initial_investment
        
        if not is_long and rsi > RSI_EXIT_THRESHOLD_SHORT:
            print(f"  📈 MOMENTUM REVERSAL (SHORT): RSI rose to {rsi:.2f}. Exiting {symbol}.")
            self.exit_position(symbol, quantity, trade_id, "Momentum Reversal (RSI)")
            self.recently_closed[symbol] = now
            return pnl, pnl_percentage, initial_investment
            
        print(f"  -> HOLDING {symbol}. No exit criteria met.")
        return pnl, pnl_percentage, initial_investment

    def exit_position(self, symbol, quantity, trade_id, reason):
        """Exits a single position and updates its status."""
        exit_order = [{
            'symbol': symbol,
            'action': 'SELL' if quantity > 0 else 'BUY',
            'quantity': abs(quantity)
        }]
        try:
            result = execute_options_strategy(exit_order, order_type="Closing")
            if result.get('status') == 'SUCCESS':
                print(f"  -> SUCCESSFULLY placed exit order for {symbol}.")
                if trade_id:
                    update_trade_status(trade_id, "CLOSED", exit_reason=reason)
            else:
                print(f"  -> FAILED to place exit order for {symbol}: {result.get('message')}")
                # Even if exit fails, we must update the status to reflect the attempt
                update_trade_status(trade_id, "EXIT_FAILED", exit_reason=f"Execution error: {result.get('message')}")
        except Exception as e:
            print(f"  -> FAILED to place exit order for {symbol}: {e}")
            # Even if exit fails, we must update the status to reflect the attempt
            update_trade_status(trade_id, "EXIT_FAILED", exit_reason=f"Execution error: {e}")

    def square_off_all(self, positions):
        """Exits all open positions."""
        for pos in positions:
            # CRITICAL FIX: Use 'symbol' key
            symbol = pos.get('symbol')
            quantity = pos.get('quantity')
            trade_id = self.find_trade_id_for_symbol(symbol)
            self.exit_position(symbol, quantity, trade_id, "Forced Square-Off")

    def find_trade_id_for_symbol(self, symbol: str) -> Optional[str]:
        """Finds the trade_id for a given symbol from the active trades file."""
        for trade_id, trade_details in self.active_trades.items():
            for leg in trade_details.get('legs', []):
                if leg.get('symbol') == symbol:
                    return trade_id
        return None

def run_intraday_scalping_position_manager():
    """Main function to instantiate and run the position manager."""
    manager = ScalpingPositionManager()
    manager.manage_positions()

if __name__ == "__main__":
    run_intraday_scalping_position_manager()