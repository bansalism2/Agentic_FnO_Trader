#!/usr/bin/env python3
"""
AlgoTrade - SIMPLE POSITION MANAGER (SCALPING)
==============================================

A minimal, high-speed, rules-based position manager for intraday NIFTY scalping.
- Only checks for profit, loss, and a cool-off period after exit.
- No RSI, trend, or indicator logic.
- Designed for maximum robustness and simplicity.
"""

import os
import sys
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import pandas as pd

# Add the parent directory to the path to import core_tools
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytz

from core_tools.execution_portfolio_tools import (
    get_portfolio_positions, get_account_margins
)
from core_tools.trade_storage import (
    get_active_trades, update_trade_status, move_trade_to_history,
    get_trade_history
)

IST_TIMEZONE = pytz.timezone('Asia/Kolkata')
PROFIT_TARGET_PCT = 15.0
STOP_LOSS_PCT = -8.0
ABSOLUTE_PROFIT_PER_LOT = 500
LOT_SIZE = 75
COOL_OFF_MINUTES = 10
EXTREME_LOSS_PCT = -30
FORCED_SQUARE_OFF_TIME = (15, 20)  # 3:20 PM IST

class SimplePositionManager:
    def __init__(self):
        self.active_trades = get_active_trades()
        self.recently_closed = {}  # symbol: datetime of last closure
        self.entry_times = {}  # symbol: datetime of entry (simulate; in prod, use actual trade storage)
        
        # Default values (fallback only)
        self.DEFAULT_STOP_LOSS_PCT = -8.0
        self.DEFAULT_PROFIT_TARGET_PCT = 8.0  # Lower profit target for more frequent wins
        self.DEFAULT_COOL_OFF_MINUTES = 10
        self.DEFAULT_MAX_HOLD_TIME_MINUTES = 30
        self.DEFAULT_MIN_HOLD_TIME_MINUTES = 3  # Minimum 3 minutes unless extreme loss
        self.INTENDED_TRADE_CHECK_MINUTES = 7  # Only consider intended trades from last 7 minutes

    def get_trade_parameters(self, trade_id):
        """Get trade-specific parameters from active_trades.json"""
        if trade_id in self.active_trades:
            trade_data = self.active_trades[trade_id]
            risk_management = trade_data.get('risk_management', {})
            
            # Parse stop loss (e.g., "2%" -> -2.0)
            stop_loss_str = risk_management.get('stop_loss', f"{abs(self.DEFAULT_STOP_LOSS_PCT)}%")
            stop_loss_pct = -float(stop_loss_str.replace('%', ''))
            
            # Parse profit target (e.g., "10%" -> 10.0)
            profit_target_str = risk_management.get('profit_target', f"{self.DEFAULT_PROFIT_TARGET_PCT}%")
            profit_target_pct = float(profit_target_str.replace('%', ''))
            
            # Parse max hold time (e.g., "15 minutes" -> 15)
            max_hold_time_str = risk_management.get('max_hold_time', f"{self.DEFAULT_MAX_HOLD_TIME_MINUTES} minutes")
            max_hold_time_minutes = int(max_hold_time_str.split()[0])
            
            # Parse min hold time (e.g., "3 minutes" -> 3) - default if not specified
            min_hold_time_str = risk_management.get('min_hold_time', f"{self.DEFAULT_MIN_HOLD_TIME_MINUTES} minutes")
            min_hold_time_minutes = int(min_hold_time_str.split()[0])
            
            return {
                'stop_loss_pct': stop_loss_pct,
                'profit_target_pct': profit_target_pct,
                'max_hold_time_minutes': max_hold_time_minutes,
                'min_hold_time_minutes': min_hold_time_minutes,
                'trade_data': trade_data
            }
        
        # Fallback to defaults
        return {
            'stop_loss_pct': self.DEFAULT_STOP_LOSS_PCT,
            'profit_target_pct': self.DEFAULT_PROFIT_TARGET_PCT,
            'max_hold_time_minutes': self.DEFAULT_MAX_HOLD_TIME_MINUTES,
            'min_hold_time_minutes': self.DEFAULT_MIN_HOLD_TIME_MINUTES,
            'trade_data': None
        }

    def manage_positions(self):
        print("\n" + "="*80)
        print("⚡ Running Simple Scalping Position Manager...")
        positions_result = get_portfolio_positions()
        if positions_result.get('status') != 'SUCCESS':
            print("❌ Could not get portfolio positions. Aborting.")
            return {'status': 'ERROR', 'message': 'Could not get portfolio positions'}
        open_positions = [
            p for p in positions_result.get('positions', [])
            if 'NIFTY' in p.get('symbol', '')
        ]
        if not open_positions:
            print("✅ No open NIFTY positions to manage.")
            return {'status': 'SUCCESS', 'decision': 'NO_POSITIONS'}
        print(f"🔎 Found {len(open_positions)} open NIFTY position(s).")
        now = datetime.now(IST_TIMEZONE)
        # Check for forced square-off
        if now.time() >= datetime.time(datetime.strptime(f"{FORCED_SQUARE_OFF_TIME[0]}:{FORCED_SQUARE_OFF_TIME[1]}", "%H:%M")):
            for pos in open_positions:
                symbol = pos.get('symbol')
                quantity = pos.get('quantity')
                trade_id = self.find_trade_id_for_symbol(symbol)
                print(f"  ⏰ FORCED SQUARE-OFF: Exiting {symbol} at 3:20 PM or later.")
                self.exit_position(symbol, quantity, trade_id, "Forced Square-Off (3:20 PM)")
                self.recently_closed[symbol] = now
            print("="*80)
            return {'status': 'SUCCESS', 'decision': 'FORCED_SQUARE_OFF'}
        for pos in open_positions:
            symbol = pos.get('symbol')
            print(f"  [DEBUG] Raw position data for {symbol}:\n{json.dumps(pos, indent=2)}")
            # Simulate entry time if not present (in prod, use actual entry time from trade storage)
            if symbol not in self.entry_times:
                self.entry_times[symbol] = now
            self.evaluate_position(pos, now)
        print("="*80)
        return {'status': 'SUCCESS', 'decision': 'MANAGEMENT_CYCLE_COMPLETE'}

    def get_live_market_price(self, symbol):
        """Get live market price for a symbol"""
        try:
            from core_tools.execution_portfolio_tools import ensure_connection_initialized
            from core_tools.connect_data_tools import _kite_instance
            
            # Ensure connection is initialized
            ensure_connection_initialized()
            
            # Use existing kite instance
            if _kite_instance is None:
                print(f"    ⚠️  No active connection for {symbol}")
                return None
            
            # Get live quote using existing connection
            quote = _kite_instance.quote(f'NFO:{symbol}')
            live_price = quote[f'NFO:{symbol}']['last_price']
            
            return live_price
            
        except Exception as e:
            print(f"    ⚠️  Error getting live price for {symbol}: {e}")
            return None

    def evaluate_position(self, pos, now):
        symbol = pos.get('symbol')
        quantity = pos.get('quantity')
        trade_id = self.find_trade_id_for_symbol(symbol)
        
        # Get live market price instead of stale broker API data
        live_price = self.get_live_market_price(symbol)
        if live_price is None:
            print(f"    ⚠️  Could not get live price for {symbol}, using broker API data")
            live_price = pos.get('last_price')
        else:
            print(f"    📊 Live market price for {symbol}: ₹{live_price}")
        
        # Get trade-specific parameters and actual entry price from trade storage
        trade_params = self.get_trade_parameters(trade_id)
        stop_loss_pct = trade_params['stop_loss_pct']
        profit_target_pct = trade_params['profit_target_pct']
        max_hold_time_minutes = trade_params['max_hold_time_minutes']
        min_hold_time_minutes = trade_params['min_hold_time_minutes']
        trade_data = trade_params.get('trade_data')
        
        # Use actual entry price from trade storage instead of broker API
        actual_entry_price = None
        if trade_data and trade_data.get('execution_details', {}).get('settlement_details', {}).get('settled_orders'):
            try:
                actual_entry_price = float(trade_data['execution_details']['settlement_details']['settled_orders'][0]['avg_price'])
                print(f"    📊 Using actual entry price from trade storage: ₹{actual_entry_price}")
            except (KeyError, ValueError, IndexError) as e:
                print(f"    ⚠️  Error getting actual entry price: {e}")
        
        # Fallback to broker API data if trade storage data not available
        if actual_entry_price is None:
            actual_entry_price = pos.get('average_price')
            print(f"    ⚠️  Using broker API entry price (may be stale): ₹{actual_entry_price}")
        
        # Calculate P&L using actual entry price and live market price
        pnl = (live_price - actual_entry_price) * quantity
        initial_investment = actual_entry_price * abs(quantity)
        pnl_percentage = (pnl / initial_investment) * 100 if initial_investment > 0 else 0
        
        print(f"  - Evaluating {symbol} (Qty: {quantity}): P&L: {pnl_percentage:.2f}% | Absolute P&L: ₹{pnl:.2f}")
        print(f"    📊 Trade Parameters: Stop Loss: {abs(stop_loss_pct)}%, Profit Target: {profit_target_pct}%, Max Hold: {max_hold_time_minutes}min, Min Hold: {min_hold_time_minutes}min")
        print(f"    💰 Entry Price: ₹{actual_entry_price} | Live Price: ₹{live_price} | P&L: ₹{pnl:.2f}")
        
        # Get actual entry time from trade storage
        entry_time = None
        if trade_data and trade_data.get('entry_time'):
            try:
                entry_time = datetime.fromisoformat(trade_data['entry_time'].replace('Z', '+00:00'))
                if entry_time.tzinfo is None:
                    entry_time = IST_TIMEZONE.localize(entry_time)
            except Exception as e:
                print(f"    ⚠️  Error parsing entry time: {e}")
                entry_time = None
        
        # Fallback to simulated entry time if not available
        if entry_time is None:
            if symbol not in self.entry_times:
                self.entry_times[symbol] = now
            entry_time = self.entry_times[symbol]
        
        minutes_since_entry = (now - entry_time).total_seconds() // 60 if entry_time else None
        
        print(f"    ⏰ Entry Time: {entry_time.strftime('%H:%M:%S')} | Minutes Since Entry: {int(minutes_since_entry) if minutes_since_entry else 'Unknown'}")
        
        # Check max hold time
        if minutes_since_entry is not None and minutes_since_entry >= max_hold_time_minutes:
            # Check if opportunity hunter wants this position before exiting
            action = 'LONG_CALL' if 'CE' in symbol else 'LONG_PUT' if 'PE' in symbol else 'UNKNOWN'
            if should_avoid_exit_due_to_intended_trade(symbol, action):
                print(f"  🤔 MAX HOLD TIME: Avoiding exit on {symbol} - Opportunity hunter wants this position")
                return
            
            print(f"  ⏰ MAX HOLD TIME REACHED: Exiting {symbol} after {int(minutes_since_entry)} minutes (max: {max_hold_time_minutes}min).")
            self.exit_position(symbol, quantity, trade_id, f"Max Hold Time ({max_hold_time_minutes} minutes)")
            self.recently_closed[symbol] = now
            return
        
        # Absolute profit target
        if abs(quantity) % LOT_SIZE == 0 and pnl >= ABSOLUTE_PROFIT_PER_LOT * (abs(quantity) // LOT_SIZE):
            # Check if opportunity hunter wants this position before exiting
            action = 'LONG_CALL' if 'CE' in symbol else 'LONG_PUT' if 'PE' in symbol else 'UNKNOWN'
            if should_avoid_exit_due_to_intended_trade(symbol, action):
                print(f"  🤔 ABSOLUTE PROFIT: Avoiding exit on {symbol} - Opportunity hunter wants this position")
                return
            
            print(f"  💰 ABSOLUTE PROFIT TARGET HIT: Exiting {symbol} at ₹{pnl:.2f} profit (Qty: {quantity}).")
            self.exit_position(symbol, quantity, trade_id, f"Absolute Profit Target (₹{ABSOLUTE_PROFIT_PER_LOT} per lot)")
            self.recently_closed[symbol] = now
            return
        
        # Percentage profit target (from trade parameters)
        if pnl_percentage >= profit_target_pct:
            # Check if opportunity hunter wants this position before exiting
            action = 'LONG_CALL' if 'CE' in symbol else 'LONG_PUT' if 'PE' in symbol else 'UNKNOWN'
            if should_avoid_exit_due_to_intended_trade(symbol, action):
                print(f"  🤔 PROFIT TARGET: Avoiding exit on {symbol} - Opportunity hunter wants this position")
                return
            
            print(f"  ✅ PROFIT TARGET HIT: Exiting {symbol} at {pnl_percentage:.2f}% profit (target: {profit_target_pct}%).")
            self.exit_position(symbol, quantity, trade_id, f"Profit Target ({profit_target_pct}%)")
            self.recently_closed[symbol] = now
            return
        
        # Minimum hold time check (unless extreme loss)
        if minutes_since_entry is not None and minutes_since_entry < min_hold_time_minutes:
            if pnl_percentage <= EXTREME_LOSS_PCT:
                # Check if opportunity hunter wants this position before exiting
                action = 'LONG_CALL' if 'CE' in symbol else 'LONG_PUT' if 'PE' in symbol else 'UNKNOWN'
                if should_avoid_exit_due_to_intended_trade(symbol, action):
                    print(f"  🤔 EXTREME LOSS: Avoiding exit on {symbol} - Opportunity hunter wants this position")
                    return
                
                print(f"  🚨 EXTREME LOSS: Exiting {symbol} at {pnl_percentage:.2f}% loss (within min hold time).")
                self.exit_position(symbol, quantity, trade_id, "Extreme Loss (Min hold override)")
                self.recently_closed[symbol] = now
                return
            elif pnl_percentage <= stop_loss_pct:
                print(f"  ⏳ MIN HOLD TIME: Holding {symbol} (opened {int(minutes_since_entry)} min ago, loss {pnl_percentage:.2f}%). Not exiting unless extreme loss.")
                return
        
        # Cool-off for loss exits only (after minimum hold time)
        if minutes_since_entry is not None and minutes_since_entry < COOL_OFF_MINUTES:
            if pnl_percentage <= EXTREME_LOSS_PCT:
                # Check if opportunity hunter wants this position before exiting
                action = 'LONG_CALL' if 'CE' in symbol else 'LONG_PUT' if 'PE' in symbol else 'UNKNOWN'
                if should_avoid_exit_due_to_intended_trade(symbol, action):
                    print(f"  🤔 EXTREME LOSS: Avoiding exit on {symbol} - Opportunity hunter wants this position")
                    return
                
                print(f"  🚨 EXTREME LOSS: Exiting {symbol} at {pnl_percentage:.2f}% loss (within {COOL_OFF_MINUTES} min of entry).")
                self.exit_position(symbol, quantity, trade_id, "Extreme Loss (Cool-off override)")
                self.recently_closed[symbol] = now
                return
            elif pnl_percentage <= stop_loss_pct:
                print(f"  ⏳ COOL-OFF: Holding {symbol} (opened {int(minutes_since_entry)} min ago, loss {pnl_percentage:.2f}%). Not exiting unless extreme loss.")
                return
        
        # Stop-loss (after minimum hold time and cool-off)
        if pnl_percentage <= stop_loss_pct:
            # Check if opportunity hunter wants this position before exiting
            action = 'LONG_CALL' if 'CE' in symbol else 'LONG_PUT' if 'PE' in symbol else 'UNKNOWN'
            if should_avoid_exit_due_to_intended_trade(symbol, action):
                print(f"  🤔 STOP-LOSS: Avoiding exit on {symbol} - Opportunity hunter wants this position")
                return
            
            print(f"  ❌ STOP-LOSS HIT: Exiting {symbol} at {pnl_percentage:.2f}% loss (stop: {abs(stop_loss_pct)}%).")
            self.exit_position(symbol, quantity, trade_id, f"Stop-Loss ({abs(stop_loss_pct)}%)")
            self.recently_closed[symbol] = now
            return
        
        # Cool-off: skip re-entry for recently closed symbols
        last_closed = self.recently_closed.get(symbol)
        if last_closed and (now - last_closed) < timedelta(minutes=COOL_OFF_MINUTES):
            print(f"  ⏳ COOL-OFF: Skipping re-entry for {symbol} (closed {int((now - last_closed).total_seconds()//60)} min ago)")
            return
        
        print(f"  -> HOLDING {symbol}. No exit criteria met.")

    def exit_position(self, symbol, quantity, trade_id, reason):
        print(f"  🔄 EXIT: {symbol} | Qty: {quantity} | Reason: {reason}")
        from core_tools.execution_portfolio_tools import execute_options_strategy
        
        action = 'SELL' if quantity > 0 else 'BUY'
        closing_leg = [{
            'symbol': symbol,
            'action': action,
            'quantity': abs(quantity)
        }]
        
        # Execute the exit order
        result = execute_options_strategy(closing_leg, order_type='Closing')
        print(f"    📝 Exit result: {json.dumps(result, indent=2)}")
        
        # Update trade storage if execution was successful
        if result.get('status') == 'BASKET_SUCCESS':
            try:
                # Import trade storage functions
                from core_tools.trade_storage import update_trade_status, move_trade_to_history
                
                # Calculate exit price from the result
                exit_price = None
                if result.get('settlement_details') and result['settlement_details'].get('settled_orders'):
                    exit_price = result['settlement_details']['settled_orders'][0].get('avg_price')
                
                # Calculate P&L
                pnl = 0.0
                if exit_price:
                    # Get the trade data to calculate P&L
                    from core_tools.trade_storage import get_active_trades
                    active_trades = get_active_trades()
                    if trade_id in active_trades:
                        trade_data = active_trades[trade_id]
                        entry_price = None
                        
                        # Get entry price from execution details
                        execution_details = trade_data.get('execution_details', {})
                        settlement_details = execution_details.get('settlement_details', {})
                        settled_orders = settlement_details.get('settled_orders', [])
                        
                        if settled_orders:
                            entry_price = settled_orders[0].get('avg_price')
                        
                        if entry_price:
                            pnl = (exit_price - entry_price) * abs(quantity)
                
                # Update trade status to closed
                update_result = update_trade_status(
                    trade_id=trade_id,
                    status='CLOSED',
                    pnl=pnl,
                    exit_reason=reason,
                    exit_price=exit_price
                )
                
                if update_result.get('status') == 'SUCCESS':
                    print(f"    ✅ Trade {trade_id} marked as CLOSED in storage")
                    
                    # Move to history
                    if trade_id in active_trades:
                        move_trade_to_history(trade_id, active_trades[trade_id])
                        print(f"    📚 Trade {trade_id} moved to history")
                else:
                    print(f"    ⚠️  Failed to update trade storage: {update_result.get('message')}")
                    
            except Exception as e:
                print(f"    ⚠️  Error updating trade storage: {str(e)}")
        else:
            print(f"    ❌ Exit execution failed: {result.get('message')}")

    def find_trade_id_for_symbol(self, symbol):
        """Find the actual trade ID for a given symbol from active trades."""
        try:
            from core_tools.trade_storage import get_active_trades
            active_trades = get_active_trades()
            
            # Look for the symbol in active trades
            for trade_id, trade_data in active_trades.items():
                legs = trade_data.get('legs', [])
                for leg in legs:
                    if leg.get('symbol') == symbol:
                        return trade_id
            
            # If not found, return None
            return None
            
        except Exception as e:
            print(f"    ⚠️  Error finding trade ID for {symbol}: {str(e)}")
            return None

def get_recent_intended_trades(symbol: str = None, minutes_back: int = 30) -> List[Dict[str, Any]]:
    """
    Get recent intended trades from opportunity hunter, optionally filtered by symbol.
    Used to check if opportunity hunter wants this position before exiting.
    """
    try:
        log_file = os.path.join(os.path.dirname(__file__), 'intended_trades.jsonl')
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

def should_avoid_exit_due_to_intended_trade(symbol: str, action: str, minutes_back: int = 7) -> bool:
    """
    Check if we should avoid exiting a position because opportunity hunter wants it.
    Returns True if we should avoid exit, False if we should proceed with exit.
    
    Args:
        symbol: The symbol to check
        action: The action type (LONG_CALL, LONG_PUT, etc.)
        minutes_back: Time window to check for intended trades (default: 7 minutes)
    """
    try:
        # Get recent intended trades for this symbol
        recent_intended = get_recent_intended_trades(symbol, minutes_back=minutes_back)
        
        if not recent_intended:
            return False  # No recent intended trades, safe to exit
        
        # Sort by timestamp to get the most recent one first
        recent_intended.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        
        # Check only the MOST RECENT intended trade
        most_recent = recent_intended[0]
        intended_action = most_recent.get('action', '')
        intended_symbol = most_recent.get('symbol', '')
        
        # If opportunity hunter's MOST RECENT intended trade matches our position, avoid exit
        if intended_symbol == symbol and intended_action == action:
            print(f"🤔 Avoiding exit on {symbol} - Opportunity hunter's MOST RECENT intended trade was {action} (within {minutes_back} minutes)")
            return True
        
        return False
        
    except Exception as e:
        print(f"⚠️ Error checking intended trades: {e}")
        return False  # On error, proceed with normal exit logic

if __name__ == "__main__":
    mgr = SimplePositionManager()
    mgr.manage_positions() 