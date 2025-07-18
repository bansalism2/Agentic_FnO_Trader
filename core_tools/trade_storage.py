#!/usr/bin/env python3
"""
Trade Storage System
===================

Common file storage system for communication between Opportunity Hunter and Position Manager.
- Opportunity Hunter writes successful trades here
- Position Manager reads and manages trades from here
"""

import json
import os
import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path

# Storage directory
STORAGE_DIR = Path(__file__).parent.parent / "trade_storage"
TRADES_FILE = STORAGE_DIR / "active_trades.json"
TRADE_HISTORY_FILE = STORAGE_DIR / "trade_history.json"
TRADE_LOG_FILE = STORAGE_DIR / "trade_log.txt"

# Ensure storage directory exists
STORAGE_DIR.mkdir(exist_ok=True)

def write_successful_trade(trade_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Write a successful trade to storage (called by Opportunity Hunter).
    
    Args:
        trade_data: Complete trade information including:
            - strategy_name: Name of the strategy
            - legs: List of strategy legs with symbols, quantities, prices
            - entry_time: Timestamp when trade was executed
            - expiry_date: Options expiry date
            - spot_price: NIFTY spot price at entry
            - strategy_type: Type of strategy (straddle, strangle, etc.)
            - risk_management: Stop loss and target levels
            - analysis_data: Market analysis that led to the trade
    
    Returns:
        Dict with status and trade_id
    """
    try:
        # Generate unique trade ID
        trade_id = f"TRADE_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(get_active_trades()) + 1}"
        
        # Add trade metadata
        trade_data.update({
            'trade_id': trade_id,
            'status': 'ACTIVE',
            'entry_time': datetime.datetime.now().isoformat(),
            'last_updated': datetime.datetime.now().isoformat(),
            'pnl': 0.0,
            'exit_time': None,
            'exit_reason': None,
            'exit_price': None
        })
        
        # Read existing trades
        active_trades = get_active_trades()
        
        # Add new trade
        active_trades[trade_id] = trade_data
        
        # Write back to file
        with open(TRADES_FILE, 'w') as f:
            json.dump(active_trades, f, indent=2)
        
        # Log the trade
        log_trade_action(trade_id, "TRADE_OPENED", trade_data)
        
        return {
            'status': 'SUCCESS',
            'trade_id': trade_id,
            'message': f'Trade {trade_id} successfully stored'
        }
        
    except Exception as e:
        return {
            'status': 'ERROR',
            'message': f'Failed to write trade: {str(e)}'
        }

def get_active_trades() -> Dict[str, Any]:
    """
    Get all active trades from storage (called by Position Manager).
    
    Returns:
        Dict of active trades with trade_id as key
    """
    try:
        if not TRADES_FILE.exists():
            return {}
        
        with open(TRADES_FILE, 'r') as f:
            return json.load(f)
            
    except Exception as e:
        print(f"Error reading active trades: {e}")
        return {}

def update_trade_status(trade_id: str, status: str, pnl: float = None, 
                       exit_reason: str = None, exit_price: float = None) -> Dict[str, Any]:
    """
    Update trade status (called by Position Manager).
    
    Args:
        trade_id: Unique trade identifier
        status: New status (ACTIVE, CLOSED, PARTIALLY_CLOSED)
        pnl: Current P&L
        exit_reason: Reason for exit (if closing)
        exit_price: Exit price (if closing)
    
    Returns:
        Dict with status
    """
    try:
        active_trades = get_active_trades()
        
        if trade_id not in active_trades:
            return {
                'status': 'ERROR',
                'message': f'Trade {trade_id} not found'
            }
        
        # Update trade data
        trade = active_trades[trade_id]
        trade['status'] = status
        trade['last_updated'] = datetime.datetime.now().isoformat()
        
        if pnl is not None:
            trade['pnl'] = pnl
        
        if status == 'CLOSED':
            trade['exit_time'] = datetime.datetime.now().isoformat()
            trade['exit_reason'] = exit_reason
            trade['exit_price'] = exit_price
            
            # Move to history
            move_trade_to_history(trade_id, trade)
            del active_trades[trade_id]
        else:
            # Update in active trades
            active_trades[trade_id] = trade
        
        # Write back to file
        with open(TRADES_FILE, 'w') as f:
            json.dump(active_trades, f, indent=2)
        
        # Log the action
        log_trade_action(trade_id, f"STATUS_UPDATED_{status}", {
            'status': status, 'pnl': pnl, 'exit_reason': exit_reason
        })
        
        return {
            'status': 'SUCCESS',
            'message': f'Trade {trade_id} status updated to {status}'
        }
        
    except Exception as e:
        return {
            'status': 'ERROR',
            'message': f'Failed to update trade status: {str(e)}'
        }

def move_trade_to_history(trade_id: str, trade_data: Dict[str, Any]) -> None:
    """Move closed trade to history file."""
    try:
        # Read existing history
        history = {}
        if TRADE_HISTORY_FILE.exists():
            with open(TRADE_HISTORY_FILE, 'r') as f:
                history = json.load(f)
        
        # Add to history
        history[trade_id] = trade_data
        
        # Write back
        with open(TRADE_HISTORY_FILE, 'w') as f:
            json.dump(history, f, indent=2)
            
    except Exception as e:
        print(f"Error moving trade to history: {e}")

def get_trade_history(days: int = 30) -> Dict[str, Any]:
    """
    Get trade history for the last N days.
    
    Args:
        days: Number of days to look back
    
    Returns:
        Dict of historical trades
    """
    try:
        if not TRADE_HISTORY_FILE.exists():
            return {}
        
        with open(TRADE_HISTORY_FILE, 'r') as f:
            history = json.load(f)
        
        # Filter by date if needed
        if days > 0:
            cutoff_date = datetime.datetime.now() - datetime.timedelta(days=days)
            filtered_history = {}
            
            for trade_id, trade in history.items():
                entry_time = datetime.datetime.fromisoformat(trade.get('entry_time', '2020-01-01'))
                if entry_time >= cutoff_date:
                    filtered_history[trade_id] = trade
            
            return filtered_history
        
        return history
        
    except Exception as e:
        print(f"Error reading trade history: {e}")
        return {}

def log_trade_action(trade_id: str, action: str, data: Dict[str, Any]) -> None:
    """Log trade actions for debugging and audit."""
    try:
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_entry = f"[{timestamp}] {trade_id} - {action}: {json.dumps(data, default=str)}\n"
        
        with open(TRADE_LOG_FILE, 'a') as f:
            f.write(log_entry)
            
    except Exception as e:
        print(f"Error logging trade action: {e}")

def get_trade_summary() -> Dict[str, Any]:
    """
    Get summary of all trades (active and recent history).
    
    Returns:
        Dict with trade summary statistics
    """
    try:
        active_trades = get_active_trades()
        recent_history = get_trade_history(days=7)
        
        total_active = len(active_trades)
        total_recent = len(recent_history)
        
        # Calculate P&L
        active_pnl = sum(trade.get('pnl', 0) for trade in active_trades.values())
        recent_pnl = sum(trade.get('pnl', 0) for trade in recent_history.values())
        
        # Strategy breakdown
        active_strategies = {}
        for trade in active_trades.values():
            strategy = trade.get('strategy_name', 'Unknown')
            active_strategies[strategy] = active_strategies.get(strategy, 0) + 1
        
        return {
            'active_trades_count': total_active,
            'active_pnl': active_pnl,
            'recent_trades_count': total_recent,
            'recent_pnl': recent_pnl,
            'active_strategies': active_strategies,
            'last_updated': datetime.datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            'status': 'ERROR',
            'message': f'Failed to get trade summary: {str(e)}'
        }

def clear_storage() -> Dict[str, Any]:
    """
    Clear all trade storage (for testing/reset purposes).
    
    Returns:
        Dict with status
    """
    try:
        if TRADES_FILE.exists():
            TRADES_FILE.unlink()
        if TRADE_HISTORY_FILE.exists():
            TRADE_HISTORY_FILE.unlink()
        if TRADE_LOG_FILE.exists():
            TRADE_LOG_FILE.unlink()
        
        return {
            'status': 'SUCCESS',
            'message': 'Trade storage cleared'
        }
        
    except Exception as e:
        return {
            'status': 'ERROR',
            'message': f'Failed to clear storage: {str(e)}'
        }

# Test function
def test_trade_storage():
    """Test the trade storage system."""
    print("Testing Trade Storage System...")
    
    # Test writing a trade
    test_trade = {
        'strategy_name': 'Long Straddle',
        'legs': [
            {'symbol': 'NIFTY2571025450CE', 'action': 'BUY', 'quantity': 75, 'price': 125.5},
            {'symbol': 'NIFTY2571025450PE', 'action': 'BUY', 'quantity': 75, 'price': 107.15}
        ],
        'expiry_date': '2025-07-10',
        'spot_price': 25461.3,
        'strategy_type': 'straddle',
        'risk_management': {
            'stop_loss': 0.5,  # 50% of premium
            'target': 2.0      # 200% of premium
        },
        'analysis_data': {
            'volatility_regime': 'COMPRESSED',
            'iv_rank': 0.25,
            'pcr': 1.2,
            'technical_signal': 'NEUTRAL'
        }
    }
    
    result = write_successful_trade(test_trade)
    print(f"Write result: {result}")
    
    # Test reading trades
    active_trades = get_active_trades()
    print(f"Active trades: {len(active_trades)}")
    
    # Test summary
    summary = get_trade_summary()
    print(f"Summary: {summary}")

if __name__ == "__main__":
    test_trade_storage() 