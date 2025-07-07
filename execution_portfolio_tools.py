#!/usr/bin/env python3
"""
NIFTY Options - Execution & Portfolio Tools (File 4 of 4)
========================================================

Trading execution, portfolio management, and reporting tools.
This file contains functions for order execution, position monitoring, and analytics.

Dependencies:
Requires nifty_connection_tools.py for market access

Author: AI Assistant
"""

import json
import datetime as dt
from typing import Dict, List, Optional, Any

# Global variable to hold connect_data_tools module
connect_data_tools = None

# Import connection tools (assumes file 1 is available)
try:
    import connect_data_tools
    from connect_data_tools import (
        initialize_connection, get_nifty_spot_price, 
        get_options_chain, analyze_options_flow,
        get_historical_volatility
    )
    
    # Initialize connection if not already done
    if not hasattr(connect_data_tools, '_kite_instance') or connect_data_tools._kite_instance is None:
        # Try to load credentials and initialize
        try:
            from dotenv import load_dotenv
            import os
            load_dotenv(dotenv_path='./.env')
            api_key = os.getenv("kite_api_key")
            api_secret = os.getenv("kite_api_secret")
            access_token = None
            try:
                with open("access_token.txt", "r") as f:
                    access_token = f.read().strip()
            except Exception:
                pass
            
            if api_key and (api_secret or access_token):
                init_result = initialize_connection(api_key, api_secret, access_token)
                print(f"Connection initialization result: {init_result}")
        except Exception as e:
            print(f"Warning: Failed to initialize connection in execution tools: {e}")
            
except ImportError as e:
    print(f"Warning: Connection tools not available. Some functions may not work. Error: {e}")
    connect_data_tools = None


def execute_options_strategy(strategy_legs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Execute a multi-leg options strategy.
    
    Args:
        strategy_legs: List of legs with symbol, action, quantity, etc.
    
    Returns:
        Dict with execution results for each leg
    """
    try:
        if connect_data_tools is None:
            return {'status': 'ERROR', 'message': 'Connection tools not available'}
        
        if not hasattr(connect_data_tools, '_kite_instance') or connect_data_tools._kite_instance is None:
            return {'status': 'ERROR', 'message': 'Connection not initialized'}
        
        if not strategy_legs:
            return {'status': 'ERROR', 'message': 'No strategy legs provided'}
        
        execution_results = []
        successful_orders = 0
        
        for i, leg in enumerate(strategy_legs):
            try:
                # Validate leg data
                required_fields = ['symbol', 'action', 'quantity']
                if not all(field in leg for field in required_fields):
                    execution_results.append({
                        'leg_number': i + 1,
                        'symbol': leg.get('symbol', 'Unknown'),
                        'status': 'FAILED',
                        'message': 'Missing required fields',
                        'order_id': None
                    })
                    continue
                
                # Place order
                order_params = {
                    'variety': 'regular',
                    'exchange': leg.get('exchange', 'NFO'),
                    'tradingsymbol': leg['symbol'],
                    'transaction_type': leg['action'],
                    'quantity': leg['quantity'],
                    'product': leg.get('product', 'MIS'),  # Default to MIS (intraday)
                    'order_type': leg.get('order_type', 'MARKET'),
                    'validity': leg.get('validity', 'DAY')
                }
                
                # Add price for limit orders
                if leg.get('order_type') == 'LIMIT' and 'price' in leg:
                    order_params['price'] = leg['price']
                
                order_id = connect_data_tools._kite_instance.place_order(**order_params)
                
                execution_results.append({
                    'leg_number': i + 1,
                    'symbol': leg['symbol'],
                    'action': leg['action'],
                    'quantity': leg['quantity'],
                    'status': 'SUCCESS',
                    'message': 'Order placed successfully',
                    'order_id': order_id
                })
                
                successful_orders += 1
                
            except Exception as e:
                execution_results.append({
                    'leg_number': i + 1,
                    'symbol': leg.get('symbol', 'Unknown'),
                    'status': 'FAILED',
                    'message': f'Order placement failed: {str(e)}',
                    'order_id': None
                })
        
        return {
            'status': 'SUCCESS' if successful_orders > 0 else 'FAILED',
            'total_legs': len(strategy_legs),
            'successful_orders': successful_orders,
            'failed_orders': len(strategy_legs) - successful_orders,
            'execution_results': execution_results,
            'timestamp': dt.datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            'status': 'ERROR',
            'message': f'Strategy execution failed: {str(e)}'
        }


def get_order_status(order_id: str) -> Dict[str, Any]:
    """
    Get status of a specific order.
    
    Args:
        order_id: Order ID to check
    
    Returns:
        Dict with order status and details
    """
    try:
        if connect_data_tools is None:
            return {'status': 'ERROR', 'message': 'Connection tools not available'}
        
        if not hasattr(connect_data_tools, '_kite_instance') or connect_data_tools._kite_instance is None:
            return {'status': 'ERROR', 'message': 'Connection not initialized'}
        
        order_history = connect_data_tools._kite_instance.order_history(order_id)
        
        if not order_history:
            return {'status': 'ERROR', 'message': 'Order not found'}
        
        latest_order = order_history[-1]  # Latest update
        
        return {
            'status': 'SUCCESS',
            'order_id': order_id,
            'order_status': latest_order.get('status'),
            'symbol': latest_order.get('tradingsymbol'),
            'quantity': latest_order.get('quantity'),
            'price': latest_order.get('price', 0),
            'average_price': latest_order.get('average_price', 0),
            'filled_quantity': latest_order.get('filled_quantity', 0),
            'pending_quantity': latest_order.get('pending_quantity', 0),
            'order_type': latest_order.get('order_type'),
            'transaction_type': latest_order.get('transaction_type'),
            'exchange': latest_order.get('exchange'),
            'product': latest_order.get('product'),
            'order_timestamp': str(latest_order.get('order_timestamp')),
            'exchange_timestamp': str(latest_order.get('exchange_timestamp')),
            'status_message': latest_order.get('status_message', ''),
            'history_count': len(order_history),
            'timestamp': dt.datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            'status': 'ERROR',
            'message': f'Failed to get order status: {str(e)}'
        }


def cancel_order(order_id: str, variety: str = 'regular') -> Dict[str, Any]:
    """
    Cancel a pending order.
    
    Args:
        order_id: Order ID to cancel
        variety: Order variety (regular, bo, co, amo)
    
    Returns:
        Dict with cancellation status
    """
    try:
        if connect_data_tools is None:
            return {'status': 'ERROR', 'message': 'Connection tools not available'}
        
        if not hasattr(connect_data_tools, '_kite_instance') or connect_data_tools._kite_instance is None:
            return {'status': 'ERROR', 'message': 'Connection not initialized'}
        
        cancelled_order_id = connect_data_tools._kite_instance.cancel_order(variety=variety, order_id=order_id)
        
        return {
            'status': 'SUCCESS',
            'message': 'Order cancelled successfully',
            'original_order_id': order_id,
            'cancelled_order_id': cancelled_order_id,
            'timestamp': dt.datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            'status': 'ERROR',
            'message': f'Order cancellation failed: {str(e)}',
            'order_id': order_id
        }


def get_portfolio_positions() -> Dict[str, Any]:
    """
    Get all open NIFTY positions and their P&L.
    
    **IMPORTANT**: Only positions with quantity > 0 are considered as "open positions".
    Positions with quantity = 0 are considered closed and are excluded from the count.
    
    Returns:
        Dict with current positions and P&L
    """
    try:
        if connect_data_tools is None:
            return {'status': 'ERROR', 'message': 'Connection tools not available'}
        
        if not hasattr(connect_data_tools, '_kite_instance') or connect_data_tools._kite_instance is None:
            return {'status': 'ERROR', 'message': 'Connection not initialized'}
        
        positions = connect_data_tools._kite_instance.positions()
        
        # Filter NIFTY options positions
        nifty_positions = []
        closed_positions = []  # Track closed positions for reference
        total_pnl = 0
        
        for pos in positions.get('net', []):
            if 'NIFTY' in pos.get('tradingsymbol', ''):
                quantity = pos.get('quantity', 0)
                pnl = pos.get('pnl', 0)
                
                position_data = {
                    'symbol': pos.get('tradingsymbol'),
                    'quantity': quantity,
                    'average_price': pos.get('average_price', 0),
                    'last_price': pos.get('last_price', 0),
                    'pnl': pnl,
                    'product': pos.get('product'),
                    'exchange': pos.get('exchange'),
                    'instrument_type': pos.get('instrument_type')
                }
                
                # Determine position status based on quantity
                if quantity > 0:
                    position_data['current_status'] = 'Open: Active position'
                    nifty_positions.append(position_data)
                    total_pnl += pnl
                elif quantity == 0:
                    position_data['current_status'] = 'Closed: No position opened'
                    closed_positions.append(position_data)
                    # Still include P&L for closed positions in total
                    total_pnl += pnl
        
        # If no open positions but closed positions exist, provide clear status
        if len(nifty_positions) == 0 and len(closed_positions) > 0:
            return {
                'status': 'SUCCESS',
                'total_positions': 0,  # No open positions
                'total_pnl': round(total_pnl, 2),
                'positions': [],  # No open positions
                'closed_positions': closed_positions,  # Reference to closed positions
                'portfolio_status': 'All positions closed - clean slate for new trades',
                'timestamp': dt.datetime.now().isoformat()
            }
        elif len(nifty_positions) == 0 and len(closed_positions) == 0:
            return {
                'status': 'SUCCESS',
                'total_positions': 0,
                'total_pnl': 0,
                'positions': [],
                'portfolio_status': 'No NIFTY positions found - clean slate for new trades',
                'timestamp': dt.datetime.now().isoformat()
            }
        else:
            return {
                'status': 'SUCCESS',
                'total_positions': len(nifty_positions),  # Only count open positions
                'total_pnl': round(total_pnl, 2),
                'positions': nifty_positions,  # Only open positions
                'closed_positions': closed_positions,  # Reference to closed positions
                'portfolio_status': f'{len(nifty_positions)} open positions, {len(closed_positions)} closed positions',
                'timestamp': dt.datetime.now().isoformat()
            }
        
    except Exception as e:
        return {
            'status': 'ERROR',
            'message': f'Failed to get positions: {str(e)}'
        }


def get_account_margins() -> Dict[str, Any]:
    """
    Get account margin information.
    
    Returns:
        Dict with margin details
        Note that if intraday_payin is there then it means that the money has been added today and might not reflect in available_cash
        and it will be reflected in the next day's available_cash but its available for trading today
    """
    try:
        if connect_data_tools is None:
            return {'status': 'ERROR', 'message': 'Connection tools not available'}
        
        if not hasattr(connect_data_tools, '_kite_instance') or connect_data_tools._kite_instance is None:
            return {'status': 'ERROR', 'message': 'Connection not initialized'}
        
        margins = connect_data_tools._kite_instance.margins()
        equity_margins = margins.get('equity', {})
        # Calculate true unrealized P&L from open NIFTY positions
        positions_result = get_portfolio_positions()
        if positions_result.get('status') == 'SUCCESS':
            unrealized_pnl = positions_result.get('total_pnl', 0)
        else:
            unrealized_pnl = 0
        return {
            'status': 'SUCCESS',
            'equity': {
                'available_cash': equity_margins.get('available', {}).get('cash', 0),
                'opening_balance': equity_margins.get('available', {}).get('opening_balance', 0),
                'live_balance': equity_margins.get('available', {}).get('live_balance', 0),
                'used_margin': equity_margins.get('used', {}).get('debits', 0),
                'intraday_payin': equity_margins.get('available', {}).get('intraday_payin', 0),
                'unrealized_pnl': unrealized_pnl
            },
            'timestamp': dt.datetime.now().isoformat()
        }
    
    except Exception as e:
        return {
            'status': 'ERROR',
            'message': f'Failed to get margins: {str(e)}'
        }


def calculate_strategy_margins(strategy_legs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate margin requirement for a strategy.
    
    Args:
        strategy_legs: List of strategy legs with order details
    
    Returns:
        Dict with margin calculations
    """
    try:
        if connect_data_tools is None:
            return {'status': 'ERROR', 'message': 'Connection tools not available'}
        
        if not hasattr(connect_data_tools, '_kite_instance') or connect_data_tools._kite_instance is None:
            return {'status': 'ERROR', 'message': 'Connection not initialized'}
        
        # Convert strategy legs to margin calculation format
        orders = []
        for leg in strategy_legs:
            order = {
                'exchange': leg.get('exchange', 'NFO'),
                'tradingsymbol': leg['symbol'],
                'quantity': leg['quantity'],
                'product': leg.get('product', 'MIS'),
                'order_type': leg.get('order_type', 'MARKET'),
                'transaction_type': leg['action']
            }
            
            if 'price' in leg:
                order['price'] = leg['price']
                
            orders.append(order)
        
        # Calculate margins
        margin_data = connect_data_tools._kite_instance.order_margins(orders)
        
        total_margin = sum(order.get('total', 0) for order in margin_data)
        
        return {
            'status': 'SUCCESS',
            'total_margin_required': round(total_margin, 2),
            'individual_margins': [
                {
                    'symbol': order.get('tradingsymbol'),
                    'margin': round(order.get('total', 0), 2),
                    'span': round(order.get('span', 0), 2),
                    'exposure': round(order.get('exposure', 0), 2)
                }
                for order in margin_data
            ],
            'timestamp': dt.datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            'status': 'ERROR',
            'message': f'Margin calculation failed: {str(e)}'
        }


def monitor_strategy_pnl(strategy_legs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Monitor real-time P&L of a strategy.
    
    Args:
        strategy_legs: List of strategy legs with entry details
    
    Returns:
        Dict with current P&L analysis
    """
    try:
        if connect_data_tools is None:
            return {'status': 'ERROR', 'message': 'Connection tools not available'}
        
        if not hasattr(connect_data_tools, '_kite_instance') or connect_data_tools._kite_instance is None:
            return {'status': 'ERROR', 'message': 'Connection not initialized'}
        
        # Get current quotes for all legs
        symbols = [f"NFO:{leg['symbol']}" for leg in strategy_legs]
        
        quotes = connect_data_tools._kite_instance.quote(symbols)
        
        total_pnl = 0
        leg_pnls = []
        
        for leg in strategy_legs:
            symbol_key = f"NFO:{leg['symbol']}"
            if symbol_key in quotes:
                current_price = quotes[symbol_key]['last_price']
                entry_price = leg['price']
                quantity = leg['quantity']
                
                if leg['action'] == 'BUY':
                    leg_pnl = (current_price - entry_price) * quantity
                else:  # SELL
                    leg_pnl = (entry_price - current_price) * quantity
                
                leg_pnls.append({
                    'symbol': leg['symbol'],
                    'action': leg['action'],
                    'quantity': quantity,
                    'entry_price': entry_price,
                    'current_price': current_price,
                    'pnl': round(leg_pnl, 2),
                    'pnl_percentage': round((leg_pnl / (entry_price * quantity)) * 100, 2) if entry_price > 0 else 0
                })
                
                total_pnl += leg_pnl
        
        return {
            'status': 'SUCCESS',
            'strategy_pnl': {
                'total_pnl': round(total_pnl, 2),
                'leg_count': len(leg_pnls),
                'profitable_legs': len([leg for leg in leg_pnls if leg['pnl'] > 0]),
                'losing_legs': len([leg for leg in leg_pnls if leg['pnl'] < 0])
            },
            'leg_details': leg_pnls,
            'timestamp': dt.datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            'status': 'ERROR',
            'message': f'Failed to monitor strategy P&L: {str(e)}'
        }


def get_orders_history(days: int = 1) -> Dict[str, Any]:
    """
    Get trading orders history.
    
    Args:
        days: Number of days to look back (1 = today only)
    
    Returns:
        Dict with orders history
    """
    try:
        if connect_data_tools is None:
            return {'status': 'ERROR', 'message': 'Connection tools not available'}
        
        if not hasattr(connect_data_tools, '_kite_instance') or connect_data_tools._kite_instance is None:
            return {'status': 'ERROR', 'message': 'Connection not initialized'}
        
        # Get all orders for today
        orders = connect_data_tools._kite_instance.orders()
        
        # Filter NIFTY options orders
        nifty_orders = []
        for order in orders:
            if 'NIFTY' in order.get('tradingsymbol', ''):
                order_data = {
                    'order_id': order.get('order_id'),
                    'symbol': order.get('tradingsymbol'),
                    'action': order.get('transaction_type'),
                    'quantity': order.get('quantity'),
                    'price': order.get('price', 0),
                    'average_price': order.get('average_price', 0),
                    'status': order.get('status'),
                    'order_type': order.get('order_type'),
                    'product': order.get('product'),
                    'filled_quantity': order.get('filled_quantity', 0),
                    'order_timestamp': str(order.get('order_timestamp', '')),
                    'exchange_timestamp': str(order.get('exchange_timestamp', ''))
                }
                nifty_orders.append(order_data)
        
        # Calculate summary statistics
        total_orders = len(nifty_orders)
        completed_orders = len([o for o in nifty_orders if o['status'] == 'COMPLETE'])
        pending_orders = len([o for o in nifty_orders if o['status'] in ['OPEN', 'TRIGGER PENDING']])
        cancelled_orders = len([o for o in nifty_orders if o['status'] == 'CANCELLED'])
        
        return {
            'status': 'SUCCESS',
            'summary': {
                'total_orders': total_orders,
                'completed_orders': completed_orders,
                'pending_orders': pending_orders,
                'cancelled_orders': cancelled_orders,
                'success_rate': round((completed_orders / total_orders) * 100, 2) if total_orders > 0 else 0
            },
            'orders': nifty_orders,
            'timestamp': dt.datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            'status': 'ERROR',
            'message': f'Failed to get orders history: {str(e)}'
        }


def get_daily_trading_summary() -> Dict[str, Any]:
    """
    Get comprehensive daily trading summary.
    
    Returns:
        Dict with daily trading metrics and summary
    """
    try:
        # Get positions
        positions_result = get_portfolio_positions()
        
        # Get margins
        margins_result = get_account_margins()
        
        # Get spot price
        spot_result = get_nifty_spot_price()
        
        # Get volatility
        volatility_result = get_historical_volatility(30)
        
        # Get options flow for current week
        flow_result = analyze_options_flow()
        
        # Get orders history
        orders_result = get_orders_history()
        
        return {
            'status': 'SUCCESS',
            'date': dt.date.today().isoformat(),
            'summary': {
                'nifty_spot': spot_result.get('spot_price', 0) if spot_result.get('status') == 'SUCCESS' else 0,
                'total_positions': positions_result.get('total_positions', 0) if positions_result.get('status') == 'SUCCESS' else 0,
                'total_pnl': positions_result.get('total_pnl', 0) if positions_result.get('status') == 'SUCCESS' else 0,
                'available_margin': margins_result.get('equity', {}).get('available_cash', 0) if margins_result.get('status') == 'SUCCESS' else 0,
                'historical_volatility': volatility_result.get('historical_volatility', 0) if volatility_result.get('status') == 'SUCCESS' else 0,
                'put_call_ratio': flow_result.get('put_call_ratio', 0) if flow_result.get('status') == 'SUCCESS' else 0,
                'total_orders_today': orders_result.get('summary', {}).get('total_orders', 0) if orders_result.get('status') == 'SUCCESS' else 0,
                'completed_orders': orders_result.get('summary', {}).get('completed_orders', 0) if orders_result.get('status') == 'SUCCESS' else 0
            },
            'detailed_data': {
                'positions': positions_result,
                'margins': margins_result,
                'volatility': volatility_result,
                'options_flow': flow_result,
                'orders_summary': orders_result.get('summary', {}) if orders_result.get('status') == 'SUCCESS' else {}
            },
            'timestamp': dt.datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            'status': 'ERROR',
            'message': f'Failed to generate daily summary: {str(e)}'
        }


def export_trading_data(filepath: str, include_orders: bool = True, 
                       include_positions: bool = True) -> Dict[str, Any]:
    """
    Export trading data to file.
    
    Args:
        filepath: File path to save data
        include_orders: Include orders history
        include_positions: Include positions data
    
    Returns:
        Dict with export status
    """
    try:
        export_data = {
            'export_date': dt.datetime.now().isoformat(),
            'data_types': []
        }
        
        # Get daily summary
        summary = get_daily_trading_summary()
        export_data['daily_summary'] = summary
        export_data['data_types'].append('daily_summary')
        
        # Include orders if requested
        if include_orders:
            orders = get_orders_history()
            export_data['orders_history'] = orders
            export_data['data_types'].append('orders_history')
        
        # Include positions if requested
        if include_positions:
            positions = get_portfolio_positions()
            export_data['positions'] = positions
            export_data['data_types'].append('positions')
        
        # Save to file
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        return {
            'status': 'SUCCESS',
            'message': f'Trading data exported to {filepath}',
            'file_path': filepath,
            'data_types_exported': export_data['data_types'],
            'export_timestamp': export_data['export_date']
        }
        
    except Exception as e:
        return {
            'status': 'ERROR',
            'message': f'Failed to export data: {str(e)}'
        }


def get_risk_metrics() -> Dict[str, Any]:
    """
    Calculate portfolio risk metrics.
    
    Returns:
        Dict with risk analysis
    """
    try:
        # Get current positions
        positions_result = get_portfolio_positions()
        margins_result = get_account_margins()
        
        if positions_result['status'] != 'SUCCESS' or margins_result['status'] != 'SUCCESS':
            return {'status': 'ERROR', 'message': 'Failed to get required data for risk calculation'}
        
        positions = positions_result['positions']
        available_cash = margins_result['equity']['available_cash']
        
        # Calculate position-level metrics
        total_exposure = 0
        max_single_loss = 0
        options_count = 0
        
        for pos in positions:
            # Calculate exposure
            exposure = abs(pos['quantity'] * pos['average_price'])
            total_exposure += exposure
            
            # Track max single position loss potential
            if pos['pnl'] < max_single_loss:
                max_single_loss = pos['pnl']
            
            # Count options positions
            if pos.get('instrument_type') in ['CE', 'PE']:
                options_count += 1
        
        # Calculate risk ratios
        portfolio_value = available_cash + total_exposure
        exposure_ratio = (total_exposure / portfolio_value) * 100 if portfolio_value > 0 else 0
        
        # Risk assessment
        risk_level = 'Low'
        if exposure_ratio > 50:
            risk_level = 'High'
        elif exposure_ratio > 25:
            risk_level = 'Medium'
        
        warnings = []
        if exposure_ratio > 70:
            warnings.append('Very high exposure ratio - consider reducing positions')
        if options_count > 10:
            warnings.append('High number of options positions - complex to manage')
        if max_single_loss < -5000:
            warnings.append('Large single position loss - monitor closely')
        
        return {
            'status': 'SUCCESS',
            'risk_metrics': {
                'total_exposure': round(total_exposure, 2),
                'available_cash': round(available_cash, 2),
                'exposure_ratio_percent': round(exposure_ratio, 2),
                'max_single_position_loss': round(max_single_loss, 2),
                'total_positions': len(positions),
                'options_positions': options_count,
                'risk_level': risk_level
            },
            'risk_warnings': warnings,
            'recommendations': [
                'Keep exposure ratio below 50% for conservative risk management',
                'Limit number of simultaneous options positions',
                'Set stop losses for all positions',
                'Monitor Greeks exposure daily',
                'Maintain adequate cash reserves'
            ],
            'timestamp': dt.datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            'status': 'ERROR',
            'message': f'Risk metrics calculation failed: {str(e)}'
        }


# Execution and Portfolio Tools Registry
EXECUTION_PORTFOLIO_TOOLS = {
    'execute_options_strategy': execute_options_strategy,
    'get_order_status': get_order_status,
    'cancel_order': cancel_order,
    'get_portfolio_positions': get_portfolio_positions,
    'get_account_margins': get_account_margins,
    'calculate_strategy_margins': calculate_strategy_margins,
    'monitor_strategy_pnl': monitor_strategy_pnl,
    'get_orders_history': get_orders_history,
    'get_daily_trading_summary': get_daily_trading_summary,
    'export_trading_data': export_trading_data,
    'get_risk_metrics': get_risk_metrics
}


# Complete NIFTY Options Tools Registry (All 4 files combined)
def get_all_nifty_tools() -> Dict[str, Any]:
    """
    Get all available NIFTY options tools across all modules.
    
    Returns:
        Dict with complete tool registry and descriptions
    """
    all_tools = {}
    
    # Import tools from other files if available
    try:
        from nifty_connection_tools import CONNECTION_DATA_TOOLS
        all_tools.update(CONNECTION_DATA_TOOLS)
    except ImportError:
        pass
    
    try:
        from nifty_calculation_tools import CALCULATION_ANALYSIS_TOOLS
        all_tools.update(CALCULATION_ANALYSIS_TOOLS)
    except ImportError:
        pass
    
    try:
        from nifty_strategy_tools import STRATEGY_CREATION_TOOLS
        all_tools.update(STRATEGY_CREATION_TOOLS)
    except ImportError:
        pass
    
    # Add execution tools
    all_tools.update(EXECUTION_PORTFOLIO_TOOLS)
    
    return {
        'status': 'SUCCESS',
        'total_tools': len(all_tools),
        'tools_registry': all_tools,
        'categories': {
            'Connection & Data': ['initialize_connection', 'authenticate_session', 'get_nifty_spot_price', 'get_nifty_expiry_dates', 'get_options_chain', 'get_historical_volatility', 'analyze_options_flow'],
            'Calculations & Analysis': ['calculate_option_greeks', 'calculate_implied_volatility', 'calculate_strategy_pnl', 'find_arbitrage_opportunities', 'calculate_portfolio_greeks', 'calculate_volatility_surface'],
            'Strategy Creation': ['create_long_straddle_strategy', 'create_short_strangle_strategy', 'create_iron_condor_strategy', 'create_butterfly_spread_strategy', 'recommend_options_strategy'],
            'Execution & Portfolio': ['execute_options_strategy', 'get_order_status', 'cancel_order', 'get_portfolio_positions', 'get_account_margins', 'calculate_strategy_margins', 'monitor_strategy_pnl', 'get_daily_trading_summary']
        },
        'usage_note': 'Each tool returns JSON-serializable data for CrewAI agent consumption',
        'timestamp': dt.datetime.now().isoformat()
    }


if __name__ == "__main__":
    from dotenv import load_dotenv
    import os
    load_dotenv(dotenv_path='./.env')
    # Load API credentials and access token from environment or files
    api_key = os.getenv("kite_api_key")
    api_secret = os.getenv("kite_api_secret")
    access_token = None
    # Try to read access_token.txt if present
    try:
        with open("access_token.txt", "r") as f:
            access_token = f.read().strip()
    except Exception:
        print("access_token.txt not found. Set access token manually if needed.")
    
    # Initialize session
    if api_key and access_token:
        result = initialize_connection(api_key, api_secret, access_token)
        print("Session init result:", result)
    else:
        print("API key or access token missing. Skipping session init.")
    
    # Test read-only functions
    print("\n--- Portfolio Positions ---")
    print(get_portfolio_positions())
    print("\n--- Account Margins ---")
    print(get_account_margins())
    print("\n--- Orders History (last 1 day) ---")
    print(get_orders_history(1))
    print("\n--- Daily Trading Summary ---")
    print(get_daily_trading_summary())
    print("\n--- Risk Metrics ---")
    print(get_risk_metrics())