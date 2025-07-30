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
from typing import Dict, List, Optional, Any, Union
import threading
import time
from enum import Enum

# Global variable to hold connect_data_tools module
connect_data_tools = None

# Global lock for basket execution
basket_execution_lock = threading.Lock()

# Global failure tracking for circuit breaker
execution_failures = []

class OrderState(Enum):
    PENDING = "pending"
    PLACED = "placed"
    PARTIAL = "partial"
    COMPLETE = "complete"
    CANCELLED = "cancelled"
    FAILED = "failed"

# Import connection tools (assumes file 1 is available)
try:
    import sys
    import os
    
    # Add current directory to path for imports
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    
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
            access_token = None
            try:
                # Try multiple possible paths for access token
                access_token_paths = [
                    "access_token.txt",  # Current directory
                    "../data/access_token.txt",  # Relative to agent_tools
                    "../../data/access_token.txt",  # Relative to core_tools
                    "./data/access_token.txt"  # Alternative relative path
                ]
                
                access_token = None
                for path in access_token_paths:
                    try:
                        with open(path, "r") as f:
                            access_token = f.read().strip()
                            print(f"✅ Successfully loaded access token from: {path}")
                            break
                    except Exception:
                        continue
                        
                if access_token is None:
                    print("❌ Could not find access_token.txt in any expected location")
                    
            except Exception as e:
                print(f"❌ Error reading access token: {e}")
                access_token = None
            
            if api_key and (api_secret or access_token):
                init_result = initialize_connection(api_key, api_secret, access_token)
                print(f"Connection initialization result: {init_result}")
        except Exception as e:
            print(f"Warning: Failed to initialize connection in execution tools: {e}")
            
except ImportError as e:
    print(f"Warning: Connection tools not available. Some functions may not work. Error: {e}")
    connect_data_tools = None


def optimize_execution_order(strategy_legs: List[Dict[str, Any]], order_type: str = "Opening") -> List[Dict[str, Any]]:
    """
    Optimize the execution order for multi-leg strategies.
    
    For Opening Positions:
    - SELL first, then BUY (premium collection first)
    - Lower strikes before higher strikes
    - PE before CE (for same strike levels)
    
    For Closing Positions:
    - BUY to close short positions first (risk elimination)
    - SELL to close long positions second
    
    Args:
        strategy_legs: List of strategy legs with order details
        order_type: "Opening" or "Closing" (default: "Opening")
    
    Returns:
        List of legs in optimized execution order
    """
    if not strategy_legs:
        return strategy_legs
    
    # Determine if this is opening or closing positions based on order_type parameter
    opening_positions = order_type.lower() == "opening"
    
    # Create a copy to avoid modifying original
    legs_copy = [leg.copy() for leg in strategy_legs]
    
    if opening_positions:
        # For opening positions: SELL first, then BUY
        print(f"🔧 Optimizing for OPENING positions (SELL first, then BUY)")
        
        # Sort by action priority: SELL before BUY
        def opening_sort_key(leg):
            action = leg.get('action', '')
            symbol = leg.get('symbol', '')
            
            # Primary: SELL before BUY
            action_priority = 0 if action == 'SELL' else 1
            
            # Secondary: Extract strike price for ordering
            strike = 0
            try:
                # Extract strike from symbol (e.g., "NIFTY2572424900PE" -> 24900)
                if 'NIFTY' in symbol:
                    # Find the strike price in the symbol
                    import re
                    strike_match = re.search(r'(\d{4,5})(PE|CE)$', symbol)
                    if strike_match:
                        strike = int(strike_match.group(1))
            except:
                pass
            
            # Tertiary: PE before CE (for same strike levels)
            option_type_priority = 0 if symbol.endswith('PE') else 1
            
            return (action_priority, strike, option_type_priority)
        
        legs_copy.sort(key=opening_sort_key)
        
    else:
        # For closing positions: BUY to close short positions first, then SELL to close long positions
        print(f"🔧 Optimizing for CLOSING positions (BUY to close short first, then SELL to close long)")
        
        # Sort by action priority: BUY before SELL (for closing)
        def closing_sort_key(leg):
            action = leg.get('action', '')
            symbol = leg.get('symbol', '')
            
            # Primary: BUY before SELL (for closing)
            action_priority = 0 if action == 'BUY' else 1
            
            # Secondary: Extract strike price for ordering
            strike = 0
            try:
                if 'NIFTY' in symbol:
                    import re
                    strike_match = re.search(r'(\d{4,5})(PE|CE)$', symbol)
                    if strike_match:
                        strike = int(strike_match.group(1))
            except:
                pass
            
            # Tertiary: PE before CE
            option_type_priority = 0 if symbol.endswith('PE') else 1
            
            return (action_priority, strike, option_type_priority)
        
        legs_copy.sort(key=closing_sort_key)
    
    return legs_copy

def wait_for_order_settlement(order_ids, max_wait_seconds=10):
    """
    Wait for orders to settle before status check with progressive delay
    """
    print(f"⏳ Waiting for {len(order_ids)} orders to settle (max {max_wait_seconds}s)...")
    
    for attempt in range(max_wait_seconds):
        time.sleep(1)
        all_settled = True
        
        for order_id in order_ids:
            try:
                status = get_order_status(order_id)
                if status.get('status') == 'SUCCESS':
                    order_status = status.get('order_status', 'UNKNOWN')
                    if order_status in ['TRIGGER PENDING', 'VALIDATION PENDING', 'OPEN']:
                        all_settled = False
                        break
            except Exception as e:
                print(f"⚠️  Error checking order {order_id}: {str(e)}")
                all_settled = False
                break
        
        if all_settled:
            print(f"✅ All orders settled after {attempt + 1} seconds")
            return True
    
    print(f"⚠️  Orders may not be fully settled after {max_wait_seconds} seconds")
    return False

def handle_partial_fills(partially_filled_orders, strategy_legs):
    """
    Handle partial fills by adjusting remaining legs or closing positions
    """
    print(f"🔄 Handling {len(partially_filled_orders)} partial fills...")
    
    for partial_fill in partially_filled_orders:
        filled_qty = partial_fill['filled_qty']
        pending_qty = partial_fill['pending_qty']
        order_info = partial_fill['order_info']
        
        print(f"📊 Partial fill: {order_info['symbol']} - Filled: {filled_qty}, Pending: {pending_qty}")
        
        # Option 1: Try to place order for remaining quantity
        if pending_qty > 0:
            try:
                remaining_leg = {
                    'symbol': order_info['symbol'],
                    'action': order_info['action'],
                    'quantity': pending_qty,
                    'exchange': 'NFO',
                    'product': 'MIS',
                    'order_type': 'MARKET'
                }
                
                print(f"🔄 Attempting to place remaining order for {pending_qty} lots...")
                remaining_order_id = connect_data_tools._kite_instance.place_order(
                    variety='regular',
                    exchange='NFO',
                    tradingsymbol=order_info['symbol'],
                    transaction_type=order_info['action'],
                    quantity=pending_qty,
                    product='MIS',
                    order_type='MARKET',
                    validity='DAY'
                )
                
                print(f"✅ Remaining order placed: {remaining_order_id}")
                
            except Exception as e:
                print(f"❌ Failed to place remaining order: {str(e)}")
                # Option 2: Close the partial position if we can't complete it
                close_partial_position(partial_fill)

def close_partial_position(partial_fill):
    """
    Close a partial position to prevent unbalanced strategy
    """
    order_info = partial_fill['order_info']
    filled_qty = partial_fill['filled_qty']
    
    print(f"🚨 Closing partial position: {order_info['symbol']} x {filled_qty}")
    
    try:
        # Determine closing action (opposite of opening action)
        closing_action = 'SELL' if order_info['action'] == 'BUY' else 'BUY'
        
        close_order_id = connect_data_tools._kite_instance.place_order(
            variety='regular',
            exchange='NFO',
            tradingsymbol=order_info['symbol'],
            transaction_type=closing_action,
            quantity=filled_qty,
            product='MIS',
            order_type='MARKET',
            validity='DAY'
        )
        
        print(f"✅ Partial position closed: {close_order_id}")
        
    except Exception as e:
        print(f"❌ Failed to close partial position: {str(e)}")

def robust_order_cancellation(order_ids):
    """
    Cancel orders with verification and retry logic
    """
    print(f"🔄 Robust cancellation for {len(order_ids)} orders...")
    cancellation_results = {}
    
    # First pass: Rapid cancellation requests
    for order_id in order_ids:
        try:
            cancel_result = cancel_order(order_id)
            if cancel_result.get('status') == 'SUCCESS':
                cancellation_results[order_id] = 'CANCEL_REQUESTED'
                print(f"✅ Cancel requested for {order_id}")
            else:
                cancellation_results[order_id] = f'CANCEL_FAILED: {cancel_result.get("message")}'
                print(f"❌ Cancel failed for {order_id}: {cancel_result.get('message')}")
        except Exception as e:
            cancellation_results[order_id] = f'CANCEL_FAILED: {str(e)}'
            print(f"❌ Cancel error for {order_id}: {str(e)}")
    
    # Second pass: Verification with retry
    time.sleep(2)  # Allow cancellations to process
    
    for order_id in order_ids:
        if cancellation_results[order_id] == 'CANCEL_REQUESTED':
            try:
                status = get_order_status(order_id)
                if status['status'] == 'SUCCESS':
                    final_status = status['order_status']
                    
                    if final_status == 'CANCELLED':
                        cancellation_results[order_id] = 'SUCCESSFULLY_CANCELLED'
                        print(f"✅ {order_id} successfully cancelled")
                    elif final_status == 'COMPLETE':
                        cancellation_results[order_id] = 'EXECUTED_BEFORE_CANCEL'
                        print(f"⚠️  {order_id} executed before cancellation")
                    else:
                        cancellation_results[order_id] = f'UNCERTAIN_STATUS: {final_status}'
                        print(f"⚠️  {order_id} uncertain: {final_status}")
                else:
                    cancellation_results[order_id] = f'STATUS_CHECK_FAILED: {status.get("message")}'
                    print(f"❌ {order_id} status check failed")
                    
            except Exception as e:
                cancellation_results[order_id] = f'STATUS_CHECK_FAILED: {str(e)}'
                print(f"❌ {order_id} status check error: {str(e)}")
    
    return cancellation_results

def validate_basket_execution(strategy_legs):
    """
    Pre-execution validation for basket orders
    """
    print("🔍 Validating basket execution prerequisites...")
    
    # Check connection
    if connect_data_tools is None:
        return {'valid': False, 'message': 'Connection tools not available'}
    
    if not hasattr(connect_data_tools, '_kite_instance') or connect_data_tools._kite_instance is None:
        return {'valid': False, 'message': 'Connection not initialized'}
    
    # Validate strategy legs
    if not strategy_legs:
        return {'valid': False, 'message': 'No strategy legs provided'}
    
    # Validate each leg
    for i, leg in enumerate(strategy_legs):
        required_fields = ['symbol', 'action', 'quantity']
        if not all(field in leg for field in required_fields):
            return {'valid': False, 'message': f'Leg {i+1} missing required fields: {required_fields}'}
        
        if leg['quantity'] <= 0:
            return {'valid': False, 'message': f'Leg {i+1} has invalid quantity: {leg["quantity"]}'}
    
    # Check circuit breaker
    circuit_breaker_result = circuit_breaker_check()
    if not circuit_breaker_result['allow_execution']:
        return {'valid': False, 'message': f'Circuit breaker: {circuit_breaker_result["reason"]}'}
    
    print("✅ Basket execution validation passed")
    return {'valid': True, 'message': 'All validations passed'}

def circuit_breaker_check():
    """
    Prevent execution if system is unstable
    """
    # Simple implementation - can be enhanced with actual failure tracking
    try:
        # Check if we can get account margins (basic connectivity test)
        margins_result = get_account_margins()
        if margins_result.get('status') != 'SUCCESS':
            return {'allow_execution': False, 'reason': 'Cannot connect to broker'}
        
        return {'allow_execution': True, 'reason': 'System stable'}
        
    except Exception as e:
        return {'allow_execution': False, 'reason': f'System error: {str(e)}'}

def place_order_with_timeout(leg, timeout_seconds=5):
    """
    Place order with cross-platform timeout protection
    """
    result = {'order_id': None, 'error': None, 'completed': False}
    
    def place_order():
        try:
            order_params = {
                'variety': 'regular',
                'exchange': leg.get('exchange', 'NFO'),
                'tradingsymbol': leg['symbol'],
                'transaction_type': leg['action'],
                'quantity': leg['quantity'],
                'product': leg.get('product', 'MIS'),
                'order_type': leg.get('order_type', 'MARKET'),
                'validity': leg.get('validity', 'DAY')
            }
            
            if leg.get('order_type') == 'LIMIT' and 'price' in leg:
                order_params['price'] = leg['price']
            
            order_id = connect_data_tools._kite_instance.place_order(**order_params)
            result['order_id'] = order_id
            result['completed'] = True
            
        except Exception as e:
            result['error'] = str(e)
            result['completed'] = True
    
    # Start order placement in thread
    thread = threading.Thread(target=place_order)
    thread.start()
    thread.join(timeout=timeout_seconds)
    
    if not result['completed']:
        raise TimeoutError(f"Order placement timed out after {timeout_seconds} seconds")
    
    if result['error']:
        raise Exception(result['error'])
    
    return result['order_id']

def verify_basket_settlement(order_results, max_wait=15):
    """
    Verify all orders in basket have settled properly
    """
    print(f"⏳ Verifying basket settlement for {len(order_results)} orders...")
    
    settled_orders = []
    failed_orders = []
    partial_fills = []
    
    # Progressive waiting with exponential backoff
    wait_intervals = [1, 2, 3, 5, 4]  # Total: 15 seconds
    
    for interval in wait_intervals:
        time.sleep(interval)
        
        for order in order_results:
            if order.get('status') != 'VERIFIED':
                try:
                    status_result = get_order_status(order['order_id'])
                    
                    if status_result['status'] == 'SUCCESS':
                        order_status = status_result['order_status']
                        filled_qty = status_result['filled_quantity']
                        total_qty = status_result['quantity']
                        
                        if order_status == 'COMPLETE' and filled_qty == total_qty:
                            order['status'] = 'VERIFIED'
                            order['avg_price'] = status_result['average_price']
                            settled_orders.append(order)
                            print(f"✅ {order['symbol']} fully settled: {filled_qty} lots")
                            
                        elif order_status == 'COMPLETE' and filled_qty < total_qty:
                            order['status'] = 'PARTIAL'
                            order['filled_qty'] = filled_qty
                            order['pending_qty'] = total_qty - filled_qty
                            partial_fills.append(order)
                            print(f"⚠️  {order['symbol']} partially filled: {filled_qty}/{total_qty}")
                            
                        elif order_status in ['CANCELLED', 'REJECTED']:
                            order['status'] = 'FAILED'
                            failed_orders.append(order)
                            print(f"❌ {order['symbol']} failed: {order_status}")
                            
                        elif order_status in ['OPEN', 'TRIGGER PENDING']:
                            print(f"⏳ {order['symbol']} still pending: {order_status}")
                            
                except Exception as e:
                    print(f"❌ Error checking {order['symbol']}: {str(e)}")
                    if interval == wait_intervals[-1]:  # Last interval
                        order['status'] = 'FAILED'
                        failed_orders.append(order)
    
    success_rate = len(settled_orders) / len(order_results) if order_results else 0
    
    print(f"📊 Settlement Summary: {len(settled_orders)} settled, {len(failed_orders)} failed, {len(partial_fills)} partial")
    
    return {
        'all_successful': len(failed_orders) == 0 and len(partial_fills) == 0,
        'settled_orders': settled_orders,
        'failed_orders': failed_orders,
        'partial_fills': partial_fills,
        'success_rate': success_rate
    }

def emergency_cancel_all(order_ids):
    """
    Emergency cancellation with aggressive retry and immediate verification
    """
    print(f"🚨 EMERGENCY CANCELLATION: {len(order_ids)} orders")
    cancellation_results = {}
    
    # First pass: Rapid cancellation requests
    for order_id in order_ids:
        try:
            cancel_order(order_id, variety='regular')
            cancellation_results[order_id] = 'CANCEL_REQUESTED'
            print(f"🔄 Cancel requested: {order_id}")
        except Exception as e:
            cancellation_results[order_id] = f'CANCEL_FAILED: {str(e)}'
            print(f"❌ Cancel failed: {order_id} - {str(e)}")
    
    # Second pass: Verification with retry
    time.sleep(2)  # Allow cancellations to process
    
    for order_id in order_ids:
        if cancellation_results[order_id] == 'CANCEL_REQUESTED':
            try:
                status = get_order_status(order_id)
                if status['status'] == 'SUCCESS':
                    final_status = status['order_status']
                    
                    if final_status == 'CANCELLED':
                        cancellation_results[order_id] = 'SUCCESSFULLY_CANCELLED'
                        print(f"✅ {order_id} successfully cancelled")
                    elif final_status == 'COMPLETE':
                        cancellation_results[order_id] = 'EXECUTED_BEFORE_CANCEL'
                        print(f"⚠️  {order_id} executed before cancellation")
                    else:
                        cancellation_results[order_id] = f'UNCERTAIN_STATUS: {final_status}'
                        print(f"⚠️  {order_id} uncertain: {final_status}")
                else:
                    cancellation_results[order_id] = f'STATUS_CHECK_FAILED: {status.get("message")}'
                    print(f"❌ {order_id} status check failed")
                    
            except Exception as e:
                cancellation_results[order_id] = f'STATUS_CHECK_FAILED: {str(e)}'
                print(f"❌ {order_id} status check error: {str(e)}')
    
    return cancellation_results

def handle_mixed_basket_results(order_results, settlement_result):
    """
    Handle mixed results (some successful, some failed)
    """
    print("🔄 Handling mixed basket results...")
    
    settled_count = len(settlement_result['settled_orders'])
    failed_count = len(settlement_result['failed_orders'])
    partial_count = len(settlement_result['partial_fills'])
    
    # If we have partial fills, try to handle them
    if partial_count > 0:
        print(f"🔄 Processing {partial_count} partial fills...")
        handle_partial_fills(settlement_result['partial_fills'], [])
    
    # If we have failed orders, we need to close successful ones
    if failed_count > 0 and settled_count > 0:
        print(f"🚨 {failed_count} orders failed, closing {settled_count} successful orders...")
        
        # Close all successful orders to prevent unbalanced positions
        for settled_order in settlement_result['settled_orders']:
            try:
                closing_action = 'SELL' if settled_order['action'] == 'BUY' else 'BUY'
                
                close_order_id = connect_data_tools._kite_instance.place_order(
                    variety='regular',
                    exchange='NFO',
                    tradingsymbol=settled_order['symbol'],
                    transaction_type=closing_action,
                    quantity=settled_order['quantity'],
                    product='MIS',
                    order_type='MARKET',
                    validity='DAY'
                )
                
                print(f"✅ Closed {settled_order['symbol']}: {close_order_id}")
                
            except Exception as e:
                print(f"❌ Failed to close {settled_order['symbol']}: {str(e)}")
    
    return {
        'status': 'MIXED_BASKET_RESULTS',
        'settled_orders': settled_count,
        'failed_orders': failed_count,
        'partial_fills': partial_count,
        'success_rate': settlement_result['success_rate'],
        'message': f'Mixed results: {settled_count} settled, {failed_count} failed, {partial_count} partial'
    }

def execute_options_strategy(strategy_legs: List[Dict[str, Any]], order_type: str = "Opening") -> Dict[str, Any]:
    """
    Execute a multi-leg options strategy with improved basket order handling.
    
    Args:
        strategy_legs: List of legs with symbol, action, quantity, etc.
        order_type: "Opening" or "Closing" (default: "Opening")
    
    Returns:
        Dict with execution results for each leg
    """
    execution_start = time.time()
    
    # Use global lock to ensure only one basket executes at a time
    with basket_execution_lock:
        try:
            # Phase 1: Pre-execution validation
            print("🔍 PHASE 1: Pre-execution validation...")
            
            validation_result = validate_basket_execution(strategy_legs)
            if not validation_result['valid']:
                return {
                    'status': 'VALIDATION_FAILED',
                    'message': validation_result['message'],
                    'execution_time': time.time() - execution_start
                }
            
            # Phase 2: Rapid order placement
            print("🚀 PHASE 2: Rapid order placement...")
            
            optimized_legs = optimize_execution_order(strategy_legs, order_type=order_type)
            print(f"📋 Optimized execution order:")
            for i, leg in enumerate(optimized_legs, 1):
                print(f"   {i}. {leg['action']} {leg['symbol']} x {leg['quantity']}")
            
            order_results = []
            
            for i, leg in enumerate(optimized_legs):
                try:
                    # Place order with timeout protection
                    order_id = place_order_with_timeout(leg, timeout_seconds=5)
                    
                    order_results.append({
                        'leg_number': i + 1,
                        'order_id': order_id,
                        'symbol': leg['symbol'],
                        'action': leg['action'],
                        'quantity': leg['quantity'],
                        'placement_time': time.time() - execution_start,
                        'status': 'PLACED'
                    })
                    
                    print(f"✅ Leg {i+1}/{len(optimized_legs)}: {order_id}")
                    
                except Exception as e:
                    print(f"❌ Leg {i+1} FAILED: {str(e)}")
                    
                    # Immediate rollback - cancel all previous orders
                    if order_results:
                        print("🔄 IMMEDIATE ROLLBACK: Cancelling all placed orders...")
                        emergency_cancellation_results = emergency_cancel_all(
                            [r['order_id'] for r in order_results]
                        )
                        
                        return {
                            'status': 'BASKET_ROLLBACK_IMMEDIATE',
                            'failed_leg': i + 1,
                            'placed_orders': len(order_results),
                            'cancellation_results': emergency_cancellation_results,
                            'execution_time': time.time() - execution_start,
                            'message': f'Leg {i+1} failed, all {len(order_results)} orders cancelled'
                        }
                    else:
                        return {
                            'status': 'FIRST_LEG_FAILED',
                            'message': f'First leg failed: {str(e)}',
                            'execution_time': time.time() - execution_start
                        }
            
            # Phase 3: Settlement verification
            print("⏳ PHASE 3: Settlement verification...")
            
            settlement_result = verify_basket_settlement(order_results, max_wait=15)
            
            if settlement_result['all_successful']:
                return {
                    'status': 'BASKET_SUCCESS',
                    'total_legs': len(strategy_legs),
                    'execution_time': time.time() - execution_start,
                    'order_results': order_results,
                    'settlement_details': settlement_result,
                    'message': f'All {len(strategy_legs)} legs executed successfully'
                }
            else:
                # Handle mixed results (some successful, some failed)
                return handle_mixed_basket_results(order_results, settlement_result)
                
        except Exception as e:
            return {
                'status': 'BASKET_EXECUTION_ERROR',
                'message': f'Basket execution error: {str(e)}',
                'execution_time': time.time() - execution_start
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
    
    **IMPORTANT**: Positions with quantity != 0 (both positive and negative) are considered as "open positions".
    - Positive quantity (> 0): Long positions (bought options)
    - Negative quantity (< 0): Short positions (sold options)
    - Zero quantity (= 0): Closed positions (no active position)
    
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
                if quantity != 0:  # Any non-zero quantity (positive or negative) is an open position
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


def execute_and_store_strategy(strategy_legs: List[Dict[str, Any]], trade_metadata: Optional[Dict[str, Any]] = None, order_type: str = "Opening") -> Dict[str, Any]:
    """
    Execute a strategy and store it in trade storage ONLY if execution is successful.
    
    Args:
        strategy_legs: List of strategy legs with order details
        trade_metadata: Additional trade information (analysis data, risk management, etc.)
                       Can also be a strategy creation result dict with 'legs' key


    Returns:
        Dict with execution result and storage status
    """
    try:
        # Handle case where trade_metadata is actually a strategy creation result
        if trade_metadata and 'legs' in trade_metadata:
            # This is a strategy creation result, extract the legs and metadata
            strategy_legs = trade_metadata['legs']
            strategy_metadata = {
                'strategy_name': trade_metadata.get('strategy_name', 'Unknown Strategy'),
                'expiry_date': trade_metadata.get('expiry_date'),
                'spot_price': trade_metadata.get('spot_price'),
                'strategy_type': trade_metadata.get('strategy_name', 'Unknown Strategy'),
                'risk_management': trade_metadata.get('strategy_metrics', {}),
                'analysis_data': {
                    'market_outlook': trade_metadata.get('market_outlook', ''),
                    'strategy_metrics': trade_metadata.get('strategy_metrics', {}),
                    'atm_strike': trade_metadata.get('atm_strike'),
                    'creation_timestamp': trade_metadata.get('timestamp')
                }
            }
        else:
            # Use provided trade_metadata as is
            strategy_metadata = trade_metadata or {}
        
        # CRITICAL: Validate margin requirements BEFORE executing any orders
        print("🔍 PRE-EXECUTION MARGIN VALIDATION...")
        
        # Use the existing calculate_strategy_margins function
        margin_result = calculate_strategy_margins(strategy_legs)
        
        if margin_result.get('status') != 'SUCCESS':
            return {
                'status': 'MARGIN_CALCULATION_FAILED',
                'execution_result': None,
                'storage_result': None,
                'message': f'Margin calculation failed: {margin_result.get("message", "Unknown error")}'
            }
        
        # Get account margins to compare
        account_margins = get_account_margins()
        if account_margins.get('status') != 'SUCCESS':
            return {
                'status': 'ACCOUNT_MARGINS_FAILED',
                'execution_result': None,
                'storage_result': None,
                'message': f'Could not fetch account margins: {account_margins.get("message", "Unknown error")}'
            }
        
        required_margin = margin_result.get('total_margin_required', 0)
        available_cash = account_margins['equity'].get('available_cash', 0)
        live_balance = account_margins['equity'].get('live_balance', 0)
        intraday_payin = account_margins['equity'].get('intraday_payin', 0)
        
        # Use conservative approach: take the lower of available_cash or live_balance
        # This accounts for unrealized P&L that might not be reflected in available_cash
        effective_available_cash = min(available_cash, live_balance) + intraday_payin
        
        # Add 20% safety buffer
        total_required = required_margin * 1.20
        
        print(f"💰 MARGIN ANALYSIS:")
        print(f"   Required Margin: ₹{required_margin:.2f}")
        print(f"   With Safety Buffer: ₹{total_required:.2f}")
        print(f"   Available Cash: ₹{available_cash:.2f}")
        print(f"   Live Balance: ₹{live_balance:.2f}")
        print(f"   Intraday Payin: ₹{intraday_payin:.2f}")
        print(f"   Effective Available (Conservative): ₹{effective_available_cash:.2f}")
        print(f"   Unrealized P&L: ₹{account_margins['equity'].get('unrealized_pnl', 0):.2f}")
        
        # Check if we have sufficient margin
        if total_required > effective_available_cash:
            return {
                'status': 'INSUFFICIENT_MARGIN',
                'execution_result': None,
                'storage_result': None,
                'margin_result': margin_result,
                'account_margins': account_margins,
                'message': f'Insufficient margin for strategy execution. Required: ₹{total_required:.2f}, Available: ₹{effective_available_cash:.2f}'
            }
        
        # CRITICAL: Check if we have sufficient margin for ALL legs
        if len(strategy_legs) > 1:
            print(f"🔍 VALIDATING MULTI-LEG STRATEGY ({len(strategy_legs)} legs)...")
            
            # Calculate individual leg margins to ensure we can execute all legs
            individual_margins = margin_result.get('individual_margins', [])
            if individual_margins:
                total_individual_margin = sum(leg.get('margin', 0) for leg in individual_margins)
                print(f"   Total Individual Margins: ₹{total_individual_margin:.2f}")
                
                # Ensure we have margin for worst-case scenario (all legs executed)
                if total_individual_margin > effective_available_cash:
                    return {
                        'status': 'INSUFFICIENT_MARGIN_FOR_ALL_LEGS',
                        'execution_result': None,
                        'storage_result': None,
                        'margin_result': margin_result,
                        'account_margins': account_margins,
                        'message': f'Insufficient margin for all {len(strategy_legs)} legs. Required: ₹{total_individual_margin:.2f}, Available: ₹{effective_available_cash:.2f}'
                    }
        
        print("✅ ALL MARGIN CHECKS PASSED - PROCEEDING WITH EXECUTION...")
        
        # First execute the strategy
        execution_result = execute_options_strategy(strategy_legs, order_type=order_type)
        
        if execution_result.get('status') not in ['SUCCESS', 'PARTIAL_SUCCESS']:
            return {
                'status': 'EXECUTION_FAILED',
                'execution_result': execution_result,
                'storage_result': None,
                'message': f'Strategy execution failed: {execution_result.get("message", "Unknown error")}'
            }
        
        # If execution successful, store the trade
        try:
            # Handle different import contexts
            try:
                from trade_storage import write_successful_trade
            except ImportError:
                from core_tools.trade_storage import write_successful_trade
            
            # Prepare trade data for storage
            trade_data = {
                'strategy_name': strategy_metadata.get('strategy_name', 'Unknown Strategy'),
                'legs': strategy_legs,
                'expiry_date': strategy_metadata.get('expiry_date'),
                'spot_price': strategy_metadata.get('spot_price'),
                'strategy_type': strategy_metadata.get('strategy_type'),
                'risk_management': strategy_metadata.get('risk_management', {}),
                'analysis_data': strategy_metadata.get('analysis_data', {}),
                'execution_details': execution_result
            }
            
            storage_result = write_successful_trade(trade_data)
            
            return {
                'status': 'SUCCESS',
                'execution_result': execution_result,
                'storage_result': storage_result,
                'trade_id': storage_result.get('trade_id'),
                'message': f'Strategy executed and stored successfully. Trade ID: {storage_result.get("trade_id")}'
            }
            
        except ImportError:
            # If trade storage is not available, still return execution success
            return {
                'status': 'EXECUTION_SUCCESS_STORAGE_UNAVAILABLE',
                'execution_result': execution_result,
                'storage_result': None,
                'message': 'Strategy executed successfully but trade storage not available'
            }
            
    except Exception as e:
        return {
            'status': 'ERROR',
            'execution_result': None,
            'storage_result': None,
            'message': f'Execute and store failed: {str(e)}'
        }


def validate_general_capital() -> Dict[str, Any]:
    """
    Wrapper function for general capital validation without specific strategy.
    
    Returns:
        Dict with general capital validation results
    """
    try:
        # Call validate_trading_capital without strategy legs for general validation
        return validate_trading_capital(strategy_legs=None, risk_percentage=5.0)
        
    except Exception as e:
        return {
            'status': 'ERROR',
            'message': f'General capital validation failed: {str(e)}',
            'validation_passed': False
        }


def validate_trading_capital(strategy_legs=None, risk_percentage=5.0):
    """
    Validate if sufficient capital is available for a trading strategy.
    
    IMPORTANT: This function includes intraday_payin amounts in the available cash calculation
    since these funds are available for trading today even though they may not reflect in
    available_cash until tomorrow.
    
    Args:
        strategy_legs: List of strategy legs with order details (can be None for general validation)
        risk_percentage: Maximum percentage of capital to risk (default 5%)
    
    Returns:
        Validation result with detailed capital analysis including intraday_payin consideration
    """
    # Handle case where strategy_legs is not provided
    if strategy_legs is None:
        strategy_legs = []
    try:
        # Get current account margins
        margins_result = get_account_margins()
        if margins_result.get('status') != 'SUCCESS':
            return {
                'status': 'ERROR',
                'message': 'Could not fetch account margins',
                'validation_passed': False
            }
        
        available_cash = margins_result['equity']['available_cash']
        live_balance = margins_result['equity']['live_balance']
        intraday_payin = margins_result['equity'].get('intraday_payin', 0)
        
        # Add intraday_payin to available cash since it's available for trading today
        # even though it might not reflect in available_cash until tomorrow
        total_available_cash = available_cash + intraday_payin
        
        # Calculate strategy margin requirement
        if strategy_legs:
            margin_result = calculate_strategy_margins(strategy_legs)
            if margin_result.get('status') != 'SUCCESS':
                # Fallback: estimate margin as 20% of option premium for simple strategies
                estimated_margin = 0
                for leg in strategy_legs:
                    if 'price' in leg and 'quantity' in leg:
                        estimated_margin += leg['price'] * leg['quantity'] * 0.20
                
                margin_result = {
                    'status': 'SUCCESS',
                    'total_margin_required': estimated_margin,
                    'note': 'Estimated margin calculation used'
                }
            
            required_margin = margin_result['total_margin_required']
            
            # Calculate premium costs for debit strategies
            total_premium_cost = 0
            for leg in strategy_legs:
                if leg['action'] == 'BUY' and 'price' in leg:
                    total_premium_cost += leg['price'] * leg['quantity']
        else:
            # General capital validation without specific strategy
            required_margin = 0
            total_premium_cost = 0
            margin_result = {
                'status': 'SUCCESS',
                'total_margin_required': 0,
                'note': 'General capital validation - no specific strategy'
            }
        
        # Calculate total capital requirement
        safety_buffer = required_margin * 0.20  # 20% safety buffer
        total_requirement = required_margin + safety_buffer + total_premium_cost
        
        # Risk validation
        max_risk_amount = live_balance * (risk_percentage / 100)
        risk_amount = max(total_premium_cost, required_margin * 0.5)  # Estimate potential loss
        
        # Validation checks
        validations = {
            'sufficient_cash': total_available_cash >= total_requirement,
            'within_risk_limit': risk_amount <= max_risk_amount,
            'reasonable_margin_usage': required_margin <= (total_available_cash * 0.8),
            'emergency_buffer': (total_available_cash - total_requirement) >= (live_balance * 0.1)
        }
        
        validation_passed = all(validations.values())
        
        # Generate warnings and recommendations
        warnings = []
        if not validations['sufficient_cash']:
            warnings.append(f"Insufficient cash: Need ₹{total_requirement:.2f}, have ₹{total_available_cash:.2f} (₹{available_cash:.2f} + ₹{intraday_payin:.2f} intraday_payin)")
        if not validations['within_risk_limit']:
            warnings.append(f"Exceeds risk limit: Risk ₹{risk_amount:.2f} > Limit ₹{max_risk_amount:.2f}")
        if not validations['reasonable_margin_usage']:
            warnings.append(f"High margin usage: ₹{required_margin:.2f} > 80% of total available cash")
        if not validations['emergency_buffer']:
            warnings.append("Insufficient emergency buffer after trade")
        
        return {
            'status': 'SUCCESS',
            'validation_passed': validation_passed,
            'capital_analysis': {
                'available_cash': available_cash,
                'intraday_payin': intraday_payin,
                'total_available_cash': total_available_cash,
                'live_balance': live_balance,
                'required_margin': required_margin,
                'premium_cost': total_premium_cost,
                'safety_buffer': safety_buffer,
                'total_requirement': total_requirement,
                'remaining_cash': total_available_cash - total_requirement,
                'risk_amount': risk_amount,
                'max_risk_allowed': max_risk_amount,
                'margin_utilization_pct': (required_margin / total_available_cash) * 100 if total_available_cash > 0 else 0
            },
            'validation_checks': validations,
            'warnings': warnings,
            'recommendations': [
                'Maintain at least 10% emergency cash buffer',
                'Keep margin utilization below 80% of total available cash (including intraday_payin)',
                f'Risk per trade should not exceed {risk_percentage}% of total capital',
                'Consider paper trading if capital is very limited',
                'Intraday_payin amounts are included in available cash for trading today'
            ],
            'margin_calculation_note': margin_result.get('note', 'Exact margin calculation used'),
            'validation_type': 'strategy_specific' if strategy_legs else 'general_capital'
        }
        
    except Exception as e:
        return {
            'status': 'ERROR',
            'message': f'Capital validation failed: {str(e)}',
            'validation_passed': False
        }


def analyze_position_conflicts_wrapper(proposed_strategy: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Wrapper function for position conflict analysis that automatically fetches current positions.
    
    Args:
        proposed_strategy: Dictionary containing proposed strategy details (optional)
    
    Returns:
        Conflict analysis with recommendations for position management
    """
    try:
        # Get current portfolio positions
        positions_result = get_portfolio_positions()
        if positions_result.get('status') != 'SUCCESS':
            return {
                'status': 'ERROR',
                'message': f'Failed to fetch current positions: {positions_result.get("message", "Unknown error")}',
                'conflicts_found': True,  # Assume conflict if we can't check
                'recommendation': 'Manual review required'
            }
        
        existing_positions = positions_result.get('positions', [])
        
        # If no proposed strategy provided, just analyze current positions
        if not proposed_strategy:
            return {
                'status': 'SUCCESS',
                'conflicts_found': False,
                'message': 'No proposed strategy provided - analyzing current positions only',
                'current_position_count': len(existing_positions),
                'analysis': {
                    'current_positions': existing_positions,
                    'recommendations': [
                        'Current positions analyzed successfully',
                        'No conflicts to check without proposed strategy'
                    ]
                },
                'recommendation': 'Current positions are manageable'
            }
        
        # Call the main conflict analysis function
        return analyze_position_conflicts(existing_positions, proposed_strategy)
        
    except Exception as e:
        return {
            'status': 'ERROR',
            'message': f'Position conflict analysis wrapper failed: {str(e)}',
            'conflicts_found': True,  # Assume conflict if analysis fails
            'recommendation': 'Manual review required'
        }


def analyze_position_conflicts(existing_positions, proposed_strategy):
    """
    Analyze potential conflicts between existing positions and a proposed new strategy.
    
    Args:
        existing_positions: List of current portfolio positions
        proposed_strategy: Dictionary containing proposed strategy details
    
    Returns:
        Conflict analysis with recommendations for position management
    """
    try:
        conflicts = {
            'directional_conflicts': [],
            'volatility_conflicts': [],
            'expiry_conflicts': [],
            'strike_conflicts': [],
            'risk_concentration': [],
            'recommendations': []
        }
        
        if not existing_positions or len(existing_positions) == 0:
            return {
                'status': 'SUCCESS',
                'conflicts_found': False,
                'message': 'No existing positions - clean slate for new trade',
                'analysis': conflicts
            }
        
        # Analyze each existing position against proposed strategy
        for position in existing_positions:
            # Directional conflict analysis
            if 'direction' in position and 'direction' in proposed_strategy:
                if position['direction'] != proposed_strategy['direction']:
                    conflicts['directional_conflicts'].append({
                        'position': position,
                        'conflict_type': 'opposite_direction',
                        'risk_level': 'high'
                    })
            
            # Expiry conflict analysis
            if 'expiry' in position and 'expiry' in proposed_strategy:
                if position['expiry'] == proposed_strategy['expiry']:
                    conflicts['expiry_conflicts'].append({
                        'position': position,
                        'conflict_type': 'same_expiry',
                        'risk_level': 'medium'
                    })
            
            # Strike conflict analysis
            if 'strike' in position and 'strike' in proposed_strategy:
                strike_diff = abs(position['strike'] - proposed_strategy['strike'])
                if strike_diff < 100:  # Close strikes create gamma risk
                    conflicts['strike_conflicts'].append({
                        'position': position,
                        'conflict_type': 'close_strikes',
                        'risk_level': 'medium',
                        'strike_distance': strike_diff
                    })
        
        # Risk concentration analysis
        total_positions = len(existing_positions) + 1  # Including proposed
        if total_positions > 3:
            conflicts['risk_concentration'].append({
                'issue': 'too_many_positions',
                'current_count': len(existing_positions),
                'proposed_count': total_positions,
                'risk_level': 'high'
            })
        
        # Generate recommendations
        if conflicts['directional_conflicts']:
            conflicts['recommendations'].append({
                'action': 'close_opposite_positions',
                'priority': 'high',
                'reason': 'Directional conflicts create hedging inefficiency'
            })
        
        if conflicts['expiry_conflicts']:
            conflicts['recommendations'].append({
                'action': 'consider_different_expiry',
                'priority': 'medium',
                'reason': 'Same expiry creates concentration risk'
            })
        
        if conflicts['strike_conflicts']:
            conflicts['recommendations'].append({
                'action': 'avoid_close_strikes',
                'priority': 'medium',
                'reason': 'Close strikes create excessive gamma risk'
            })
        
        if conflicts['risk_concentration']:
            conflicts['recommendations'].append({
                'action': 'manage_existing_positions',
                'priority': 'high',
                'reason': 'Too many positions create management complexity'
            })
        
        conflicts_found = any([
            conflicts['directional_conflicts'],
            conflicts['volatility_conflicts'],
            conflicts['expiry_conflicts'],
            conflicts['strike_conflicts'],
            conflicts['risk_concentration']
        ])
        
        return {
            'status': 'SUCCESS',
            'conflicts_found': conflicts_found,
            'total_conflicts': sum(len(conflicts[key]) for key in conflicts if key != 'recommendations'),
            'analysis': conflicts,
            'recommendation': 'Proceed with caution' if conflicts_found else 'No conflicts detected'
        }
        
    except Exception as e:
        return {
            'status': 'ERROR',
            'message': f'Position conflict analysis failed: {str(e)}',
            'conflicts_found': True,  # Assume conflict if analysis fails
            'recommendation': 'Manual review required'
        }


def calculate_realistic_pricing(strike_price, option_type, expiry, transaction_type, current_bid, current_ask, market_volatility="normal"):
    """
    Calculate realistic execution prices accounting for bid-ask spreads and slippage.
    
    Args:
        strike_price: Option strike price
        option_type: 'CE' or 'PE'
        expiry: Expiry date
        transaction_type: 'buy' or 'sell'
        current_bid: Current bid price
        current_ask: Current ask price
        market_volatility: 'low', 'normal', 'high' for buffer adjustment
    
    Returns:
        Recommended order price with buffer, spread analysis, and execution probability
    """
    try:
        bid = float(current_bid)
        ask = float(current_ask)
        mid_price = (bid + ask) / 2
        spread_pct = ((ask - bid) / mid_price) * 100 if mid_price > 0 else 0
        
        # Determine buffer based on market conditions
        if market_volatility == "high":
            buffer_pct = 5
            buffer_absolute = max(0.25, mid_price * 0.05)
        elif market_volatility == "low":
            buffer_pct = 2
            buffer_absolute = max(0.05, mid_price * 0.02)
        else:  # normal
            buffer_pct = 3
            buffer_absolute = max(0.10, mid_price * 0.03)
        
        if transaction_type.lower() == "buy":
            recommended_price = ask + buffer_absolute
            execution_probability = "High" if spread_pct < 5 else "Medium" if spread_pct < 10 else "Low"
        else:  # sell
            recommended_price = bid - buffer_absolute
            execution_probability = "High" if spread_pct < 5 else "Medium" if spread_pct < 10 else "Low"
        
        # Round to nearest 0.05 (NSE tick size)
        recommended_price = round(recommended_price * 20) / 20
        
        analysis = {
            "recommended_price": recommended_price,
            "current_bid": bid,
            "current_ask": ask,
            "spread_percentage": round(spread_pct, 2),
            "buffer_applied": round(buffer_absolute, 2),
            "execution_probability": execution_probability,
            "warning": "Reject trade" if spread_pct > 8 else "Proceed with caution" if spread_pct > 5 else "Good liquidity"
        }
        
        return analysis
        
    except Exception as e:
        return {"error": f"Pricing calculation failed: {str(e)}"}


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
    'get_risk_metrics': get_risk_metrics,
    'execute_and_store_strategy': execute_and_store_strategy
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
    
    # Load API credentials and access token from environment or files
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