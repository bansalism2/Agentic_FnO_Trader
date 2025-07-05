#!/usr/bin/env python3
"""
NIFTY Options - Strategy Creation Tools (File 3 of 4)
====================================================

Options strategy creation and recommendation tools.
This file contains functions to create and analyze various options strategies.

Dependencies:
Requires nifty_connection_tools.py for market data

Author: AI Assistant
"""

import datetime as dt
from typing import Dict, List, Optional, Any
import os
from dotenv import load_dotenv

# Import connection tools (assumes file 1 is available)
try:
    from connect_data_tools import (
        initialize_connection,
        authenticate_session,
        get_nifty_spot_price,
        get_nifty_expiry_dates,
        get_options_chain,
        get_historical_volatility,
        analyze_options_flow
    )
    CONNECTION_TOOLS_AVAILABLE = True
except ImportError:
    print("Warning: Connection tools not available. Some functions may not work.")
    CONNECTION_TOOLS_AVAILABLE = False


def create_long_straddle_strategy(expiry_date: str = None, expiry_type: str = 'weekly',
                                quantity: int = 25) -> Dict[str, Any]:
    """
    Create a Long Straddle options strategy.
    
    Args:
        expiry_date: Specific expiry (YYYY-MM-DD) or None for auto-selection
        expiry_type: 'weekly' or 'monthly' if expiry_date is None
        quantity: Number of lots (multiples of 25)
    
    Returns:
        Dict with strategy details and execution plan
    """
    try:
        # Get expiry date if not provided
        if not expiry_date:
            expiry_result = get_nifty_expiry_dates(expiry_type)
            if expiry_result['status'] != 'SUCCESS' or not expiry_result['expiries']:
                return {'status': 'ERROR', 'message': 'No expiry dates available'}
            expiry_date = expiry_result['expiries'][0]
        
        # Get spot price
        spot_result = get_nifty_spot_price()
        if spot_result['status'] != 'SUCCESS':
            return spot_result
        
        spot_price = spot_result['spot_price']
        
        # Get options chain
        chain_result = get_options_chain(expiry_date, expiry_type, strike_range=5)
        if chain_result['status'] != 'SUCCESS':
            return chain_result
        
        # Find ATM strike
        atm_strike = chain_result['atm_strike']
        chain_data = chain_result['options_chain']
        
        # Find ATM options
        atm_row = next((row for row in chain_data if row['strike'] == atm_strike), None)
        if not atm_row or 'CE_symbol' not in atm_row or 'PE_symbol' not in atm_row:
            return {'status': 'ERROR', 'message': 'ATM options not found'}
        
        call_price = atm_row.get('CE_ltp', 0)
        put_price = atm_row.get('PE_ltp', 0)
        
        if call_price <= 0 or put_price <= 0:
            return {'status': 'ERROR', 'message': 'Invalid option prices'}
        
        # Calculate strategy metrics
        total_premium = (call_price + put_price) * quantity
        breakeven_upper = atm_strike + (call_price + put_price)
        breakeven_lower = atm_strike - (call_price + put_price)
        
        # Create strategy legs
        legs = [
            {
                'symbol': atm_row['CE_symbol'],
                'action': 'BUY',
                'quantity': quantity,
                'strike': atm_strike,
                'option_type': 'CE',
                'price': call_price,
                'exchange': 'NFO'
            },
            {
                'symbol': atm_row['PE_symbol'],
                'action': 'BUY',
                'quantity': quantity,
                'strike': atm_strike,
                'option_type': 'PE',
                'price': put_price,
                'exchange': 'NFO'
            }
        ]
        
        return {
            'status': 'SUCCESS',
            'strategy_name': f'Long Straddle ({expiry_type.title()})',
            'expiry_date': expiry_date,
            'spot_price': spot_price,
            'atm_strike': atm_strike,
            'legs': legs,
            'strategy_metrics': {
                'total_premium_paid': round(total_premium, 2),
                'max_profit': 'Unlimited',
                'max_loss': round(total_premium, 2),
                'breakeven_points': [round(breakeven_lower, 2), round(breakeven_upper, 2)],
                'probability_of_profit': 'Depends on movement > ' + str(round(call_price + put_price, 2))
            },
            'market_outlook': 'High volatility expected, direction uncertain',
            'timestamp': dt.datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            'status': 'ERROR',
            'message': f'Strategy creation failed: {str(e)}'
        }


def create_short_strangle_strategy(expiry_date: str = None, expiry_type: str = 'weekly',
                                 otm_distance: int = 100, quantity: int = 25) -> Dict[str, Any]:
    """
    Create a Short Strangle options strategy.
    
    Args:
        expiry_date: Specific expiry (YYYY-MM-DD) or None for auto-selection
        expiry_type: 'weekly' or 'monthly' if expiry_date is None
        otm_distance: Distance from ATM for OTM options
        quantity: Number of lots
    
    Returns:
        Dict with strategy details and execution plan
    """
    try:
        # Get expiry and spot price
        if not expiry_date:
            expiry_result = get_nifty_expiry_dates(expiry_type)
            if expiry_result['status'] != 'SUCCESS' or not expiry_result['expiries']:
                return {'status': 'ERROR', 'message': 'No expiry dates available'}
            expiry_date = expiry_result['expiries'][0]
        
        spot_result = get_nifty_spot_price()
        if spot_result['status'] != 'SUCCESS':
            return spot_result
        
        spot_price = spot_result['spot_price']
        
        # Get options chain
        chain_result = get_options_chain(expiry_date, expiry_type, strike_range=10)
        if chain_result['status'] != 'SUCCESS':
            return chain_result
        
        # Calculate OTM strikes
        atm_strike = chain_result['atm_strike']
        call_strike = atm_strike + otm_distance
        put_strike = atm_strike - otm_distance
        
        chain_data = chain_result['options_chain']
        
        # Find OTM options
        call_row = next((row for row in chain_data if row['strike'] == call_strike), None)
        put_row = next((row for row in chain_data if row['strike'] == put_strike), None)
        
        if not call_row or not put_row or 'CE_symbol' not in call_row or 'PE_symbol' not in put_row:
            return {'status': 'ERROR', 'message': 'OTM options not found'}
        
        call_price = call_row.get('CE_ltp', 0)
        put_price = put_row.get('PE_ltp', 0)
        
        if call_price <= 0 or put_price <= 0:
            return {'status': 'ERROR', 'message': 'Invalid option prices'}
        
        # Calculate strategy metrics
        total_premium = (call_price + put_price) * quantity
        breakeven_upper = call_strike + (call_price + put_price)
        breakeven_lower = put_strike - (call_price + put_price)
        
        # Create strategy legs
        legs = [
            {
                'symbol': call_row['CE_symbol'],
                'action': 'SELL',
                'quantity': quantity,
                'strike': call_strike,
                'option_type': 'CE',
                'price': call_price,
                'exchange': 'NFO'
            },
            {
                'symbol': put_row['PE_symbol'],
                'action': 'SELL',
                'quantity': quantity,
                'strike': put_strike,
                'option_type': 'PE',
                'price': put_price,
                'exchange': 'NFO'
            }
        ]
        
        return {
            'status': 'SUCCESS',
            'strategy_name': f'Short Strangle ({expiry_type.title()})',
            'expiry_date': expiry_date,
            'spot_price': spot_price,
            'atm_strike': atm_strike,
            'legs': legs,
            'strategy_metrics': {
                'total_premium_received': round(total_premium, 2),
                'max_profit': round(total_premium, 2),
                'max_loss': 'Unlimited (but limited practically)',
                'breakeven_points': [round(breakeven_lower, 2), round(breakeven_upper, 2)],
                'profit_range': f'{round(put_strike, 2)} to {round(call_strike, 2)}'
            },
            'market_outlook': 'Low volatility expected, range-bound movement',
            'timestamp': dt.datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            'status': 'ERROR',
            'message': f'Strategy creation failed: {str(e)}'
        }


def create_iron_condor_strategy(expiry_date: str = None, expiry_type: str = 'weekly',
                              wing_width: int = 100, quantity: int = 25) -> Dict[str, Any]:
    """
    Create an Iron Condor options strategy.
    
    Args:
        expiry_date: Specific expiry (YYYY-MM-DD) or None for auto-selection
        expiry_type: 'weekly' or 'monthly' if expiry_date is None
        wing_width: Width between strikes
        quantity: Number of lots
    
    Returns:
        Dict with strategy details and execution plan
    """
    try:
        # Get expiry and spot price
        if not expiry_date:
            expiry_result = get_nifty_expiry_dates(expiry_type)
            if expiry_result['status'] != 'SUCCESS' or not expiry_result['expiries']:
                return {'status': 'ERROR', 'message': 'No expiry dates available'}
            expiry_date = expiry_result['expiries'][0]
        
        spot_result = get_nifty_spot_price()
        if spot_result['status'] != 'SUCCESS':
            return spot_result
        
        spot_price = spot_result['spot_price']
        
        # Get options chain
        chain_result = get_options_chain(expiry_date, expiry_type, strike_range=15)
        if chain_result['status'] != 'SUCCESS':
            return chain_result
        
        # Calculate Iron Condor strikes
        atm_strike = chain_result['atm_strike']
        put_sell_strike = atm_strike - wing_width      # Sell Put
        put_buy_strike = atm_strike - (2 * wing_width) # Buy Put
        call_sell_strike = atm_strike + wing_width     # Sell Call
        call_buy_strike = atm_strike + (2 * wing_width) # Buy Call
        
        chain_data = chain_result['options_chain']
        
        # Find all required options
        strikes_needed = [put_buy_strike, put_sell_strike, call_sell_strike, call_buy_strike]
        option_legs = []
        
        for strike in strikes_needed:
            row = next((r for r in chain_data if r['strike'] == strike), None)
            if not row:
                return {'status': 'ERROR', 'message': f'Strike {strike} not found in chain'}
            
            # Determine action based on Iron Condor structure
            if strike == put_buy_strike:
                action, option_type = 'BUY', 'PE'
                symbol = row.get('PE_symbol')
                price = row.get('PE_ltp', 0)
            elif strike == put_sell_strike:
                action, option_type = 'SELL', 'PE'
                symbol = row.get('PE_symbol')
                price = row.get('PE_ltp', 0)
            elif strike == call_sell_strike:
                action, option_type = 'SELL', 'CE'
                symbol = row.get('CE_symbol')
                price = row.get('CE_ltp', 0)
            else:  # call_buy_strike
                action, option_type = 'BUY', 'CE'
                symbol = row.get('CE_symbol')
                price = row.get('CE_ltp', 0)
            
            if not symbol or price <= 0:
                return {'status': 'ERROR', 'message': f'Invalid option data for strike {strike}'}
            
            option_legs.append({
                'symbol': symbol,
                'action': action,
                'quantity': quantity,
                'strike': strike,
                'option_type': option_type,
                'price': price,
                'exchange': 'NFO'
            })
        
        # Calculate net premium (received - paid)
        net_premium = 0
        for leg in option_legs:
            if leg['action'] == 'SELL':
                net_premium += leg['price'] * quantity
            else:
                net_premium -= leg['price'] * quantity
        
        max_profit = net_premium
        max_loss = (wing_width * quantity) - net_premium
        
        # Breakeven points
        breakeven_lower = put_sell_strike - (net_premium / quantity)
        breakeven_upper = call_sell_strike + (net_premium / quantity)
        
        return {
            'status': 'SUCCESS',
            'strategy_name': f'Iron Condor ({expiry_type.title()})',
            'expiry_date': expiry_date,
            'spot_price': spot_price,
            'atm_strike': atm_strike,
            'legs': option_legs,
            'strategy_metrics': {
                'net_premium': round(net_premium, 2),
                'max_profit': round(max_profit, 2),
                'max_loss': round(max_loss, 2),
                'breakeven_points': [round(breakeven_lower, 2), round(breakeven_upper, 2)],
                'profit_range': f'{round(put_sell_strike, 2)} to {round(call_sell_strike, 2)}',
                'wing_width': wing_width
            },
            'market_outlook': 'Neutral - expecting range-bound movement',
            'timestamp': dt.datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            'status': 'ERROR',
            'message': f'Iron Condor strategy creation failed: {str(e)}'
        }


def create_butterfly_spread_strategy(expiry_date: str = None, expiry_type: str = 'weekly',
                                   option_type: str = 'CE', wing_width: int = 100,
                                   quantity: int = 25) -> Dict[str, Any]:
    """
    Create a Butterfly Spread options strategy.
    
    Args:
        expiry_date: Specific expiry (YYYY-MM-DD) or None for auto-selection
        expiry_type: 'weekly' or 'monthly' if expiry_date is None
        option_type: 'CE' or 'PE' for call or put butterfly
        wing_width: Distance between strikes
        quantity: Number of lots
    
    Returns:
        Dict with strategy details and execution plan
    """
    try:
        # Get expiry and spot price
        if not expiry_date:
            expiry_result = get_nifty_expiry_dates(expiry_type)
            if expiry_result['status'] != 'SUCCESS' or not expiry_result['expiries']:
                return {'status': 'ERROR', 'message': 'No expiry dates available'}
            expiry_date = expiry_result['expiries'][0]
        
        spot_result = get_nifty_spot_price()
        if spot_result['status'] != 'SUCCESS':
            return spot_result
        
        spot_price = spot_result['spot_price']
        
        # Get options chain
        chain_result = get_options_chain(expiry_date, expiry_type, strike_range=10)
        if chain_result['status'] != 'SUCCESS':
            return chain_result
        
        # Calculate butterfly strikes
        atm_strike = chain_result['atm_strike']
        lower_strike = atm_strike - wing_width
        middle_strike = atm_strike
        upper_strike = atm_strike + wing_width
        
        chain_data = chain_result['options_chain']
        
        # Find required options
        strikes_data = {}
        for strike in [lower_strike, middle_strike, upper_strike]:
            row = next((r for r in chain_data if r['strike'] == strike), None)
            if not row:
                return {'status': 'ERROR', 'message': f'Strike {strike} not found in chain'}
            strikes_data[strike] = row
        
        # Create butterfly legs: Buy-Sell-Sell-Buy (middle sold twice)
        legs = []
        net_premium = 0
        
        # Buy lower strike
        lower_symbol = strikes_data[lower_strike].get(f'{option_type}_symbol')
        lower_price = strikes_data[lower_strike].get(f'{option_type}_ltp', 0)
        if not lower_symbol or lower_price <= 0:
            return {'status': 'ERROR', 'message': f'Invalid {option_type} option for lower strike'}
        
        legs.append({
            'symbol': lower_symbol,
            'action': 'BUY',
            'quantity': quantity,
            'strike': lower_strike,
            'option_type': option_type,
            'price': lower_price,
            'exchange': 'NFO'
        })
        net_premium -= lower_price * quantity
        
        # Sell 2x middle strike
        middle_symbol = strikes_data[middle_strike].get(f'{option_type}_symbol')
        middle_price = strikes_data[middle_strike].get(f'{option_type}_ltp', 0)
        if not middle_symbol or middle_price <= 0:
            return {'status': 'ERROR', 'message': f'Invalid {option_type} option for middle strike'}
        
        legs.append({
            'symbol': middle_symbol,
            'action': 'SELL',
            'quantity': quantity * 2,
            'strike': middle_strike,
            'option_type': option_type,
            'price': middle_price,
            'exchange': 'NFO'
        })
        net_premium += middle_price * quantity * 2
        
        # Buy upper strike
        upper_symbol = strikes_data[upper_strike].get(f'{option_type}_symbol')
        upper_price = strikes_data[upper_strike].get(f'{option_type}_ltp', 0)
        if not upper_symbol or upper_price <= 0:
            return {'status': 'ERROR', 'message': f'Invalid {option_type} option for upper strike'}
        
        legs.append({
            'symbol': upper_symbol,
            'action': 'BUY',
            'quantity': quantity,
            'strike': upper_strike,
            'option_type': option_type,
            'price': upper_price,
            'exchange': 'NFO'
        })
        net_premium -= upper_price * quantity
        
        # Calculate max profit/loss
        max_profit = (wing_width * quantity) + net_premium if net_premium < 0 else net_premium
        max_loss = abs(net_premium) if net_premium < 0 else (wing_width * quantity) - net_premium
        
        return {
            'status': 'SUCCESS',
            'strategy_name': f'{option_type} Butterfly Spread ({expiry_type.title()})',
            'expiry_date': expiry_date,
            'spot_price': spot_price,
            'center_strike': middle_strike,
            'legs': legs,
            'strategy_metrics': {
                'net_premium': round(net_premium, 2),
                'max_profit': round(max_profit, 2),
                'max_loss': round(max_loss, 2),
                'profit_at_expiry': middle_strike,
                'wing_width': wing_width,
                'strike_range': f'{lower_strike} - {middle_strike} - {upper_strike}'
            },
            'market_outlook': f'Neutral - expecting price to stay near {middle_strike}',
            'timestamp': dt.datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            'status': 'ERROR',
            'message': f'Butterfly strategy creation failed: {str(e)}'
        }


def create_ratio_spread_strategy(expiry_date: str = None, expiry_type: str = 'weekly',
                               option_type: str = 'CE', ratio: str = '1x2',
                               quantity: int = 25) -> Dict[str, Any]:
    """
    Create a Ratio Spread strategy (1x2 or 1x3).
    
    Args:
        expiry_date: Specific expiry (YYYY-MM-DD) or None for auto-selection
        expiry_type: 'weekly' or 'monthly' if expiry_date is None
        option_type: 'CE' or 'PE'
        ratio: '1x2' or '1x3' ratio
        quantity: Base quantity (will be multiplied by ratio)
    
    Returns:
        Dict with strategy details and execution plan
    """
    try:
        # Get expiry and spot price
        if not expiry_date:
            expiry_result = get_nifty_expiry_dates(expiry_type)
            if expiry_result['status'] != 'SUCCESS' or not expiry_result['expiries']:
                return {'status': 'ERROR', 'message': 'No expiry dates available'}
            expiry_date = expiry_result['expiries'][0]
        
        spot_result = get_nifty_spot_price()
        if spot_result['status'] != 'SUCCESS':
            return spot_result
        
        spot_price = spot_result['spot_price']
        
        # Get options chain
        chain_result = get_options_chain(expiry_date, expiry_type, strike_range=8)
        if chain_result['status'] != 'SUCCESS':
            return chain_result
        
        # Define strikes based on option type
        atm_strike = chain_result['atm_strike']
        if option_type == 'CE':
            buy_strike = atm_strike
            sell_strike = atm_strike + 100  # OTM
        else:
            buy_strike = atm_strike
            sell_strike = atm_strike - 100  # OTM
        
        chain_data = chain_result['options_chain']
        
        # Find required options
        buy_row = next((r for r in chain_data if r['strike'] == buy_strike), None)
        sell_row = next((r for r in chain_data if r['strike'] == sell_strike), None)
        
        if not buy_row or not sell_row:
            return {'status': 'ERROR', 'message': 'Required strikes not found in chain'}
        
        buy_symbol = buy_row.get(f'{option_type}_symbol')
        sell_symbol = sell_row.get(f'{option_type}_symbol')
        buy_price = buy_row.get(f'{option_type}_ltp', 0)
        sell_price = sell_row.get(f'{option_type}_ltp', 0)
        
        if not all([buy_symbol, sell_symbol, buy_price > 0, sell_price > 0]):
            return {'status': 'ERROR', 'message': 'Invalid option data for ratio spread'}
        
        # Determine sell multiplier
        sell_multiplier = 2 if ratio == '1x2' else 3
        
        legs = [
            {
                'symbol': buy_symbol,
                'action': 'BUY',
                'quantity': quantity,
                'strike': buy_strike,
                'option_type': option_type,
                'price': buy_price,
                'exchange': 'NFO'
            },
            {
                'symbol': sell_symbol,
                'action': 'SELL',
                'quantity': quantity * sell_multiplier,
                'strike': sell_strike,
                'option_type': option_type,
                'price': sell_price,
                'exchange': 'NFO'
            }
        ]
        
        net_premium = (sell_price * quantity * sell_multiplier) - (buy_price * quantity)
        
        return {
            'status': 'SUCCESS',
            'strategy_name': f'{ratio} {option_type} Ratio Spread ({expiry_type.title()})',
            'expiry_date': expiry_date,
            'spot_price': spot_price,
            'legs': legs,
            'strategy_metrics': {
                'net_premium': round(net_premium, 2),
                'max_profit': round(net_premium + (abs(sell_strike - buy_strike) * quantity), 2),
                'max_loss': 'Unlimited' if sell_multiplier > 1 else round(net_premium, 2),
                'ratio': ratio,
                'buy_strike': buy_strike,
                'sell_strike': sell_strike
            },
            'market_outlook': f'Moderately {"bullish" if option_type == "CE" else "bearish"}',
            'timestamp': dt.datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            'status': 'ERROR',
            'message': f'Ratio spread strategy creation failed: {str(e)}'
        }


def recommend_options_strategy(market_outlook: str, volatility_outlook: str,
                             time_horizon: str = 'short', risk_tolerance: str = 'medium') -> Dict[str, Any]:
    """
    Recommend options strategies based on market outlook and parameters.
    
    Args:
        market_outlook: 'bullish', 'bearish', 'neutral'
        volatility_outlook: 'high', 'low', 'increasing', 'decreasing'
        time_horizon: 'short' (weekly), 'medium' (monthly)
        risk_tolerance: 'low', 'medium', 'high'
    
    Returns:
        Dict with recommended strategies and reasoning
    """
    try:
        recommendations = []
        
        # Strategy matrix based on outlook
        if market_outlook.lower() == 'neutral':
            if volatility_outlook.lower() in ['low', 'decreasing']:
                if risk_tolerance.lower() == 'low':
                    recommendations.append({
                        'strategy': 'Iron Condor',
                        'reason': 'Limited risk, profits from low volatility and range-bound movement',
                        'risk_level': 'Low',
                        'profit_potential': 'Limited but consistent',
                        'function_call': 'create_iron_condor_strategy',
                        'suitability_score': 9
                    })
                else:
                    recommendations.append({
                        'strategy': 'Short Strangle',
                        'reason': 'Higher premium collection in low volatility environment',
                        'risk_level': 'Medium-High',
                        'profit_potential': 'Good',
                        'function_call': 'create_short_strangle_strategy',
                        'suitability_score': 8
                    })
            
            elif volatility_outlook.lower() in ['high', 'increasing']:
                recommendations.append({
                    'strategy': 'Long Straddle',
                    'reason': 'Benefits from high volatility regardless of direction',
                    'risk_level': 'Medium',
                    'profit_potential': 'High',
                    'function_call': 'create_long_straddle_strategy',
                    'suitability_score': 9
                })
                
                recommendations.append({
                    'strategy': 'Butterfly Spread',
                    'reason': 'Limited risk way to benefit from volatility decrease',
                    'risk_level': 'Low',
                    'profit_potential': 'Limited',
                    'function_call': 'create_butterfly_spread_strategy',
                    'suitability_score': 7
                })
        
        elif market_outlook.lower() == 'bullish':
            recommendations.append({
                'strategy': 'Call Butterfly Spread',
                'reason': 'Profits if market moves to specific upside target',
                'risk_level': 'Low',
                'profit_potential': 'Medium',
                'function_call': 'create_butterfly_spread_strategy',
                'parameters': {'option_type': 'CE'},
                'suitability_score': 8
            })
            
            if volatility_outlook.lower() in ['low', 'decreasing']:
                recommendations.append({
                    'strategy': 'Call Ratio Spread',
                    'reason': 'Bullish strategy with income generation',
                    'risk_level': 'Medium',
                    'profit_potential': 'Good',
                    'function_call': 'create_ratio_spread_strategy',
                    'parameters': {'option_type': 'CE'},
                    'suitability_score': 7
                })
        
        elif market_outlook.lower() == 'bearish':
            recommendations.append({
                'strategy': 'Put Butterfly Spread',
                'reason': 'Profits if market moves to specific downside target',
                'risk_level': 'Low',
                'profit_potential': 'Medium',
                'function_call': 'create_butterfly_spread_strategy',
                'parameters': {'option_type': 'PE'},
                'suitability_score': 8
            })
            
            if volatility_outlook.lower() in ['low', 'decreasing']:
                recommendations.append({
                    'strategy': 'Put Ratio Spread',
                    'reason': 'Bearish strategy with controlled risk',
                    'risk_level': 'Medium',
                    'profit_potential': 'Good',
                    'function_call': 'create_ratio_spread_strategy',
                    'parameters': {'option_type': 'PE'},
                    'suitability_score': 7
                })
        
        # Time horizon adjustments
        expiry_recommendation = 'weekly' if time_horizon.lower() == 'short' else 'monthly'
        
        # Sort recommendations by suitability score
        recommendations.sort(key=lambda x: x.get('suitability_score', 0), reverse=True)
        
        # Add general guidance
        general_guidance = {
            'expiry_type': expiry_recommendation,
            'reasoning': f'{time_horizon.title()} time horizon suggests {expiry_recommendation} expiries',
            'risk_considerations': [
                'Weekly options have higher Gamma (more sensitive to price moves)',
                'Monthly options have lower time decay but require larger moves',
                'Consider position sizing based on account size',
                'Always have exit plan before entering',
                'Monitor Greeks and adjust positions as needed'
            ],
            'market_conditions': {
                'best_for_high_iv': ['Short Strangle', 'Iron Condor'],
                'best_for_low_iv': ['Long Straddle', 'Long Strangle'],
                'best_for_trending': ['Ratio Spreads', 'Directional Butterflies'],
                'best_for_range_bound': ['Iron Condor', 'Short Straddle']
            }
        }
        
        if not recommendations:
            recommendations.append({
                'strategy': 'Market Assessment',
                'reason': 'Current market conditions require careful analysis',
                'risk_level': 'Variable',
                'profit_potential': 'Depends on execution',
                'function_call': 'analyze_options_flow',
                'note': 'Consider waiting for clearer market signals',
                'suitability_score': 5
            })
        
        return {
            'status': 'SUCCESS',
            'input_parameters': {
                'market_outlook': market_outlook,
                'volatility_outlook': volatility_outlook,
                'time_horizon': time_horizon,
                'risk_tolerance': risk_tolerance
            },
            'recommendations': recommendations,
            'general_guidance': general_guidance,
            'total_strategies': len(recommendations),
            'top_recommendation': recommendations[0] if recommendations else None,
            'timestamp': dt.datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            'status': 'ERROR',
            'message': f'Strategy recommendation failed: {str(e)}'
        }


def analyze_strategy_greeks(strategy_legs: List[Dict[str, Any]], spot_price: float) -> Dict[str, Any]:
    """
    Analyze Greeks for a complete strategy.
    
    Args:
        strategy_legs: List of strategy legs with option details
        spot_price: Current spot price
    
    Returns:
        Dict with strategy Greeks analysis
    """
    try:
        # Import calculation tools
        from nifty_calculation_tools import calculate_option_greeks
        
        total_delta = 0
        total_gamma = 0
        total_theta = 0
        total_vega = 0
        
        leg_analysis = []
        
        for leg in strategy_legs:
            # Get leg details
            strike = leg.get('strike')
            option_type = leg.get('option_type')
            action = leg.get('action')
            quantity = leg.get('quantity')
            
            # Calculate Greeks for this leg (using generic expiry)
            expiry_date = (dt.date.today() + dt.timedelta(days=7)).strftime('%Y-%m-%d')
            
            greeks_result = calculate_option_greeks(
                spot_price=spot_price,
                strike=strike,
                expiry_date=expiry_date,
                option_type=option_type
            )
            
            if greeks_result['status'] == 'SUCCESS':
                greeks = greeks_result['greeks']
                
                # Apply sign based on action (BUY = +, SELL = -)
                multiplier = quantity if action == 'BUY' else -quantity
                
                leg_delta = greeks['delta'] * multiplier
                leg_gamma = greeks['gamma'] * multiplier
                leg_theta = greeks['theta_daily'] * multiplier
                leg_vega = greeks['vega'] * multiplier
                
                total_delta += leg_delta
                total_gamma += leg_gamma
                total_theta += leg_theta
                total_vega += leg_vega
                
                leg_analysis.append({
                    'symbol': leg.get('symbol', 'Unknown'),
                    'strike': strike,
                    'option_type': option_type,
                    'action': action,
                    'quantity': quantity,
                    'greeks': {
                        'delta': round(leg_delta, 4),
                        'gamma': round(leg_gamma, 6),
                        'theta': round(leg_theta, 2),
                        'vega': round(leg_vega, 2)
                    }
                })
        
        # Risk analysis
        risk_metrics = {
            'delta_neutral': abs(total_delta) < 0.1,
            'gamma_risk': 'High' if abs(total_gamma) > 0.01 else 'Low',
            'theta_decay': 'Positive' if total_theta > 0 else 'Negative',
            'vega_exposure': 'High' if abs(total_vega) > 10 else 'Low'
        }
        
        return {
            'status': 'SUCCESS',
            'strategy_greeks': {
                'total_delta': round(total_delta, 4),
                'total_gamma': round(total_gamma, 6),
                'total_theta': round(total_theta, 2),
                'total_vega': round(total_vega, 2)
            },
            'leg_analysis': leg_analysis,
            'risk_metrics': risk_metrics,
            'recommendations': {
                'hedge_delta': 'Consider delta hedging' if abs(total_delta) > 0.5 else 'Delta exposure manageable',
                'gamma_warning': 'High gamma - position sensitive to price moves' if abs(total_gamma) > 0.01 else 'Gamma risk controlled',
                'time_decay': 'Time working in favor' if total_theta > 0 else 'Time working against position'
            },
            'timestamp': dt.datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            'status': 'ERROR',
            'message': f'Strategy Greeks analysis failed: {str(e)}'
        }


# Strategy Creation Tools Registry
STRATEGY_CREATION_TOOLS = {
    'create_long_straddle_strategy': create_long_straddle_strategy,
    'create_short_strangle_strategy': create_short_strangle_strategy,
    'create_iron_condor_strategy': create_iron_condor_strategy,
    'create_butterfly_spread_strategy': create_butterfly_spread_strategy,
    'create_ratio_spread_strategy': create_ratio_spread_strategy,
    'recommend_options_strategy': recommend_options_strategy,
    'analyze_strategy_greeks': analyze_strategy_greeks
}


if __name__ == "__main__":
    """
    Test strategy creation tools
    """
    print("=== NIFTY Options - Strategy Creation Tools Test ===\n")
    
    # Test strategy recommendation
    recommendation_result = recommend_options_strategy(
        market_outlook='neutral',
        volatility_outlook='low',
        time_horizon='short',
        risk_tolerance='medium'
    )
    print(f"Strategy Recommendation: {recommendation_result['status']}")
    if recommendation_result['status'] == 'SUCCESS':
        print(f"Total recommendations: {recommendation_result['total_strategies']}")
        if recommendation_result['recommendations']:
            top_rec = recommendation_result['recommendations'][0]
            print(f"Top recommendation: {top_rec['strategy']}")
            print(f"Reason: {top_rec['reason']}")
    
    # Note: Other strategy tests require market connection
    print("\n✅ Strategy Creation Tools loaded!")
    print("Note: Full testing requires active market connection")

    # Load environment variables
    load_dotenv(dotenv_path='./.env')
    api_key = os.getenv("kite_api_key")
    api_secret = os.getenv("kite_api_secret")
    access_token = None
    if os.path.exists("access_token.txt"):
        with open("access_token.txt", "r") as f:
            access_token = f.read().strip()
    # Initialize connection if tools are available
    if CONNECTION_TOOLS_AVAILABLE:
        print("Initializing Kite Connect session...")
        conn_result = initialize_connection(api_key, api_secret, access_token)
        print("Connection:", conn_result)
        # Fetch and print live NIFTY spot price
        spot = get_nifty_spot_price()
        print("Live NIFTY Spot Price:", spot)
        # Fetch and print live options chain
        chain = get_options_chain()
        print("Live Options Chain:", chain)
    else:
        print("Connection tools not available. Skipping live data workflow.")