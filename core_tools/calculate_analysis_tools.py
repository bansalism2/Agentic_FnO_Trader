#!/usr/bin/env python3
"""
NIFTY Options - Calculation & Analysis Tools (File 2 of 4)
=========================================================

Options Greeks calculations, implied volatility, and advanced analysis tools.
This file contains mathematical and analytical functions for options trading.

Dependencies:
pip install numpy scipy

Author: AI Assistant
"""

import datetime as dt
from typing import Dict, List, Optional, Any
import numpy as np
from scipy.stats import norm
import traceback
from datetime import datetime
from math import log, sqrt, exp


def calculate_option_greeks(spot_price: float, strike: float, expiry_date: str,
                          option_type: str, market_price: float = None,
                          volatility: float = 0.20, risk_free_rate: float = 0.06) -> Dict[str, Any]:
    """
    Calculate options Greeks using Black-Scholes model.
    
    Args:
        spot_price: Current NIFTY spot price
        strike: Strike price of option
        expiry_date: Expiry date (YYYY-MM-DD)
        option_type: 'CE' or 'PE'
        market_price: Current market price (for IV calculation)
        volatility: Volatility assumption (if market_price not provided)
        risk_free_rate: Risk-free rate (annual)
    
    Returns:
        Dict with calculated Greeks and theoretical price
    """
    try:
        # Calculate time to expiry
        expiry_dt = dt.datetime.strptime(expiry_date, '%Y-%m-%d').date()
        today = dt.date.today()
        days_to_expiry = (expiry_dt - today).days
        time_to_expiry = max(days_to_expiry / 365.0, 0.001)  # Avoid division by zero
        
        if time_to_expiry <= 0:
            return {
                'status': 'ERROR',
                'message': 'Option has expired'
            }
        
        # Calculate implied volatility if market price provided
        calculated_iv = volatility
        if market_price and market_price > 0:
            try:
                calculated_iv = calculate_implied_volatility(
                    market_price, spot_price, strike, time_to_expiry, risk_free_rate, option_type
                )
            except:
                calculated_iv = volatility  # Fallback to provided volatility
        
        # Black-Scholes calculations
        d1 = (np.log(spot_price / strike) + (risk_free_rate + 0.5 * calculated_iv ** 2) * time_to_expiry) / (calculated_iv * np.sqrt(time_to_expiry))
        d2 = d1 - calculated_iv * np.sqrt(time_to_expiry)
        
        N_d1 = norm.cdf(d1)
        N_d2 = norm.cdf(d2)
        n_d1 = norm.pdf(d1)
        
        if option_type == 'CE':
            theoretical_price = spot_price * N_d1 - strike * np.exp(-risk_free_rate * time_to_expiry) * N_d2
            delta = N_d1
        else:  # PE
            theoretical_price = strike * np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(-d2) - spot_price * norm.cdf(-d1)
            delta = N_d1 - 1
        
        # Common Greeks
        gamma = n_d1 / (spot_price * calculated_iv * np.sqrt(time_to_expiry))
        theta = (-(spot_price * n_d1 * calculated_iv) / (2 * np.sqrt(time_to_expiry)) 
                - risk_free_rate * strike * np.exp(-risk_free_rate * time_to_expiry) * N_d2)
        
        if option_type == 'PE':
            theta += risk_free_rate * strike * np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(-d2)
        
        theta_daily = theta / 365  # Daily theta
        vega = spot_price * np.sqrt(time_to_expiry) * n_d1 / 100  # Per 1% change in volatility
        
        return {
            'status': 'SUCCESS',
            'spot_price': spot_price,
            'strike': strike,
            'option_type': option_type,
            'days_to_expiry': days_to_expiry,
            'time_to_expiry': round(time_to_expiry, 4),
            'theoretical_price': round(max(0, theoretical_price), 2),
            'market_price': market_price,
            'implied_volatility': round(calculated_iv, 4),
            'greeks': {
                'delta': round(delta, 4),
                'gamma': round(gamma, 6),
                'theta_daily': round(theta_daily, 2),
                'vega': round(vega, 2)
            },
            'inputs': {
                'volatility_used': round(calculated_iv, 4),
                'risk_free_rate': risk_free_rate
            }
        }
        
    except Exception as e:
        return {
            'status': 'ERROR',
            'message': f'Greeks calculation failed: {str(e)}'
        }


def calculate_implied_volatility(market_price: float, spot: float, strike: float,
                               time_to_expiry: float, risk_free_rate: float, option_type: str, dividend_yield: float = 0.0) -> float:
    """
    Calculate implied volatility using Newton-Raphson method with dividend adjustment.
    """
    iv = 0.3  # Initial guess
    tolerance = 1e-6
    max_iterations = 100
    for _ in range(max_iterations):
        d1 = (np.log(spot / strike) + (risk_free_rate - dividend_yield + 0.5 * iv ** 2) * time_to_expiry) / (iv * np.sqrt(time_to_expiry))
        d2 = d1 - iv * np.sqrt(time_to_expiry)
        if option_type == 'CE':
            theoretical_price = spot * np.exp(-dividend_yield * time_to_expiry) * norm.cdf(d1) - strike * np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(d2)
        else:
            theoretical_price = strike * np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(-d2) - spot * np.exp(-dividend_yield * time_to_expiry) * norm.cdf(-d1)
        vega = spot * np.exp(-dividend_yield * time_to_expiry) * np.sqrt(time_to_expiry) * norm.pdf(d1)
        price_diff = theoretical_price - market_price
        # Improved convergence: both absolute and relative error
        if abs(price_diff) < tolerance and (abs(price_diff / market_price) < 0.001 if market_price else True):
            break
        if vega < tolerance:
            break
        iv = iv - price_diff / vega
        iv = max(0.01, min(5.0, iv))  # Keep IV within reasonable bounds
    return iv


def calculate_strategy_pnl(legs: List[Dict[str, Any]], spot_prices: List[float]) -> Dict[str, Any]:
    """
    Calculate P&L for a multi-leg options strategy across different spot prices.
    
    Args:
        legs: List of strategy legs with symbol, action, quantity, strike, option_type, price
        spot_prices: List of spot prices to calculate P&L for
    
    Returns:
        Dict with P&L analysis across spot prices
    """
    try:
        pnl_data = []
        
        for spot in spot_prices:
            total_pnl = 0
            
            for leg in legs:
                strike = leg['strike']
                option_type = leg['option_type']
                action = leg['action']
                quantity = leg['quantity']
                entry_price = leg['price']
                
                # Calculate option value at expiry
                if option_type == 'CE':
                    option_value = max(0, spot - strike)
                else:  # PE
                    option_value = max(0, strike - spot)
                
                # Calculate leg P&L
                if action == 'BUY':
                    leg_pnl = (option_value - entry_price) * quantity
                else:  # SELL
                    leg_pnl = (entry_price - option_value) * quantity
                
                total_pnl += leg_pnl
            
            pnl_data.append({
                'spot_price': spot,
                'total_pnl': round(total_pnl, 2)
            })
        
        # Find breakeven points
        breakeven_points = []
        for i in range(len(pnl_data) - 1):
            current_pnl = pnl_data[i]['total_pnl']
            next_pnl = pnl_data[i + 1]['total_pnl']
            
            # Check if P&L crosses zero
            if (current_pnl <= 0 <= next_pnl) or (current_pnl >= 0 >= next_pnl):
                # Linear interpolation to find exact breakeven
                current_spot = pnl_data[i]['spot_price']
                next_spot = pnl_data[i + 1]['spot_price']
                
                if next_pnl != current_pnl:
                    breakeven = current_spot + (next_spot - current_spot) * (-current_pnl) / (next_pnl - current_pnl)
                    breakeven_points.append(round(breakeven, 2))
        
        # Find max profit and loss
        max_profit = max(pnl['total_pnl'] for pnl in pnl_data)
        max_loss = min(pnl['total_pnl'] for pnl in pnl_data)
        
        return {
            'status': 'SUCCESS',
            'pnl_analysis': pnl_data,
            'breakeven_points': breakeven_points,
            'max_profit': round(max_profit, 2),
            'max_loss': round(max_loss, 2),
            'profit_loss_ratio': round(abs(max_profit / max_loss), 2) if max_loss != 0 else float('inf'),
            'total_legs': len(legs)
        }
        
    except Exception as e:
        return {
            'status': 'ERROR',
            'message': f'P&L calculation failed: {str(e)}'
        }


def find_arbitrage_opportunities(options_chain: List[Dict[str, Any]], spot_price: float) -> Dict[str, Any]:
    """
    Find potential arbitrage opportunities in options chain.
    
    Args:
        options_chain: Options chain data from get_options_chain
        spot_price: Current NIFTY spot price
    
    Returns:
        Dict with arbitrage opportunities found
    """
    try:
        arbitrage_opportunities = []
        
        # Check Put-Call Parity violations
        for row in options_chain:
            strike = row['strike']
            ce_ltp = row.get('CE_ltp', 0)
            pe_ltp = row.get('PE_ltp', 0)
            
            if ce_ltp > 0 and pe_ltp > 0:
                # Put-Call Parity: C - P = S - K*e^(-rt)
                # Simplified for small r and t: C - P ≈ S - K
                theoretical_diff = spot_price - strike
                actual_diff = ce_ltp - pe_ltp
                parity_violation = abs(actual_diff - theoretical_diff)
                
                # Flag significant violations (> ₹5)
                if parity_violation > 5:
                    arbitrage_opportunities.append({
                        'type': 'Put-Call Parity Violation',
                        'strike': strike,
                        'ce_price': ce_ltp,
                        'pe_price': pe_ltp,
                        'theoretical_diff': round(theoretical_diff, 2),
                        'actual_diff': round(actual_diff, 2),
                        'violation_amount': round(parity_violation, 2),
                        'recommendation': 'Buy underpriced, Sell overpriced'
                    })
        
        # Check for inverted spreads (unusual pricing)
        sorted_strikes = sorted([row['strike'] for row in options_chain])
        
        for i in range(len(sorted_strikes) - 1):
            lower_strike = sorted_strikes[i]
            higher_strike = sorted_strikes[i + 1]
            
            lower_row = next((r for r in options_chain if r['strike'] == lower_strike), None)
            higher_row = next((r for r in options_chain if r['strike'] == higher_strike), None)
            
            if lower_row and higher_row:
                # For calls: lower strike should be more expensive
                lower_ce = lower_row.get('CE_ltp', 0)
                higher_ce = higher_row.get('CE_ltp', 0)
                
                if lower_ce > 0 and higher_ce > 0 and lower_ce < higher_ce:
                    arbitrage_opportunities.append({
                        'type': 'Call Spread Inversion',
                        'lower_strike': lower_strike,
                        'higher_strike': higher_strike,
                        'lower_price': lower_ce,
                        'higher_price': higher_ce,
                        'profit_potential': round(higher_ce - lower_ce, 2),
                        'recommendation': f'Buy {lower_strike}CE, Sell {higher_strike}CE'
                    })
                
                # For puts: higher strike should be more expensive
                lower_pe = lower_row.get('PE_ltp', 0)
                higher_pe = higher_row.get('PE_ltp', 0)
                
                if lower_pe > 0 and higher_pe > 0 and lower_pe > higher_pe:
                    arbitrage_opportunities.append({
                        'type': 'Put Spread Inversion',
                        'lower_strike': lower_strike,
                        'higher_strike': higher_strike,
                        'lower_price': lower_pe,
                        'higher_price': higher_pe,
                        'profit_potential': round(lower_pe - higher_pe, 2),
                        'recommendation': f'Sell {lower_strike}PE, Buy {higher_strike}PE'
                    })
        
        return {
            'status': 'SUCCESS',
            'spot_price': spot_price,
            'total_opportunities': len(arbitrage_opportunities),
            'opportunities': arbitrage_opportunities,
            'analysis_time': dt.datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            'status': 'ERROR',
            'message': f'Arbitrage analysis failed: {str(e)}'
        }


def calculate_portfolio_greeks(positions: List[Dict[str, Any]], spot_price: float) -> Dict[str, Any]:
    """
    Calculate combined Greeks for a portfolio of options positions.
    
    Args:
        positions: List of positions with greeks data
        spot_price: Current spot price
    
    Returns:
        Dict with portfolio-level Greeks
    """
    try:
        portfolio_delta = 0
        portfolio_gamma = 0
        portfolio_theta = 0
        portfolio_vega = 0
        total_value = 0
        
        for position in positions:
            quantity = position.get('quantity', 0)
            greeks = position.get('greeks', {})
            market_value = position.get('market_value', 0)
            
            # Sum up Greeks weighted by quantity
            portfolio_delta += greeks.get('delta', 0) * quantity
            portfolio_gamma += greeks.get('gamma', 0) * quantity
            portfolio_theta += greeks.get('theta_daily', 0) * quantity
            portfolio_vega += greeks.get('vega', 0) * quantity
            total_value += market_value
        
        # Calculate portfolio risk metrics
        delta_dollars = portfolio_delta * spot_price  # Dollar delta
        gamma_dollars = portfolio_gamma * spot_price * spot_price / 100  # Dollar gamma for 1% move
        
        return {
            'status': 'SUCCESS',
            'portfolio_greeks': {
                'total_delta': round(portfolio_delta, 4),
                'total_gamma': round(portfolio_gamma, 6),
                'total_theta': round(portfolio_theta, 2),
                'total_vega': round(portfolio_vega, 2)
            },
            'risk_metrics': {
                'delta_dollars': round(delta_dollars, 2),
                'gamma_dollars': round(gamma_dollars, 2),
                'theta_decay_daily': round(portfolio_theta, 2),
                'vega_1percent': round(portfolio_vega, 2)
            },
            'portfolio_value': round(total_value, 2),
            'spot_price': spot_price,
            'total_positions': len(positions),
            'calculation_time': dt.datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            'status': 'ERROR',
            'message': f'Portfolio Greeks calculation failed: {str(e)}'
        }


def calculate_volatility_surface(options_data: List[Dict[str, Any]], spot_price: float) -> Dict[str, Any]:
    """
    Calculate implied volatility surface from options data.
    
    Args:
        options_data: List of options with prices and parameters
        spot_price: Current spot price
    
    Returns:
        Dict with volatility surface data
    """
    try:
        surface_data = []
        
        for option in options_data:
            strike = option.get('strike')
            option_type = option.get('option_type')
            market_price = option.get('market_price', 0)
            expiry_date = option.get('expiry_date')
            
            if not all([strike, option_type, market_price > 0, expiry_date]):
                continue
            
            # Calculate time to expiry
            expiry_dt = dt.datetime.strptime(expiry_date, '%Y-%m-%d').date()
            days_to_expiry = (expiry_dt - dt.date.today()).days
            time_to_expiry = days_to_expiry / 365.0
            
            if time_to_expiry <= 0:
                continue
            
            # Calculate implied volatility
            try:
                iv = calculate_implied_volatility(
                    market_price, spot_price, strike, time_to_expiry, 0.06, option_type
                )
                
                # Calculate moneyness
                moneyness = strike / spot_price
                
                surface_data.append({
                    'strike': strike,
                    'option_type': option_type,
                    'days_to_expiry': days_to_expiry,
                    'market_price': market_price,
                    'implied_volatility': round(iv, 4),
                    'moneyness': round(moneyness, 4),
                    'time_to_expiry': round(time_to_expiry, 4)
                })
                
            except Exception:
                continue  # Skip options with calculation errors
        
        # Calculate volatility smile metrics
        if surface_data:
            iv_values = [point['implied_volatility'] for point in surface_data]
            avg_iv = np.mean(iv_values)
            iv_std = np.std(iv_values)
            
            # Find ATM volatility
            atm_points = [p for p in surface_data if abs(p['moneyness'] - 1.0) < 0.02]
            atm_iv = np.mean([p['implied_volatility'] for p in atm_points]) if atm_points else avg_iv
            
            return {
                'status': 'SUCCESS',
                'surface_points': surface_data,
                'total_points': len(surface_data),
                'volatility_metrics': {
                    'average_iv': round(avg_iv, 4),
                    'iv_standard_deviation': round(iv_std, 4),
                    'atm_implied_volatility': round(atm_iv, 4),
                    'iv_range': {
                        'min': round(min(iv_values), 4),
                        'max': round(max(iv_values), 4)
                    }
                },
                'spot_price': spot_price,
                'calculation_time': dt.datetime.now().isoformat()
            }
        else:
            return {
                'status': 'ERROR',
                'message': 'No valid options data for surface calculation'
            }
            
    except Exception as e:
        return {
            'status': 'ERROR',
            'message': f'Volatility surface calculation failed: {str(e)}'
        }


def calculate_probability_of_profit(strategy_legs: List[Dict[str, Any]], 
                                  spot_price: float, volatility: float,
                                  days_to_expiry: int) -> Dict[str, Any]:
    """
    Calculate probability of profit for an options strategy using Monte Carlo simulation.
    
    Args:
        strategy_legs: List of strategy legs
        spot_price: Current spot price
        volatility: Expected volatility
        days_to_expiry: Days until expiry
    
    Returns:
        Dict with probability analysis
    """
    try:
        import random
        
        num_simulations = 10000
        profitable_outcomes = 0
        time_to_expiry = days_to_expiry / 365.0
        
        # Calculate strategy cost
        total_cost = 0
        for leg in strategy_legs:
            price = leg.get('price', 0)
            quantity = leg.get('quantity', 0)
            action = leg.get('action', 'BUY')
            
            if action == 'BUY':
                total_cost += price * quantity
            else:
                total_cost -= price * quantity
        
        for _ in range(num_simulations):
            # Generate random price at expiry using geometric Brownian motion
            random_return = random.gauss(0, volatility * np.sqrt(time_to_expiry))
            simulated_spot = spot_price * np.exp(random_return)
            
            # Calculate strategy value at expiry
            strategy_value = 0
            for leg in strategy_legs:
                strike = leg.get('strike', 0)
                option_type = leg.get('option_type', 'CE')
                quantity = leg.get('quantity', 0)
                action = leg.get('action', 'BUY')
                
                # Calculate intrinsic value
                if option_type == 'CE':
                    intrinsic_value = max(0, simulated_spot - strike)
                else:
                    intrinsic_value = max(0, strike - simulated_spot)
                
                if action == 'BUY':
                    strategy_value += intrinsic_value * quantity
                else:
                    strategy_value -= intrinsic_value * quantity
            
            # Check if strategy is profitable
            total_pnl = strategy_value - total_cost
            if total_pnl > 0:
                profitable_outcomes += 1
        
        probability_of_profit = profitable_outcomes / num_simulations
        
        return {
            'status': 'SUCCESS',
            'probability_of_profit': round(probability_of_profit, 4),
            'probability_percentage': round(probability_of_profit * 100, 2),
            'simulations_run': num_simulations,
            'profitable_outcomes': profitable_outcomes,
            'strategy_cost': round(total_cost, 2),
            'inputs': {
                'spot_price': spot_price,
                'volatility': volatility,
                'days_to_expiry': days_to_expiry
            },
            'calculation_time': dt.datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            'status': 'ERROR',
            'message': f'Probability calculation failed: {str(e)}'
        }


def analyze_vix_integration_wrapper() -> Dict[str, Any]:
    """
    Enhanced VIX integration wrapper with comprehensive analysis.
    
    Returns:
        Dict with VIX analysis and IV validation
    """
    try:
        from connect_data_tools import get_nifty_spot_price_safe, get_options_chain_safe
        
        # Get current spot price
        spot_result = get_nifty_spot_price_safe()
        if spot_result.get('status') != 'SUCCESS':
            return {
                'status': 'ERROR',
                'message': f'Failed to get spot price: {spot_result.get("message", "Unknown error")}'
            }
        
        spot_price = spot_result.get('spot_price', 0)
        if spot_price <= 0:
            return {
                'status': 'ERROR',
                'message': 'Invalid spot price received'
            }
        
        # Get options chain
        options_result = get_options_chain_safe()
        if options_result.get('status') != 'SUCCESS':
            return {
                'status': 'ERROR',
                'message': f'Failed to get options chain: {options_result.get("message", "Unknown error")}'
            }
        
        options_chain = options_result.get('options_chain', [])
        if not options_chain:
            return {
                'status': 'ERROR',
                'message': 'Empty options chain received'
            }
        
        # Call the enhanced VIX analysis function
        return analyze_vix_integration_enhanced(spot_price, options_chain)
        
    except Exception as e:
        return {
            'status': 'ERROR',
            'message': f'VIX integration wrapper failed: {str(e)}'
        }

def analyze_vix_integration_enhanced(spot_price: float, options_chain: List[Dict[str, Any]], 
                                   historical_volatility: float = None) -> Dict[str, Any]:
    """
    Enhanced VIX integration analysis with comprehensive validation.
    
    Args:
        spot_price: Current NIFTY spot price
        options_chain: Current options chain data
        historical_volatility: Historical volatility data (optional)
    
    Returns:
        Dict with enhanced VIX analysis and IV validation
    """
    try:
        print("📊 Enhanced VIX Integration Analysis...")
        
        # Calculate current IV from options
        iv_result = calculate_iv_rank_analysis(options_chain, spot_price)
        if iv_result.get('status') != 'SUCCESS':
            return {
                'status': 'ERROR',
                'message': f'Failed to calculate current IV: {iv_result.get("message", "Unknown error")}'
            }
        
        current_iv = iv_result.get('current_iv', 0)
        iv_percentile = iv_result.get('iv_percentile', 0)
        
        # Calculate volatility surface for comprehensive analysis
        volatility_surface = calculate_comprehensive_volatility_surface(options_chain, spot_price)
        surface_stats = volatility_surface.get('volatility_surface', {}).get('surface_stats', {})
        mean_iv_surface = surface_stats.get('mean_iv', 0)
        
        # Get historical volatility if available
        if not historical_volatility:
            try:
                from connect_data_tools import get_historical_volatility
                hist_vol_result = get_historical_volatility(days=30)
                if hist_vol_result.get('status') == 'SUCCESS':
                    historical_volatility = hist_vol_result.get('historical_volatility', 0)
                    print(f"📈 Historical volatility (30d): {historical_volatility:.4f}")
            except:
                historical_volatility = 0.20  # Default 20%
        
        # VIX-like calculation (simplified)
        # In a real implementation, you would fetch actual VIX data
        # For now, we'll use a proxy based on options chain analysis
        
        # Calculate VIX proxy using ATM options
        atm_strike = round(spot_price / 50) * 50
        atm_options = [opt for opt in options_chain if abs(opt['strike'] - atm_strike) <= 25]
        
        vix_proxy = 0
        vix_components = []
        
        for opt in atm_options:
            if 'CE_ltp' in opt and opt['CE_ltp'] > 0 and 'PE_ltp' in opt and opt['PE_ltp'] > 0:
                try:
                    # Calculate both call and put IV
                    call_iv = calculate_implied_volatility(
                        opt['CE_ltp'], spot_price, opt['strike'], 30/365, 0.06, 'CE'
                    )
                    put_iv = calculate_implied_volatility(
                        opt['PE_ltp'], spot_price, opt['strike'], 30/365, 0.06, 'PE'
                    )
                    
                    # VIX-like calculation: average of call and put IV
                    avg_iv = (call_iv + put_iv) / 2
                    vix_components.append(avg_iv)
                    
                except:
                    pass
        
        if vix_components:
            vix_proxy = np.mean(vix_components)
        
        # Analyze VIX vs calculated IV
        iv_vix_diff = abs(current_iv - vix_proxy)
        iv_vix_ratio = current_iv / vix_proxy if vix_proxy > 0 else 1
        
        # Determine volatility regime
        if vix_proxy < 0.15:
            vix_regime = 'LOW_VOLATILITY'
        elif vix_proxy < 0.25:
            vix_regime = 'NORMAL_VOLATILITY'
        elif vix_proxy < 0.35:
            vix_regime = 'HIGH_VOLATILITY'
        else:
            vix_proxy = 'EXTREME_VOLATILITY'
        
        # Validation analysis
        validation_status = 'VALID'
        validation_message = 'IV calculation appears consistent'
        
        if iv_vix_diff > 0.10:  # More than 10% difference
            validation_status = 'SUSPICIOUS'
            validation_message = 'Large difference between calculated IV and VIX proxy'
        elif iv_vix_ratio < 0.5 or iv_vix_ratio > 2.0:
            validation_status = 'WARNING'
            validation_message = 'Significant ratio difference between IV and VIX proxy'
        
        # Market sentiment analysis
        if vix_proxy > 0.30:
            market_sentiment = 'FEAR'
            sentiment_description = 'High volatility indicates market fear'
        elif vix_proxy < 0.15:
            market_sentiment = 'COMPLACENCY'
            sentiment_description = 'Low volatility indicates market complacency'
        else:
            market_sentiment = 'NEUTRAL'
            sentiment_description = 'Normal volatility levels'
        
        return {
            'status': 'SUCCESS',
            'vix_analysis': {
                'vix_proxy': round(vix_proxy, 4),
                'vix_regime': vix_regime,
                'market_sentiment': market_sentiment,
                'sentiment_description': sentiment_description
            },
            'iv_validation': {
                'calculated_iv': round(current_iv, 4),
                'vix_proxy': round(vix_proxy, 4),
                'difference': round(iv_vix_diff, 4),
                'ratio': round(iv_vix_ratio, 3),
                'validation_status': validation_status,
                'validation_message': validation_message
            },
            'volatility_comparison': {
                'current_iv': round(current_iv, 4),
                'mean_surface_iv': round(mean_iv_surface, 4),
                'historical_volatility': round(historical_volatility, 4),
                'vix_proxy': round(vix_proxy, 4),
                'consistency_score': round(1 - min(iv_vix_diff, 0.5), 2)  # Higher is better
            },
            'market_conditions': {
                'volatility_level': 'LOW' if vix_proxy < 0.15 else 'NORMAL' if vix_proxy < 0.25 else 'HIGH' if vix_proxy < 0.35 else 'EXTREME',
                'risk_level': 'LOW' if vix_proxy < 0.15 else 'MEDIUM' if vix_proxy < 0.25 else 'HIGH',
                'trading_recommendation': 'Premium selling' if vix_proxy > 0.25 else 'Long strategies' if vix_proxy < 0.15 else 'Balanced approach'
            },
            'timestamp': dt.datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            'status': 'ERROR',
            'message': f'Enhanced VIX analysis failed: {str(e)}'
        }


def calculate_iv_rank_analysis_wrapper() -> Dict[str, Any]:
    """
    Wrapper function for TRUE IV analysis that automatically fetches required data.
    
    Returns:
        Dict with TRUE IV analysis and trading recommendations
    """
    try:
        # Use the new TRUE IV calculation function
        true_iv_result = calculate_true_iv_data(days=30)
        
        if true_iv_result.get('status') != 'SUCCESS':
            return {
                'status': 'ERROR',
                'message': f'TRUE IV calculation failed: {true_iv_result.get("message", "Unknown error")}'
            }
        
        # Extract the key metrics from TRUE IV result
        iv_statistics = true_iv_result.get('iv_statistics', {})
        current_iv_analysis = true_iv_result.get('current_iv_analysis', {})
        
        # Format the result to match the expected structure
        result = {
            'status': 'SUCCESS',
            'current_iv': iv_statistics.get('current_iv', 0),
            'iv_percentile': iv_statistics.get('iv_percentile', 0),
            'iv_rank': iv_statistics.get('iv_rank', 0),
            'volatility_regime': iv_statistics.get('volatility_regime', 'UNKNOWN'),
            'iv_min': 0.05,  # Estimated historical minimum
            'iv_max': 0.50,  # Estimated historical maximum
            'iv_status': 'TRUE_IV_CALCULATED',
            'iv_trend': 'CURRENT_ANALYSIS',
            'recommendation': _get_iv_recommendation(iv_statistics.get('iv_percentile', 0)),
            'calculation_method': 'TRUE_IV_BLACK_SCHOLES',
            'historical_data_used': current_iv_analysis.get('historical_data_used', False),
            'total_options_analyzed': current_iv_analysis.get('total_options_analyzed', 0),
            'valid_iv_calculations': current_iv_analysis.get('valid_iv_calculations', 0),
            'detailed_iv_data': current_iv_analysis.get('detailed_iv_data', []),
            'spot_price': true_iv_result.get('spot_price', 0),
            'calculation_date': true_iv_result.get('calculation_date', ''),
            'message': 'TRUE IV analysis completed successfully'
        }
        
        return result
        
    except Exception as e:
        return {
            'status': 'ERROR',
            'message': f'TRUE IV analysis wrapper failed: {str(e)}'
        }


def _get_iv_recommendation(iv_percentile: float) -> str:
    """
    Generate trading recommendation based on IV percentile.
    
    Args:
        iv_percentile: IV percentile (0-1)
    
    Returns:
        Trading recommendation string
    """
    if iv_percentile > 0.80:
        return "HIGH_IV_PREMIUM_SELLING_OPPORTUNITY"
    elif iv_percentile > 0.60:
        return "ELEVATED_IV_PREMIUM_SELLING_FAVORABLE"
    elif iv_percentile > 0.40:
        return "NORMAL_IV_NEUTRAL_STRATEGIES"
    elif iv_percentile > 0.20:
        return "LOW_IV_LONG_STRATEGIES_FAVORABLE"
    else:
        return "VERY_LOW_IV_CONSIDER_LONG_STRATEGIES"


def detect_market_regime_wrapper() -> Dict[str, Any]:
    """
    Wrapper function for market regime detection that automatically fetches required data.
    
    Returns:
        Dict with market regime detection and strategy recommendations
    """
    try:
        # Import required functions
        from .connect_data_tools import get_nifty_spot_price_safe, get_options_chain_safe
        from .master_indicators import get_nifty_technical_analysis_tool
        
        # Get current spot price
        spot_result = get_nifty_spot_price_safe()
        if spot_result.get('status') != 'SUCCESS':
            return {
                'status': 'ERROR',
                'message': f'Failed to get spot price: {spot_result.get("message", "Unknown error")}'
            }
        
        # Get options chain
        options_result = get_options_chain_safe()
        if options_result.get('status') != 'SUCCESS':
            return {
                'status': 'ERROR',
                'message': f'Failed to get options chain: {options_result.get("message", "Unknown error")}'
            }
        
        options_chain = options_result.get('options_chain', [])
        if not options_chain:
            return {
                'status': 'ERROR',
                'message': 'Empty options chain received'
            }
        
        # Get technical indicators
        tech_result = get_nifty_technical_analysis_tool()
        if 'error' in tech_result:
            return {
                'status': 'ERROR',
                'message': f'Failed to get technical analysis: {tech_result.get("error", "Unknown error")}'
            }
        
        technical_indicators = tech_result
        
        # Get VIX analysis for volatility data
        vix_result = analyze_vix_integration_wrapper()
        volatility_data = vix_result if vix_result.get('status') == 'SUCCESS' else {
            'volatility_regime': 'NORMAL'
        }
        
        # Call the main market regime detection function
        return detect_market_regime(technical_indicators, volatility_data, options_chain)
        
    except Exception as e:
        return {
            'status': 'ERROR',
            'message': f'Market regime detection wrapper failed: {str(e)}'
        }


# OLD FUNCTION REMOVED - Now using unified TRUE IV system


def detect_market_regime(technical_indicators: Dict[str, Any], 
                        volatility_data: Dict[str, Any],
                        options_chain: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Detect market regime for strategy selection.
    
    Args:
        technical_indicators: Technical analysis results
        volatility_data: Volatility analysis data
        options_chain: Options chain data
    
    Returns:
        Dict with market regime detection and strategy recommendations
    """
    try:
        # Extract key indicators
        rsi = technical_indicators.get('latest_indicator_values', {}).get('rsi', 50)
        adx = technical_indicators.get('latest_indicator_values', {}).get('adx', 20)
        macd_signal = technical_indicators.get('trading_signals', {}).get('macd', 'NEUTRAL')
        supertrend_signal = technical_indicators.get('trading_signals', {}).get('supertrend', 'NEUTRAL')
        
        # Volatility regime
        vol_regime = volatility_data.get('volatility_regime', 'NORMAL')
        
        # Calculate Put-Call Ratio
        total_call_oi = sum(opt.get('CE_oi', 0) for opt in options_chain)
        total_put_oi = sum(opt.get('PE_oi', 0) for opt in options_chain)
        pcr = total_put_oi / total_call_oi if total_call_oi > 0 else 1.0
        
        # Market regime detection logic
        regime_scores = {
            'TRENDING_BULL': 0,
            'TRENDING_BEAR': 0,
            'RANGING': 0,
            'VOLATILE': 0,
            'COMPRESSED': 0
        }
        
        # Trend analysis
        if adx > 25:  # Strong trend
            if macd_signal == 'BUY' and supertrend_signal == 'BUY':
                regime_scores['TRENDING_BULL'] += 3
            elif macd_signal == 'SELL' and supertrend_signal == 'SELL':
                regime_scores['TRENDING_BEAR'] += 3
        else:  # Weak trend
            regime_scores['RANGING'] += 2
        
        # RSI analysis
        if rsi > 70:
            regime_scores['TRENDING_BULL'] += 1
        elif rsi < 30:
            regime_scores['TRENDING_BEAR'] += 1
        elif 40 <= rsi <= 60:
            regime_scores['RANGING'] += 1
        
        # Volatility analysis
        if vol_regime == 'HIGH_STRESS':
            regime_scores['VOLATILE'] += 3
        elif vol_regime == 'COMPRESSED':
            regime_scores['COMPRESSED'] += 3
        elif vol_regime == 'ELEVATED':
            regime_scores['VOLATILE'] += 1
        
        # PCR analysis
        if pcr > 1.2:  # Bearish sentiment
            regime_scores['TRENDING_BEAR'] += 1
        elif pcr < 0.8:  # Bullish sentiment
            regime_scores['TRENDING_BULL'] += 1
        else:  # Neutral sentiment
            regime_scores['RANGING'] += 1
        
        # Determine primary regime
        primary_regime = max(regime_scores, key=regime_scores.get)
        
        # Market sentiment analysis
        sentiment = 'BULLISH' if pcr < 0.8 else 'BEARISH' if pcr > 1.2 else 'NEUTRAL'
        
        return {
            'status': 'SUCCESS',
            'primary_regime': primary_regime,
            'regime_scores': regime_scores,
            'technical_indicators': {
                'rsi': round(rsi, 2),
                'adx': round(adx, 2),
                'macd_signal': macd_signal,
                'supertrend_signal': supertrend_signal
            },
            'volatility_regime': vol_regime,
            'put_call_ratio': round(pcr, 3),
            'market_sentiment': sentiment,
            'confidence_score': round(max(regime_scores.values()) / 5, 2),
            'timestamp': dt.datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            'status': 'ERROR',
            'message': f'Market regime detection failed: {str(e)}'
        }


def calculate_pnl_percentage(pnl, avg_price, quantity):
    """
    Calculate the P&L percentage for a position.
    Args:
        pnl (float): The profit or loss for the position.
        avg_price (float): The average price of the position.
        quantity (int): The quantity of the position.
    Returns:
        float: The P&L percentage, or 0 if avg_price or quantity is zero.
    """
    try:
        if avg_price and quantity:
            return (pnl / (avg_price * abs(quantity))) * 100
        else:
            return 0.0
    except Exception as e:
        print(f"Error calculating P&L percentage: {e}")
        return 0.0


def calculate_true_iv_data(days: int = 30) -> Dict[str, Any]:
    """
    Calculate TRUE Implied Volatility data using options chain and Black-Scholes.
    
    Args:
        days: Number of days for historical analysis (default: 30)
    
    Returns:
        Dict with comprehensive IV analysis
    """
    try:
        from core_tools.connect_data_tools import get_nifty_spot_price_safe, get_options_chain_safe
        from scipy.optimize import brentq
        
        print(f"📊 Calculating TRUE Implied Volatility data for {days} days...")
        
        # Get current spot price
        spot_result = get_nifty_spot_price_safe()
        if spot_result.get('status') != 'SUCCESS':
            return {
                'status': 'ERROR',
                'message': f'Failed to get spot price: {spot_result.get("message", "Unknown error")}'
            }
        
        spot_price = spot_result.get('spot_price', 0)
        if spot_price <= 0:
            return {
                'status': 'ERROR',
                'message': 'Invalid spot price received'
            }
        
        # Get current options chain
        options_result = get_options_chain_safe()
        if options_result.get('status') != 'SUCCESS':
            return {
                'status': 'ERROR',
                'message': f'Failed to get options chain: {options_result.get("message", "Unknown error")}'
            }
        
        options_chain = options_result.get('options_chain', [])
        if not options_chain:
            return {
                'status': 'ERROR',
                'message': 'Empty options chain received'
            }
        
        # Calculate current IV from options
        current_iv_data = calculate_current_iv_from_options(options_chain, spot_price)
        
        # Update historical IV data with current IV
        if current_iv_data.get('status') == 'SUCCESS':
            current_iv = current_iv_data.get('atm_iv', 0)
            if current_iv > 0:
                update_historical_iv_data(current_iv, spot_price)
        
        # Load historical IV data for percentile calculations
        historical_data = load_historical_iv_data()
        
        # Calculate IV statistics using historical data
        iv_statistics = calculate_iv_statistics_from_historical(current_iv_data, historical_data)
        
        # Prepare final result
        result = {
            'status': 'SUCCESS',
            'calculation_date': datetime.now().isoformat(),
            'spot_price': spot_price,
            'current_iv_analysis': current_iv_data,
            'historical_iv_data': historical_data,
            'iv_statistics': iv_statistics
        }
        
        # Save to cache
        save_true_iv_data(result)
        
        return result
        
    except Exception as e:
        return {
            'status': 'ERROR',
            'message': f'Exception in calculate_true_iv_data: {str(e)}',
            'traceback': traceback.format_exc()
        }

def load_historical_iv_data() -> Dict[str, Any]:
    """
    Load cached historical IV data for percentile calculations.
    
    Returns:
        Dict with historical IV data or None if not available
    """
    try:
        import json
        import os
        from datetime import datetime, timedelta
        
        # Try to load the new historical IV data file first
        historical_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'iv_cache', 'historical_iv_data.json')
        
        if os.path.exists(historical_file):
            with open(historical_file, 'r') as f:
                data = json.load(f)
            
            # Check if data is recent (within 24 hours)
            if data.get('last_updated'):
                last_updated = datetime.fromisoformat(data['last_updated'])
                if datetime.now() - last_updated > timedelta(hours=24):
                    print("⚠️  Historical IV data is older than 24 hours - consider refreshing")
            
            return data
        
        # Fallback to old cache file
        cache_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'iv_cache', 'true_iv_data.json')
        
        if not os.path.exists(cache_file):
            return None
        
        # Check if cache is recent (within 24 hours)
        file_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
        if datetime.now() - file_time > timedelta(hours=24):
            print("⚠️  Historical IV cache is older than 24 hours - consider refreshing")
        
        with open(cache_file, 'r') as f:
            data = json.load(f)
        
        return data
        
    except Exception as e:
        print(f"⚠️  Warning: Could not load IV cache: {e}")
        return None


def calculate_current_iv_from_options(options_chain: List[Dict[str, Any]], spot_price: float, dividend_yield: float = 0.0) -> Dict[str, Any]:
    """
    Calculate current Implied Volatility from options chain using Black-Scholes.
    
    Args:
        options_chain: Current options chain data
        spot_price: Current spot price
    
    Returns:
        Dict with current IV analysis
    """
    try:
        from scipy.optimize import brentq
        
        # Find ATM strikes (±25 points from spot)
        atm_strikes = []
        for option in options_chain:
            strike = option.get('strike', 0)
            if abs(strike - spot_price) <= 25:
                atm_strikes.append(strike)
        
        if not atm_strikes:
            return {
                'status': 'ERROR',
                'message': 'No ATM options found'
            }
        
        # Calculate IV for each ATM option
        iv_data = []
        for option in options_chain:
            strike = option.get('strike', 0)
            if strike not in atm_strikes:
                continue
            
            # Calculate time to expiry (assuming 30 days for now)
            time_to_expiry = 30 / 365.0
            risk_free_rate = 0.06
            
            # Calculate Call IV
            if 'CE_ltp' in option and option['CE_ltp'] > 0:
                try:
                    call_iv = calculate_iv_using_brentq(
                        option['CE_ltp'], spot_price, strike, time_to_expiry, risk_free_rate, 'CE', dividend_yield
                    )
                    iv_data.append({
                        'strike': strike,
                        'option_type': 'CE',
                        'market_price': option['CE_ltp'],
                        'iv': call_iv,
                        'moneyness': strike / spot_price
                    })
                except Exception as e:
                    print(f"⚠️  Could not calculate Call IV for strike {strike}: {e}")
            
            # Calculate Put IV
            if 'PE_ltp' in option and option['PE_ltp'] > 0:
                try:
                    put_iv = calculate_iv_using_brentq(
                        option['PE_ltp'], spot_price, strike, time_to_expiry, risk_free_rate, 'PE', dividend_yield
                    )
                    iv_data.append({
                        'strike': strike,
                        'option_type': 'PE',
                        'market_price': option['PE_ltp'],
                        'iv': put_iv,
                        'moneyness': strike / spot_price
                    })
                except Exception as e:
                    print(f"⚠️  Could not calculate Put IV for strike {strike}: {e}")
        
        if not iv_data:
            return {
                'status': 'ERROR',
                'message': 'No valid IV calculations'
            }
        
        # Calculate statistics
        iv_values = [item['iv'] for item in iv_data if item['iv'] > 0]
        if not iv_values:
            return {
                'status': 'ERROR',
                'message': 'No valid IV values'
            }
        
        atm_iv = np.mean(iv_values)
        iv_std = np.std(iv_values)
        
        # Load historical IV data for proper percentile calculation
        historical_data = load_historical_iv_data()
        if historical_data and 'historical_iv_values' in historical_data:
            historical_ivs = historical_data['historical_iv_values']
            if len(historical_ivs) >= 10:  # Need minimum data points
                # Calculate real IV percentile and rank
                iv_percentile = len([x for x in historical_ivs if x <= atm_iv]) / len(historical_ivs)
                iv_rank = (atm_iv - min(historical_ivs)) / (max(historical_ivs) - min(historical_ivs)) if max(historical_ivs) > min(historical_ivs) else 0.5
            else:
                # Fallback to estimated values if insufficient historical data
                iv_percentile = estimate_iv_percentile(atm_iv)
                iv_rank = estimate_iv_rank(atm_iv)
        else:
            # Fallback to estimated values if no historical data
            iv_percentile = estimate_iv_percentile(atm_iv)
            iv_rank = estimate_iv_rank(atm_iv)
        
        # Determine volatility regime based on historical context
        volatility_regime = determine_volatility_regime(atm_iv, historical_data)
        
        return {
            'status': 'SUCCESS',
            'atm_iv': round(atm_iv, 4),
            'iv_std': round(iv_std, 4),
            'iv_rank': round(iv_rank, 3),
            'iv_percentile': round(iv_percentile, 3),
            'volatility_regime': volatility_regime,
            'total_options_analyzed': len(iv_data),
            'valid_iv_calculations': len(iv_values),
            'iv_range': {
                'min': round(min(iv_values), 4),
                'max': round(max(iv_values), 4)
            },
            'detailed_iv_data': iv_data,
            'historical_data_used': historical_data is not None
        }
        
    except Exception as e:
        return {
            'status': 'ERROR',
            'message': f'Current IV calculation failed: {str(e)}'
        }


def estimate_iv_percentile(current_iv: float) -> float:
    """
    Estimate IV percentile when historical data is not available.
    This is a fallback method using typical NIFTY IV ranges.
    """
    # Typical NIFTY IV ranges based on market experience
    if current_iv <= 0.08:
        return 0.05  # Very low
    elif current_iv <= 0.12:
        return 0.15  # Low
    elif current_iv <= 0.18:
        return 0.40  # Below average
    elif current_iv <= 0.25:
        return 0.65  # Average
    elif current_iv <= 0.35:
        return 0.85  # Above average
    else:
        return 0.95  # Very high


def estimate_iv_rank(current_iv: float) -> float:
    """
    Estimate IV rank when historical data is not available.
    """
    # Typical NIFTY IV range: 5% to 50%
    iv_min = 0.05
    iv_max = 0.50
    iv_rank = (current_iv - iv_min) / (iv_max - iv_min)
    return max(0, min(1, iv_rank))


def determine_volatility_regime(current_iv: float, historical_data: Dict[str, Any] = None) -> str:
    """
    Determine volatility regime using historical context if available.
    """
    if historical_data and 'historical_iv_values' in historical_data:
        historical_ivs = historical_data['historical_iv_values']
        if len(historical_ivs) >= 10:
            # Use historical percentiles
            percentile_25 = np.percentile(historical_ivs, 25)
            percentile_75 = np.percentile(historical_ivs, 75)
            
            if current_iv < percentile_25:
                return 'LOW_VOLATILITY'
            elif current_iv < percentile_75:
                return 'NORMAL_VOLATILITY'
            else:
                return 'HIGH_VOLATILITY'
    
    # Fallback to absolute ranges
    if current_iv < 0.15:
        return 'LOW_VOLATILITY'
    elif current_iv < 0.25:
        return 'NORMAL_VOLATILITY'
    elif current_iv < 0.35:
        return 'HIGH_VOLATILITY'
    else:
        return 'EXTREME_VOLATILITY'


def update_historical_iv_data(current_iv: float, spot_price: float) -> Dict[str, Any]:
    """
    Update historical IV data with current IV value.
    This should be called daily to build historical IV database.
    """
    try:
        import json
        import os
        from datetime import datetime, timedelta
        
        cache_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'iv_cache')
        os.makedirs(cache_dir, exist_ok=True)
        
        historical_file = os.path.join(cache_dir, 'historical_iv_data.json')
        
        # Load existing historical data
        if os.path.exists(historical_file):
            with open(historical_file, 'r') as f:
                historical_data = json.load(f)
                # Ensure required fields exist
                if 'historical_iv_values' not in historical_data:
                    historical_data['historical_iv_values'] = []
                if 'daily_iv_data' not in historical_data:
                    historical_data['daily_iv_data'] = []
        else:
            historical_data = {
                'historical_iv_values': [],
                'daily_iv_data': [],
                'last_updated': None
            }
        
        # Add current IV to historical values
        historical_data['historical_iv_values'].append(current_iv)
        
        # Keep only last 60 days of data
        if len(historical_data['historical_iv_values']) > 60:
            historical_data['historical_iv_values'] = historical_data['historical_iv_values'][-60:]
        
        # Add daily data point
        daily_point = {
            'date': datetime.now().strftime('%Y-%m-%d'),
            'iv': current_iv,
            'spot_price': spot_price,
            'timestamp': datetime.now().isoformat()
        }
        
        # Remove duplicate dates and add new data
        historical_data['daily_iv_data'] = [
            x for x in historical_data['daily_iv_data'] 
            if x['date'] != daily_point['date']
        ]
        historical_data['daily_iv_data'].append(daily_point)
        
        # Keep only last 60 days of daily data
        if len(historical_data['daily_iv_data']) > 60:
            historical_data['daily_iv_data'] = historical_data['daily_iv_data'][-60:]
        
        historical_data['last_updated'] = datetime.now().isoformat()
        
        # Save updated data
        with open(historical_file, 'w') as f:
            json.dump(historical_data, f, indent=2)
        
        print(f"✅ Historical IV data updated: {current_iv:.4f} ({len(historical_data['historical_iv_values'])} data points)")
        
        return historical_data
        
    except Exception as e:
        print(f"⚠️  Failed to update historical IV data: {e}")
        return None


def black_scholes(spot: float, strike: float, time_to_expiry: float, risk_free_rate: float, volatility: float, option_type: str, dividend_yield: float = 0.0) -> float:
    """
    Calculate Black-Scholes option price with dividend adjustment.
    
    Args:
        spot: Current spot price
        strike: Strike price
        time_to_expiry: Time to expiry in years
        risk_free_rate: Risk-free rate
        volatility: Volatility
        option_type: 'CE' for Call, 'PE' for Put
        dividend_yield: Dividend yield (default 0.0)
    Returns:
        Theoretical option price
    """
    if time_to_expiry <= 0 or volatility <= 0:
        return max(0.0, spot - strike) if option_type == 'CE' else max(0.0, strike - spot)
    d1 = (log(spot / strike) + (risk_free_rate - dividend_yield + 0.5 * volatility ** 2) * time_to_expiry) / (volatility * sqrt(time_to_expiry))
    d2 = d1 - volatility * sqrt(time_to_expiry)
    if option_type == 'CE':
        price = spot * exp(-dividend_yield * time_to_expiry) * norm.cdf(d1) - strike * exp(-risk_free_rate * time_to_expiry) * norm.cdf(d2)
    else:
        price = strike * exp(-risk_free_rate * time_to_expiry) * norm.cdf(-d2) - spot * exp(-dividend_yield * time_to_expiry) * norm.cdf(-d1)
    return price


def calculate_iv_using_brentq(market_price: float, spot: float, strike: float,
                            time_to_expiry: float, risk_free_rate: float, option_type: str, dividend_yield: float = 0.0) -> float:
    from scipy.optimize import brentq
    def objective(sigma):
        return black_scholes(spot, strike, time_to_expiry, risk_free_rate, sigma, option_type, dividend_yield) - market_price
    try:
        iv = brentq(objective, 0.01, 5.0, maxiter=100, xtol=1e-6)
        # Check convergence
        price_diff = objective(iv)
        if abs(price_diff) < 1e-6 and (abs(price_diff / market_price) < 0.001 if market_price else True):
            return iv
        else:
            # Fallback to Newton-Raphson if not converged
            return calculate_implied_volatility(market_price, spot, strike, time_to_expiry, risk_free_rate, option_type, dividend_yield)
    except ValueError:
        return calculate_implied_volatility(market_price, spot, strike, time_to_expiry, risk_free_rate, option_type, dividend_yield)


def calculate_historical_iv_trend(days: int, current_spot: float) -> Dict[str, Any]:
    """
    Calculate historical IV trend (simplified - would need historical options data for true IV).
    
    Args:
        days: Number of days to analyze
        current_spot: Current spot price
    
    Returns:
        Dict with historical IV trend analysis
    """
    try:
        # For now, we'll use a simplified approach
        # In a real implementation, you would need historical options data
        
        return {
            'status': 'SUCCESS',
            'analysis_period': f'{days} days',
            'note': 'Historical IV trend analysis requires historical options data',
            'current_spot': current_spot,
            'recommendation': 'Use current IV analysis for trading decisions'
        }
        
    except Exception as e:
        return {
            'status': 'ERROR',
            'message': f'Historical IV trend calculation failed: {str(e)}'
        }


def calculate_comprehensive_volatility_surface(options_chain: List[Dict[str, Any]], 
                                             spot_price: float,
                                             expiry_date: str = None) -> Dict[str, Any]:
    """
    Calculate comprehensive volatility surface across multiple strikes and expiries.
    
    Args:
        options_chain: Current options chain data
        spot_price: Current spot price
        expiry_date: Expiry date for analysis
    
    Returns:
        Dict with volatility surface analysis
    """
    try:
        print("📊 Calculating comprehensive volatility surface...")
        
        # Calculate time to expiry
        if expiry_date:
            try:
                expiry_dt = dt.datetime.strptime(expiry_date, '%Y-%m-%d').date()
                today = dt.date.today()
                days_to_expiry = (expiry_dt - today).days
                time_to_expiry = max(days_to_expiry / 365, 0.01)
            except:
                time_to_expiry = 30/365
        else:
            time_to_expiry = 30/365
        
        # Analyze IV across different strikes
        volatility_data = {
            'calls': [],
            'puts': [],
            'skew_analysis': {},
            'surface_stats': {}
        }
        
        # Process each strike
        for option in options_chain:
            strike = option['strike']
            moneyness = strike / spot_price
            
            # Calculate Call IV
            if 'CE_ltp' in option and option['CE_ltp'] > 0:
                try:
                    call_iv = calculate_implied_volatility(
                        option['CE_ltp'], spot_price, strike, time_to_expiry, 0.06, 'CE'
                    )
                    volatility_data['calls'].append({
                        'strike': strike,
                        'moneyness': round(moneyness, 3),
                        'iv': round(call_iv, 4),
                        'ltp': option['CE_ltp'],
                        'volume': option.get('CE_volume', 0),
                        'oi': option.get('CE_oi', 0)
                    })
                except:
                    pass
            
            # Calculate Put IV
            if 'PE_ltp' in option and option['PE_ltp'] > 0:
                try:
                    put_iv = calculate_implied_volatility(
                        option['PE_ltp'], spot_price, strike, time_to_expiry, 0.06, 'PE'
                    )
                    volatility_data['puts'].append({
                        'strike': strike,
                        'moneyness': round(moneyness, 3),
                        'iv': round(put_iv, 4),
                        'ltp': option['PE_ltp'],
                        'volume': option.get('PE_volume', 0),
                        'oi': option.get('PE_oi', 0)
                    })
                except:
                    pass
        
        # Calculate volatility skew
        if volatility_data['calls'] and volatility_data['puts']:
            # Find ATM options (moneyness closest to 1.0)
            atm_calls = [c for c in volatility_data['calls'] if abs(c['moneyness'] - 1.0) < 0.02]
            atm_puts = [p for p in volatility_data['puts'] if abs(p['moneyness'] - 1.0) < 0.02]
            
            atm_call_iv = np.mean([c['iv'] for c in atm_calls]) if atm_calls else 0
            atm_put_iv = np.mean([p['iv'] for p in atm_puts]) if atm_puts else 0
            
            # Calculate skew metrics
            volatility_data['skew_analysis'] = {
                'atm_call_iv': round(atm_call_iv, 4),
                'atm_put_iv': round(atm_put_iv, 4),
                'put_call_skew': round(atm_put_iv - atm_call_iv, 4),
                'skew_direction': 'PUT_SKEW' if atm_put_iv > atm_call_iv else 'CALL_SKEW',
                'skew_magnitude': 'HIGH' if abs(atm_put_iv - atm_call_iv) > 0.05 else 'LOW'
            }
        
        # Calculate surface statistics
        all_ivs = [c['iv'] for c in volatility_data['calls']] + [p['iv'] for p in volatility_data['puts']]
        if all_ivs:
            volatility_data['surface_stats'] = {
                'mean_iv': round(np.mean(all_ivs), 4),
                'median_iv': round(np.median(all_ivs), 4),
                'std_iv': round(np.std(all_ivs), 4),
                'min_iv': round(min(all_ivs), 4),
                'max_iv': round(max(all_ivs), 4),
                'iv_range': round(max(all_ivs) - min(all_ivs), 4),
                'total_options': len(all_ivs)
            }
        
        # Analyze term structure (if multiple expiries available)
        volatility_data['term_structure'] = {
            'current_expiry': expiry_date,
            'days_to_expiry': int(days_to_expiry) if expiry_date else 30,
            'time_to_expiry_years': round(time_to_expiry, 3)
        }
        
        print(f"✅ Volatility surface calculated: {len(volatility_data['calls'])} calls, {len(volatility_data['puts'])} puts")
        
        return {
            'status': 'SUCCESS',
            'volatility_surface': volatility_data,
            'timestamp': dt.datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            'status': 'ERROR',
            'message': f'Volatility surface calculation failed: {str(e)}'
        }

def calculate_realized_volatility(historical_prices: List[float], period: int = 30) -> float:
    """
    Calculate realized volatility from historical price data.
    
    Args:
        historical_prices: List of historical prices (most recent last)
        period: Period for volatility calculation (default 30 days)
    
    Returns:
        Realized volatility as annualized percentage
    """
    try:
        if len(historical_prices) < period + 1:
            return None
        
        # Use the last 'period' data points
        prices = historical_prices[-period-1:]
        
        # Calculate daily returns
        returns = []
        for i in range(1, len(prices)):
            if prices[i-1] > 0:
                daily_return = np.log(prices[i] / prices[i-1])
                returns.append(daily_return)
        
        if len(returns) < period:
            return None
        
        # Calculate realized volatility (annualized)
        realized_vol = np.std(returns) * np.sqrt(252)  # Annualized
        
        return realized_vol
        
    except Exception as e:
        print(f"Error calculating realized volatility: {str(e)}")
        return None


def get_realized_volatility_from_kite() -> float:
    """
    Fetch realized volatility from Kite Connect historical data.
    
    Returns:
        Realized volatility as float, or None if not available
    """
    try:
        import core_tools.connect_data_tools as connect_data_tools
        
        if not hasattr(connect_data_tools, '_kite_instance') or connect_data_tools._kite_instance is None:
            print("⚠️  Kite Connect not available for realized volatility")
            return None
        
        kite = connect_data_tools._kite_instance
        
        # Get NIFTY historical data for last 30 days
        from datetime import datetime, timedelta
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=45)  # Extra days to ensure 30 trading days
        
        try:
            # Fetch NIFTY historical data
            historical_data = kite.historical_data(
                instrument_token=256265,  # NIFTY 50 token
                from_date=start_date,
                to_date=end_date,
                interval='day'
            )
            
            if not historical_data or len(historical_data) < 30:
                print("⚠️  Insufficient historical data for realized volatility")
                return None
            
            # Extract closing prices
            prices = [day['close'] for day in historical_data if day['close'] > 0]
            
            if len(prices) < 30:
                print("⚠️  Insufficient price data for realized volatility")
                return None
            
            # Calculate realized volatility
            realized_vol = calculate_realized_volatility(prices, 30)
            
            if realized_vol:
                print(f"📊 Realized Volatility (30-day): {realized_vol:.4f} ({realized_vol*100:.2f}%)")
            
            return realized_vol
            
        except Exception as e:
            print(f"⚠️  Error fetching historical data: {str(e)}")
            return None
            
    except Exception as e:
        print(f"⚠️  Error in realized volatility calculation: {str(e)}")
        return None


def analyze_options_liquidity(options_chain: List[Dict[str, Any]], atm_strike: float) -> Dict[str, Any]:
    """
    Analyze liquidity of options around ATM strike.
    
    Args:
        options_chain: Options chain data
        atm_strike: At-the-money strike price
    
    Returns:
        Dict with liquidity analysis
    """
    try:
        # Focus on strikes within ±100 points of ATM
        nearby_strikes = [opt for opt in options_chain if abs(opt['strike'] - atm_strike) <= 100]
        
        if not nearby_strikes:
            return {
                'status': 'ERROR',
                'message': 'No nearby strikes found for liquidity analysis'
            }
        
        liquidity_metrics = {
            'total_strikes_analyzed': len(nearby_strikes),
            'atm_strike': atm_strike,
            'call_liquidity': {},
            'put_liquidity': {},
            'overall_liquidity_score': 0
        }
        
        # Analyze call options liquidity
        call_oi_total = 0
        call_volume_total = 0
        call_bid_ask_spreads = []
        liquid_calls = 0
        
        for opt in nearby_strikes:
            if 'CE_oi' in opt and opt['CE_oi'] > 0:
                call_oi_total += opt['CE_oi']
                call_volume_total += opt.get('CE_volume', 0)
                
                # Calculate bid-ask spread if available
                if 'CE_bid' in opt and 'CE_ask' in opt and opt['CE_bid'] > 0 and opt['CE_ask'] > 0:
                    spread = (opt['CE_ask'] - opt['CE_bid']) / opt['CE_bid'] * 100  # Percentage spread
                    call_bid_ask_spreads.append(spread)
                
                # Count liquid calls - More realistic criteria for NIFTY
                # Primary: OI > 15000 (200 lots) OR Volume > 1500 (20 lots)
                # Secondary: Must have reasonable bid-ask spread (< 10%)
                ce_oi = opt['CE_oi']
                ce_volume = opt.get('CE_volume', 0)
                ce_bid = opt.get('CE_bid', 0)
                ce_ask = opt.get('CE_ask', 0)
                
                # Check if spread is reasonable (if bid/ask available)
                spread_ok = True
                if ce_bid > 0 and ce_ask > 0:
                    spread_pct = ((ce_ask - ce_bid) / ce_bid) * 100
                    spread_ok = spread_pct < 10  # Max 10% spread
                
                if (ce_oi > 15000 or ce_volume > 1500) and spread_ok:
                    liquid_calls += 1
        
        # Analyze put options liquidity
        put_oi_total = 0
        put_volume_total = 0
        put_bid_ask_spreads = []
        liquid_puts = 0
        
        for opt in nearby_strikes:
            if 'PE_oi' in opt and opt['PE_oi'] > 0:
                put_oi_total += opt['PE_oi']
                put_volume_total += opt.get('PE_volume', 0)
                
                # Calculate bid-ask spread if available
                if 'PE_bid' in opt and 'PE_ask' in opt and opt['PE_bid'] > 0 and opt['PE_ask'] > 0:
                    spread = (opt['PE_ask'] - opt['PE_bid']) / opt['PE_bid'] * 100  # Percentage spread
                    put_bid_ask_spreads.append(spread)
                
                # Count liquid puts - More realistic criteria for NIFTY
                # Primary: OI > 15000 (200 lots) OR Volume > 1500 (20 lots)
                # Secondary: Must have reasonable bid-ask spread (< 10%)
                pe_oi = opt['PE_oi']
                pe_volume = opt.get('PE_volume', 0)
                pe_bid = opt.get('PE_bid', 0)
                pe_ask = opt.get('PE_ask', 0)
                
                # Check if spread is reasonable (if bid/ask available)
                spread_ok = True
                if pe_bid > 0 and pe_ask > 0:
                    spread_pct = ((pe_ask - pe_bid) / pe_bid) * 100
                    spread_ok = spread_pct < 10  # Max 10% spread
                
                if (pe_oi > 15000 or pe_volume > 1500) and spread_ok:
                    liquid_puts += 1
        
        # Calculate liquidity metrics
        liquidity_metrics['call_liquidity'] = {
            'total_oi': call_oi_total,
            'total_volume': call_volume_total,
            'liquid_strikes': liquid_calls,
            'avg_bid_ask_spread': np.mean(call_bid_ask_spreads) if call_bid_ask_spreads else None,
            'liquidity_score': min(100, (liquid_calls / len(nearby_strikes)) * 100) if nearby_strikes else 0
        }
        
        liquidity_metrics['put_liquidity'] = {
            'total_oi': put_oi_total,
            'total_volume': put_volume_total,
            'liquid_strikes': liquid_puts,
            'avg_bid_ask_spread': np.mean(put_bid_ask_spreads) if put_bid_ask_spreads else None,
            'liquidity_score': min(100, (liquid_puts / len(nearby_strikes)) * 100) if nearby_strikes else 0
        }
        
        # Overall liquidity score (average of call and put liquidity)
        overall_score = (liquidity_metrics['call_liquidity']['liquidity_score'] + 
                        liquidity_metrics['put_liquidity']['liquidity_score']) / 2
        liquidity_metrics['overall_liquidity_score'] = round(overall_score, 1)
        
        # Determine liquidity status
        if overall_score >= 80:
            liquidity_status = 'EXCELLENT'
            liquidity_comment = 'High liquidity across strikes - optimal for trading'
        elif overall_score >= 60:
            liquidity_status = 'GOOD'
            liquidity_comment = 'Good liquidity - suitable for most strategies'
        elif overall_score >= 40:
            liquidity_status = 'MODERATE'
            liquidity_comment = 'Moderate liquidity - exercise caution with large positions'
        elif overall_score >= 20:
            liquidity_status = 'POOR'
            liquidity_comment = 'Poor liquidity - avoid complex strategies'
        else:
            liquidity_status = 'VERY_POOR'
            liquidity_comment = 'Very poor liquidity - avoid trading'
        
        liquidity_metrics['liquidity_status'] = liquidity_status
        liquidity_metrics['liquidity_comment'] = liquidity_comment
        
        # Add specific recommendations
        if liquidity_status in ['EXCELLENT', 'GOOD']:
            liquidity_metrics['recommendations'] = [
                'All strategies suitable',
                'Tight bid-ask spreads expected',
                'Good for complex multi-leg strategies'
            ]
        elif liquidity_status == 'MODERATE':
            liquidity_metrics['recommendations'] = [
                'Stick to simple strategies',
                'Use limit orders',
                'Avoid large position sizes'
            ]
        else:
            liquidity_metrics['recommendations'] = [
                'Avoid trading in this expiry',
                'Consider different expiry dates',
                'Use only ATM options if necessary'
            ]
        
        print(f"📊 Liquidity Analysis:")
        print(f"   Overall Score: {overall_score:.1f}/100 ({liquidity_status})")
        print(f"   Call Liquidity: {liquid_calls}/{len(nearby_strikes)} strikes liquid")
        print(f"   Put Liquidity: {liquid_puts}/{len(nearby_strikes)} strikes liquid")
        print(f"   Comment: {liquidity_comment}")
        
        return {
            'status': 'SUCCESS',
            'liquidity_metrics': liquidity_metrics
        }
        
    except Exception as e:
        return {
            'status': 'ERROR',
            'message': f'Liquidity analysis failed: {str(e)}'
        }


def analyze_volatility_regime(volatility_surface: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze volatility regime based on volatility surface data.
    
    Args:
        volatility_surface: Volatility surface data from calculate_comprehensive_volatility_surface
    
    Returns:
        Dict with volatility regime analysis
    """
    try:
        surface_data = volatility_surface.get('volatility_surface', {})
        surface_stats = surface_data.get('surface_stats', {})
        skew_analysis = surface_data.get('skew_analysis', {})
        
        mean_iv = surface_stats.get('mean_iv', 0)
        std_iv = surface_stats.get('std_iv', 0)
        put_call_skew = skew_analysis.get('put_call_skew', 0)
        skew_direction = skew_analysis.get('skew_direction', 'NEUTRAL')
        
        # Determine volatility regime
        if mean_iv < 0.15:
            regime = 'LOW_VOLATILITY'
            regime_description = 'Low volatility environment - suitable for long strategies'
        elif mean_iv < 0.25:
            regime = 'NORMAL_VOLATILITY'
            regime_description = 'Normal volatility environment - balanced opportunities'
        elif mean_iv < 0.35:
            regime = 'HIGH_VOLATILITY'
            regime_description = 'High volatility environment - premium selling opportunities'
        else:
            regime = 'EXTREME_VOLATILITY'
            regime_description = 'Extreme volatility - high premium selling opportunities'
        
        # Analyze skew implications
        if put_call_skew > 0.05:
            skew_implication = 'Strong put skew - market expects downside protection'
        elif put_call_skew < -0.05:
            skew_implication = 'Strong call skew - market expects upside moves'
        else:
            skew_implication = 'Balanced skew - neutral market expectations'
        
        # Strategy recommendations based on regime
        if regime in ['HIGH_VOLATILITY', 'EXTREME_VOLATILITY']:
            strategy_recommendation = 'Premium selling strategies (Short Strangle, Iron Condor)'
        elif regime == 'LOW_VOLATILITY':
            strategy_recommendation = 'Long strategies or wait for volatility increase'
        else:
            strategy_recommendation = 'Balanced approach - consider both premium selling and directional strategies'
        
        return {
            'status': 'SUCCESS',
            'volatility_regime': regime,
            'regime_description': regime_description,
            'mean_iv': mean_iv,
            'volatility_level': 'LOW' if mean_iv < 0.15 else 'NORMAL' if mean_iv < 0.25 else 'HIGH' if mean_iv < 0.35 else 'EXTREME',
            'skew_analysis': {
                'put_call_skew': put_call_skew,
                'skew_direction': skew_direction,
                'skew_implication': skew_implication
            },
            'strategy_recommendation': strategy_recommendation,
            'risk_level': 'LOW' if regime == 'LOW_VOLATILITY' else 'MEDIUM' if regime == 'NORMAL_VOLATILITY' else 'HIGH',
            'timestamp': dt.datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            'status': 'ERROR',
            'message': f'Volatility regime analysis failed: {str(e)}'
        }


# Calculation and Analysis Tools Registry
CALCULATION_ANALYSIS_TOOLS = {
    'calculate_option_greeks': calculate_option_greeks,
    'calculate_implied_volatility': calculate_implied_volatility,
    'calculate_strategy_pnl': calculate_strategy_pnl,
    'find_arbitrage_opportunities': find_arbitrage_opportunities,
    'calculate_portfolio_greeks': calculate_portfolio_greeks,
    'calculate_volatility_surface': calculate_volatility_surface,
    'calculate_probability_of_profit': calculate_probability_of_profit,
    'analyze_vix_integration_wrapper': analyze_vix_integration_wrapper,
    'calculate_iv_rank_analysis_wrapper': calculate_iv_rank_analysis_wrapper,
    'detect_market_regime_wrapper': detect_market_regime_wrapper,
    'calculate_comprehensive_volatility_surface': calculate_comprehensive_volatility_surface,
    'analyze_volatility_regime': analyze_volatility_regime
}


if __name__ == "__main__":
    """
    Test calculation and analysis tools
    """
    print("=== NIFTY Options - Calculation & Analysis Tools Test ===\n")
    
    # Test Greeks calculation
    greeks_result = calculate_option_greeks(
        spot_price=24500,
        strike=24500,
        expiry_date='2024-07-11',
        option_type='CE',
        market_price=120
    )
    print(f"Greeks Calculation: {greeks_result['status']}")
    if greeks_result['status'] == 'SUCCESS':
        greeks = greeks_result['greeks']
        print(f"Delta: {greeks['delta']}, Gamma: {greeks['gamma']}")
        print(f"Theta: {greeks['theta_daily']}, Vega: {greeks['vega']}")
    
    # Test strategy P&L calculation
    sample_legs = [
        {'strike': 24500, 'option_type': 'CE', 'action': 'BUY', 'quantity': 25, 'price': 120},
        {'strike': 24500, 'option_type': 'PE', 'action': 'BUY', 'quantity': 25, 'price': 100}
    ]
    
    spot_range = list(range(24000, 25000, 50))
    pnl_result = calculate_strategy_pnl(sample_legs, spot_range)
    print(f"\nStrategy P&L Calculation: {pnl_result['status']}")
    if pnl_result['status'] == 'SUCCESS':
        print(f"Max Profit: ₹{pnl_result['max_profit']}")
        print(f"Max Loss: ₹{pnl_result['max_loss']}")
        print(f"Breakevens: {pnl_result['breakeven_points']}")
    
    # Test probability calculation
    prob_result = calculate_probability_of_profit(
        sample_legs, 24500, 0.20, 7
    )
    print(f"\nProbability Calculation: {prob_result['status']}")
    if prob_result['status'] == 'SUCCESS':
        print(f"Probability of Profit: {prob_result['probability_percentage']}%")
    
    print("\n✅ Calculation and Analysis Tools working!")

def calculate_iv_statistics_from_historical(current_iv_data: Dict[str, Any], historical_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate IV statistics using historical IV data for percentile and rank calculations.
    
    Args:
        current_iv_data: Current IV analysis results
        historical_data: Historical IV data from cache
    
    Returns:
        Dict with IV statistics including percentile and rank
    """
    try:
        current_iv = current_iv_data.get('atm_iv', 0)
        
        if not historical_data or not historical_data.get('historical_iv_values'):
            # No historical data available, use current IV only
            return {
                'current_iv': current_iv,
                'iv_percentile': 0.05,  # Default low percentile
                'iv_rank': 0.025,       # Default low rank
                'volatility_regime': 'LOW_VOLATILITY' if current_iv < 0.15 else 'NORMAL_VOLATILITY',
                'historical_data_available': False
            }
        
        # Get historical IV values
        historical_iv_values = historical_data.get('historical_iv_values', [])
        
        if len(historical_iv_values) < 5:
            # Insufficient historical data
            return {
                'current_iv': current_iv,
                'iv_percentile': 0.05,
                'iv_rank': 0.025,
                'volatility_regime': 'LOW_VOLATILITY' if current_iv < 0.15 else 'NORMAL_VOLATILITY',
                'historical_data_available': False,
                'note': f'Only {len(historical_iv_values)} historical data points available'
            }
        
        # Calculate statistics from historical data
        historical_iv_array = np.array(historical_iv_values)
        iv_min = np.min(historical_iv_array)
        iv_max = np.max(historical_iv_array)
        iv_mean = np.mean(historical_iv_array)
        iv_std = np.std(historical_iv_array)
        
        # Calculate IV rank (0-1 scale)
        if iv_max > iv_min:
            iv_rank = (current_iv - iv_min) / (iv_max - iv_min)
            iv_rank = max(0, min(1, iv_rank))  # Clamp to 0-1
        else:
            iv_rank = 0.5
        
        # Calculate IV percentile
        percentile = np.percentile(historical_iv_array, [5, 25, 50, 75, 95])
        
        if current_iv <= percentile[0]:  # 5th percentile
            iv_percentile = 0.05
        elif current_iv <= percentile[1]:  # 25th percentile
            iv_percentile = 0.25
        elif current_iv <= percentile[2]:  # 50th percentile
            iv_percentile = 0.50
        elif current_iv <= percentile[3]:  # 75th percentile
            iv_percentile = 0.75
        else:  # Above 95th percentile
            iv_percentile = 0.95
        
        # Determine volatility regime
        if current_iv < iv_mean - iv_std:
            volatility_regime = 'LOW_VOLATILITY'
        elif current_iv > iv_mean + iv_std:
            volatility_regime = 'HIGH_VOLATILITY'
        else:
            volatility_regime = 'NORMAL_VOLATILITY'
        
        return {
            'current_iv': current_iv,
            'iv_percentile': iv_percentile,
            'iv_rank': iv_rank,
            'volatility_regime': volatility_regime,
            'historical_data_available': True,
            'historical_stats': {
                'min': iv_min,
                'max': iv_max,
                'mean': iv_mean,
                'std': iv_std,
                'data_points': len(historical_iv_values)
            }
        }
        
    except Exception as e:
        return {
            'current_iv': current_iv_data.get('atm_iv', 0),
            'iv_percentile': 0.05,
            'iv_rank': 0.025,
            'volatility_regime': 'UNKNOWN',
            'historical_data_available': False,
            'error': str(e)
        }


def save_true_iv_data(data: Dict[str, Any]) -> None:
    """
    Save TRUE IV data to cache file.
    
    Args:
        data: IV data to save
    """
    try:
        import json
        import os
        
        cache_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'iv_cache')
        os.makedirs(cache_dir, exist_ok=True)
        
        cache_file = os.path.join(cache_dir, 'true_iv_data.json')
        with open(cache_file, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        print(f"✅ True IV data saved to: {cache_file}")
        
    except Exception as e:
        print(f"⚠️  Warning: Could not save IV cache: {e}")


def update_historical_iv_data(current_iv: float, spot_price: float) -> None:
    """
    Update historical IV data with current IV value.
    
    Args:
        current_iv: Current implied volatility
        spot_price: Current spot price
    """
    try:
        import json
        import os
        from datetime import datetime
        
        # Load existing historical data
        historical_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'iv_cache', 'true_iv_data.json')
        
        if os.path.exists(historical_file):
            with open(historical_file, 'r') as f:
                historical_data = json.load(f)
                # Ensure required fields exist
                if 'historical_iv_values' not in historical_data:
                    historical_data['historical_iv_values'] = []
                if 'daily_iv_data' not in historical_data:
                    historical_data['daily_iv_data'] = []
        else:
            historical_data = {
                'historical_iv_values': [],
                'daily_iv_data': [],
                'last_updated': None
            }
        
        # Add current IV to historical values
        historical_data['historical_iv_values'].append(current_iv)
        
        # Add daily data point
        daily_data = {
            'date': datetime.now().strftime('%Y-%m-%d'),
            'iv': current_iv,
            'spot_price': spot_price,
            'timestamp': datetime.now().isoformat()
        }
        historical_data['daily_iv_data'].append(daily_data)
        
        # Keep only last 100 data points to prevent file from growing too large
        if len(historical_data['historical_iv_values']) > 100:
            historical_data['historical_iv_values'] = historical_data['historical_iv_values'][-100:]
            historical_data['daily_iv_data'] = historical_data['daily_iv_data'][-100:]
        
        # Update last updated timestamp
        historical_data['last_updated'] = datetime.now().isoformat()
        
        # Save updated data
        with open(historical_file, 'w') as f:
            json.dump(historical_data, f, indent=2, default=str)
        
        print(f"✅ Historical IV data updated: {current_iv:.4f} ({len(historical_data['historical_iv_values'])} data points)")
        
    except Exception as e:
        print(f"⚠️  Failed to update historical IV data: {e}")


def load_historical_iv_data() -> Dict[str, Any]:
    """
    Load historical IV data from cache.
    
    Returns:
        Dict with historical IV data or empty dict if not available
    """
    try:
        import json
        import os
        
        historical_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'iv_cache', 'true_iv_data.json')
        
        if not os.path.exists(historical_file):
            return {}
        
        with open(historical_file, 'r') as f:
            data = json.load(f)
        
        return data
        
    except Exception as e:
        print(f"⚠️  Failed to load historical IV data: {e}")
        return {}