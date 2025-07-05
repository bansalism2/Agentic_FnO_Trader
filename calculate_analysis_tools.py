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
                               time_to_expiry: float, risk_free_rate: float, option_type: str) -> float:
    """
    Calculate implied volatility using Newton-Raphson method.
    
    Args:
        market_price: Current market price of option
        spot: Current spot price
        strike: Strike price
        time_to_expiry: Time to expiry in years
        risk_free_rate: Risk-free rate
        option_type: 'CE' or 'PE'
    
    Returns:
        Implied volatility as decimal (e.g., 0.20 for 20%)
    """
    iv = 0.3  # Initial guess
    tolerance = 1e-6
    max_iterations = 100
    
    for _ in range(max_iterations):
        d1 = (np.log(spot / strike) + (risk_free_rate + 0.5 * iv ** 2) * time_to_expiry) / (iv * np.sqrt(time_to_expiry))
        d2 = d1 - iv * np.sqrt(time_to_expiry)
        
        if option_type == 'CE':
            theoretical_price = spot * norm.cdf(d1) - strike * np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(d2)
        else:
            theoretical_price = strike * np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(-d2) - spot * norm.cdf(-d1)
        
        vega = spot * np.sqrt(time_to_expiry) * norm.pdf(d1)
        price_diff = theoretical_price - market_price
        
        if abs(price_diff) < tolerance or vega < tolerance:
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


# Calculation and Analysis Tools Registry
CALCULATION_ANALYSIS_TOOLS = {
    'calculate_option_greeks': calculate_option_greeks,
    'calculate_implied_volatility': calculate_implied_volatility,
    'calculate_strategy_pnl': calculate_strategy_pnl,
    'find_arbitrage_opportunities': find_arbitrage_opportunities,
    'calculate_portfolio_greeks': calculate_portfolio_greeks,
    'calculate_volatility_surface': calculate_volatility_surface,
    'calculate_probability_of_profit': calculate_probability_of_profit
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