#!/usr/bin/env python3
"""
Simple test to check if basic margin functions work
"""

import sys
import os
sys.path.append('..')

from dotenv import load_dotenv
from pathlib import Path

# Load .env file from agent_tools (parent of main_agents)
dotenv_path = Path(__file__).parent.parent / '.env'
print(f'Loading .env from: {dotenv_path}')
load_dotenv(dotenv_path)

# Load credentials from environment
api_key = os.environ.get('kite_api_key', None)
api_secret = os.environ.get('kite_api_secret', None)

print(f'API Key loaded: {api_key is not None}')
print(f'API Secret loaded: {api_secret is not None}')

# Load access token from file
access_token_path = '../data/access_token.txt'
print(f'Loading access token from: {os.path.abspath(access_token_path)}')
with open(access_token_path, 'r') as f:
    access_token = f.read().strip()

print(f'Access Token loaded: {len(access_token) > 0}')

# Initialize connection
from core_tools.connect_data_tools import initialize_connection
print("Initializing connection...")
connection_result = initialize_connection(api_key, api_secret, access_token)
print(f'Connection result: {connection_result}')

if connection_result.get('status') == 'SUCCESS':
    print("✅ Connection successful!")
    
    # Test basic functions
    from core_tools.execution_portfolio_tools import get_account_margins
    print("\nTesting get_account_margins...")
    margins = get_account_margins()
    print(f'Account margins: {margins}')
    
    # Test margin calculation with simple data
    from core_tools.execution_portfolio_tools import calculate_strategy_margins
    
    # Simple test legs
    test_legs = [
        {
            'symbol': 'NIFTY2572424250PE',
            'action': 'BUY',
            'quantity': 75,
            'exchange': 'NFO',
            'product': 'MIS',
            'order_type': 'MARKET'
        }
    ]
    
    print(f"\nTesting calculate_strategy_margins with {len(test_legs)} leg...")
    margin_result = calculate_strategy_margins(test_legs)
    print(f'Margin result: {margin_result}')
    
else:
    print("❌ Connection failed!") 