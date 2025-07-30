import sys
import os
import traceback

# Add the parent directory to the path to import core_tools
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print('--- test_true_iv_calculation.py START ---')
print('Python version:', sys.version)
print('sys.path:', sys.path)
try:
    from dotenv import load_dotenv
    print('dotenv imported successfully')
except Exception as e:
    print('Failed to import dotenv:', e)
    traceback.print_exc()
    sys.exit(1)

# Load environment variables
try:
    load_dotenv('../.env')
    print('.env loaded')
except Exception as e:
    print('Failed to load .env:', e)
    traceback.print_exc()

# Read access token
access_token = None
access_token_path = '../data/access_token.txt'
try:
    if os.path.exists(access_token_path):
        with open(access_token_path, 'r') as f:
            access_token = f.read().strip()
    print('Access token loaded:', bool(access_token))
except Exception as e:
    print('Failed to read access token:', e)
    traceback.print_exc()

api_key = os.getenv('kite_api_key')
api_secret = os.getenv('kite_api_secret')
print('API key:', api_key)
print('API secret:', bool(api_secret))

try:
    from core_tools.connect_data_tools import initialize_connection
    print('Imported initialize_connection')
except Exception as e:
    print('Failed to import initialize_connection:', e)
    traceback.print_exc()
    sys.exit(1)

try:
    print('Initializing Kite Connect...')
    result = initialize_connection(api_key, api_secret, access_token)
    print('Kite Connect init result:', result)
except Exception as e:
    print('Failed to initialize Kite Connect:', e)
    traceback.print_exc()

try:
    from core_tools.calculate_analysis_tools import calculate_true_iv_data
    print('Imported calculate_true_iv_data')
except Exception as e:
    print('Failed to import calculate_true_iv_data:', e)
    traceback.print_exc()
    sys.exit(1)

try:
    print('Calculating TRUE IV...')
    result = calculate_true_iv_data()
    print('Status:', result.get('status'))
    print('Current IV:', result.get('iv_statistics', {}).get('current_iv'))
    print('IV Percentile:', result.get('iv_statistics', {}).get('iv_percentile'))
    print('Volatility Regime:', result.get('iv_statistics', {}).get('volatility_regime'))
    print('Full Result:', result)
except Exception as e:
    print('Error during IV calculation:', e)
    traceback.print_exc() 