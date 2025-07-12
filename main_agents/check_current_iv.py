#!/usr/bin/env python3
"""
Check Current IV Values (with Kite Connect initialization)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from dotenv import load_dotenv

# Load credentials
load_dotenv(dotenv_path='../.env')
api_key = os.getenv("kite_api_key")
api_secret = os.getenv("kite_api_secret")
access_token = None
try:
    with open("../data/access_token.txt", "r") as f:
        access_token = f.read().strip()
except Exception as e:
    print("Could not read ../data/access_token.txt:", e)

print(f"API Key loaded: {api_key[:10] if api_key else 'None'}...")
print(f"API Secret loaded: {api_secret[:10] if api_secret else 'None'}...")
print(f"Access Token loaded: {access_token[:20] if access_token else 'None'}...")

# Initialize Kite Connect session
try:
    from core_tools.connect_data_tools import initialize_connection
    init_result = initialize_connection(api_key, api_secret, access_token)
    print("Kite Connect initialization result:", init_result)
except Exception as e:
    print(f"Kite Connect initialization failed: {e}")

def check_current_iv():
    """Check current IV values"""
    try:
        from core_tools.calculate_analysis_tools import calculate_iv_rank_analysis_wrapper
        
        print("🔍 Checking Current IV Values...")
        print("=" * 50)
        
        iv_rank = calculate_iv_rank_analysis_wrapper()
        
        print(f"Raw IV result: {iv_rank}")
        print(f"Current IV: {iv_rank.get('current_iv', 'N/A')}")
        print(f"IV Percentile: {iv_rank.get('iv_percentile', 'N/A')}%")
        print(f"IV Rank: {iv_rank.get('iv_rank', 'N/A')}")
        print(f"Historical IV Range: {iv_rank.get('iv_min', 'N/A')} - {iv_rank.get('iv_max', 'N/A')}")
        
        print("\n📊 IV Analysis:")
        print(f"   Current IV Status: {iv_rank.get('iv_status', 'N/A')}")
        print(f"   IV Trend: {iv_rank.get('iv_trend', 'N/A')}")
        print(f"   Recommendation: {iv_rank.get('recommendation', 'N/A')}")
        
        print("\n🎯 Threshold Analysis:")
        current_iv = iv_rank.get('current_iv', 0)
        iv_percentile = iv_rank.get('iv_percentile', 0)
        
        print(f"   Emergency IV > 25%: {'✅ YES' if current_iv > 25 else '❌ NO'} (Current: {current_iv})")
        print(f"   Emergency IV percentile > 75%: {'✅ YES' if iv_percentile > 75 else '❌ NO'} (Current: {iv_percentile:.1f}%)")
        print(f"   Emergency IV percentile > 65%: {'✅ YES' if iv_percentile > 65 else '❌ NO'} (Current: {iv_percentile:.1f}%)")
        print(f"   Quick check IV < 25% skip: {'✅ YES' if iv_percentile < 25 else '❌ NO'} (Current: {iv_percentile:.1f}%)")
        print(f"   Quick check IV > 60% proceed: {'✅ YES' if iv_percentile > 60 else '❌ NO'} (Current: {iv_percentile:.1f}%)")
        print(f"   Strategy IV > 45% premium selling: {'✅ YES' if iv_percentile > 45 else '❌ NO'} (Current: {iv_percentile:.1f}%)")
        print(f"   Strategy IV < 35% long strategies: {'✅ YES' if iv_percentile < 35 else '❌ NO'} (Current: {iv_percentile:.1f}%)")
        
        print("=" * 50)
        
    except Exception as e:
        print(f"❌ Error checking IV: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_current_iv() 