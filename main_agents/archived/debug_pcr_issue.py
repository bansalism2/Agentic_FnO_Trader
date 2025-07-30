#!/usr/bin/env python3
"""
Debug PCR Technical Analysis Issue
==================================

This script will help identify why PCR technical analysis is failing with
"Insufficient data for indicator calculation" error.
"""

import sys
import os
sys.path.append('..')

from dotenv import load_dotenv

# Load credentials from .env and access_token.txt (same as main workflow)
load_dotenv(dotenv_path='../.env')
api_key = os.getenv("kite_api_key")
api_secret = os.getenv("kite_api_secret")
access_token = None
try:
    with open("../data/access_token.txt", "r") as f:
        access_token = f.read().strip()
        print(f"✅ Successfully loaded access token: {access_token[:10]}...")
except Exception as e:
    print(f"❌ Could not read ../data/access_token.txt: {e}")

def test_pcr_technical_analysis():
    """Test PCR technical analysis step by step"""
    print("🔍 DEBUGGING PCR TECHNICAL ANALYSIS ISSUE")
    print("=" * 60)
    
    try:
        # Step 1: Test Kite Connection
        print("\n1️⃣ Testing Kite Connection...")
        from core_tools.connect_data_tools import initialize_connection, get_nifty_spot_price_safe
        
        # Initialize connection with loaded credentials
        conn_result = initialize_connection(api_key, api_secret, access_token)
        print(f"Connection result: {conn_result}")
        
        # Test spot price
        spot_result = get_nifty_spot_price_safe()
        print(f"Spot price result: {spot_result}")
        
        # Step 2: Test OHLCV Data Fetch (at least 5 days)
        print("\n2️⃣ Testing OHLCV Data Fetch (5+ days)...")
        from core_tools.master_indicators import fetch_nifty_ohlcv_data, get_nifty_technical_analysis
        
        try:
            days = 5
            print(f"Testing with {days} days, 15minute interval...")
            df = fetch_nifty_ohlcv_data(days=days, interval="15minute")
            print(f"DataFrame shape: {df.shape}")
            print(f"DataFrame columns: {df.columns.tolist()}")
            print(f"DataFrame head:\n{df.head()}")
            
            # Filter to most recent 30 rows if needed
            min_rows = 30
            if len(df) > min_rows:
                df_recent = df.tail(min_rows)
                print(f"Filtered to most recent {min_rows} rows for indicator calculation.")
            else:
                df_recent = df
                print(f"Only {len(df)} rows available. Might be insufficient for some indicators.")
            
            print(f"Rows used for indicator calculation: {len(df_recent)}")
            
        except Exception as e:
            print(f"❌ OHLCV data fetch failed: {str(e)}")
        
        # Step 3: Test Technical Analysis with more days
        print("\n3️⃣ Testing Technical Analysis (5+ days)...")
        try:
            tech_result = get_nifty_technical_analysis(days=days, interval="15minute")
            print(f"Technical analysis result: {tech_result}")
            
            if 'error' in tech_result:
                print(f"❌ Technical analysis failed: {tech_result['error']}")
            else:
                print("✅ Technical analysis successful")
                
        except Exception as e:
            print(f"❌ Technical analysis exception: {str(e)}")
        
        # Step 4: Test PCR Analysis
        print("\n4️⃣ Testing PCR Analysis...")
        from core_tools.master_indicators import calculate_pcr_technical_analysis_wrapper
        
        try:
            pcr_result = calculate_pcr_technical_analysis_wrapper()
            print(f"PCR analysis result: {pcr_result}")
            
            if pcr_result.get('status') == 'ERROR':
                print(f"❌ PCR analysis failed: {pcr_result.get('message')}")
            else:
                print("✅ PCR analysis successful")
                
        except Exception as e:
            print(f"❌ PCR analysis exception: {str(e)}")
        
        # Step 5: Test with Daily Data (for completeness)
        print("\n5️⃣ Testing with Daily Data...")
        try:
            print("Testing with 45 days of daily data...")
            daily_result = get_nifty_technical_analysis(days=45, interval="day")
            print(f"Daily technical analysis: {daily_result}")
            
            if 'error' not in daily_result:
                print("✅ Daily technical analysis successful")
            else:
                print(f"❌ Daily technical analysis failed: {daily_result['error']}")
                
        except Exception as e:
            print(f"❌ Daily analysis exception: {str(e)}")
        
    except Exception as e:
        print(f"❌ Overall test failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_pcr_technical_analysis() 