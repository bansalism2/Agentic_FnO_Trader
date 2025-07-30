#!/usr/bin/env python3
"""
Generate Historical IV Data for AlgoTrade

This script calculates and stores historical IV data for accurate percentile calculations.
Run this once to establish the historical IV baseline.

Usage:
    python generate_historical_iv.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def main():
    """Generate historical IV data"""
    try:
        print("🚀 Generating Historical IV Data for AlgoTrade")
        print("=" * 60)
        
        # Import the function
        from core_tools.calculate_analysis_tools import calculate_true_iv_data
        
        # Calculate historical IV data (252 trading days = ~1 year)
        print("📊 Calculating historical IV data for 252 trading days...")
        result = calculate_true_iv_data(days=252)
        
        if result.get('status') == 'SUCCESS':
            print("\n✅ Historical IV data generated successfully!")
            print("=" * 60)
            
            # Display summary
            stats_30d = result.get('statistics_30d', {})
            stats_60d = result.get('statistics_60d', {})
            current_conditions = result.get('current_market_conditions', {})
            
            print("📈 30-Day Rolling Volatility Statistics:")
            print(f"   Min: {stats_30d.get('min', 'N/A')}")
            print(f"   Max: {stats_30d.get('max', 'N/A')}")
            print(f"   Mean: {stats_30d.get('mean', 'N/A')}")
            print(f"   Std: {stats_30d.get('std', 'N/A')}")
            
            print("\n📊 30-Day Percentiles:")
            percentiles_30d = stats_30d.get('percentiles', {})
            for pct, value in percentiles_30d.items():
                print(f"   {pct}: {value:.4f}")
            
            print(f"\n🎯 Current Market Conditions:")
            print(f"   Recent 30D IV: {current_conditions.get('recent_30d_iv', 'N/A')}")
            print(f"   Recent 60D IV: {current_conditions.get('recent_60d_iv', 'N/A')}")
            print(f"   Volatility Trend: {current_conditions.get('volatility_trend', 'N/A')}")
            
            print(f"\n📅 Data Period: {result.get('data_period', 'N/A')}")
            print(f"📊 Total Days: {result.get('total_days', 'N/A')}")
            print(f"💾 Cache Location: ../data/iv_cache/historical_iv_data.json")
            
            print("\n" + "=" * 60)
            print("✅ Historical IV data is now available for accurate percentile calculations!")
            print("🔄 The system will automatically use this data for IV analysis.")
            print("⏰ Cache will be refreshed automatically when older than 24 hours.")
            
        else:
            print(f"❌ Failed to generate historical IV data: {result.get('message', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"❌ Error generating historical IV data: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 Historical IV data generation completed successfully!")
    else:
        print("\n💥 Historical IV data generation failed!")
        sys.exit(1) 