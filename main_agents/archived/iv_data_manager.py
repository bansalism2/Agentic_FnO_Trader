#!/usr/bin/env python3
"""
IV Data Manager for AlgoTrade

This module manages historical IV data, provides real-time updates, and ensures data freshness.
It automatically refreshes data daily and provides comprehensive IV analysis.

Usage:
    from iv_data_manager import IVDataManager
    manager = IVDataManager()
    manager.refresh_if_needed()
    analysis = manager.get_comprehensive_analysis()
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Add current directory to path for imports
# sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
# sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'core_tools'))
# sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

class IVDataManager:
    """
    Manages historical IV data and provides comprehensive analysis.
    """
    
    def __init__(self, cache_dir: str = None):
        """
        Initialize IV Data Manager.
        
        Args:
            cache_dir: Directory to store IV cache files
        """
        if cache_dir is None:
            cache_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'iv_cache')
        
        self.cache_dir = cache_dir
        self.historical_data_file = os.path.join(cache_dir, 'historical_iv_data.json')
        self.last_update_file = os.path.join(cache_dir, 'last_update.json')
        
        # Ensure cache directory exists
        os.makedirs(cache_dir, exist_ok=True)
        
        # Setup logging
        self.logger = self._setup_logging()
        self.logger.info("IVDataManager version 2.0 initialized.")
        
        # Initialize connection if not already done
        # self._ensure_connection()
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for IV data manager."""
        logger = logging.getLogger('IVDataManager')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _ensure_connection(self):
        """Ensure Kite Connect connection is initialized."""
        try:
            # This logic is problematic and redundant. The connection should be managed
            # by the agent or driver script that instantiates this manager.
            # Removing the direct dependency on a singleton _kite_instance.
            pass
        except Exception as e:
            self.logger.error(f"Error in _ensure_connection (now disabled): {e}")
    
    def is_data_fresh(self, max_age_hours: int = 24) -> bool:
        """
        Check if historical IV data is fresh.
        
        Args:
            max_age_hours: Maximum age in hours before data is considered stale
            
        Returns:
            True if data is fresh, False otherwise
        """
        try:
            if not os.path.exists(self.last_update_file):
                return False
            
            with open(self.last_update_file, 'r') as f:
                last_update = json.load(f)
            
            last_update_time = datetime.fromisoformat(last_update.get('timestamp', ''))
            age_hours = (datetime.now() - last_update_time).total_seconds() / 3600
            
            return age_hours < max_age_hours
            
        except Exception as e:
            self.logger.warning(f"Error checking data freshness: {e}")
            return False
    
    def refresh_if_needed(self, force_refresh: bool = False) -> bool:
        """
        Refresh historical IV data if needed.
        
        Args:
            force_refresh: Force refresh even if data is fresh
            
        Returns:
            True if refresh was successful or not needed, False otherwise
        """
        try:
            if not force_refresh and self.is_data_fresh():
                self.logger.info("Historical IV data is fresh, no refresh needed")
                return True
            
            self.logger.info("Refreshing historical IV data...")
            
            # Import the calculation function
            from core_tools.calculate_analysis_tools import calculate_true_iv_data
            
            # Calculate new historical data
            result = calculate_true_iv_data(days=252)
            
            if result.get('status') == 'SUCCESS':
                # Save last update timestamp
                last_update = {
                    'timestamp': datetime.now().isoformat(),
                    'status': 'SUCCESS',
                    'data_points': result.get('total_days', 0)
                }
                
                with open(self.last_update_file, 'w') as f:
                    json.dump(last_update, f, indent=2)
                
                self.logger.info("Historical IV data refreshed successfully")
                return True
            else:
                self.logger.error(f"Failed to refresh historical IV data: {result.get('message', 'Unknown error')}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error refreshing historical IV data: {e}")
            return False
    
    def get_historical_data(self) -> Optional[Dict[str, Any]]:
        """
        Get historical IV data.
        
        Returns:
            Historical IV data or None if not available
        """
        try:
            if not os.path.exists(self.historical_data_file):
                return None
            
            with open(self.historical_data_file, 'r') as f:
                data = json.load(f)
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error loading historical IV data: {e}")
            return None
    
    def get_comprehensive_analysis(self) -> Dict[str, Any]:
        """
        Get comprehensive IV analysis including historical data and current market conditions.
        
        Returns:
            Comprehensive IV analysis
        """
        try:
            # Ensure data is fresh
            if not self.refresh_if_needed():
                return {
                    'status': 'ERROR',
                    'message': 'Failed to refresh historical IV data'
                }
            
            # Get historical data
            historical_data = self.get_historical_data()
            if not historical_data:
                return {
                    'status': 'ERROR',
                    'message': 'No historical IV data available'
                }
            
            # Get current IV analysis
            from core_tools.calculate_analysis_tools import calculate_iv_rank_analysis_wrapper
            current_iv_result = calculate_iv_rank_analysis_wrapper()
            
            # Get volatility surface analysis
            from core_tools.connect_data_tools import get_nifty_spot_price_safe, get_options_chain_safe
            spot_result = get_nifty_spot_price_safe()
            options_result = get_options_chain_safe()
            
            if spot_result.get('status') != 'SUCCESS' or options_result.get('status') != 'SUCCESS':
                return {
                    'status': 'ERROR',
                    'message': 'Failed to get current market data'
                }
            
            spot_price = spot_result.get('spot_price', 0)
            options_chain = options_result.get('options_chain', [])
            
            # Calculate volatility surface
            from core_tools.calculate_analysis_tools import calculate_comprehensive_volatility_surface, analyze_volatility_regime
            volatility_surface = calculate_comprehensive_volatility_surface(options_chain, spot_price)
            regime_analysis = analyze_volatility_regime(volatility_surface)
            
            # Get VIX analysis
            from core_tools.calculate_analysis_tools import analyze_vix_integration_wrapper
            vix_analysis = analyze_vix_integration_wrapper()
            
            # Compile comprehensive analysis
            analysis = {
                'status': 'SUCCESS',
                'timestamp': datetime.now().isoformat(),
                'data_freshness': {
                    'last_update': self.get_last_update_time(),
                    'is_fresh': self.is_data_fresh(),
                    'age_hours': self.get_data_age_hours()
                },
                'current_iv_analysis': current_iv_result,
                'volatility_surface': volatility_surface,
                'regime_analysis': regime_analysis,
                'vix_analysis': vix_analysis,
                'historical_context': {
                    'data_period': historical_data.get('data_period', 'N/A'),
                    'total_days': historical_data.get('total_days', 0),
                    'statistics_30d': historical_data.get('statistics_30d', {}),
                    'current_market_conditions': historical_data.get('current_market_conditions', {})
                },
                'summary': self._generate_summary(current_iv_result, regime_analysis, vix_analysis)
            }
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error in comprehensive analysis: {e}")
            return {
                'status': 'ERROR',
                'message': f'Comprehensive analysis failed: {str(e)}'
            }
    
    def get_last_update_time(self) -> str:
        """Get the last update timestamp."""
        try:
            if os.path.exists(self.last_update_file):
                with open(self.last_update_file, 'r') as f:
                    data = json.load(f)
                return data.get('timestamp', 'Unknown')
            return 'Never'
        except:
            return 'Unknown'
    
    def get_data_age_hours(self) -> float:
        """Get the age of the data in hours."""
        try:
            last_update = self.get_last_update_time()
            if last_update == 'Never' or last_update == 'Unknown':
                return float('inf')
            
            last_update_time = datetime.fromisoformat(last_update)
            age_hours = (datetime.now() - last_update_time).total_seconds() / 3600
            return round(age_hours, 2)
        except:
            return float('inf')
    
    def _generate_summary(self, current_iv: Dict[str, Any], 
                         regime: Dict[str, Any], 
                         vix: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a summary of the IV analysis."""
        try:
            current_iv_value = current_iv.get('current_iv', 0)
            iv_percentile = current_iv.get('iv_percentile', 0)
            regime_type = regime.get('volatility_regime', 'UNKNOWN')
            vix_proxy = vix.get('vix_analysis', {}).get('vix_proxy', 0)
            
            # Determine overall market condition
            if current_iv_value < 0.15 and iv_percentile < 0.25:
                market_condition = 'LOW_VOLATILITY_COMPLACENCY'
                recommendation = 'Consider long strategies or wait for volatility increase'
            elif current_iv_value > 0.25 and iv_percentile > 0.75:
                market_condition = 'HIGH_VOLATILITY_OPPORTUNITY'
                recommendation = 'Premium selling strategies recommended'
            else:
                market_condition = 'NORMAL_VOLATILITY'
                recommendation = 'Balanced approach - evaluate specific opportunities'
            
            return {
                'market_condition': market_condition,
                'current_iv': current_iv_value,
                'iv_percentile': iv_percentile,
                'volatility_regime': regime_type,
                'vix_proxy': vix_proxy,
                'recommendation': recommendation,
                'risk_level': 'LOW' if current_iv_value < 0.15 else 'MEDIUM' if current_iv_value < 0.25 else 'HIGH'
            }
            
        except Exception as e:
            return {
                'market_condition': 'UNKNOWN',
                'recommendation': 'Unable to generate summary',
                'error': str(e)
            }

def main():
    """Main function for testing IV Data Manager."""
    try:
        print("🚀 Testing IV Data Manager...")
        
        manager = IVDataManager()
        
        # Refresh data if needed
        print("📊 Checking data freshness...")
        if manager.refresh_if_needed():
            print("✅ Data is fresh or refreshed successfully")
        else:
            print("❌ Failed to refresh data")
            return
        
        # Get comprehensive analysis
        print("🔍 Getting comprehensive analysis...")
        analysis = manager.get_comprehensive_analysis()
        
        if analysis.get('status') == 'SUCCESS':
            print("✅ Comprehensive analysis completed")
            
            # Display summary
            summary = analysis.get('summary', {})
            print(f"\n📋 SUMMARY:")
            print(f"   Market Condition: {summary.get('market_condition', 'N/A')}")
            print(f"   Current IV: {summary.get('current_iv', 'N/A')}")
            print(f"   IV Percentile: {summary.get('iv_percentile', 'N/A')}")
            print(f"   Volatility Regime: {summary.get('volatility_regime', 'N/A')}")
            print(f"   VIX Proxy: {summary.get('vix_proxy', 'N/A')}")
            print(f"   Recommendation: {summary.get('recommendation', 'N/A')}")
            print(f"   Risk Level: {summary.get('risk_level', 'N/A')}")
            
        else:
            print(f"❌ Analysis failed: {analysis.get('message', 'Unknown error')}")
            
    except Exception as e:
        print(f"❌ Error in main: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 