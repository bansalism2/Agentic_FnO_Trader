#!/usr/bin/env python3
"""
Enhanced Dashboard for AlgoTrade System

Real-time dashboard showing:
1. Indicator History - Market conditions and technical indicators
2. Opportunity Hunter Activity - What trades it's considering/taking
3. Position Manager Status - Current positions and management decisions

@author: AlgoTrade Team
"""

import json
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
import pytz

# Add the parent directory to the path to import core_tools
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core_tools.execution_portfolio_tools import get_portfolio_positions
from core_tools.trade_storage import get_active_trades, get_trade_history

# Configure logging
logger = logging.getLogger(__name__)

class EnhancedDashboard:
    def __init__(self):
        """
        Initialize the enhanced dashboard
        """
        self.ist_timezone = pytz.timezone('Asia/Kolkata')
        self.data_dir = Path(__file__).parent.parent / 'data'
        self.main_agents_dir = Path(__file__).parent.parent / 'main_agents'
        
        # Dashboard data structure
        self.dashboard_data = {
            'timestamp': datetime.now(self.ist_timezone).strftime('%Y-%m-%d %H:%M:%S %Z'),
            'market_conditions': {},
            'opportunity_hunter': {},
            'position_manager': {},
            'active_positions': [],
            'recent_trades': [],
            'system_status': {}
        }
    
    def load_indicator_history(self, lookback_minutes: int = 30) -> List[Dict]:
        """
        Load recent indicator history
        """
        try:
            indicator_file = self.data_dir / 'indicator_history.jsonl'
            if not indicator_file.exists():
                return []
            
            cutoff_time = datetime.now(self.ist_timezone) - timedelta(minutes=lookback_minutes)
            recent_indicators = []
            
            with open(indicator_file, 'r') as f:
                for line in f:
                    try:
                        indicator = json.loads(line.strip())
                        indicator_time = datetime.fromisoformat(indicator['timestamp'].replace('Z', '+00:00'))
                        if indicator_time.tzinfo is None:
                            indicator_time = self.ist_timezone.localize(indicator_time)
                        
                        if indicator_time >= cutoff_time:
                            recent_indicators.append(indicator)
                    except:
                        continue
            
            return recent_indicators[-20:]  # Return last 20 entries
            
        except Exception as e:
            logger.error(f"Error loading indicator history: {e}")
            return []
    
    def load_intended_trades(self, lookback_minutes: int = 30) -> List[Dict]:
        """
        Load recent intended trades from opportunity hunter
        """
        try:
            intended_file = self.data_dir / 'intended_trades.jsonl'
            if not intended_file.exists():
                return []
            
            cutoff_time = datetime.now(self.ist_timezone) - timedelta(minutes=lookback_minutes)
            recent_intended = []
            
            with open(intended_file, 'r') as f:
                for line in f:
                    try:
                        intended = json.loads(line.strip())
                        intended_time = datetime.fromisoformat(intended['timestamp'])
                        if intended_time.tzinfo is None:
                            intended_time = self.ist_timezone.localize(intended_time)
                        
                        if intended_time >= cutoff_time:
                            recent_intended.append(intended)
                    except:
                        continue
            
            return recent_intended[-10:]  # Return last 10 entries
            
        except Exception as e:
            logger.error(f"Error loading intended trades: {e}")
            return []
    
    def get_current_positions(self) -> List[Dict]:
        """
        Get current portfolio positions
        """
        try:
            positions_result = get_portfolio_positions()
            if positions_result.get('status') == 'SUCCESS':
                return positions_result.get('positions', [])
            return []
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return []
    
    def get_active_trades_data(self) -> Dict:
        """
        Get active trades from storage
        """
        try:
            return get_active_trades()
        except Exception as e:
            logger.error(f"Error getting active trades: {e}")
            return {}
    
    def update_dashboard(self):
        """
        Update all dashboard data
        """
        try:
            # Update timestamp
            self.dashboard_data['timestamp'] = datetime.now(self.ist_timezone).strftime('%Y-%m-%d %H:%M:%S %Z')
            
            # Load indicator history
            indicators = self.load_indicator_history(30)
            if indicators:
                latest_indicator = indicators[-1]
                self.dashboard_data['market_conditions'] = {
                    'latest_rsi': latest_indicator.get('rsi', 0),
                    'latest_adx': latest_indicator.get('adx', 0),
                    'latest_macd_signal': latest_indicator.get('macd_signal', 'NEUTRAL'),
                    'market_regime': latest_indicator.get('market_regime', 'UNKNOWN'),
                    'trend_signal': latest_indicator.get('trend_signal', 'NEUTRAL'),
                    'recent_indicators_count': len(indicators),
                    'indicator_trend': self._analyze_indicator_trend(indicators)
                }
            
            # Load intended trades
            intended_trades = self.load_intended_trades(30)
            if intended_trades:
                latest_intended = intended_trades[-1]
                self.dashboard_data['opportunity_hunter'] = {
                    'latest_intended_symbol': latest_intended.get('symbol', 'N/A'),
                    'latest_intended_action': latest_intended.get('action', 'N/A'),
                    'latest_intended_reason': latest_intended.get('reason', 'N/A'),
                    'recent_intended_count': len(intended_trades),
                    'recent_intended_trades': intended_trades[-5:]  # Last 5
                }
            
            # Get current positions
            positions = self.get_current_positions()
            nifty_positions = [p for p in positions if 'NIFTY' in p.get('symbol', '')]
            self.dashboard_data['active_positions'] = nifty_positions
            
            # Get active trades
            active_trades = self.get_active_trades_data()
            self.dashboard_data['position_manager'] = {
                'active_positions_count': len(nifty_positions),
                'active_trades_count': len(active_trades),
                'positions': nifty_positions,
                'active_trades': active_trades
            }
            
            # System status
            self.dashboard_data['system_status'] = {
                'market_open': self._is_market_open(),
                'data_files_exist': {
                    'indicator_history': (self.data_dir / 'indicator_history.jsonl').exists(),
                    'intended_trades': (self.data_dir / 'intended_trades.jsonl').exists()
                },
                'last_update': datetime.now(self.ist_timezone).strftime('%Y-%m-%d %H:%M:%S %Z')
            }
            
        except Exception as e:
            logger.error(f"Error updating dashboard: {e}")
    
    def _analyze_indicator_trend(self, indicators: List[Dict]) -> str:
        """
        Analyze trend from recent indicators
        """
        if len(indicators) < 3:
            return "INSUFFICIENT_DATA"
        
        recent_rsi = [i.get('rsi', 0) for i in indicators[-3:]]
        recent_adx = [i.get('adx', 0) for i in indicators[-3:]]
        
        # Simple trend analysis
        rsi_trend = "RISING" if recent_rsi[-1] > recent_rsi[0] else "FALLING" if recent_rsi[-1] < recent_rsi[0] else "FLAT"
        adx_trend = "RISING" if recent_adx[-1] > recent_adx[0] else "FALLING" if recent_adx[-1] < recent_adx[0] else "FLAT"
        
        return f"RSI_{rsi_trend}_ADX_{adx_trend}"
    
    def _is_market_open(self) -> bool:
        """
        Check if market is currently open
        """
        now = datetime.now(self.ist_timezone)
        current_time = now.time()
        
        # Market hours: 9:15 AM to 3:30 PM IST
        market_start = datetime.strptime("09:15", "%H:%M").time()
        market_end = datetime.strptime("15:30", "%H:%M").time()
        
        return market_start <= current_time <= market_end
    
    def print_dashboard(self):
        """
        Print the enhanced dashboard
        """
        print("\n" + "="*120)
        print("🚀 ALGOTRADE ENHANCED DASHBOARD")
        print("="*120)
        print(f"📅 Last Updated: {self.dashboard_data['timestamp']}")
        print(f"🌐 Market Status: {'🟢 OPEN' if self.dashboard_data['system_status']['market_open'] else '🔴 CLOSED'}")
        
        # Market Conditions
        if self.dashboard_data['market_conditions']:
            mc = self.dashboard_data['market_conditions']
            print(f"\n📊 MARKET CONDITIONS:")
            print(f"  RSI: {mc['latest_rsi']:.2f}")
            print(f"  ADX: {mc['latest_adx']:.2f}")
            print(f"  MACD Signal: {mc['latest_macd_signal']}")
            print(f"  Market Regime: {mc['market_regime']}")
            print(f"  Trend Signal: {mc['trend_signal']}")
            print(f"  Indicator Trend: {mc['indicator_trend']}")
            print(f"  Recent Indicators: {mc['recent_indicators_count']} entries")
        
        # Opportunity Hunter
        if self.dashboard_data['opportunity_hunter']:
            oh = self.dashboard_data['opportunity_hunter']
            print(f"\n🎯 OPPORTUNITY HUNTER:")
            print(f"  Latest Intended: {oh['latest_intended_symbol']} - {oh['latest_intended_action']}")
            print(f"  Reason: {oh['latest_intended_reason']}")
            print(f"  Recent Intended Trades: {oh['recent_intended_count']}")
            
            if oh['recent_intended_trades']:
                print(f"  Last 5 Intended Trades:")
                for trade in oh['recent_intended_trades']:
                    print(f"    [{trade['timestamp'][11:19]}] {trade['symbol']} - {trade['action']}")
        
        # Position Manager
        if self.dashboard_data['position_manager']:
            pm = self.dashboard_data['position_manager']
            print(f"\n⚡ POSITION MANAGER:")
            print(f"  Active Positions: {pm['active_positions_count']}")
            print(f"  Active Trades: {pm['active_trades_count']}")
            
            if pm['positions']:
                print(f"  Current Positions:")
                for pos in pm['positions']:
                    symbol = pos.get('symbol', 'N/A')
                    quantity = pos.get('quantity', 0)
                    avg_price = pos.get('average_price', 0)
                    pnl = pos.get('pnl', 0)
                    print(f"    {symbol}: Qty={quantity}, Avg={avg_price:.2f}, P&L={pnl:.2f}")
            
            if pm['active_trades']:
                print(f"  Active Trades:")
                for trade_id, trade_data in pm['active_trades'].items():
                    status = trade_data.get('status', 'UNKNOWN')
                    entry_time = trade_data.get('entry_time', 'N/A')
                    print(f"    {trade_id}: {status} (Entry: {entry_time})")
        
        # System Status
        ss = self.dashboard_data['system_status']
        print(f"\n🔧 SYSTEM STATUS:")
        print(f"  Data Files:")
        for file_name, exists in ss['data_files_exist'].items():
            status = "✅" if exists else "❌"
            print(f"    {status} {file_name}")
        
        print("="*120)
    
    def save_dashboard(self, filename: str = None):
        """
        Save dashboard data to file
        """
        if filename is None:
            filename = f"enhanced_dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        try:
            dashboard_file = Path(__file__).parent / filename
            with open(dashboard_file, 'w') as f:
                json.dump(self.dashboard_data, f, indent=2, default=str)
            
            logger.info(f"Dashboard saved: {dashboard_file}")
            return dashboard_file
        except Exception as e:
            logger.error(f"Error saving dashboard: {e}")
            return None

def main():
    """
    Main function to run the enhanced dashboard
    """
    dashboard = EnhancedDashboard()
    dashboard.update_dashboard()
    dashboard.print_dashboard()
    
    # Save dashboard
    dashboard.save_dashboard()

if __name__ == "__main__":
    main() 