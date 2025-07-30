#!/usr/bin/env python3
"""
Web-based Dashboard for AlgoTrade System

Flask web application providing real-time visualization of:
1. Indicator History - Market conditions and technical indicators
2. Opportunity Hunter Activity - What trades it's considering/taking
3. Position Manager Status - Current positions and management decisions

@author: AlgoTrade Team
"""

from flask import Flask, render_template, jsonify, request
import json
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
import pytz
import threading
import time

# Add the parent directory to the path to import core_tools
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core_tools.execution_portfolio_tools import get_portfolio_positions
from core_tools.trade_storage import get_active_trades, get_trade_history

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

class WebDashboard:
    def __init__(self):
        """
        Initialize the web dashboard
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
            
            return recent_indicators[-50:]  # Return last 50 entries for charts
            
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
            
            return recent_intended[-20:]  # Return last 20 entries
            
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
            indicators = self.load_indicator_history(60)  # 1 hour lookback for charts
            if indicators:
                latest_indicator = indicators[-1]
                self.dashboard_data['market_conditions'] = {
                    'latest_rsi': latest_indicator.get('rsi', 0),
                    'latest_adx': latest_indicator.get('adx', 0),
                    'latest_macd_signal': latest_indicator.get('macd_signal', 'NEUTRAL'),
                    'market_regime': latest_indicator.get('market_regime', 'UNKNOWN'),
                    'trend_signal': latest_indicator.get('trend_signal', 'NEUTRAL'),
                    'recent_indicators_count': len(indicators),
                    'indicator_trend': self._analyze_indicator_trend(indicators),
                    'chart_data': self._prepare_chart_data(indicators)
                }
            
            # Load intended trades
            intended_trades = self.load_intended_trades(60)
            if intended_trades:
                latest_intended = intended_trades[-1]
                self.dashboard_data['opportunity_hunter'] = {
                    'latest_intended_symbol': latest_intended.get('symbol', 'N/A'),
                    'latest_intended_action': latest_intended.get('action', 'N/A'),
                    'latest_intended_reason': latest_intended.get('reason', 'N/A'),
                    'recent_intended_count': len(intended_trades),
                    'recent_intended_trades': intended_trades[-10:],  # Last 10
                    'chart_data': self._prepare_intended_trades_chart(intended_trades)
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
                'active_trades': active_trades,
                'chart_data': self._prepare_positions_chart(nifty_positions)
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
    
    def _prepare_chart_data(self, indicators: List[Dict]) -> Dict:
        """
        Prepare chart data for indicators
        """
        if not indicators:
            return {}
        
        timestamps = []
        rsi_values = []
        adx_values = []
        
        for indicator in indicators:
            try:
                timestamp = datetime.fromisoformat(indicator['timestamp'].replace('Z', '+00:00'))
                if timestamp.tzinfo is None:
                    timestamp = self.ist_timezone.localize(timestamp)
                
                timestamps.append(timestamp.strftime('%H:%M:%S'))
                rsi_values.append(indicator.get('rsi', 0))
                adx_values.append(indicator.get('adx', 0))
            except:
                continue
        
        return {
            'timestamps': timestamps,
            'rsi': rsi_values,
            'adx': adx_values
        }
    
    def _prepare_intended_trades_chart(self, intended_trades: List[Dict]) -> Dict:
        """
        Prepare chart data for intended trades
        """
        if not intended_trades:
            return {}
        
        timestamps = []
        actions = []
        symbols = []
        
        for trade in intended_trades:
            try:
                timestamp = datetime.fromisoformat(trade['timestamp'])
                if timestamp.tzinfo is None:
                    timestamp = self.ist_timezone.localize(timestamp)
                
                timestamps.append(timestamp.strftime('%H:%M:%S'))
                actions.append(trade.get('action', 'UNKNOWN'))
                symbols.append(trade.get('symbol', 'UNKNOWN'))
            except:
                continue
        
        return {
            'timestamps': timestamps,
            'actions': actions,
            'symbols': symbols
        }
    
    def _prepare_positions_chart(self, positions: List[Dict]) -> Dict:
        """
        Prepare chart data for positions
        """
        if not positions:
            return {}
        
        symbols = []
        quantities = []
        pnl_values = []
        
        for position in positions:
            symbols.append(position.get('symbol', 'UNKNOWN'))
            quantities.append(abs(position.get('quantity', 0)))
            pnl_values.append(position.get('pnl', 0))
        
        return {
            'symbols': symbols,
            'quantities': quantities,
            'pnl': pnl_values
        }
    
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
    
    def get_dashboard_data(self) -> Dict:
        """
        Get current dashboard data
        """
        self.update_dashboard()
        return self.dashboard_data

# Global dashboard instance
dashboard = WebDashboard()

@app.route('/')
def index():
    """
    Main dashboard page
    """
    return render_template('web_dashboard.html')

@app.route('/api/dashboard')
def get_dashboard():
    """
    API endpoint to get dashboard data
    """
    return jsonify(dashboard.get_dashboard_data())

@app.route('/api/indicators')
def get_indicators():
    """
    API endpoint to get indicator data for charts
    """
    indicators = dashboard.load_indicator_history(60)
    chart_data = dashboard._prepare_chart_data(indicators)
    return jsonify(chart_data)

@app.route('/api/intended_trades')
def get_intended_trades():
    """
    API endpoint to get intended trades data
    """
    intended_trades = dashboard.load_intended_trades(60)
    chart_data = dashboard._prepare_intended_trades_chart(intended_trades)
    return jsonify(chart_data)

@app.route('/api/positions')
def get_positions():
    """
    API endpoint to get positions data
    """
    positions = dashboard.get_current_positions()
    nifty_positions = [p for p in positions if 'NIFTY' in p.get('symbol', '')]
    chart_data = dashboard._prepare_positions_chart(nifty_positions)
    return jsonify({
        'positions': nifty_positions,
        'chart_data': chart_data
    })

def start_auto_refresh():
    """
    Background thread to auto-refresh dashboard data
    """
    while True:
        try:
            dashboard.update_dashboard()
            time.sleep(30)  # Refresh every 30 seconds
        except Exception as e:
            logger.error(f"Error in auto-refresh: {e}")
            time.sleep(60)  # Wait longer on error

if __name__ == '__main__':
    # Start auto-refresh thread
    refresh_thread = threading.Thread(target=start_auto_refresh, daemon=True)
    refresh_thread.start()
    
    # Start Flask app
    print("🚀 Starting AlgoTrade Web Dashboard...")
    print("📊 Open your browser and go to: http://localhost:5001")
    print("⏰ Dashboard will auto-refresh every 30 seconds")
    print("🛑 Press Ctrl+C to stop")
    
    app.run(debug=True, host='0.0.0.0', port=5001) 