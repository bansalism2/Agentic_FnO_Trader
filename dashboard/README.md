# AlgoTrade Enhanced Dashboard

A comprehensive real-time dashboard for monitoring your AlgoTrade system, showing market conditions, opportunity hunter activity, and position manager status.

## Features

### 📊 Market Conditions
- **RSI, ADX, MACD** - Latest technical indicator values
- **Market Regime** - Current market regime (TRENDING_BULL, RANGING, etc.)
- **Trend Signal** - Current trend direction (LONG, SHORT, WAIT)
- **Indicator Trend** - Analysis of recent indicator movements
- **Recent Indicators Count** - Number of recent data points

### 🎯 Opportunity Hunter Activity
- **Latest Intended Trade** - What the opportunity hunter would have done
- **Reason** - Why the trade was intended
- **Recent Intended Trades** - Last 5 intended trades with timestamps
- **Recent Count** - Total number of recent intended trades

### ⚡ Position Manager Status
- **Active Positions** - Current portfolio positions with details
- **Active Trades** - Trades being managed by the position manager
- **Position Details** - Symbol, quantity, average price, P&L
- **Trade Status** - Status of each active trade

### 🔧 System Status
- **Market Open/Closed** - Real-time market status
- **Data Files** - Status of indicator_history.jsonl and intended_trades.jsonl
- **Last Update** - Timestamp of last dashboard update

## Usage

### Quick Start
```bash
# Run the dashboard with auto-refresh (30-second intervals)
./launch_dashboard.sh

# Or run directly with Python
python enhanced_dashboard.py

# Run with auto-refresh
python run_enhanced_dashboard.py
```

### Manual Run
```bash
cd agent_tools/dashboard
python enhanced_dashboard.py
```

## Dashboard Output Example

```
============================================================================================================
🚀 ALGOTRADE ENHANCED DASHBOARD
============================================================================================================
📅 Last Updated: 2025-07-30 10:30:15 IST
🌐 Market Status: 🟢 OPEN

📊 MARKET CONDITIONS:
  RSI: 45.23
  ADX: 28.45
  MACD Signal: NEUTRAL
  Market Regime: TRENDING_BULL
  Trend Signal: LONG
  Indicator Trend: RSI_RISING_ADX_FLAT
  Recent Indicators: 15 entries

🎯 OPPORTUNITY HUNTER:
  Latest Intended: NIFTY2580724900CE - LONG_CALL
  Reason: Would have taken LONG CALL (trend+regime aligned)
  Recent Intended Trades: 3
  Last 5 Intended Trades:
    [10:30:12] NIFTY2580724900CE - LONG_CALL
    [10:29:45] NIFTY2580724900PE - LONG_PUT
    [10:29:18] NIFTY2580724900CE - LONG_CALL

⚡ POSITION MANAGER:
  Active Positions: 1
  Active Trades: 1
  Current Positions:
    NIFTY2580724900CE: Qty=50, Avg=125.50, P&L=2.75
  Active Trades:
    trade_001: ACTIVE (Entry: 2025-07-30T10:25:30)

🔧 SYSTEM STATUS:
  Data Files:
    ✅ indicator_history
    ✅ intended_trades
============================================================================================================
```

## Files

- `enhanced_dashboard.py` - Main dashboard class and functionality
- `run_enhanced_dashboard.py` - Auto-refresh version (30-second intervals)
- `launch_dashboard.sh` - Shell script launcher
- `README.md` - This documentation

## Data Sources

The dashboard reads from:
- `../data/indicator_history.jsonl` - Market conditions and technical indicators
- `../data/intended_trades.jsonl` - Opportunity hunter intended trades
- Broker API - Current portfolio positions
- Trade storage - Active trades being managed

## Auto-Refresh

The dashboard automatically refreshes every 30 seconds when using `run_enhanced_dashboard.py`. Press `Ctrl+C` to stop.

## Troubleshooting

### Common Issues

1. **"Data Files: ❌"** - Check that the data files exist in the `../data/` directory
2. **"Connection failed"** - Check your API credentials and network connection
3. **"No positions found"** - This is normal when no trades are active

### Data File Locations
- `indicator_history.jsonl` - Created by opportunity hunter during market analysis
- `intended_trades.jsonl` - Created when opportunity hunter defers trades due to existing positions

## Integration

The dashboard integrates with your existing AlgoTrade system:
- **Opportunity Hunter** - Logs intended trades when deferring
- **Position Manager** - Manages positions and provides status
- **Market Analysis** - Provides technical indicators and market conditions

This gives you a complete real-time view of your trading system's activity! 