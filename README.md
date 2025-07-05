# 🤖 AlgoTrade - AI-Powered F&O Trading System

An intelligent algorithmic trading system that uses CrewAI agents to automate F&O (Futures & Options) trading on Indian markets. Features real-time market analysis, risk management, and automated position management with dynamic scheduling.

## 🎯 Overview

AlgoTrade is a sophisticated automated trading system designed for Indian F&O markets with a focus on capital preservation and risk management. The system uses multiple AI agents to analyze markets, manage risk, and execute trading strategies while maintaining strict controls.

## 🔧 Key Features

### 🤖 Multi-Agent AI System
- **Market Analyst**: Short-term market dynamics and volatility analysis
- **Risk Manager**: Portfolio risk assessment and capital management
- **Strategy Executor**: High-conviction trade execution with timing expertise

### 📊 Trading Capabilities
- Real-time NIFTY F&O analysis and trading decisions
- Technical analysis with multiple indicators (RSI, MACD, Bollinger Bands, etc.)
- Options chain analysis and flow monitoring
- Greeks calculation and risk assessment
- Portfolio position monitoring and automated closures

### ⚡ Automation Features
- **Dynamic Scheduling**: 15-minute intervals (9:30 AM - 2:30 PM), 5-minute intervals (2:30 PM - 3:30 PM)
- **Time-Based Restrictions**: No new trades after 2:30 PM, focus on position management
- **Automated Token Management**: Zerodha Kite Connect access token refresh
- **Market Hours Detection**: Automatic execution during trading hours only

### 🛡️ Risk Management
- Maximum 3 active positions at any time
- Risk per trade limited to 5-8% of capital
- Real-time margin and capital validation
- Automated stop-loss and profit booking
- Conservative approach with strict risk controls

## 🚀 Quick Start

### Prerequisites
- Python 3.12
- Zerodha Kite Connect account
- OpenAI API key

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/AlgoTrade.git
   cd AlgoTrade/AlgoTradeActive/AlgoTradeAgent/agent_tools
   ```

2. **Set up conda environment**
   ```bash
   conda create -n kite_auto python=3.12
   conda activate kite_auto
   ```

3. **Install dependencies**
   ```bash
   pip install crewai openai kiteconnect schedule pytz python-dotenv
   pip install pandas numpy ta-lib selenium pyotp
   ```

4. **Configure environment variables**
   Create a `.env` file in the `agent_tools` directory:
   ```env
   OPENAI_API_KEY=your_openai_api_key
   kite_api_key=your_kite_api_key
   kite_api_secret=your_kite_api_secret
   kite_user_id=your_kite_user_id
   kite_password=your_kite_password
   kite_totp_secret=your_kite_totp_secret
   ```

### Usage

#### 1. Test Mode (Single Execution)
```bash
python3 crew_driver.py --single --test
```

#### 2. Production Mode (Continuous Scheduling)
```bash
python3 crew_driver.py
```

#### 3. Manual Agent Execution
```bash
python3 crew_agent.py
```

## 📋 System Architecture

### Crew Driver (`crew_driver.py`)
- **Market Hours Detection**: 9:30 AM - 3:30 PM IST, weekdays only
- **Dynamic Scheduling**: 
  - 9:30 AM - 2:30 PM: Every 15 minutes (trading mode)
  - 2:30 PM - 3:30 PM: Every 5 minutes (position management mode)
- **Token Management**: Automatic Kite Connect token refresh
- **Error Handling**: Graceful failure recovery and logging

### Crew Agent (`crew_agent.py`)
- **Portfolio Review**: Analyze existing positions and take corrective actions
- **Market Analysis**: Short-term market dynamics and opportunity identification
- **Risk Assessment**: Capital validation and position sizing
- **Strategy Execution**: High-conviction trade execution with strict criteria

### Technical Analysis (`master_indicators.py`)
- **Multiple Indicators**: RSI, MACD, Bollinger Bands, ADX, ATR, etc.
- **Pattern Recognition**: Doji, Hammer, Shooting Star, Maru Bozu
- **Support/Resistance**: Pivot points and trend analysis
- **Data Validation**: Interval limits and historical data constraints

## 🔄 Trading Workflow

### 1. Market Hours Check
- Verifies if market is open (9:30 AM - 3:30 PM IST, weekdays)
- Skips execution outside trading hours

### 2. Token Refresh
- Automatically refreshes Zerodha access token
- Handles authentication failures gracefully

### 3. Portfolio Review
- Analyzes existing positions
- Takes corrective actions (profit booking, stop-loss)
- Manages time decay and expiry risks

### 4. Market Analysis
- Short-term market dynamics (1-5 day horizon)
- Volatility analysis and options flow
- Technical indicator assessment

### 5. Risk Assessment
- Capital validation and margin requirements
- Position sizing based on conviction levels
- Risk-reward ratio evaluation

### 6. Strategy Execution
- **Before 2:30 PM**: New trade evaluation and execution
- **After 2:30 PM**: Position management and closures only
- Strict criteria for trade execution

## 📊 Trading Strategy

### Conservative Approach
- **Capital Preservation First**: Protect capital before seeking growth
- **High Conviction Only**: Execute only exceptional opportunities
- **Risk Management**: Strict position sizing and stop-losses

### F&O Specific Rules
- **Time Decay Management**: Close positions with <7 days to expiry
- **Volatility Exits**: Close long volatility positions if IV drops >20%
- **Liquidity Assessment**: Minimum 500 OI and 50+ daily volume
- **Spread Control**: Bid-ask spread <8% of option price

### Position Management
- **Profit Booking**: Close at 25-40% of max profit
- **Stop Losses**: 2x premium for long options, 50% credit for short options
- **Greeks Management**: Reduce delta >50%, gamma >30%
- **Expiry Week**: Aggressive position management

## ⚙️ Configuration

### Market Hours
```python
MARKET_START_TIME = "09:30"  # 9:30 AM IST
MARKET_END_TIME = "15:30"    # 3:30 PM IST
POSITION_MANAGEMENT_START = "14:30"  # 2:30 PM IST
```

### Risk Parameters
```python
MAX_POSITIONS = 3
MAX_RISK_PER_TRADE = 0.08  # 8% of capital
MIN_PROFIT_BOOKING = 0.25   # 25% of max profit
```

### Scheduling
```python
NORMAL_INTERVAL = 15  # minutes
POSITION_MANAGEMENT_INTERVAL = 5  # minutes
```

## 📈 Data Requirements

### Historical Data Limits
- **minute**: Maximum 60 days
- **3minute**: Maximum 100 days
- **5minute**: Maximum 100 days
- **10minute**: Maximum 100 days
- **15minute**: Maximum 200 days
- **30minute**: Maximum 200 days
- **60minute**: Maximum 400 days
- **day**: Maximum 2000 days

### Technical Indicators
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands
- ADX (Average Directional Index)
- ATR (Average True Range)
- Pivot Points
- Candlestick Patterns

## 🛠️ Development

### Project Structure
```
AlgoTradeActive/
├── AlgoTradeAgent/
│   └── agent_tools/
│       ├── crew_driver.py          # Main driver with scheduling
│       ├── crew_agent.py           # AI agents and tasks
│       ├── master_indicators.py    # Technical analysis
│       ├── connect_data_tools.py   # Data connection utilities
│       ├── calculate_analysis_tools.py  # Analysis calculations
│       ├── execution_portfolio_tools.py # Portfolio management
│       ├── strategy_creation_tools.py   # Strategy creation
│       ├── get_access_token.py     # Token management
│       └── logs/                   # Execution logs
```

### Adding New Features
1. Create new tool functions in appropriate modules
2. Register tools in `crew_agent.py` using `@tool` decorator
3. Update agent backstories and task descriptions
4. Test with `--single --test` mode

## 📝 Logging

### Log Files
- **Location**: `logs/crew_driver_YYYYMMDD.log`
- **Format**: Timestamp, Level, Message
- **Rotation**: Daily log files

### Log Levels
- **INFO**: Normal operations and status updates
- **WARNING**: Non-critical issues (token refresh failures)
- **ERROR**: Critical errors and exceptions

## ⚠️ Important Disclaimers

### Risk Warning
- **Trading involves substantial risk of loss**
- **Past performance does not guarantee future results**
- **This system is for educational purposes only**
- **Use at your own risk and responsibility**

### Legal Notice
- This software is provided "as is" without warranty
- Users are responsible for compliance with local trading regulations
- Not financial advice - consult qualified professionals
- Educational use only - not for commercial trading

### Technical Limitations
- Depends on external APIs (Zerodha, OpenAI)
- Market conditions may affect performance
- System may not work during market holidays
- Requires stable internet connection

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

For issues and questions:
1. Check the [Issues](https://github.com/yourusername/AlgoTrade/issues) page
2. Review the logs in `logs/` directory
3. Ensure all dependencies are installed correctly
4. Verify API keys and credentials

## 🙏 Acknowledgments

- [CrewAI](https://github.com/joaomdmoura/crewAI) for the multi-agent framework
- [Zerodha](https://kite.trade/) for the trading API
- [OpenAI](https://openai.com/) for the AI capabilities
- [TA-Lib](https://ta-lib.org/) for technical analysis

---

**⚠️ Disclaimer: This software is for educational purposes only. Trading involves risk of loss. Use at your own risk.** 
