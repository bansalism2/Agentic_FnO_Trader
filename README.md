# 🤖 AlgoTrade - AI-Powered F&O Trading System

An intelligent algorithmic trading system that uses CrewAI agents to automate F&O (Futures & Options) trading on Indian markets. Features real-time market analysis, risk management, and automated position management with advanced fast-track analysis and market regime detection.

## 🎯 Overview

AlgoTrade is a sophisticated automated trading system designed for Indian F&O markets with a focus on capital preservation and premium selling strategies. The system uses multiple AI agents with hierarchical fast-track analysis to identify high-probability trading opportunities while maintaining strict risk controls.

## 🔧 Key Features

### 🤖 Multi-Agent AI System
- **Fast-Track Opportunity Hunter**: Hierarchical analysis with reduced over-analysis paralysis
- **Ultra-Conservative Position Manager**: Time-decay focused position management
- **Market Regime Detection**: Advanced market state analysis for strategy selection
- **Pre-Market Analysis**: Global market data integration for gap predictions

### 📊 Trading Capabilities
- **Premium Selling First**: Prioritizes short strangles, iron condors, and credit spreads
- **Market Regime Analysis**: RANGING, TRENDING, VOLATILE regime detection
- **Fast-Track Analysis**: Quick go/no-go decisions before detailed analysis
- **Real-time NIFTY F&O**: Options chain analysis with IV rank and PCR monitoring
- **Technical Analysis**: Multi-timeframe analysis (intraday + daily)

### ⚡ Automation Features
- **Optimized Scheduling**: 15-minute intervals for both opportunity hunting and position management
- **Market Hours**: 9:15 AM - 3:30 PM IST (weekdays only)
- **No New Trades**: After 2:30 PM, focus on position management only
- **Auto Cleanup**: Active trades cleared at 3:30 PM (broker auto-close at 3:20 PM)
- **Robust Error Handling**: Skip failed runs, wait 2 minutes, no retries

### 🛡️ Risk Management
- **Capital Preservation**: Ultra-conservative approach with patience discipline
- **Premium Selling Focus**: Leverage time decay (theta) advantages
- **Market Regime Based**: Strategy selection based on market conditions
- **Fast-Track Validation**: Quick strategy suitability checks

## 🚀 Quick Start

### Prerequisites
- Python 3.12
- Zerodha Kite Connect account
- OpenAI API key (or Anthropic/Google Gemini)

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
   pip install pandas numpy ta-lib selenium pyotp yfinance
   ```

4. **Configure environment variables**
   Create a `.env` file in the `agent_tools` directory:
   ```env
   # LLM Configuration (choose one)
   LLM_MODEL=gpt-4o  # or claude-3-sonnet-20240229, gemini/gemini-2.5-pro
   OPENAI_API_KEY=your_openai_api_key
   ANTHROPIC_API_KEY=your_anthropic_api_key  # if using Claude
   GEMINI_API_KEY=your_gemini_api_key        # if using Gemini
   
   # Zerodha Configuration
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
# Fast-track opportunity hunter
python3 opportunity_hunter_agent_fast_track.py

# Position manager
python3 position_manager_agent.py
```

## 📋 System Architecture

### Crew Driver (`crew_driver.py`)
- **Market Hours**: 9:15 AM - 3:30 PM IST, weekdays only
- **Scheduling**: 
  - 9:00 AM: Pre-market data fetch
  - Every 15 minutes: Fast-track Opportunity Hunter + Position Manager
  - 3:30 PM: Market close cleanup (clear active_trades.json)
- **Error Handling**: Skip failed runs, wait 2 minutes, no retries
- **Token Management**: Automatic Kite Connect token refresh

### Fast-Track Opportunity Hunter (`opportunity_hunter_agent_fast_track.py`)
- **Hierarchical Analysis**: Quick go/no-go decisions before detailed analysis
- **Premium Selling First**: Short strangles, iron condors, credit spreads
- **Market Regime Based**: Strategy selection based on market conditions
- **Fast-Track Tools**: Emergency fast-track for obvious opportunities
- **Strategy Validation**: Iron condor and short strangle suitability checks

### Ultra-Conservative Position Manager (`position_manager_agent.py`)
- **Time Decay Focus**: Optimized for premium selling strategies
- **Ultra-Conservative**: Capital preservation with patience discipline
- **Position Monitoring**: Real-time position tracking and management
- **Risk Controls**: Strict position sizing and stop-losses

### Market Regime Detection (`calculate_analysis_tools.py`)
- **Regime Classification**: RANGING, TRENDING_BULL, TRENDING_BEAR, VOLATILE, COMPRESSED
- **Multi-Factor Analysis**: ADX, RSI, MACD, Supertrend, PCR, Volatility
- **Confidence Scoring**: Regime confidence with detailed breakdown
- **Strategy Recommendations**: Regime-based strategy selection

### Pre-Market Analysis (`data/pre_market_data.py`)
- **Global Markets**: S&P 500, Nasdaq, Nikkei, Hang Seng, Crude Oil, Gold, VIX
- **Gap Prediction**: NIFTY gap prediction with confidence scoring
- **Market Sentiment**: Global sentiment analysis
- **Trading Signals**: Actionable recommendations based on global data

## 🔄 Trading Workflow

### 1. Pre-Market Analysis (9:00 AM)
- Fetch global market data (US, Asia, commodities)
- Predict NIFTY gap and market sentiment
- Generate trading recommendations

### 2. Market Hours Check (9:15 AM - 3:30 PM)
- Verify market is open (weekdays only)
- Skip execution outside trading hours

### 3. Fast-Track Opportunity Analysis (Every 15 minutes)
- **Hierarchical Analysis**: Quick market condition assessment
- **Emergency Fast-Track**: Obvious opportunity detection
- **Strategy Validation**: Iron condor and short strangle checks
- **Premium Selling Focus**: Prioritize theta decay strategies

### 4. Position Management (Every 15 minutes)
- Monitor existing positions
- Manage time decay and expiry risks
- Execute profit booking and stop-losses
- Ultra-conservative approach

### 5. Market Close Cleanup (3:30 PM)
- Clear active_trades.json
- All positions auto-closed by broker at 3:20 PM

## 📊 Trading Strategy

### Premium Selling First Approach
- **Why Premium Selling**: Earn money from time decay (theta)
- **Strategy Priority**:
  1. Short Strangle (preferred premium selling)
  2. Iron Condor (range-bound premium selling)
  3. Bull Put Spread (bullish premium selling)
  4. Bear Call Spread (bearish premium selling)
  5. Calendar Spread (time decay premium selling)
  6. Long Strategies (only if IV <30% AND strong directional signal)

### Market Regime Based Strategy Selection
- **RANGING**: Iron condors, short strangles, calendar spreads
- **TRENDING**: Directional credit spreads, trend-following strategies
- **VOLATILE**: Reduced position sizes, focus on premium selling
- **COMPRESSED**: Calendar spreads, long volatility strategies

### Fast-Track Analysis Logic
- **Emergency Fast-Track**: Obvious opportunities (high IV, clear signals)
- **Normal Fast-Track**: Standard market condition analysis
- **Detailed Analysis**: Only when fast-track suggests potential
- **Strategy Validation**: Iron condor and short strangle suitability

### Risk Management
- **Capital Preservation**: Patience discipline - better to miss 10 good trades than lose on 1 bad trade
- **Position Limits**: Maximum 5 positions (ultra-conservative)
- **Time Restrictions**: No new trades after 2:30 PM
- **Market Regime**: Adjust strategy based on market conditions

## ⚙️ Configuration

### Market Hours
```python
MARKET_START_TIME = "09:15"  # 9:15 AM IST
MARKET_END_TIME = "15:30"    # 3:30 PM IST
NO_NEW_TRADES_AFTER = "14:30"  # 2:30 PM IST
```

### Scheduling
```python
OPPORTUNITY_HUNTER_INTERVAL = 15  # minutes
POSITION_MANAGER_INTERVAL = 15    # minutes
PRE_MARKET_FETCH = "09:00"        # 9:00 AM IST
MARKET_CLOSE_CLEANUP = "15:30"    # 3:30 PM IST
```

### Fast-Track Parameters
```python
EMERGENCY_FAST_TRACK_THRESHOLD = 0.8  # 80% confidence
NORMAL_FAST_TRACK_THRESHOLD = 0.6     # 60% confidence
DETAILED_ANALYSIS_THRESHOLD = 0.4     # 40% confidence
```

## 📈 Data Requirements

### Market Data Sources
- **Zerodha Kite Connect**: Real-time NIFTY F&O data
- **Yahoo Finance**: Global market data for pre-market analysis
- **Technical Indicators**: RSI, MACD, ADX, Supertrend, Bollinger Bands
- **Options Chain**: Live options data with Greeks and flow analysis

### Historical Data Limits
- **minute**: Maximum 60 days
- **3minute**: Maximum 100 days
- **5minute**: Maximum 100 days
- **10minute**: Maximum 100 days
- **15minute**: Maximum 200 days
- **30minute**: Maximum 200 days
- **60minute**: Maximum 400 days
- **day**: Maximum 2000 days

## 🛠️ Development

### Project Structure
```
AlgoTradeActive/
├── AlgoTradeAgent/
│   └── agent_tools/
│       ├── main_agents/
│       │   ├── crew_driver.py                    # Main driver with scheduling
│       │   ├── opportunity_hunter_agent_fast_track.py  # Fast-track opportunity hunter
│       │   ├── position_manager_agent.py         # Ultra-conservative position manager
│       │   ├── opportunity_hunter_agent_optimized.py   # Optimized opportunity hunter
│       │   └── opportunity_hunter_agent.py       # Original opportunity hunter
│       ├── core_tools/
│       │   ├── connect_data_tools.py             # Data connection utilities
│       │   ├── calculate_analysis_tools.py       # Analysis calculations & regime detection
│       │   ├── execution_portfolio_tools.py      # Portfolio management
│       │   ├── strategy_creation_tools.py        # Strategy creation
│       │   └── master_indicators.py              # Technical analysis
│       ├── data/
│       │   └── pre_market_data.py                # Pre-market global data analysis
│       ├── utilities/
│       │   └── get_access_token.py               # Token management
│       └── trade_storage/
│           ├── active_trades.json                # Active positions
│           └── trade_history.json                # Trade history
```

### Adding New Features
1. Create new tool functions in appropriate `core_tools` modules
2. Register tools in agent files using `@tool` decorator
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

## 🔍 Market Regime Detection

### Regime Classification
- **RANGING**: Sideways market with weak trends (ADX < 25)
- **TRENDING_BULL**: Strong uptrend with bullish signals
- **TRENDING_BEAR**: Strong downtrend with bearish signals
- **VOLATILE**: High volatility with stress conditions
- **COMPRESSED**: Low volatility with compressed conditions

### Scoring Factors
- **ADX**: Trend strength (threshold: 25)
- **RSI**: Momentum (overbought > 70, oversold < 30)
- **MACD/Supertrend**: Trend signals
- **Put-Call Ratio**: Market sentiment
- **Volatility Regime**: Market stress levels

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
- Depends on external APIs (Zerodha, OpenAI/Anthropic/Google)
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
- [Anthropic](https://anthropic.com/) for Claude AI
- [Google](https://ai.google.dev/) for Gemini AI
- [TA-Lib](https://ta-lib.org/) for technical analysis

---

**⚠️ Disclaimer: This software is for educational purposes only. Trading involves risk of loss. Use at your own risk.** 