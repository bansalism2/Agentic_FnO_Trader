#!/usr/bin/env python3
"""
Enhanced Portfolio History Module
=================================

Improvements over the basic version:
1. More sophisticated trend detection
2. Better error handling
3. Configurable thresholds
4. Detailed pattern recognition
"""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional

# ============================================================================
# CONFIGURATION
# ============================================================================

HISTORY_FILE = os.path.join(os.path.dirname(__file__), "portfolio_history.json")
MAX_SNAPSHOTS = 6

# ADJUSTED CONFIGURATION
TREND_CONFIG = {
    'profit_erosion_threshold': 5,       # REDUCED: 5% decline from peak (was 7%)
    'loss_acceleration_threshold': 25,    # KEEP: Working well
    'failed_recovery_attempts': 2,       # KEEP: OK
    'min_duration_minutes': 30,         # REDUCED: 30 minutes (was 45) for faster detection
    'catastrophic_loss_threshold': 30,   # KEEP: OK
    'min_profit_for_erosion': 8,        # NEW: Minimum profit before tracking erosion (was 10%)
}

portfolio_memory = {
    "snapshots": []
}

# ============================================================================
# CORE FUNCTIONS
# ============================================================================

def load_portfolio_history():
    """Load portfolio history with better error handling"""
    global portfolio_memory
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r") as f:
                data = json.load(f)
                # Validate data structure
                if isinstance(data, dict) and "snapshots" in data:
                    portfolio_memory = data
                else:
                    print("Warning: Invalid portfolio history format, resetting")
                    portfolio_memory = {"snapshots": []}
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"Warning: Could not load portfolio history: {e}")
            portfolio_memory = {"snapshots": []}
    else:
        portfolio_memory = {"snapshots": []}

def record_portfolio_snapshot(total_portfolio_pnl_pct, strategies):
    """
    Enhanced snapshot recording with validation
    """
    global portfolio_memory
    
    # Validate inputs
    if not isinstance(strategies, dict):
        print("Warning: Invalid strategies data type")
        return False
    
    try:
        snapshot = {
            "timestamp": datetime.now().isoformat(),
            "total_portfolio_pnl_pct": float(total_portfolio_pnl_pct),
            "strategies": {k: v for k, v in strategies.items() if isinstance(v, dict)}
        }
        
        portfolio_memory["snapshots"].append(snapshot)
        
        # Keep only the last MAX_SNAPSHOTS
        if len(portfolio_memory["snapshots"]) > MAX_SNAPSHOTS:
            portfolio_memory["snapshots"] = portfolio_memory["snapshots"][-MAX_SNAPSHOTS:]
        
        # Save to disk with error handling
        try:
            os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)
            with open(HISTORY_FILE, "w") as f:
                json.dump(portfolio_memory, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving portfolio history: {e}")
            return False
            
    except Exception as e:
        print(f"Error recording snapshot: {e}")
        return False

def detect_trend_patterns():
    """
    Enhanced trend detection with specific pattern recognition
    """
    if len(portfolio_memory["snapshots"]) < 3:
        return {}

    issues = {}
    
    # Get time duration
    first_time = datetime.fromisoformat(portfolio_memory["snapshots"][0]["timestamp"])
    last_time = datetime.fromisoformat(portfolio_memory["snapshots"][-1]["timestamp"])
    duration_minutes = (last_time - first_time).total_seconds() / 60
    
    # Portfolio-level analysis
    portfolio_pnls = [snap["total_portfolio_pnl_pct"] for snap in portfolio_memory["snapshots"]]
    portfolio_issue = analyze_pnl_trend(portfolio_pnls, duration_minutes, "PORTFOLIO")
    if portfolio_issue:
        issues["portfolio"] = portfolio_issue
    
    # Strategy-level analysis
    last_snapshot = portfolio_memory["snapshots"][-1]
    for trade_id in last_snapshot["strategies"]:
        strategy_pnls = []
        strategy_name = ""
        
        for snap in portfolio_memory["snapshots"]:
            if trade_id in snap["strategies"]:
                strategy_pnls.append(snap["strategies"][trade_id]["pnl_pct"])
                strategy_name = snap["strategies"][trade_id].get("strategy_name", "Unknown")
        
        if len(strategy_pnls) >= 3:
            strategy_issue = analyze_pnl_trend(strategy_pnls, duration_minutes, strategy_name)
            if strategy_issue:
                issues[trade_id] = strategy_issue
    
    return issues

def analyze_pnl_trend(pnl_series: list, duration_minutes: float, entity_name: str):
    """
    Improved version with better thresholds and logic
    """
    if len(pnl_series) < 3:
        return None
    current_pnl = pnl_series[-1]
    initial_pnl = pnl_series[0]
    peak_pnl = max(pnl_series)
    # Pattern 1: Profit Erosion (FIXED)
    if (peak_pnl > TREND_CONFIG['min_profit_for_erosion'] and
        current_pnl < peak_pnl - TREND_CONFIG['profit_erosion_threshold'] and
        duration_minutes >= TREND_CONFIG['min_duration_minutes']):
        decline_pct = ((peak_pnl - current_pnl) / peak_pnl) * 100
        return f"PROFIT_EROSION: {entity_name} declined {decline_pct:.1f}% from peak {peak_pnl:.1f}% to {current_pnl:.1f}%"
    # Pattern 2: Loss Acceleration (WORKING - NO CHANGE)
    if (current_pnl < -TREND_CONFIG['loss_acceleration_threshold'] and
        current_pnl < initial_pnl and
        is_accelerating_loss(pnl_series)):
        return f"LOSS_ACCELERATION: {entity_name} loss accelerating from {initial_pnl:.1f}% to {current_pnl:.1f}%"
    # Pattern 3: Failed Recovery (FIXED)
    if (current_pnl < -5 and
        duration_minutes >= TREND_CONFIG['min_duration_minutes']):
        failed_recoveries = count_failed_recoveries(pnl_series)
        if failed_recoveries >= TREND_CONFIG['failed_recovery_attempts']:
            return f"FAILED_RECOVERY: {entity_name} has {failed_recoveries} failed recovery attempts, current: {current_pnl:.1f}%"
    # Pattern 4: Catastrophic Loss (NO CHANGE)
    if current_pnl < -TREND_CONFIG['catastrophic_loss_threshold']:
        return f"CATASTROPHIC_LOSS: {entity_name} at {current_pnl:.1f}% loss - immediate exit recommended"
    return None

def is_accelerating_loss(pnl_series: List[float]) -> bool:
    """Check if losses are accelerating (getting worse faster)"""
    if len(pnl_series) < 3:
        return False
    
    # Check if each interval shows larger negative changes
    changes = [pnl_series[i] - pnl_series[i-1] for i in range(1, len(pnl_series))]
    negative_changes = [c for c in changes if c < 0]
    
    # Loss acceleration: more than half the changes are negative and getting larger
    if len(negative_changes) >= len(changes) / 2:
        return abs(negative_changes[-1]) > abs(negative_changes[0]) if len(negative_changes) > 1 else True
    
    return False

def count_failed_recoveries(pnl_series: list) -> int:
    """
    More sensitive failed recovery detection
    """
    if len(pnl_series) < 3:
        return 0
    failed_count = 0
    for i in range(2, len(pnl_series)):
        prev_prev = pnl_series[i-2]
        prev = pnl_series[i-1]
        current = pnl_series[i]
        if (prev_prev < 0 and prev > prev_prev and current < prev):
            failed_count += 1
    return failed_count

def get_strategy_trend_history(trade_id: str) -> List[Dict]:
    """
    Enhanced strategy trend history with validation
    """
    history = []
    
    for snap in portfolio_memory["snapshots"]:
        if trade_id in snap.get("strategies", {}):
            strategy_data = snap["strategies"][trade_id]
            if isinstance(strategy_data, dict) and "pnl_pct" in strategy_data:
                history.append({
                    "timestamp": snap["timestamp"],
                    "pnl_pct": strategy_data["pnl_pct"]
                })
    
    return history

def get_trend_summary() -> Dict:
    """
    Get a summary of current trends for reporting
    """
    issues = detect_trend_patterns()
    
    summary = {
        "total_strategies_tracked": 0,
        "strategies_with_issues": 0,
        "portfolio_issue": None,
        "critical_issues": [],
        "warning_issues": [],
        "snapshot_count": len(portfolio_memory["snapshots"]),
        "data_duration_minutes": 0
    }
    
    if len(portfolio_memory["snapshots"]) >= 2:
        first_time = datetime.fromisoformat(portfolio_memory["snapshots"][0]["timestamp"])
        last_time = datetime.fromisoformat(portfolio_memory["snapshots"][-1]["timestamp"])
        summary["data_duration_minutes"] = (last_time - first_time).total_seconds() / 60
    
    # Count strategies
    if portfolio_memory["snapshots"]:
        last_snapshot = portfolio_memory["snapshots"][-1]
        summary["total_strategies_tracked"] = len(last_snapshot.get("strategies", {}))
    
    # Categorize issues
    for entity, issue in issues.items():
        if entity == "portfolio":
            summary["portfolio_issue"] = issue
        else:
            summary["strategies_with_issues"] += 1
            
            if "CATASTROPHIC" in issue or "LOSS_ACCELERATION" in issue:
                summary["critical_issues"].append({
                    "trade_id": entity,
                    "issue": issue
                })
            else:
                summary["warning_issues"].append({
                    "trade_id": entity,
                    "issue": issue
                })
    
    return summary

def reset_portfolio_history():
    """Reset portfolio history (useful for testing)"""
    global portfolio_memory
    portfolio_memory = {"snapshots": []}
    try:
        if os.path.exists(HISTORY_FILE):
            os.remove(HISTORY_FILE)
    except Exception as e:
        print(f"Warning: Could not remove history file: {e}")

# ============================================================================
# TESTING FUNCTIONS
# ============================================================================

def simulate_trend_scenario(scenario_name: str):
    """
    Simulate different trend scenarios for testing.
    If scenario_name is 'all', run all scenarios.
    """
    scenarios = {
        "profit_erosion": [
            {"total": 15.0, "strategy1": {"pnl_pct": 18.0, "strategy_name": "Iron Condor"}},
            {"total": 12.0, "strategy1": {"pnl_pct": 14.5, "strategy_name": "Iron Condor"}},
            {"total": 8.0, "strategy1": {"pnl_pct": 10.2, "strategy_name": "Iron Condor"}},
            {"total": 4.0, "strategy1": {"pnl_pct": 5.8, "strategy_name": "Iron Condor"}},
        ],
        "loss_acceleration": [
            {"total": -8.0, "strategy1": {"pnl_pct": -12.0, "strategy_name": "Long Straddle"}},
            {"total": -15.0, "strategy1": {"pnl_pct": -18.0, "strategy_name": "Long Straddle"}},
            {"total": -24.0, "strategy1": {"pnl_pct": -28.0, "strategy_name": "Long Straddle"}},
            {"total": -32.0, "strategy1": {"pnl_pct": -35.0, "strategy_name": "Long Straddle"}},
        ],
        "failed_recovery": [
            {"total": -15.0, "strategy1": {"pnl_pct": -20.0, "strategy_name": "Short Strangle"}},
            {"total": -10.0, "strategy1": {"pnl_pct": -12.0, "strategy_name": "Short Strangle"}},
            {"total": -18.0, "strategy1": {"pnl_pct": -22.0, "strategy_name": "Short Strangle"}},
            {"total": -8.0, "strategy1": {"pnl_pct": -10.0, "strategy_name": "Short Strangle"}},
            {"total": -20.0, "strategy1": {"pnl_pct": -25.0, "strategy_name": "Short Strangle"}},
        ]
    }

    if scenario_name == "all":
        for name in scenarios:
            print(f"\n=== Simulating {name} scenario ===")
            reset_portfolio_history()
            for data_point in scenarios[name]:
                total_pnl = data_point["total"]
                strategies = {k: v for k, v in data_point.items() if k != "total"}
                record_portfolio_snapshot(total_pnl, strategies)
            issues = detect_trend_patterns()
            summary = get_trend_summary()
            print(f"Issues detected: {issues}")
            print(f"Summary: {summary}")
        return

    if scenario_name not in scenarios:
        print(f"Unknown scenario: {scenario_name}")
        return

    print(f"Simulating {scenario_name} scenario...")
    reset_portfolio_history()
    for data_point in scenarios[scenario_name]:
        total_pnl = data_point["total"]
        strategies = {k: v for k, v in data_point.items() if k != "total"}
        record_portfolio_snapshot(total_pnl, strategies)
    issues = detect_trend_patterns()
    summary = get_trend_summary()
    print(f"Issues detected: {issues}")
    print(f"Summary: {summary}")

# Add threshold summary and test functions for direct script testing

def print_threshold_comparison():
    print("\nThreshold Adjustments Summary:")
    print("=" * 50)
    print("Setting                     | Old Value | New Value | Reason")
    print("-" * 50)
    print("Profit erosion threshold    |    7%     |    5%     | More sensitive")
    print("Min profit for erosion      |   10%     |    8%     | Lower barrier")
    print("Min duration (minutes)      |   45      |   30      | Faster detection")
    print("Failed recovery min loss    |   -10%    |   -5%     | Catch smaller losses")
    print("Min data points (recovery)  |    4      |    3      | Faster detection")
    print("-" * 50)
    print("Loss acceleration threshold |   25%     |   25%     | Working fine (no change)")
    print("Catastrophic loss threshold |   30%     |   30%     | Working fine (no change)")

def test_adjusted_thresholds():
    print("Testing Adjusted Thresholds...")
    print("=" * 50)
    # Test 1: Profit Erosion
    print("\n1. Testing Profit Erosion (should trigger):")
    pnl_series = [15.0, 12.0, 10.0, 8.0]  # 15% to 8% = 47% decline
    result = analyze_pnl_trend(pnl_series, 35, "Test Strategy")
    print(f"   Result: {result}")
    print(f"   Expected: Should detect profit erosion")
    # Test 2: Failed Recovery
    print("\n2. Testing Failed Recovery (should trigger):")
    pnl_series = [-15.0, -10.0, -18.0, -8.0, -20.0]  # Clear up/down pattern
    failed_recoveries = count_failed_recoveries(pnl_series)
    result = analyze_pnl_trend(pnl_series, 35, "Test Strategy")
    print(f"   Failed recoveries count: {failed_recoveries}")
    print(f"   Result: {result}")
    print(f"   Expected: Should detect 2 failed recoveries")
    # Test 3: Loss Acceleration (should still work)
    print("\n3. Testing Loss Acceleration (should still work):")
    pnl_series = [-12.0, -18.0, -28.0, -35.0]
    result = analyze_pnl_trend(pnl_series, 35, "Test Strategy")
    print(f"   Result: {result}")
    print(f"   Expected: Should detect loss acceleration")

if __name__ == "__main__":
    print_threshold_comparison()
    test_adjusted_thresholds()

# Load history on import
load_portfolio_history()