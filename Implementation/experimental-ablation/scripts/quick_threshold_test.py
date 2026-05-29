#!/usr/bin/env python3
"""
Quick threshold test - minimal overhead.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add parent to path
sys.path.insert(0, str(Path(__file__).parents[2]))

from core.backtest import backtest_pairs, BacktestConfig, USCosts
from core.entry import ZScoreThreshold, OUThreshold


def test_thresholds():
    """Test different thresholds on US market 2021."""
    
    print("Loading prices...")
    prices = pd.read_parquet("../data/us_market.parquet")
    test_prices = prices["2021-01-01":"2021-12-31"]
    
    # Hardcode top 5 correlated pairs (skip selector to save time)
    top_pairs = [
        ("AAPL", "MSFT"),
        ("GOOGL", "MSFT"),
        ("V", "MA"),
        ("COST", "WMT"),
        ("JPM", "BAC")
    ]
    
    print(f"Testing on 2021: {len(test_prices)} days, {len(top_pairs)} pairs")
    print()
    
    config = BacktestConfig(costs=USCosts)
    
    print("ZSCORE THRESHOLDS")
    print("-" * 60)
    for entry_z in [1.5, 2.0, 2.5, 3.0]:
        signal = ZScoreThreshold(lookback=252, entry_z=entry_z, exit_z=0.5)
        bt = backtest_pairs(test_prices, top_pairs, signal, config)
        status = "✅" if bt.n_trades > 0 else "❌"
        print(f"{status} entry_z={entry_z:.1f}: {bt.n_trades:3} trades | Sharpe {bt.net_sharpe:+.3f}")
    
    print()
    print("OU THRESHOLDS")
    print("-" * 60)
    for entry_k in [0.3, 0.5, 0.8, 1.0, 1.2, 1.5]:
        signal = OUThreshold(lookback=252, entry_k=entry_k, exit_k=0.2)
        bt = backtest_pairs(test_prices, top_pairs, signal, config)
        status = "✅" if bt.n_trades > 0 else "❌"
        print(f"{status} entry_k={entry_k:.1f}: {bt.n_trades:3} trades | Sharpe {bt.net_sharpe:+.3f}")
    
    print("\nDone!")


if __name__ == "__main__":
    test_thresholds()
