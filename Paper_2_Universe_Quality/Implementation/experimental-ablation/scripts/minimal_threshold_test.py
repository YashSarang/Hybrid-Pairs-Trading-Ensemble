#!/usr/bin/env python3
"""
Ultra-minimal threshold test - direct signal testing only.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def zscore_signal(spread, lookback=252, entry_z=2.0, exit_z=0.5):
    """Simplified ZScore signal generator."""
    mu = spread.rolling(lookback).mean()
    std = spread.rolling(lookback).std()
    z = (spread - mu) / std
    
    sig = pd.Series(0, index=spread.index)
    sig[z > entry_z] = -1
    sig[z < -entry_z] = +1
    sig[z.abs() < exit_z] = 0
    
    return sig.ffill().fillna(0).astype(int)


def ou_signal(spread, lookback=252, entry_k=1.0, exit_k=0.2):
    """Simplified OU signal generator."""
    # AR(1) estimation
    y = spread - spread.rolling(lookback).mean()
    y_lag = y.shift(1)
    
    # Rolling regression
    def ar1_coef(window):
        if len(window) < 10:
            return np.nan
        y_curr = window.iloc[1:]
        y_prev = window.iloc[:-1].values
        if np.std(y_prev) < 1e-8:
            return 0.0
        return np.corrcoef(y_curr, y_prev)[0, 1] * (np.std(y_curr) / np.std(y_prev))
    
    phi = y.rolling(lookback).apply(lambda w: ar1_coef(pd.Series(w)), raw=False)
    
    # Scaled deviation
    std = spread.rolling(lookback).std()
    dev = spread / std
    scaled_dev = dev / (1 - phi.abs() + 0.01)
    
    sig = pd.Series(0, index=spread.index)
    sig[scaled_dev > entry_k] = -1
    sig[scaled_dev < -entry_k] = +1
    sig[scaled_dev.abs() < exit_k] = 0
    
    return sig.ffill().fillna(0).astype(int)


def count_trades(signal):
    """Count trade entries (signal changes from 0 to ±1 or between ±1)."""
    changes = signal.diff().fillna(0)
    entries = (changes.abs() > 0).sum()
    return entries


def test_pair_thresholds(prices, ticker_a, ticker_b):
    """Test thresholds on a single pair."""
    spread = prices[ticker_a] - prices[ticker_b]
    
    results = {
        "zscore": [],
        "ou": []
    }
    
    # ZScore
    for entry_z in [1.5, 2.0, 2.5, 3.0]:
        sig = zscore_signal(spread, entry_z=entry_z)
        trades = count_trades(sig)
        results["zscore"].append((entry_z, trades))
    
    # OU
    for entry_k in [0.3, 0.5, 0.8, 1.0, 1.2, 1.5]:
        sig = ou_signal(spread, entry_k=entry_k)
        trades = count_trades(sig)
        results["ou"].append((entry_k, trades))
    
    return results


def main():
    print("Loading prices...")
    prices = pd.read_parquet("../data/us_market.parquet")
    test_prices = prices["2021-01-01":"2021-12-31"]
    
    print(f"Test period: {test_prices.index[0]} to {test_prices.index[-1]} ({len(test_prices)} days)")
    print()
    
    # Test on AAPL_MSFT (highly correlated, should trade)
    pair = ("AAPL", "MSFT")
    print(f"Testing pair: {pair[0]}_{pair[1]}")
    print()
    
    results = test_pair_thresholds(test_prices, *pair)
    
    print("ZSCORE THRESHOLDS")
    print("-" * 50)
    for entry_z, trades in results["zscore"]:
        status = "✅" if trades > 0 else "❌"
        print(f"{status} entry_z={entry_z:.1f}: {trades:3} signal changes")
    
    print()
    print("OU THRESHOLDS")
    print("-" * 50)
    for entry_k, trades in results["ou"]:
        status = "✅" if trades > 0 else "❌"
        print(f"{status} entry_k={entry_k:.1f}: {trades:3} signal changes")
    
    print()
    print("✅ Done!")


if __name__ == "__main__":
    main()
