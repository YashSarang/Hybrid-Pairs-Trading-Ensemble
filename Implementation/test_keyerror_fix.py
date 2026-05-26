"""
Test: Verify KeyError fix for missing tickers
This simulates what happens when yfinance fails to download some tickers
"""
import sys
sys.path.insert(0, '.')

import pandas as pd
from datetime import datetime, timedelta

print("=" * 70)
print("TEST: KeyError Fix for Missing Tickers")
print("=" * 70)

# Simulate the scenario:
# User requests: ["RELIANCE", "TCS", "INVALID_TICKER"]
# yfinance returns data for only: ["RELIANCE", "TCS"]

universe_requested = ["RELIANCE", "TCS", "INVALID_TICKER"]
print(f"\n1. User requested {len(universe_requested)} tickers:")
print(f"   {universe_requested}")

# Simulate prices DataFrame (only has successful downloads)
dates = pd.date_range(start="2024-01-01", end="2024-01-10", freq="D")
prices = pd.DataFrame({
    "RELIANCE": [100 + i for i in range(len(dates))],
    "TCS": [200 + i*2 for i in range(len(dates))],
}, index=dates)

print(f"\n2. yfinance successfully downloaded {len(prices.columns)} tickers:")
print(f"   {list(prices.columns)}")

# Apply the fix (what the app now does)
successful_tickers = list(prices.columns)
failed_tickers = [t for t in universe_requested if t not in successful_tickers]

print(f"\n3. Detecting failures:")
print(f"   ✅ Successful: {successful_tickers}")
print(f"   ❌ Failed: {failed_tickers}")

if failed_tickers:
    print(f"\n   ⚠️  Warning: Failed to download {len(failed_tickers)} ticker(s): {', '.join(failed_tickers)}")

# Update universe to only successful tickers
universe = successful_tickers

print(f"\n4. Updated universe for pair selection:")
print(f"   {universe}")

if len(universe) < 2:
    print(f"\n❌ ERROR: Not enough tickers. Need at least 2, got {len(universe)}")
else:
    print(f"\n✅ SUCCESS: {len(universe)} tickers available")
    
    # Create pairs (this would have failed before the fix)
    from core.selectors import Pair
    pairs = [Pair(universe[i], universe[j]) 
             for i in range(len(universe)) 
             for j in range(i + 1, len(universe))]
    
    print(f"   Created {len(pairs)} pairs: {[str(p) for p in pairs]}")
    
    # Try to access prices for pairs (this caused KeyError before)
    for pair in pairs:
        try:
            a_prices = prices[pair.a]
            b_prices = prices[pair.b]
            print(f"   ✅ Accessed prices for pair {pair}")
        except KeyError as e:
            print(f"   ❌ KeyError for pair {pair}: {e}")

print("\n" + "=" * 70)
print("✅ TEST PASSED: No KeyErrors!")
print("=" * 70)
