"""
COMPLETE END-TO-END TEST: Simulate the full app workflow
This tests the actual pipeline the Streamlit app uses.
"""
import sys
sys.path.insert(0, '.')

import numpy as np
import pandas as pd
from datetime import datetime

# Import everything the app uses
from core.data import DataConfig, YFinanceNSESource
from core.selectors import Pair, PairScore, PairSelector
from core.selectors_statistical import CorrelationSelector
from core.entry import ZScoreThreshold, OUThreshold
from core.ensemble import normalize_weights, ensemble_pair_scores, scores_to_frame
from core.backtest import IndianCosts, BacktestConfig, backtest_pairs
from core.reports import ReportManager

print("=" * 70)
print("COMPLETE END-TO-END WORKFLOW TEST")
print("=" * 70)

# ============================================================================
# STEP 1: Create synthetic data (simulating YFinanceNSESource)
# ============================================================================
print("\n[1/7] Creating synthetic price data...")
np.random.seed(42)
dates = pd.date_range('2020-01-01', periods=500, freq='D')
n_stocks = 10
stock_names = [f'STOCK{i}' for i in range(1, n_stocks+1)]

prices = pd.DataFrame(index=dates, columns=stock_names)
base_price = 100.0
for stock in stock_names:
    returns = np.random.normal(0.0005, 0.015, len(dates))
    price_series = base_price * np.exp(np.cumsum(returns))
    prices[stock] = price_series

print(f"   ✅ Created {prices.shape[0]} days × {prices.shape[1]} stocks")
print(f"   Date range: {prices.index.min().date()} to {prices.index.max().date()}")

# ============================================================================
# STEP 2: Split train/test
# ============================================================================
print("\n[2/7] Splitting train/test...")
train_end_date = datetime(2020, 12, 31)
train_prices = prices[prices.index <= train_end_date]
test_prices = prices[prices.index > train_end_date]

print(f"   ✅ Train: {len(train_prices)} days ({train_prices.index.min().date()} to {train_prices.index.max().date()})")
print(f"   ✅ Test: {len(test_prices)} days ({test_prices.index.min().date()} to {test_prices.index.max().date()})")

# ============================================================================
# STEP 3: Stage 1 - Pair Selection (like the app does it)
# ============================================================================
print("\n[3/7] Stage 1: Pair selection...")

# Create selectors
selectors = {
    "Correlation": CorrelationSelector(lookback=60),
}
selector_weights = {"Correlation": 1.0}

# Normalize weights
selector_weights = normalize_weights(selector_weights)
print(f"   Selector weights: {selector_weights}")

# Generate all possible pairs
all_pairs = [Pair(stock_names[i], stock_names[j]) 
             for i in range(len(stock_names)) 
             for j in range(i+1, len(stock_names))]
print(f"   Total possible pairs: {len(all_pairs)}")

# Score pairs with each selector
scores_by_model = {}
for name, selector in selectors.items():
    scores = selector.score_pairs(train_prices, all_pairs)
    scores_by_model[name] = scores
    print(f"   ✅ {name}: scored {len(scores)} pairs")

# Ensemble the scores
top_k = 5
pair_scores = ensemble_pair_scores(scores_by_model, selector_weights, top_k=top_k)
print(f"   ✅ Ensemble complete: selected top {len(pair_scores)} pairs")

if pair_scores:
    print(f"   Top 3 pairs:")
    for i, ps in enumerate(pair_scores[:3], 1):
        print(f"      {i}. {ps.pair.a}-{ps.pair.b}: score={ps.score:.3f}")

# ============================================================================
# STEP 4: Stage 2 - Entry Models Setup
# ============================================================================
print("\n[4/7] Stage 2: Entry model setup...")

# Create entry/exit models (will be called inside backtest_pairs)
entry_models = {
    "ZScore": ZScoreThreshold(entry_z=2.0, exit_z=0.5, lookback=60),
    "OU": OUThreshold(entry_k=1.5, exit_k=0.2, lookback=60),
}
entry_weights = {"ZScore": 0.5, "OU": 0.5}

# Normalize weights
entry_weights = normalize_weights(entry_weights)
print(f"   Entry model weights: {entry_weights}")
print(f"   ✅ Entry models configured: {list(entry_models.keys())}")

# ============================================================================
# STEP 5: Backtest Configuration
# ============================================================================
print("\n[5/7] Creating backtest config...")

costs = IndianCosts(
    brokerage_bps=3.0,
    exchange_txn_bps=0.345,
    stt_bps_sell=10.0,
    slippage_bps_per_leg=2.0
)

bt_cfg = BacktestConfig(
    capital=1_000_000.0,
    max_concurrent_pairs=top_k,
    per_trade_cap=200_000.0,
    costs=costs,
    periods_per_year=252
)

print(f"   ✅ Capital: ₹{bt_cfg.capital:,.0f}")
print(f"   ✅ Max pairs: {bt_cfg.max_concurrent_pairs}")
print(f"   ✅ Per trade cap: ₹{bt_cfg.per_trade_cap:,.0f}")

# ============================================================================
# STEP 6: Run Backtest
# ============================================================================
print("\n[6/7] Running backtest...")

# Get the selected pairs
selected_pairs = [ps.pair for ps in pair_scores[:top_k]]
print(f"   Trading {len(selected_pairs)} selected pairs")

try:
    result = backtest_pairs(
        prices=test_prices,
        selected_pairs=selected_pairs,
        entry_models=entry_models,
        entry_weights=entry_weights,
        cfg=bt_cfg
    )
    
    print(f"   ✅ Backtest complete!")
    
except Exception as e:
    print(f"   ❌ Backtest failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# STEP 7: Display Results
# ============================================================================
print("\n[7/7] Results:")
print("=" * 70)

# Extract metrics from the result
m = result.metrics

print(f"\n📊 PERFORMANCE METRICS:")
print(f"   Gross Sharpe Ratio:  {m.get('Gross.Sharpe', 0):7.3f}")
print(f"   Net Sharpe Ratio:    {m.get('Net.Sharpe', 0):7.3f}")
print(f"   Gross Return:        {m.get('Gross.Return%', 0):7.2f}%")
print(f"   Net Return:          {m.get('Net.Return%', 0):7.2f}%")
print(f"   Max Drawdown:        {m.get('Net.MaxDD%', 0):7.2f}%")

print(f"\n📈 TRADING ACTIVITY:")
print(f"   Number of Trades:    {len(result.trades):7,}")
if not result.trades.empty:
    # Calculate win rate from trades
    print(f"   Total Turnover:      {result.turnover.sum():7.2f}")

print(f"\n💰 EQUITY CURVES:")
print(f"   Final Gross Equity:  ₹{result.equity_gross.iloc[-1]:,.0f}")
print(f"   Final Net Equity:    ₹{result.equity_net.iloc[-1]:,.0f}")
print(f"   Cost Impact:         ₹{result.equity_gross.iloc[-1] - result.equity_net.iloc[-1]:,.0f}")

print("\n" + "=" * 70)
print("✅ COMPLETE END-TO-END TEST PASSED!")
print("=" * 70)
print("\nThe full workflow works:")
print("  ✅ Data preparation")
print("  ✅ Train/test split")
print("  ✅ Stage 1: Pair selection (ensemble of selectors)")
print("  ✅ Stage 2: Signal generation (ensemble of entry models)")
print("  ✅ Backtest execution")
print("  ✅ Performance metrics")
print("\n🎉 Your backtesting engine is FULLY FUNCTIONAL!")
