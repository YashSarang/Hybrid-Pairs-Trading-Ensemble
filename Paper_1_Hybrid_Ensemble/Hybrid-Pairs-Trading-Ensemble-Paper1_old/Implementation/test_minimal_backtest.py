"""
Minimal test: Can we run a backtest with synthetic data?
This verifies the core engine works without needing real market data.
"""
import sys
sys.path.insert(0, '.')

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from core.data import DataConfig
from core.backtest import BacktestConfig, IndianCosts, backtest_pairs
from core.selectors_statistical import CorrelationSelector
from core.entry import ZScoreThreshold
from core.ensemble import ensemble_pair_scores, ensemble_signals

print("=" * 70)
print("MINIMAL BACKTEST TEST (Synthetic Data)")
print("=" * 70)

# 1. Create synthetic price data (5 stocks, 500 days)
print("\n1. Creating synthetic data...")
np.random.seed(42)
dates = pd.date_range('2020-01-01', periods=500, freq='D')
n_stocks = 5
stock_names = [f'STOCK{i}' for i in range(1, n_stocks+1)]

# Generate correlated random walks
prices = pd.DataFrame(index=dates, columns=stock_names)
base_price = 100.0
for stock in stock_names:
    returns = np.random.normal(0.0005, 0.02, len(dates))  # slight positive drift, 2% daily vol
    price_series = base_price * np.exp(np.cumsum(returns))
    prices[stock] = price_series

print(f"   ✅ Created {prices.shape[0]} days × {prices.shape[1]} stocks")
print(f"   Date range: {prices.index.min()} to {prices.index.max()}")

# 2. Create configs
print("\n2. Creating configs...")
data_cfg = DataConfig(
    start=datetime(2020, 1, 1),
    end=datetime(2021, 12, 31),
    freq='1D'
)

costs = IndianCosts()

bt_cfg = BacktestConfig(
    capital=1_000_000.0,
    max_concurrent_pairs=3,
    per_trade_cap=100_000.0,
    costs=costs,
    periods_per_year=252
)

train_end_date = datetime(2020, 12, 31)

print(f"   ✅ DataConfig: {data_cfg.freq}")
print(f"   ✅ BacktestConfig: capital={bt_cfg.capital:,.0f}")
print(f"   ✅ IndianCosts: brokerage={costs.brokerage_bps}bps, exchange={costs.exchange_txn_bps}bps")

# 3. Create selectors and entry models
print("\n3. Creating models...")
selectors = [CorrelationSelector(lookback=60)]
selector_weights = [1.0]

entry_models = [ZScoreThreshold(entry_z=2.0, exit_z=0.5, lookback=20)]
entry_weights = [1.0]

print(f"   ✅ Selectors: {len(selectors)}")
print(f"   ✅ Entry models: {len(entry_models)}")

# 4. Split train/test
print("\n4. Splitting train/test...")
train_prices = prices[prices.index <= train_end_date]
test_prices = prices[prices.index > train_end_date]

print(f"   ✅ Train: {len(train_prices)} days")
print(f"   ✅ Test: {len(test_prices)} days")

# 5. Stage 1: Score pairs
print("\n5. Stage 1: Scoring pairs...")
try:
    pair_scores = ensemble_pair_scores(
        selectors=selectors,
        weights=selector_weights,
        prices=train_prices
    )
    print(f"   ✅ Scored {len(pair_scores)} pairs")
    if pair_scores:
        print(f"   Top 3 pairs: {pair_scores[:3]}")
except Exception as e:
    print(f"   ❌ FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 6. Stage 2: Generate signals
print("\n6. Stage 2: Generating signals...")
try:
    if not pair_scores:
        print("   ⚠️  No pairs scored, using all possible pairs")
        pairs = [(stock_names[i], stock_names[j]) 
                 for i in range(len(stock_names)) 
                 for j in range(i+1, len(stock_names))]
    else:
        pairs = [ps.pair for ps in pair_scores[:bt_cfg.max_concurrent_pairs]]
    
    print(f"   Selected pairs: {pairs}")
    
    signals = ensemble_signals(
        entry_models=entry_models,
        weights=entry_weights,
        prices=test_prices,
        pairs=pairs
    )
    print(f"   ✅ Generated signals: {signals.shape}")
    print(f"   Non-zero signals: {(signals != 0).sum().sum()}")
except Exception as e:
    print(f"   ❌ FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 7. Run backtest
print("\n7. Running backtest...")
try:
    result = backtest_pairs(
        prices=test_prices,
        signals=signals,
        config=bt_cfg
    )
    
    print(f"   ✅ Backtest complete!")
    print(f"\n   RESULTS:")
    print(f"   --------")
    print(f"   Gross Sharpe: {result.gross_sharpe:.3f}")
    print(f"   Net Sharpe:   {result.net_sharpe:.3f}")
    print(f"   Gross Return: {result.gross_return_pct:.2f}%")
    print(f"   Net Return:   {result.net_return_pct:.2f}%")
    print(f"   Max Drawdown: {result.max_drawdown_pct:.2f}%")
    print(f"   Num Trades:   {result.num_trades}")
    
except Exception as e:
    print(f"   ❌ FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 70)
print("✅ ALL TESTS PASSED — Core engine works!")
print("=" * 70)
