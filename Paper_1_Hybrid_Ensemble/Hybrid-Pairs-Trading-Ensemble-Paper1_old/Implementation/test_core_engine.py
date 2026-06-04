"""
SIMPLIFIED TEST: Check core engine imports and basic functionality
"""
import sys
sys.path.insert(0, '.')

print("=" * 70)
print("CORE ENGINE TEST — Import & Basic Functionality")
print("=" * 70)

# Test 1: Imports
print("\n1. TESTING IMPORTS...")
try:
    from core.data import DataConfig
    print("   ✅ DataConfig")
except Exception as e:
    print(f"   ❌ DataConfig: {e}")
    sys.exit(1)

try:
    from core.backtest import BacktestConfig, IndianCosts, backtest_pairs, BacktestResult
    print("   ✅ BacktestConfig, IndianCosts, backtest_pairs, BacktestResult")
except Exception as e:
    print(f"   ❌ backtest: {e}")
    sys.exit(1)

try:
    from core.selectors_statistical import CorrelationSelector
    print("   ✅ CorrelationSelector")
except Exception as e:
    print(f"   ❌ CorrelationSelector: {e}")
    sys.exit(1)

try:
    from core.entry import ZScoreThreshold, OUThreshold
    print("   ✅ ZScoreThreshold, OUThreshold")
except Exception as e:
    print(f"   ❌ entry models: {e}")
    sys.exit(1)

try:
    from core.ensemble import ensemble_pair_scores, ensemble_signals
    print("   ✅ ensemble_pair_scores, ensemble_signals")
except Exception as e:
    print(f"   ❌ ensemble: {e}")
    sys.exit(1)

try:
    from core.selectors import Pair, PairScore
    print("   ✅ Pair, PairScore")
except Exception as e:
    print(f"   ❌ selectors base: {e}")
    sys.exit(1)

# Test 2: Create configs
print("\n2. TESTING CONFIG CREATION...")
try:
    from datetime import datetime
    
    data_cfg = DataConfig(
        start=datetime(2020, 1, 1),
        end=datetime(2021, 12, 31),
        freq='1D'
    )
    print(f"   ✅ DataConfig created: freq={data_cfg.freq}")
    
    costs = IndianCosts()
    print(f"   ✅ IndianCosts created: brokerage={costs.brokerage_bps}bps")
    
    bt_cfg = BacktestConfig(
        capital=1_000_000.0,
        max_concurrent_pairs=5,
        per_trade_cap=200_000.0,
        costs=costs
    )
    print(f"   ✅ BacktestConfig created: capital={bt_cfg.capital:,.0f}")
    
except Exception as e:
    print(f"   ❌ Config creation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Create models
print("\n3. TESTING MODEL CREATION...")
try:
    selector = CorrelationSelector(lookback=60)
    print(f"   ✅ CorrelationSelector created: lookback={selector.lookback}")
    
    zscore = ZScoreThreshold(entry_z=2.0, exit_z=0.5, lookback=60)
    print(f"   ✅ ZScoreThreshold created: entry_z={zscore.entry_z}")
    
    ou = OUThreshold(entry_k=2.0, exit_k=0.5, lookback=60)
    print(f"   ✅ OUThreshold created: entry_k={ou.entry_k}")
    
except Exception as e:
    print(f"   ❌ Model creation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Test pair scoring (minimal)
print("\n4. TESTING PAIR SCORING...")
try:
    import pandas as pd
    import numpy as np
    
    # Create minimal synthetic data
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    prices = pd.DataFrame({
        'A': 100 + np.cumsum(np.random.randn(100) * 0.5),
        'B': 100 + np.cumsum(np.random.randn(100) * 0.5),
        'C': 100 + np.cumsum(np.random.randn(100) * 0.5),
    }, index=dates)
    
    # Score all possible pairs
    candidates = [Pair('A', 'B'), Pair('A', 'C'), Pair('B', 'C')]
    scores = selector.score_pairs(prices, candidates)
    
    print(f"   ✅ Scored {len(scores)} pairs")
    for ps in scores:
        print(f"      {ps.pair.a}-{ps.pair.b}: score={ps.score:.3f}")
    
except Exception as e:
    print(f"   ❌ Pair scoring failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Test signal generation
print("\n5. TESTING SIGNAL GENERATION...")
try:
    # Generate signals for one pair
    test_prices = prices.iloc[60:]  # Use last 40 days for testing
    pair = Pair('A', 'B')
    
    signals_df = zscore.generate_signals(test_prices, [pair])
    
    print(f"   ✅ Generated signals: shape={signals_df.shape}")
    print(f"      Non-zero signals: {(signals_df != 0).sum().sum()}")
    print(f"      Signal range: [{signals_df.min().min():.1f}, {signals_df.max().max():.1f}]")
    
except Exception as e:
    print(f"   ❌ Signal generation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Run minimal backtest
print("\n6. TESTING BACKTEST ENGINE...")
try:
    # Run backtest with the generated signals
    result = backtest_pairs(
        prices=test_prices,
        signals=signals_df,
        config=bt_cfg
    )
    
    print(f"   ✅ Backtest complete!")
    print(f"      Gross Sharpe: {result.gross_sharpe:.3f}")
    print(f"      Net Sharpe:   {result.net_sharpe:.3f}")
    print(f"      Gross Return: {result.gross_return_pct:.2f}%")
    print(f"      Net Return:   {result.net_return_pct:.2f}%")
    print(f"      Num Trades:   {result.num_trades}")
    
except Exception as e:
    print(f"   ❌ Backtest failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 70)
print("✅ ALL TESTS PASSED — Core engine works!")
print("=" * 70)
print("\nThe backtesting engine and all core modules are functional.")
print("You can now run the Streamlit app or experiments.")
