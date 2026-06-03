#!/usr/bin/env python3
"""
Test different signal thresholds to find working parameters.
Run locally before submitting to cluster.
"""

import sys
import json
from pathlib import Path
import pandas as pd

# Add parent to path
sys.path.insert(0, str(Path(__file__).parents[2]))

from core.selectors import CorrelationSelector
from core.backtest import backtest_pairs, BacktestConfig, USCosts
from core.entry import ZScoreThreshold, OUThreshold


def quick_test_thresholds(prices_path: str):
    """Test different thresholds on US market with 2021 data only."""
    
    print("=" * 80)
    print("SIGNAL THRESHOLD DEBUGGING")
    print("=" * 80)
    print()
    
    # Load US prices
    prices = pd.read_parquet(prices_path)
    print(f"Loaded {len(prices.columns)} tickers, {len(prices)} days")
    print(f"Period: {prices.index[0]} to {prices.index[-1]}")
    print()
    
    # Use 2020 train / 2021 test (single fold for speed)
    train_prices = prices["2020-01-01":"2020-12-31"]
    test_prices = prices["2021-01-01":"2021-12-31"]
    
    print(f"Train: {train_prices.index[0]} to {train_prices.index[-1]} ({len(train_prices)} days)")
    print(f"Test:  {test_prices.index[0]} to {test_prices.index[-1]} ({len(test_prices)} days)")
    print()
    
    # Select top 5 pairs quickly (correlation only)
    print("Selecting pairs (correlation selector)...")
    selector = CorrelationSelector()
    selector.fit(train_prices)
    
    candidates = []
    tickers = train_prices.columns.tolist()
    for i, t1 in enumerate(tickers):
        for t2 in tickers[i+1:]:
            candidates.append((t1, t2))
    
    scored = selector.score_pairs(train_prices, candidates)
    top_pairs = [p.pair for p in scored[:5]]
    print(f"Selected {len(top_pairs)} pairs: {', '.join('_'.join(p) for p in top_pairs)}")
    print()
    
    # Test different thresholds
    config = BacktestConfig(costs=USCosts)
    
    results = []
    
    print("=" * 80)
    print("TESTING ZSCORE THRESHOLDS")
    print("=" * 80)
    
    for entry_z in [1.5, 2.0, 2.5, 3.0]:
        signal = ZScoreThreshold(lookback=252, entry_z=entry_z, exit_z=0.5)
        
        bt = backtest_pairs(
            prices=test_prices,
            pairs=top_pairs,
            signal_model=signal,
            config=config
        )
        
        results.append({
            "signal": "zscore",
            "entry_z": entry_z,
            "exit_z": 0.5,
            "trades": bt.n_trades,
            "net_sharpe": bt.net_sharpe,
            "gross_sharpe": bt.gross_sharpe,
            "net_return": bt.net_return,
            "max_dd": bt.max_dd
        })
        
        status = "✅" if bt.n_trades > 0 else "❌"
        print(f"{status} entry_z={entry_z:.1f}: {bt.n_trades:3} trades | "
              f"Sharpe {bt.net_sharpe:+.3f} | Return {bt.net_return:+.2%} | DD {bt.max_dd:.1%}")
    
    print()
    print("=" * 80)
    print("TESTING OU THRESHOLDS")
    print("=" * 80)
    
    for entry_k in [0.3, 0.5, 0.8, 1.0, 1.2, 1.5]:
        signal = OUThreshold(lookback=252, entry_k=entry_k, exit_k=0.2)
        
        bt = backtest_pairs(
            prices=test_prices,
            pairs=top_pairs,
            signal_model=signal,
            config=config
        )
        
        results.append({
            "signal": "ou",
            "entry_k": entry_k,
            "exit_k": 0.2,
            "trades": bt.n_trades,
            "net_sharpe": bt.net_sharpe,
            "gross_sharpe": bt.gross_sharpe,
            "net_return": bt.net_return,
            "max_dd": bt.max_dd
        })
        
        status = "✅" if bt.n_trades > 0 else "❌"
        print(f"{status} entry_k={entry_k:.1f}: {bt.n_trades:3} trades | "
              f"Sharpe {bt.net_sharpe:+.3f} | Return {bt.net_return:+.2%} | DD {bt.max_dd:.1%}")
    
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    # Find best for each signal type
    zscore_results = [r for r in results if r["signal"] == "zscore"]
    ou_results = [r for r in results if r["signal"] == "ou"]
    
    zscore_with_trades = [r for r in zscore_results if r["trades"] > 0]
    ou_with_trades = [r for r in ou_results if r["trades"] > 0]
    
    print("\n📊 ZSCORE RESULTS:")
    if zscore_with_trades:
        best_z = max(zscore_with_trades, key=lambda x: x["net_sharpe"])
        print(f"   ✅ Best: entry_z={best_z['entry_z']:.1f} → {best_z['trades']} trades, Sharpe {best_z['net_sharpe']:+.3f}")
        print(f"   📈 Recommended: entry_z={best_z['entry_z']:.1f}, exit_z=0.5")
    else:
        print("   ❌ NO thresholds produced trades!")
    
    print("\n📊 OU RESULTS:")
    if ou_with_trades:
        best_ou = max(ou_with_trades, key=lambda x: x["net_sharpe"])
        print(f"   ✅ Best: entry_k={best_ou['entry_k']:.1f} → {best_ou['trades']} trades, Sharpe {best_ou['net_sharpe']:+.3f}")
        print(f"   📈 Recommended: entry_k={best_ou['entry_k']:.1f}, exit_k=0.2")
    else:
        print("   ❌ NO thresholds produced trades!")
    
    # Save results
    output_file = Path(__file__).parent.parent / "results" / "threshold_debug.json"
    output_file.parent.mkdir(exist_ok=True)
    with open(output_file, "w") as f:
        json.dump({
            "test_period": "2021",
            "pairs_tested": len(top_pairs),
            "results": results,
            "best_zscore": best_z if zscore_with_trades else None,
            "best_ou": best_ou if ou_with_trades else None
        }, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_file}")
    print()


if __name__ == "__main__":
    prices_path = Path(__file__).parent.parent / "data" / "us_market.parquet"
    
    if not prices_path.exists():
        print(f"❌ Error: {prices_path} not found!")
        print("   Run fetch_market_data.py first.")
        sys.exit(1)
    
    quick_test_thresholds(str(prices_path))
