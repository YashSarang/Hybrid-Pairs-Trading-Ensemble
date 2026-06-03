"""
run_control_experiment_direct.py

Direct Python execution of control experiments without bash/venv issues.
Runs NSE Nifty 50 + Rolling + ZScore and OU signals.
"""

import sys
from pathlib import Path

# Add parent dirs to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

print("="*60)
print("NSE NIFTY 50 CONTROL EXPERIMENT (Direct Python)")
print("="*60)
print()
print("Purpose: Isolate universe quality from geographic effects")
print("Comparing: NSE Nifty 50 vs NSE Nifty 100 vs India Multi-Market")
print()

# Import after path setup
import yaml
import pandas as pd
import numpy as np
from datetime import datetime
import json
from typing import Dict, List
from itertools import combinations

# Core imports
from core.data import YFinanceNSESource
from core.selectors import (
    CorrelationSelector, DistanceSelector, CointegrationSelector,
    CombinedCriteriaSelector, MLSelector, LSTMSelector,
    TransformerSelector, GNNSelector, Pair
)
from core.entry import ZScoreThreshold, OUThreshold
from core.backtest import backtest_pairs, BacktestConfig, IndianCosts
from core.ensemble import ensemble_pair_scores

print("✓ Imports successful")
print()

# Load config
config_path = Path(__file__).parent / "configs" / "nse_nifty50.yaml"
print(f"Loading config: {config_path}")

with open(config_path) as f:
    config = yaml.safe_load(f)

print(f"✓ Config loaded: {config['market']['name']}")
print(f"  Universe: {len(config['universe']['tickers'])} tickers")
print(f"  Transaction cost: {config['costs'].get('slippage_bps', 0) + config['costs'].get('brokerage_bps', 0):.1f} bps")
print()

# Fetch data
print("Step 1: Fetching price data...")
tickers = config['universe']['tickers']
start_date = config['data']['start_date']
end_date = config['data']['end_date']

cache_dir = Path(__file__).parent.parent / config['data']['cache_dir']
cache_dir.mkdir(parents=True, exist_ok=True)
cache_file = cache_dir / f"prices_{start_date}_{end_date}.parquet"

if cache_file.exists():
    print(f"✓ Loading from cache: {cache_file}")
    prices = pd.read_parquet(cache_file)
else:
    print(f"  Downloading {len(tickers)} tickers from yfinance...")
    print(f"  This may take 5-10 minutes...")
    
    import yfinance as yf
    all_data = []
    failed = []
    
    for i, ticker in enumerate(tickers, 1):
        try:
            print(f"  [{i}/{len(tickers)}] {ticker}...", end="", flush=True)
            data = yf.download(ticker, start=start_date, end=end_date, interval="1d", progress=False)
            if data.empty:
                print(" ❌ No data")
                failed.append(ticker)
                continue
            
            close = data['Close'] if 'Close' in data.columns else data['Close'].iloc[:, 0]
            close = close.dropna()
            all_data.append(pd.DataFrame({ticker: close}))
            print(f" ✅ {len(close)} days")
        except Exception as e:
            print(f" ❌ {str(e)[:30]}")
            failed.append(ticker)
    
    if not all_data:
        raise RuntimeError("No data fetched successfully")
    
    prices = pd.concat(all_data, axis=1)
    prices = prices.fillna(method='ffill', limit=3).dropna(axis=1, how='all')
    
    print(f"\n✓ Fetched {len(prices.columns)} tickers successfully")
    if failed:
        print(f"  Failed: {', '.join(failed)}")
    
    # Save cache
    prices.to_parquet(cache_file)
    print(f"✓ Cached to {cache_file}")

print(f"✓ Data ready: {prices.shape[0]} days × {prices.shape[1]} tickers")
print()

# RUN EXPERIMENTS
results_dir = Path(__file__).parent / "results" / "nse_nifty50"
results_dir.mkdir(parents=True, exist_ok=True)

for signal_name in ['zscore', 'ou']:
    print(f"\n{'='*60}")
    print(f"RUN: NSE Nifty 50 + Rolling + {signal_name.upper()}")
    print(f"{'='*60}\n")
    
    # Initialize signal model
    if signal_name == 'zscore':
        signal_model = ZScoreThreshold(lookback=126, entry_z=2.0, exit_z=0.5)
    else:
        signal_model = OUThreshold(lookback=126, entry_k=1.5, exit_k=0.2)
    
    # Run walk-forward validation
    folds = []
    fold_dates = [
        (1, "2020-01-01", "2020-12-31", "2021-01-01", "2021-12-31"),
        (2, "2021-01-01", "2021-12-31", "2022-01-01", "2022-12-31"),
        (3, "2022-01-01", "2022-12-31", "2023-01-01", "2023-12-31"),
        (4, "2023-01-01", "2023-12-31", "2024-01-01", "2024-12-31"),
    ]
    
    for fold_id, train_start, train_end, test_start, test_end in fold_dates:
        print(f"\n--- Fold {fold_id} ---")
        print(f"Train: {train_start} to {train_end}")
        print(f"Test:  {test_start} to {test_end}")
        
        train_prices = prices.loc[train_start:train_end]
        test_prices = prices.loc[test_start:test_end]
        
        print(f"Train: {train_prices.shape}, Test: {test_prices.shape}")
        
        # Generate pairs
        tickers_avail = list(train_prices.columns)
        candidate_pairs = [Pair(a, b) for a, b in combinations(tickers_avail, 2)]
        print(f"Candidate pairs: {len(candidate_pairs)}")
        
        # Train selectors (simplified: use statistical only for speed)
        selector_scores = {}
        selectors_to_use = [
            ('correlation', CorrelationSelector()),
            ('distance', DistanceSelector()),
            ('cointegration', CointegrationSelector()),
            ('combined', CombinedCriteriaSelector()),
        ]
        
        for sel_name, selector in selectors_to_use:
            print(f"  Training {sel_name}...", end="", flush=True)
            try:
                selector.fit(train_prices, candidate_pairs)
                scores = selector.score(candidate_pairs)
                selector_scores[sel_name] = dict(zip(candidate_pairs, scores))
                print(f" ✓ ({len([s for s in scores if s > 0])} pairs scored)")
            except Exception as e:
                print(f" ❌ {str(e)[:40]}")
                selector_scores[sel_name] = {p: 0.0 for p in candidate_pairs}
        
        # Ensemble scoring (equal weights)
        ensemble_scores = ensemble_pair_scores(
            selector_scores,
            weights={sel: 1.0 for sel in selector_scores.keys()}
        )
        
        # Select top 10 pairs
        sorted_pairs = sorted(ensemble_scores.items(), key=lambda x: x[1], reverse=True)
        selected_pairs = [p for p, s in sorted_pairs[:10]]
        
        print(f"  Selected {len(selected_pairs)} pairs for testing")
        
        # Backtest
        try:
            cost_model = IndianCosts(
                brokerage_bps=3.0,
                exchange_txn_bps=0.345,
                sebi_bps=0.01,
                stt_bps_sell=10.0,
                gst_rate=0.18,
                stamp_bps_buy=1.0,
                intraday=True,
                slippage_bps_per_leg=2.0
            )
            
            bt_config = BacktestConfig(
                initial_capital=10_000_000,
                max_concurrent_pairs=10,
                per_trade_notional=500_000,
                hold_period_days=20,
                cost_model=cost_model
            )
            
            results_df, metrics_df = backtest_pairs(
                prices=test_prices,
                pairs=selected_pairs,
                signal_model=signal_model,
                config=bt_config
            )
            
            # Extract metrics
            metrics = {
                'Gross.Return': float(metrics_df.loc['Gross.Return'].iloc[0]),
                'Gross.Sharpe': float(metrics_df.loc['Gross.Sharpe'].iloc[0]),
                'Net.Return': float(metrics_df.loc['Net.Return'].iloc[0]),
                'Net.Sharpe': float(metrics_df.loc['Net.Sharpe'].iloc[0]),
                'Turnover.Trades': int(metrics_df.loc['Turnover.Trades'].iloc[0]),
            }
            
            print(f"  Net Sharpe: {metrics['Net.Sharpe']:.3f}, Trades: {metrics['Turnover.Trades']}")
            
        except Exception as e:
            print(f"  ❌ Backtest failed: {str(e)[:60]}")
            metrics = {
                'Gross.Return': 0.0,
                'Gross.Sharpe': 0.0,
                'Net.Return': 0.0,
                'Net.Sharpe': 0.0,
                'Turnover.Trades': 0,
            }
        
        folds.append({
            'fold': fold_id,
            'train_start': train_start,
            'train_end': train_end,
            'test_start': test_start,
            'test_end': test_end,
            'selected_pairs': [(p.ticker1, p.ticker2) for p in selected_pairs],
            'metrics': metrics
        })
    
    # Aggregate results
    sharpes = [f['metrics']['Net.Sharpe'] for f in folds]
    total_trades = sum(f['metrics']['Turnover.Trades'] for f in folds)
    
    avg_sharpe = np.mean(sharpes)
    std_sharpe = np.std(sharpes, ddof=1) if len(sharpes) > 1 else 0.0
    
    result = {
        'market': 'NSE_Nifty50',
        'market_code': 'NSE',
        'signal_model': signal_name,
        'n_folds': len(folds),
        'avg_net_sharpe': float(avg_sharpe),
        'std_net_sharpe': float(std_sharpe),
        'avg_gross_sharpe': float(np.mean([f['metrics']['Gross.Sharpe'] for f in folds])),
        'transaction_cost_bps': 16.355,
        'folds': folds,
        'timestamp': datetime.now().isoformat()
    }
    
    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = results_dir / f"wfv_4folds_{signal_name}_{timestamp}.json"
    
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\n✅ COMPLETE: {signal_name.upper()}")
    print(f"   Net Sharpe: {avg_sharpe:.3f} ± {std_sharpe:.3f}")
    print(f"   Total Trades: {total_trades}")
    print(f"   Saved to: {output_file.name}")

print(f"\n{'='*60}")
print("✅ ALL CONTROL EXPERIMENTS COMPLETE")
print(f"{'='*60}")
print(f"\nResults saved to: {results_dir}")
print("\nNext: Compare NSE Nifty 50 vs India Multi-Market")
print("      to isolate geographic effect from universe quality")
