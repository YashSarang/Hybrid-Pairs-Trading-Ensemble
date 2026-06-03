"""
run_control_experiment_cached.py

Uses CACHED India data (same tickers as NSE Nifty 50) to run control experiments.
Bypasses yfinance fetch failures.
"""

import sys
from pathlib import Path
import shutil

# Add parent dirs to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

print("="*60)
print("NSE NIFTY 50 CONTROL EXPERIMENT (Cached Data)")
print("="*60)
print()
print("Purpose: Isolate universe quality from geographic effects")
print("Comparing: NSE Nifty 50 vs NSE Nifty 100 vs India Multi-Market")
print()
print("Note: Using cached India data (same 35 tickers)")
print()

# Setup cache directories
base_dir = Path(__file__).parent
india_cache = base_dir / "data" / "india" / "prices_2020-01-01_2025-05-01.parquet"
nse_cache_dir = base_dir / "data" / "nse_nifty50"
nse_cache_file = nse_cache_dir / "prices_2020-01-01_2025-05-01.parquet"

# Copy India data to NSE Nifty 50 cache (already done by SLURM script, skip)
print(f"✓ Using cache: {nse_cache_file}")

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

# Load data (will use cache)
print("Step 1: Loading price data from cache...")
prices = pd.read_parquet(nse_cache_file)
print(f"✓ Data loaded: {prices.shape[0]} days × {prices.shape[1]} tickers")
print()

# RUN EXPERIMENTS
results_dir = Path(__file__).parent / "results" / "nse_nifty50"
results_dir.mkdir(parents=True, exist_ok=True)

for signal_name in ['zscore', 'ou']:
    print(f"\n{'='*60}")
    print(f"RUN: NSE Nifty 50 + Rolling + {signal_name.upper()}")
    print(f"{'='*60}\n")
    
    start_time = datetime.now()
    
    # Create signal
    if signal_name == 'zscore':
        signal = ZScoreThreshold(lookback=126, entry_z=2.0, exit_z=0.5)
    else:  # ou
        signal = OUThreshold(lookback=126, entry_k=1.5, exit_k=0.2)
    
    print(f"Signal: {signal_name}")
    print(f"  Lookback: 126")
    print()
    
    # Setup selectors (ensemble of 8)
    print("Setting up selectors (ensemble of 8)...")
    selectors = {
        "Correlation": CorrelationSelector(),
        "Distance": DistanceSelector(),
        "Cointegration": CointegrationSelector(),
        "Combined": CombinedCriteriaSelector(),
        "ML": MLSelector(),
        "LSTM": LSTMSelector(),
        "Transformer": TransformerSelector(),
        "GNN": GNNSelector()
    }
    
    # Walk-forward validation (4 folds)
    n_splits = 4
    total_days = len(prices)
    fold_size = total_days // (n_splits + 1)
    
    print(f"Walk-forward validation: {n_splits} folds")
    print(f"  Total days: {total_days}")
    print(f"  Fold size: {fold_size}")
    print()
    
    all_trades = []
    fold_results = []
    
    for fold_idx in range(n_splits):
        train_start = 0
        train_end = (fold_idx + 1) * fold_size
        test_start = train_end
        test_end = test_start + fold_size
        
        train_data = prices.iloc[train_start:train_end]
        test_data = prices.iloc[test_start:test_end]
        
        print(f"\nFold {fold_idx + 1}/{n_splits}:")
        print(f"  Train: {train_data.index[0].date()} to {train_data.index[-1].date()} ({len(train_data)} days)")
        print(f"  Test:  {test_data.index[0].date()} to {test_data.index[-1].date()} ({len(test_data)} days)")
        
        # Generate all candidate pairs
        tickers = list(train_data.columns)
        candidates = [Pair(a, b) for i, a in enumerate(tickers) for b in tickers[i+1:]]
        print(f"  Candidate pairs: {len(candidates)}")
        
        # Score pairs with each selector
        print(f"  Running ensemble pair selection...")
        selector_scores = {}
        for sel_name, selector in selectors.items():
            try:
                scores = selector.score_pairs(train_data, candidates)
                selector_scores[sel_name] = scores
                print(f"    ✓ {sel_name}: {len(scores)} pairs scored")
            except Exception as e:
                print(f"    ✗ {sel_name} failed: {e}")
                selector_scores[sel_name] = []
        
        # Ensemble aggregation
        if not selector_scores or all(len(s) == 0 for s in selector_scores.values()):
            print(f"  ❌ All selectors failed")
            fold_results.append({'fold': fold_idx + 1, 'error': 'All selectors failed'})
            continue
        
        try:
            weights = {name: 1.0 for name in selectors.keys()}
            aggregated = ensemble_pair_scores(selector_scores, weights, top_k=30)
            top_pairs = [ps.pair for ps in aggregated]
            print(f"  ✓ Selected {len(top_pairs)} pairs")
        except Exception as e:
            print(f"  ❌ Ensemble aggregation failed: {e}")
            fold_results.append({'fold': fold_idx + 1, 'error': str(e)})
            continue
        
        # Backtest
        print(f"  Backtesting {len(top_pairs)} pairs on test set...")
        
        try:
            bt_config = BacktestConfig(
                capital=100000,
                per_trade_cap=20000,
                costs=IndianCosts(slippage_bps_per_leg=2.0),
                periods_per_year=252,
                min_hold_bars=30
            )
            
            # Create entry models dict
            entry_models = {signal_name: signal}
            entry_weights = {signal_name: 1.0}
            
            result = backtest_pairs(
                prices=test_data,
                selected_pairs=top_pairs,
                entry_models=entry_models,
                entry_weights=entry_weights,
                cfg=bt_config
            )
            
            # Extract metrics
            metrics = {
                'sharpe_ratio': result.metrics.get('Sharpe', 0.0),
                'total_return': result.metrics.get('Return', 0.0),
                'max_drawdown': result.metrics.get('MaxDrawdown', 0.0),
                'n_trades': len(result.trades)
            }
            
            all_trades.extend(result.trades)
            fold_results.append({
                'fold': fold_idx + 1,
                'train_start': str(train_data.index[0].date()),
                'train_end': str(train_data.index[-1].date()),
                'test_start': str(test_data.index[0].date()),
                'test_end': str(test_data.index[-1].date()),
                'n_trades': len(result.trades),
                'sharpe': metrics['sharpe_ratio'],
                'total_return': metrics['total_return'],
                'max_drawdown': metrics['max_drawdown']
            })
            
            print(f"  ✓ Fold complete:")
            print(f"    Trades: {len(result.trades)}")
            print(f"    Sharpe: {metrics['sharpe_ratio']:.3f}")
            print(f"    Return: {metrics['total_return']:.2%}")
            
        except Exception as e:
            print(f"  ❌ Backtest failed: {e}")
            import traceback
            traceback.print_exc()
            fold_results.append({
                'fold': fold_idx + 1,
                'error': str(e)
            })
    
    # Aggregate results across folds
    elapsed = (datetime.now() - start_time).total_seconds()
    
    # Calculate overall metrics
    if fold_results and any('sharpe' in f for f in fold_results):
        valid_folds = [f for f in fold_results if 'sharpe' in f]
        avg_sharpe = np.mean([f['sharpe'] for f in valid_folds])
        total_trades = sum([f['n_trades'] for f in valid_folds])
    else:
        avg_sharpe = 0.0
        total_trades = 0
    
    result = {
        'experiment': f'nse_nifty50_rolling_{signal_name}',
        'timestamp': datetime.now().isoformat(),
        'config': {
            'market': 'NSE_Nifty50',
            'universe_size': len(config['universe']['tickers']),
            'signal': signal_name,
            'window_type': 'rolling',
            'n_folds': n_splits,
            'transaction_cost_bps': 5.0,
            'min_hold_days': 30
        },
        'results': {
            'avg_sharpe_ratio': avg_sharpe,
            'total_trades': total_trades,
            'fold_results': fold_results
        },
        'runtime_seconds': elapsed
    }
    
    # Save results
    output_file = results_dir / f"wfv_4folds_{signal_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"EXPERIMENT COMPLETE: {signal_name.upper()}")
    print(f"{'='*60}")
    print(f"Avg Sharpe Ratio: {avg_sharpe:.3f}")
    print(f"Total Trades: {total_trades}")
    print(f"Runtime: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"Results saved: {output_file}")
    print()

print("\n" + "="*60)
print("ALL EXPERIMENTS COMPLETE")
print("="*60)
print(f"\nResults directory: {results_dir}")
print(f"\nNext steps:")
print(f"1. Download results: scp yash.sarang@kalpana:~/Hybrid-Pairs-Trading-Ensemble/Implementation/experimental-ablation/results/nse_nifty50/*.json ./results/nse_nifty50/")
print(f"2. Analyze and compare to baselines")
print(f"3. Choose scenario (A/B/C) and reframe thesis")
