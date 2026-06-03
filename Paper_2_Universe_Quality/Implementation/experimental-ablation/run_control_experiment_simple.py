"""
run_control_experiment_simple.py

Statistical selectors only (no ML) to avoid TensorFlow hangs.
"""

import sys
from pathlib import Path

# Add parent dirs to path
sys.path.insert(0, str(Path(__file__).parent.parent))

print("="*60)
print("NSE NIFTY 50 CONTROL EXPERIMENT (Statistical Only)")
print("="*60)
print()

# Import only statistical selectors
import yaml
import pandas as pd
import numpy as np
from datetime import datetime
import json

from core.selectors_statistical import (
    CorrelationSelector, DistanceSelector,
    CointegrationSelector, CombinedCriteriaSelector, Pair
)
from core.selectors_base import PairScore
from core.entry import ZScoreThreshold, OUThreshold
from core.backtest import backtest_pairs, BacktestConfig, IndianCosts
from core.ensemble import ensemble_pair_scores

print("✓ Imports successful (statistical selectors only)")
print()

# Load config
config_path = Path(__file__).parent / "configs" / "nse_nifty50.yaml"
with open(config_path) as f:
    config = yaml.safe_load(f)

print(f"✓ Config loaded: {config['market']['name']}")
print()

# Load data
cache_file = Path(__file__).parent / "data" / "nse_nifty50" / "prices_2020-01-01_2025-05-01.parquet"
prices = pd.read_parquet(cache_file)
print(f"✓ Data loaded: {prices.shape[0]} days × {prices.shape[1]} tickers")
print()

# Results directory
results_dir = Path(__file__).parent / "results" / "nse_nifty50"
results_dir.mkdir(parents=True, exist_ok=True)

# RUN EXPERIMENTS
for signal_name in ['zscore', 'ou']:
    print(f"\n{'='*60}")
    print(f"RUN: NSE Nifty 50 + Rolling + {signal_name.upper()}")
    print(f"{'='*60}\n")
    
    start_time = datetime.now()
    
    # Create signal
    if signal_name == 'zscore':
        signal = ZScoreThreshold(lookback=126, entry_z=2.0, exit_z=0.5)
    else:
        signal = OUThreshold(lookback=126, entry_k=1.5, exit_k=0.2)
    
    # Setup statistical selectors only (4 instead of 8)
    selectors = {
        "Correlation": CorrelationSelector(),
        "Distance": DistanceSelector(),
        "Cointegration": CointegrationSelector(),
        "Combined": CombinedCriteriaSelector()
    }
    
    # Walk-forward validation
    n_splits = 4
    fold_size = len(prices) // (n_splits + 1)
    
    print(f"Walk-forward validation: {n_splits} folds")
    print(f"  Fold size: {fold_size} days")
    print()
    
    fold_results = []
    
    for fold_idx in range(n_splits):
        train_end = (fold_idx + 1) * fold_size
        test_start = train_end
        test_end = test_start + fold_size
        
        train_data = prices.iloc[:train_end]
        test_data = prices.iloc[test_start:test_end]
        
        print(f"\nFold {fold_idx + 1}/{n_splits}:")
        print(f"  Train: {len(train_data)} days, Test: {len(test_data)} days")
        
        # Generate candidate pairs
        tickers = list(train_data.columns)
        candidates = [Pair(a, b) for i, a in enumerate(tickers) for b in tickers[i+1:]]
        print(f"  Candidates: {len(candidates)} pairs")
        
        # Score with each selector
        selector_scores = {}
        for sel_name, selector in selectors.items():
            try:
                scores = selector.score_pairs(train_data, candidates)
                selector_scores[sel_name] = scores
                print(f"    ✓ {sel_name}: {len(scores)} scored")
            except Exception as e:
                print(f"    ✗ {sel_name}: {e}")
                selector_scores[sel_name] = []
        
        if not selector_scores or all(len(s) == 0 for s in selector_scores.values()):
            print(f"  ❌ All selectors failed")
            fold_results.append({'fold': fold_idx + 1, 'error': 'All selectors failed'})
            continue
        
        # Ensemble
        try:
            weights = {name: 1.0 for name in selectors.keys()}
            aggregated = ensemble_pair_scores(selector_scores, weights, top_k=30)
            top_pairs = [ps.pair for ps in aggregated]
            print(f"  ✓ Selected {len(top_pairs)} pairs")
        except Exception as e:
            print(f"  ❌ Ensemble failed: {e}")
            fold_results.append({'fold': fold_idx + 1, 'error': str(e)})
            continue
        
        # Backtest
        try:
            bt_config = BacktestConfig(
                capital=100000,
                per_trade_cap=20000,
                costs=IndianCosts(slippage_bps_per_leg=2.0),
                periods_per_year=252,
                min_hold_bars=30
            )
            
            entry_models = {signal_name: signal}
            entry_weights = {signal_name: 1.0}
            
            result = backtest_pairs(
                prices=test_data,
                selected_pairs=top_pairs,
                entry_models=entry_models,
                entry_weights=entry_weights,
                cfg=bt_config
            )
            
            metrics = {
                'sharpe': result.metrics.get('Net.Sharpe', 0.0),
                'return': result.metrics.get('Net.Return', 0.0),
                'n_trades': int(result.metrics.get('Turnover.Trades', 0))
            }
            
            fold_results.append({
                'fold': fold_idx + 1,
                'sharpe': metrics['sharpe'],
                'return': metrics['return'],
                'n_trades': metrics['n_trades']
            })
            
            print(f"  ✓ Sharpe: {metrics['sharpe']:.3f}, Trades: {metrics['n_trades']}")
            
        except Exception as e:
            print(f"  ❌ Backtest failed: {e}")
            fold_results.append({'fold': fold_idx + 1, 'error': str(e)})
    
    # Aggregate
    elapsed = (datetime.now() - start_time).total_seconds()
    valid_folds = [f for f in fold_results if 'sharpe' in f]
    avg_sharpe = np.mean([f['sharpe'] for f in valid_folds]) if valid_folds else 0.0
    total_trades = sum([f['n_trades'] for f in valid_folds]) if valid_folds else 0
    
    result = {
        'experiment': f'nse_nifty50_rolling_{signal_name}_statistical_only',
        'timestamp': datetime.now().isoformat(),
        'config': {
            'market': 'NSE_Nifty50',
            'signal': signal_name,
            'selectors': list(selectors.keys()),
            'n_folds': n_splits
        },
        'results': {
            'avg_sharpe': avg_sharpe,
            'total_trades': total_trades,
            'folds': fold_results
        },
        'runtime_seconds': elapsed
    }
    
    output_file = results_dir / f"wfv_4folds_{signal_name}_statistical_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"EXPERIMENT COMPLETE: {signal_name.upper()}")
    print(f"{'='*60}")
    print(f"Avg Sharpe: {avg_sharpe:.3f}")
    print(f"Total Trades: {total_trades}")
    print(f"Runtime: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"Saved: {output_file.name}")
    print()

print("\n" + "="*60)
print("ALL EXPERIMENTS COMPLETE")
print("="*60)
