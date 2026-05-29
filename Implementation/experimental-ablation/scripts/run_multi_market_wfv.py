#!/usr/bin/env python3
"""
run_multi_market_wfv.py

Adapt experiments/walk_forward.py for multi-market experimental ablation.
Runs 6-fold WFV for a given market with the full 8-selector ensemble.

Usage:
    python run_multi_market_wfv.py --market us --n_folds 6
    python run_multi_market_wfv.py --market brazil --selectors lstm correlation --n_folds 6
"""

import argparse
import sys
import json
from pathlib import Path
from datetime import datetime
import yaml
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.data import YFinanceNSESource
from core.selectors import (
    CorrelationSelector, DistanceSelector, CointegrationSelector,
    CombinedCriteriaSelector, MLSelector, LSTMSelector,
    TransformerSelector, GNNSelector, Pair
)
from core.entry import ZScoreThreshold, OUThreshold, KalmanHedge, MLSignal
from core.backtest import backtest_pairs, BacktestConfig, IndianCosts
from core.ensemble import ensemble_pair_scores, ensemble_signals

def load_config(market: str) -> Dict:
    """Load market YAML config."""
    config_path = Path(__file__).parent.parent / "configs" / f"{market}.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_cached_prices(market: str, config: Dict) -> pd.DataFrame:
    """Load cached price data from Parquet."""
    cache_dir = Path(__file__).parent.parent.parent / config['data']['cache_dir']
    start = config['data']['start_date']
    end = config['data']['end_date']
    cache_file = cache_dir / f"prices_{start}_{end}.parquet"
    
    if not cache_file.exists():
        raise FileNotFoundError(
            f"Cache not found: {cache_file}\n"
            f"Run: python fetch_market_data.py --market {market}"
        )
    
    return pd.read_parquet(cache_file)

def build_cost_model(config: Dict):
    """Build market-specific cost model."""
    costs_cfg = config['costs']
    market_code = config['market']['code']
    
    if market_code == 'NSE':
        return IndianCosts(
            brokerage_bps=costs_cfg['brokerage_bps'],
            exchange_txn_bps=costs_cfg.get('exchange_bps', 0.322),
            sebi_bps=costs_cfg['sebi_bps'],
            stt_bps_sell=costs_cfg.get('stt_bps', 10.0),
            gst_rate=costs_cfg['gst_rate'],
            stamp_bps_buy=costs_cfg.get('stamp_bps', 1.5),
            intraday=True,
            slippage_bps_per_leg=costs_cfg.get('slippage_bps', 2.0)
        )
    else:
        # Generic cost model for non-India markets
        # Sum all bps fields
        total_bps = sum(
            v for k, v in costs_cfg.items()
            if k.endswith('_bps') and isinstance(v, (int, float))
        )
        return IndianCosts(
            brokerage_bps=total_bps / 4,
            exchange_txn_bps=total_bps / 4,
            sebi_bps=0,
            stt_bps_sell=0,
            gst_rate=0,
            stamp_bps_buy=0,
            intraday=True,
            slippage_bps_per_leg=costs_cfg.get('slippage_bps', 2.0)
        )

def get_selector(name: str):
    """Instantiate selector by name."""
    selector_map = {
        'correlation': CorrelationSelector,
        'distance': DistanceSelector,
        'cointegration': CointegrationSelector,
        'combined': CombinedCriteriaSelector,
        'ml': MLSelector,
        'lstm': LSTMSelector,
        'transformer': TransformerSelector,
        'gnn': GNNSelector
    }
    return selector_map[name]()

def get_signal_model(name: str, **kwargs):
    """Instantiate signal model by name."""
    signal_map = {
        'zscore': ZScoreThreshold,
        'ou': OUThreshold,
        'kalman': KalmanHedge,
        'ml': MLSignal
    }
    return signal_map[name](**kwargs)

def run_fold(
    fold_idx: int,
    train_start: str, train_end: str,
    test_start: str, test_end: str,
    prices: pd.DataFrame,
    config: Dict,
    selector_names: List[str],
    signal_model: str = 'zscore'
) -> Dict:
    """Run a single WFV fold."""
    
    print(f"\n{'='*60}")
    print(f"FOLD {fold_idx}")
    print(f"Train: {train_start} to {train_end}")
    print(f"Test:  {test_start} to {test_end}")
    print(f"{'='*60}\n")
    
    # Split data
    train_prices = prices.loc[train_start:train_end]
    test_prices = prices.loc[test_start:test_end]
    
    print(f"Train shape: {train_prices.shape}")
    print(f"Test shape:  {test_prices.shape}\n")
    
    # Generate candidate pairs (combinatorial)
    tickers = list(prices.columns)
    from itertools import combinations
    candidate_pairs = [Pair(a, b) for a, b in combinations(tickers, 2)]
    
    print(f"Candidate pairs: {len(candidate_pairs)}\n")
    
    # Train selectors
    selector_weights = config['selectors']['weights']
    selector_scores = {}
    
    for sel_name in selector_names:
        if sel_name not in selector_weights:
            print(f"⚠️  Skipping {sel_name} (not in config weights)")
            continue
        
        print(f"Training {sel_name}...", end=" ", flush=True)
        
        try:
            selector = get_selector(sel_name)
            selector.fit(train_prices)
            scores = selector.score_pairs(train_prices, candidate_pairs)
            selector_scores[sel_name] = scores
            print(f"✅ {len([s for s in scores if s.score > 0])} pairs scored")
        except Exception as e:
            print(f"❌ Error: {str(e)[:60]}")
            selector_scores[sel_name] = []
    
    # Ensemble pair scores
    print(f"\nEnsembling {len(selector_scores)} selectors...")
    ensemble_scores = ensemble_pair_scores(
        selector_scores,
        {k: selector_weights[k] for k in selector_scores.keys()},
        top_k=config['backtest']['max_concurrent_pairs']
    )
    
    # Extract top-K pairs (ensemble_pair_scores returns List[PairScore])
    selected_pairs = [ps.pair for ps in ensemble_scores]
    
    print(f"Selected pairs: {len(selected_pairs)} (target {config['backtest']['max_concurrent_pairs']})\n")
    
    if not selected_pairs:
        print("❌ No pairs selected, skipping fold\n")
        return {
            'fold': fold_idx,
            'train_start': train_start,
            'train_end': train_end,
            'test_start': test_start,
            'test_end': test_end,
            'selected_pairs': 0,
            'net_sharpe': 0.0,
            'gross_sharpe': 0.0,
            'error': 'No pairs selected'
        }
    
    # Backtest on test period
    print(f"\nBacktesting {len(selected_pairs)} pairs on test period...")
    
    bt_config = BacktestConfig(
        capital=config['backtest']['initial_capital'],
        max_concurrent_pairs=config['backtest']['max_concurrent_pairs'],
        per_trade_cap=config['backtest']['per_trade_notional'],
        costs=build_cost_model(config),
        periods_per_year=252,  # Daily data
        min_hold_bars=config['backtest']['hold_period_days']
    )
    
    # Build entry models based on signal_model choice
    if signal_model == 'zscore':
        entry_models = {
            "ZScore": ZScoreThreshold(lookback=126, entry_z=2.0, exit_z=0.5)
        }
    elif signal_model == 'ou':
        entry_models = {
            "OU": OUThreshold(lookback=126, entry_k=1.5, exit_k=0.2)
        }
    elif signal_model == 'kalman':
        entry_models = {
            "Kalman": KalmanHedge()
        }
    elif signal_model == 'ml':
        entry_models = {
            "ML": MLSignal()
        }
    else:
        raise ValueError(f"Unknown signal model: {signal_model}")
    
    entry_weights = {list(entry_models.keys())[0]: 1.0}
    
    results = backtest_pairs(
        prices=test_prices,
        selected_pairs=selected_pairs,
        entry_models=entry_models,
        entry_weights=entry_weights,
        cfg=bt_config
    )
    
    # Extract metrics
    metrics = results.metrics
    
    print(f"\n{'='*40}")
    print(f"FOLD {fold_idx} RESULTS")
    print(f"{'='*40}")
    print(f"Net Sharpe:   {metrics['Net.Sharpe']:.3f}")
    print(f"Gross Sharpe: {metrics['Gross.Sharpe']:.3f}")
    print(f"Max DD:       {metrics['Net.MaxDrawdown']*100:.1f}%")
    print(f"Total Trades: {int(metrics['Turnover.Trades'])}")
    print(f"{'='*40}\n")
    
    return {
        'fold': fold_idx,
        'train_start': train_start,
        'train_end': train_end,
        'test_start': test_start,
        'test_end': test_end,
        'selected_pairs': len(selected_pairs),
        'pairs': [f"{p.a}_{p.b}" for p in selected_pairs],
        'selector_scores': {
            sel: {f"{ps.pair.a}_{ps.pair.b}": float(ps.score) for ps in scores}
            for sel, scores in selector_scores.items()
        },
        'metrics': {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                   for k, v in metrics.items()}
    }

def run_walk_forward(market: str, selector_names: List[str], n_folds: int, signal_model: str = 'zscore'):
    """Run full walk-forward validation for a market."""
    
    config = load_config(market)
    prices = load_cached_prices(market, config)
    
    print(f"\n{'='*60}")
    print(f"Multi-Market WFV: {config['market']['name'].upper()}")
    print(f"{'='*60}")
    print(f"Universe: {prices.shape[1]} tickers")
    print(f"Period: {prices.index[0].date()} to {prices.index[-1].date()}")
    print(f"Selectors: {', '.join(selector_names)}")
    print(f"Folds: {n_folds}")
    print(f"{'='*60}\n")
    
    # WFV fold definitions (from config)
    wfv_cfg = config['walk_forward']
    # Walk-forward folds (1-year train, 1-year test, rolling)
    # Matches thesis structure: each fold tests on a full calendar year
    # Note: 2024 test truncated to available data (2024-01-01 to 2025-04-30)
    folds = [
        (1, "2020-01-01", "2020-12-31", "2021-01-01", "2021-12-31"),
        (2, "2021-01-01", "2021-12-31", "2022-01-01", "2022-12-31"),
        (3, "2022-01-01", "2022-12-31", "2023-01-01", "2023-12-31"),
        (4, "2023-01-01", "2023-12-31", "2024-01-01", "2025-04-30"),  # 16 months (limited by data)
    ][:n_folds]
    
    fold_results = []
    
    for fold_idx, train_start, train_end, test_start, test_end in folds:
        result = run_fold(
            fold_idx, train_start, train_end, test_start, test_end,
            prices, config, selector_names, signal_model
        )
        fold_results.append(result)
    
    # Aggregate results
    net_sharpes = [r['metrics']['Net.Sharpe'] for r in fold_results if 'metrics' in r]
    gross_sharpes = [r['metrics']['Gross.Sharpe'] for r in fold_results if 'metrics' in r]
    
    summary = {
        'market': config['market']['name'],
        'market_code': config['market']['code'],
        'signal_model': signal_model,
        'n_folds': len(fold_results),
        'selectors': selector_names,
        'avg_net_sharpe': float(np.mean(net_sharpes)) if net_sharpes else 0.0,
        'std_net_sharpe': float(np.std(net_sharpes)) if net_sharpes else 0.0,
        'avg_gross_sharpe': float(np.mean(gross_sharpes)) if gross_sharpes else 0.0,
        'transaction_cost_bps': sum(
            v for k, v in config['costs'].items()
            if k.endswith('_bps') and isinstance(v, (int, float))
        ),
        'folds': fold_results,
        'timestamp': datetime.now().isoformat()
    }
    
    # Save results
    results_dir = Path(__file__).parent.parent / "results" / market
    results_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = results_dir / f"wfv_{n_folds}folds_{signal_model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"SUMMARY: {config['market']['name'].upper()}")
    print(f"{'='*60}")
    print(f"Avg Net Sharpe:   {summary['avg_net_sharpe']:.3f} ± {summary['std_net_sharpe']:.3f}")
    print(f"Avg Gross Sharpe: {summary['avg_gross_sharpe']:.3f}")
    print(f"Cost Drag:        {summary['avg_gross_sharpe'] - summary['avg_net_sharpe']:.3f}")
    print(f"Transaction Cost: {summary['transaction_cost_bps']:.1f} bps")
    print(f"\n💾 Saved: {output_file}")
    print(f"{'='*60}\n")

def main():
    parser = argparse.ArgumentParser(description="Multi-market WFV runner")
    parser.add_argument(
        '--market',
        required=True,
        choices=['india', 'us', 'brazil', 'uk'],
        help='Market to run'
    )
    parser.add_argument(
        '--selectors',
        nargs='+',
        default=['correlation', 'distance', 'cointegration', 'combined', 'ml', 'lstm', 'transformer', 'gnn'],
        help='Selectors to use (default: all 8)'
    )
    parser.add_argument(
        '--n_folds',
        type=int,
        default=4,
        help='Number of folds (default: 4)'
    )
    parser.add_argument(
        '--signal_model',
        choices=['zscore', 'ou', 'kalman', 'ml'],
        default='zscore',
        help='Signal model to use (default: zscore)'
    )
    
    args = parser.parse_args()
    
    run_walk_forward(args.market, args.selectors, args.n_folds, args.signal_model)

if __name__ == "__main__":
    main()
