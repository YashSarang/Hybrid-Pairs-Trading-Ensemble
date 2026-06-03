"""
experiments/walk_forward_rolling.py
====================================
ROLLING-WINDOW Walk-Forward Validation (Comparison to Expanding-Window)
------------------------------------------------------------------------

PURPOSE
-------
Re-run thesis E1-E6 experiments using ROLLING 12-month training windows
(matching the multi-market methodology) instead of EXPANDING windows.

This allows apples-to-apples comparison:
  - Expanding (original thesis): Train 2016-2019 → Test 2020
  - Rolling (this script):        Train 2020    → Test 2021

METHODOLOGY DIFFERENCE
----------------------
Original walk_forward.py:
  - Expanding window: training grows each fold (4, 5, 6 years)
  - More historical data per fold
  - Academic WFV standard

This script (rolling):
  - Fixed 12-month training window
  - Matches real-world deployment (models retrained annually)
  - Matches multi-market experiments for fair comparison

FOLD STRUCTURE (Rolling 12-month)
----------------------------------
  Fold 1:  Train 2019  |  Test 2020  (E1 equivalent)
  Fold 2:  Train 2020  |  Test 2021  (E2 equivalent)
  Fold 3:  Train 2021  |  Test 2022  (E3 equivalent)
  Fold 4:  Train 2022  |  Test 2023  (E4 equivalent)
  Fold 5:  Train 2023  |  Test 2024  (E5 equivalent)
  Fold 6:  Train 2024  |  Test 2025  (E6 equivalent)

SIGNAL PARAMETERS
-----------------
  - ZScoreThreshold: lookback=126, entry_z=2.0, exit_z=0.5
  - OUThreshold:     lookback=126, entry_k=1.5, exit_k=0.2
  
  (Matches multi-market experiments exactly)

OUTPUTS
-------
Saved to: experiments/results/rolling_window_validation_20260529/
  - walk_forward_rolling_<timestamp>.json
  - Metrics: Per-fold Sharpe, trades, cost drag
  - Aggregate: Mean ± std across folds

USAGE
-----
  python experiments/walk_forward_rolling.py
  python experiments/walk_forward_rolling.py --top-k 15
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# Add parent to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.backtest import BacktestConfig, IndianCosts, backtest_pairs
from core.data import DataConfig, YFinanceNSESource
from core.entry import ZScoreThreshold, OUThreshold
from core.selectors import (
    CorrelationSelector,
    DistanceSelector,
    CointegrationSelector,
    CombinedCriteriaSelector,
    MLSelector,
    LSTMSelector,
    TransformerSelector,
    GNNSelector,
)
from experiments.config import NSE_UNIVERSE

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


def build_selectors() -> Dict:
    """Build all 8 selectors (matching thesis ensemble)."""
    return {
        "Correlation": CorrelationSelector(),
        "Distance": DistanceSelector(),
        "Cointegration": CointegrationSelector(),
        "Combined": CombinedCriteriaSelector(),
        "ML": MLSelector(),
        "LSTM": LSTMSelector(),
        "Transformer": TransformerSelector(),
        "GNN": GNNSelector(),
    }


def run_rolling_fold(
    fold_idx: int,
    train_start: str,
    train_end: str,
    test_start: str,
    test_end: str,
    prices: pd.DataFrame,
    selectors: Dict,
    top_k: int,
) -> dict:
    """
    Execute one rolling-window fold.
    
    Train: 12 months (e.g., 2020-01-01 to 2020-12-31)
    Test:  12 months (e.g., 2021-01-01 to 2021-12-31)
    """
    logger.info(f"\n{'='*70}")
    logger.info(f"FOLD {fold_idx}: {test_start} to {test_end}")
    logger.info(f"  Train: {train_start} to {train_end}")
    logger.info(f"{'='*70}")
    
    # Split data
    train_prices = prices.loc[train_start:train_end]
    test_prices = prices.loc[test_start:test_end]
    
    logger.info(f"Train: {len(train_prices)} bars, Test: {len(test_prices)} bars")
    
    # Stage 1: Pair Selection (on training data only)
    sel_start = time.time()
    
    # Generate all candidate pairs
    from itertools import combinations
    from core.selectors_base import Pair
    candidates = [Pair(a, b) for a, b in combinations(prices.columns.tolist(), 2)]
    
    logger.info(f"  Generated {len(candidates)} candidate pairs")
    
    # Run all selectors
    selector_scores = {}
    for sel_name, selector in selectors.items():
        logger.info(f"  Running {sel_name} selector...")
        try:
            selector.fit(train_prices)
            scores = selector.score_pairs(train_prices, candidates)
            selector_scores[sel_name] = scores
            logger.info(f"    ✓ {len(scores)} pairs scored")
        except Exception as e:
            logger.warning(f"    ✗ {sel_name} failed: {e}")
            selector_scores[sel_name] = []
    
    # Aggregate scores via ensemble (equal-weight voting)
    from core.ensemble import ensemble_pair_scores
    
    if not selector_scores or all(len(s) == 0 for s in selector_scores.values()):
        raise RuntimeError("All selectors failed — cannot proceed")
    
    # Equal weights for all selectors
    weights = {name: 1.0 for name in selectors.keys()}
    
    # Get top-k pairs
    aggregated = ensemble_pair_scores(selector_scores, weights, top_k=top_k)
    selected_pairs = [ps.pair for ps in aggregated]
    
    sel_time = time.time() - sel_start
    logger.info(f"  Selected {len(selected_pairs)} pairs in {sel_time:.1f}s")
    logger.info(f"  Top 5: {[f'{p.a}-{p.b}' for p in selected_pairs[:5]]}")
    
    # Stage 2: Backtest (on test data)
    bt_start = time.time()
    
    # Build entry models with fixed lookback=126
    entry_models = {
        "ZScore": ZScoreThreshold(lookback=126, entry_z=2.0, exit_z=0.5),
        "OU": OUThreshold(lookback=126, entry_k=1.5, exit_k=0.2),
    }
    
    entry_weights = {
        "ZScore": 0.5,
        "OU": 0.5,
    }
    
    bt_config = BacktestConfig(
        per_trade_cap=10_000_000,  # 1 Crore INR
        costs=IndianCosts(),
        periods_per_year=252,
        min_hold_bars=30,  # 30-day minimum hold
    )
    
    results = backtest_pairs(
        prices=test_prices,
        selected_pairs=selected_pairs,
        entry_models=entry_models,
        entry_weights=entry_weights,
        cfg=bt_config,
    )
    
    bt_time = time.time() - bt_start
    
    metrics = results.metrics
    
    logger.info(f"\n{'='*40}")
    logger.info(f"FOLD {fold_idx} RESULTS")
    logger.info(f"{'='*40}")
    logger.info(f"Net Sharpe:   {metrics['Net.Sharpe']:.3f}")
    logger.info(f"Gross Sharpe: {metrics['Gross.Sharpe']:.3f}")
    logger.info(f"Max DD:       {metrics['Net.MaxDrawdown']*100:.1f}%")
    logger.info(f"Total Trades: {int(metrics['Turnover.Trades'])}")
    logger.info(f"Cost Drag:    {metrics['Net.Sharpe'] - metrics['Gross.Sharpe']:.3f} Sharpe units")
    logger.info(f"{'='*40}\n")
    
    return {
        "fold": fold_idx,
        "name": f"Fold{fold_idx}_{test_start[:4]}",
        "train_start": train_start,
        "train_end": train_end,
        "test_start": test_start,
        "test_end": test_end,
        "selected_pairs": len(selected_pairs),
        "pairs": [f"{p.a}-{p.b}" for p in selected_pairs],
        "selector_counts": {k: len(v) for k, v in selector_scores.items()},  # Pair count per selector
        "sel_time_s": sel_time,
        "bt_time_s": bt_time,
        "gross_sharpe": metrics["Gross.Sharpe"],
        "net_sharpe": metrics["Net.Sharpe"],
        "gross_ann_ret_pct": metrics["Gross.Return"] * 100,
        "net_ann_ret_pct": metrics["Net.Return"] * 100,
        "gross_maxdd_pct": metrics["Gross.MaxDrawdown"] * 100,
        "net_maxdd_pct": metrics["Net.MaxDrawdown"] * 100,
        "total_trades": int(metrics["Turnover.Trades"]),
        "trades_per_year": int(metrics["Turnover.Trades"]),  # Already annualized
        "cost_drag_sharpe": metrics["Net.Sharpe"] - metrics["Gross.Sharpe"],
        "n_bars_oos": len(test_prices),
    }


def main():
    parser = argparse.ArgumentParser(description="Rolling-window WFV (thesis validation)")
    parser.add_argument("--top-k", type=int, default=10, help="Number of pairs to select per fold")
    args = parser.parse_args()
    
    logger.info("="*70)
    logger.info("ROLLING-WINDOW WALK-FORWARD VALIDATION")
    logger.info("Methodology: Fixed 12-month training windows")
    logger.info("Signal Parameters: lookback=126 (both ZScore and OU)")
    logger.info("="*70)
    
    # Load NSE data (2019-2025 for all 6 folds)
    logger.info("\\n📥 Loading NSE Nifty 100 data...")
    
    # Fetch prices
    data_cfg = DataConfig(
        start=datetime(2019, 1, 1),  # Include 2019 for Fold 1 training
        end=datetime(2025, 12, 31),   # Include 2025 for Fold 6 testing
        freq="1D"
    )
    
    prices = YFinanceNSESource().get_prices(NSE_UNIVERSE, data_cfg)
    
    # Filter for coverage
    coverage = prices.notna().mean()
    prices = prices[coverage[coverage >= 0.80].index]
    
    logger.info(f"✓ Loaded {len(prices.columns)} tickers, {len(prices)} days")
    
    # Build selectors once
    logger.info("\n🔧 Building ensemble selectors...")
    selectors = build_selectors()
    logger.info(f"✓ {len(selectors)} selectors ready")
    
    # Define rolling folds (12-month train, 12-month test)
    # Matches thesis E1-E6 test years exactly
    folds = [
        (1, "2019-01-01", "2019-12-31", "2020-01-01", "2020-12-31"),  # E1: 2020 test
        (2, "2020-01-01", "2020-12-31", "2021-01-01", "2021-12-31"),  # E2: 2021 test
        (3, "2021-01-01", "2021-12-31", "2022-01-01", "2022-12-31"),  # E3: 2022 test
        (4, "2022-01-01", "2022-12-31", "2023-01-01", "2023-12-31"),  # E4: 2023 test
        (5, "2023-01-01", "2023-12-31", "2024-01-01", "2024-12-31"),  # E5: 2024 test
        (6, "2024-01-01", "2024-12-31", "2025-01-01", "2025-12-31"),  # E6: 2025 test
    ]
    
    # Execute folds
    fold_results = []
    for fold_idx, train_start, train_end, test_start, test_end in folds:
        result = run_rolling_fold(
            fold_idx=fold_idx,
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
            prices=prices,
            selectors=selectors,
            top_k=args.top_k,
        )
        fold_results.append(result)
    
    # Aggregate statistics
    net_sharpes = [f["net_sharpe"] for f in fold_results]
    gross_sharpes = [f["gross_sharpe"] for f in fold_results]
    total_trades = sum(f["total_trades"] for f in fold_results)
    
    avg_net_sharpe = np.mean(net_sharpes)
    std_net_sharpe = np.std(net_sharpes, ddof=1) if len(net_sharpes) > 1 else 0
    avg_gross_sharpe = np.mean(gross_sharpes)
    
    logger.info("\n" + "="*70)
    logger.info("AGGREGATE RESULTS (All Folds)")
    logger.info("="*70)
    logger.info(f"Avg Net Sharpe:   {avg_net_sharpe:+.3f} ± {std_net_sharpe:.3f}")
    logger.info(f"Avg Gross Sharpe: {avg_gross_sharpe:+.3f}")
    logger.info(f"Total Trades:     {total_trades}")
    logger.info(f"Avg Trades/Fold:  {total_trades / len(fold_results):.0f}")
    logger.info(f"Positive Folds:   {sum(1 for s in net_sharpes if s > 0)}/{len(net_sharpes)}")
    logger.info("="*70)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(__file__).parent / "results" / "rolling_window_validation_20260529"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / f"walk_forward_rolling_{timestamp}.json"
    
    output_data = {
        "methodology": "rolling_12month",
        "description": "Rolling-window WFV matching multi-market experiments",
        "signal_parameters": {
            "ZScore": {"lookback": 126, "entry_z": 2.0, "exit_z": 0.5},
            "OU": {"lookback": 126, "entry_k": 1.5, "exit_k": 0.2},
        },
        "top_k": args.top_k,
        "n_folds": len(fold_results),
        "avg_net_sharpe": avg_net_sharpe,
        "std_net_sharpe": std_net_sharpe,
        "avg_gross_sharpe": avg_gross_sharpe,
        "total_trades": total_trades,
        "folds": fold_results,
    }
    
    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)
    
    logger.info(f"\n✅ Results saved to: {output_file}")
    logger.info("\nComparison instructions:")
    logger.info("  - Original expanding-window: experiments/results/walk_forward_20260506_104613.json")
    logger.info(f"  - New rolling-window:        {output_file}")
    logger.info("\nNext: Compare methodologies and decide which to use in thesis Chapter 3")


if __name__ == "__main__":
    main()
