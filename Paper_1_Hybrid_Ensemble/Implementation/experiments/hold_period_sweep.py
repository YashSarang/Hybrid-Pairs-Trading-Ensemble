"""
experiments/hold_period_sweep.py
==================================
Experiment E2 — Minimum Hold Period Sweep
------------------------------------------
Finds the optimal minimum holding period that maximises net Sharpe on daily
NSE pairs trading data.

BACKGROUND
----------
Experiment E1 (freq_comparison.py) showed:
  - Daily gross Sharpe = 1.144  (pairs are genuinely mean-reverting, Hurst 0.19)
  - Daily net  Sharpe = -2.294  (strategy loses money after costs)
  - Cost drag = 16.29 pp/year   (673 trades/year * ~60bps round-trip)

The pairs are good; the signal layer is the problem.  The ensemble of four
signal models (ZScore, OU, Kalman, MLSignal) generates position changes too
frequently: roughly one change every 3–4 trading days per pair on average.
At 60 bps round-trip, each unnecessary transition costs ~0.6% of notional.

HYPOTHESIS
----------
Enforcing a minimum holding period (min_hold_bars) will:
  1. Cut trades/year proportionally.
  2. Leave gross alpha largely intact (mean-reversion operates on multi-day
     horizons; a few days of forced holding rarely strands a position).
  3. Bring net Sharpe positive beyond some threshold hold.

We sweep min_hold_bars in [0, 1, 2, 3, 5, 7, 10] trading days.

DESIGN
------
To isolate the hold-period effect from selector randomness:
  - Data fetched ONCE.
  - Selectors run ONCE -> fixed top-K pairs.
  - Backtest re-run N times, one per hold value, with identical everything
    except min_hold_bars.

This is the correct controlled-experiment design.  Sweeping hold periods
with different selector seeds would confound the two effects.

WHAT THE RESULT WILL BE USED FOR
---------------------------------
The optimal hold period from this sweep becomes the DEFAULT min_hold_bars
in BacktestConfig for ALL subsequent experiments (E3 ablation, E4 walk-
forward, E5 benchmarks).  It is a methodological parameter, not a
tuned hyperparameter, so it is set before any test-set data is touched.

Usage
-----
    python experiments/hold_period_sweep.py
    python experiments/hold_period_sweep.py --mode full --top-k 15

Outputs
-------
    experiments/results/hold_period_sweep_<YYYYMMDD_HHMMSS>.json
    Console: formatted table + recommendation
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import date, datetime, timedelta
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.backtest import BacktestConfig, IndianCosts, backtest_pairs
from core.data import DataConfig, YFinanceNSESource
from core.ensemble import ensemble_pair_scores
from core.entry import KalmanHedge, MLSignal, OUThreshold, ZScoreThreshold
from core.selectors import (
    CombinedCriteriaSelector,
    CointegrationSelector,
    CorrelationSelector,
    DistanceSelector,
    GNNSelector,
    LSTMSelector,
    MLSelector,
    Pair,
    TransformerSelector,
)

from experiments.config import (
    DEFAULT_CAPITAL,
    DEFAULT_MAX_PAIRS,
    DEFAULT_PER_PAIR,
    DEFAULT_S1_WEIGHTS,
    DEFAULT_S2_WEIGHTS,
    DEFAULT_TOP_K,
    MAIN_END,
    MAIN_START,
    NSE_UNIVERSE,
    PERIODS_PER_YEAR,
    STAT_ML_S1_WEIGHTS,
    STAT_ONLY_S1_WEIGHTS,
)

# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Hold periods to sweep (trading days).
# 0 = baseline (no constraint), matches E1.
# Upper bound 10 = 2 trading weeks; beyond this we expect diminishing returns
# because mean-reversion half-lives are typically 10-30 days for these pairs.
HOLD_VALUES = [0, 1, 2, 3, 5, 7, 10]

_MODE_WEIGHTS = {
    "full":      DEFAULT_S1_WEIGHTS,
    "stat_ml":   STAT_ML_S1_WEIGHTS,
    "stat_only": STAT_ONLY_S1_WEIGHTS,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_selectors(mode: str) -> dict:
    weights = _MODE_WEIGHTS[mode]
    sel: dict = {}
    if weights.get("Correlation", 0):   sel["Correlation"]   = CorrelationSelector()
    if weights.get("Distance", 0):      sel["Distance"]      = DistanceSelector()
    if weights.get("Cointegration", 0): sel["Cointegration"] = CointegrationSelector()
    if weights.get("Combined", 0):      sel["Combined"]      = CombinedCriteriaSelector()
    if weights.get("ML", 0):            sel["ML"]            = MLSelector()
    if weights.get("LSTM", 0):          sel["LSTM"]          = LSTMSelector()
    if weights.get("Transformer", 0):   sel["Transformer"]   = TransformerSelector()
    if weights.get("GNN", 0):           sel["GNN"]           = GNNSelector()
    return sel


def _build_entry_models() -> dict:
    return {
        "ZScore": ZScoreThreshold(),
        "OU":     OUThreshold(),
        "Kalman": KalmanHedge(),
        "ML":     MLSignal(),
    }


def _annualise(total_return: float, n_bars: int, ppy: int) -> float:
    if n_bars == 0 or not np.isfinite(total_return) or total_return <= -1.0:
        return float("nan")
    return round(float(((1.0 + total_return) ** (ppy / n_bars) - 1.0) * 100), 4)


def _tpy(total_trades: int, n_bars: int, ppy: int) -> float:
    if n_bars == 0:
        return float("nan")
    return round(float(total_trades) / (n_bars / ppy), 2)


# ---------------------------------------------------------------------------
# Stage 1 — select pairs (run once)
# ---------------------------------------------------------------------------

def select_pairs(
    prices: pd.DataFrame,
    mode: str,
    top_k: int,
) -> list[Pair]:
    log.info("[Stage 1] Pair selection ...")
    candidates = [Pair(a, b) for a, b in combinations(list(prices.columns), 2)]
    log.info(f"  {len(candidates)} candidates from {len(prices.columns)} tickers")

    selectors = _build_selectors(mode)
    weights   = _MODE_WEIGHTS[mode]
    scores_by_model: dict = {}

    for name, sel in selectors.items():
        if weights.get(name, 0) == 0:
            continue
        t0 = time.time()
        log.info(f"  [{name}] fitting + scoring ...")
        try:
            sel.fit(prices)
            scores_by_model[name] = sel.score_pairs(prices, candidates)
            log.info(f"  [{name}] done in {time.time() - t0:.1f}s")
        except Exception as exc:
            log.warning(f"  [{name}] FAILED: {exc}")

    if not scores_by_model:
        raise RuntimeError("All selectors failed.")

    aggregated = ensemble_pair_scores(scores_by_model, weights, top_k=top_k)
    selected   = [ps.pair for ps in aggregated]
    display    = [f"{p.a.split('.')[0]}-{p.b.split('.')[0]}" for p in selected]
    log.info(f"  Selected {len(selected)} pairs: {display}")
    return selected


# ---------------------------------------------------------------------------
# Stage 2 — backtest at one hold period
# ---------------------------------------------------------------------------

def run_one_hold(
    prices: pd.DataFrame,
    selected: list[Pair],
    min_hold: int,
    ppy: int,
) -> dict:
    entry_models = _build_entry_models()
    bt_cfg = BacktestConfig(
        capital=DEFAULT_CAPITAL,
        max_concurrent_pairs=min(len(selected), DEFAULT_MAX_PAIRS),
        per_trade_cap=DEFAULT_PER_PAIR,
        costs=IndianCosts(),
        periods_per_year=ppy,
        min_hold_bars=min_hold,
    )
    result = backtest_pairs(prices, selected, entry_models, DEFAULT_S2_WEIGHTS, bt_cfg)

    m      = result.metrics
    n_bars = len(result.pnl_gross)
    total_trades = int(m["Turnover.Trades"])

    gross_ann = _annualise(m["Gross.Return"], n_bars, ppy)
    net_ann   = _annualise(m["Net.Return"],   n_bars, ppy)
    cost_drag = round(gross_ann - net_ann, 4) if np.isfinite(gross_ann) and np.isfinite(net_ann) else float("nan")

    return {
        "min_hold_bars":    min_hold,
        "gross_sharpe":     round(m["Gross.Sharpe"],      4),
        "gross_ann_ret_pct": gross_ann,
        "gross_max_dd_pct": round(m["Gross.MaxDrawdown"] * 100, 4),
        "net_sharpe":       round(m["Net.Sharpe"],        4),
        "net_ann_ret_pct":  net_ann,
        "net_max_dd_pct":   round(m["Net.MaxDrawdown"] * 100, 4),
        "total_trades":     total_trades,
        "trades_per_year":  _tpy(total_trades, n_bars, ppy),
        "cost_drag_pp":     cost_drag,
    }


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

_COLS = [
    ("Hold (days)",        "min_hold_bars",      "d"),
    ("Gross Sharpe",       "gross_sharpe",       ".3f"),
    ("Gross Ret %",        "gross_ann_ret_pct",  ".2f"),
    ("Net Sharpe",         "net_sharpe",         ".3f"),
    ("Net Ret %",          "net_ann_ret_pct",    ".2f"),
    ("Net MaxDD %",        "net_max_dd_pct",     ".2f"),
    ("Trades/Yr",          "trades_per_year",    ".1f"),
    ("Cost Drag pp",       "cost_drag_pp",       ".2f"),
]


def print_sweep(rows: list[dict], optimal_hold: int) -> None:
    col_w = 14
    header = "".join(f"{label:>{col_w}}" for label, _, _ in _COLS)
    sep    = "=" * (col_w * len(_COLS))

    print(f"\n{sep}")
    print("  HOLD PERIOD SWEEP RESULTS  (daily data, stat_only selectors)")
    print(sep)
    print(header)
    print("-" * (col_w * len(_COLS)))

    for row in rows:
        line = ""
        for label, key, fmt in _COLS:
            val = row.get(key, float("nan"))
            try:
                line += f"{val:>{col_w}{fmt}}"
            except (TypeError, ValueError):
                line += f"{'N/A':>{col_w}}"
        marker = "  <-- optimal" if row["min_hold_bars"] == optimal_hold else ""
        print(line + marker)

    print(sep)
    print(f"\n  Recommendation: min_hold_bars = {optimal_hold} trading days")
    print(f"  Add this as the default to BacktestConfig in all subsequent experiments.\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="E2: Sweep min_hold_bars to find optimal hold period",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mode", choices=["full", "stat_ml", "stat_only"],
                        default="stat_only")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument(
        "--hold-values", nargs="+", type=int, default=None,
        metavar="N",
        help="Override the hold period sweep values (e.g. --hold-values 0 5 10 15 20 30)",
    )
    args = parser.parse_args()

    global HOLD_VALUES
    if args.hold_values is not None:
        HOLD_VALUES = sorted(set(args.hold_values))

    log.info("=" * 64)
    log.info(f"E2 Hold Period Sweep  |  mode={args.mode}  |  top_k={args.top_k}")
    log.info(f"Hold values: {HOLD_VALUES}")
    log.info("=" * 64)

    # ---- 1. Fetch data (once) ----
    log.info("[Data] Fetching daily prices ...")
    cfg    = DataConfig(start=MAIN_START, end=MAIN_END, freq="1D")
    prices = YFinanceNSESource().get_prices(NSE_UNIVERSE, cfg)
    # Drop tickers with >20% missing bars
    coverage = prices.notna().mean()
    prices   = prices[coverage[coverage >= 0.80].index]
    n_bars   = len(prices)
    log.info(f"  {len(prices.columns)} tickers x {n_bars} bars  "
             f"({prices.index[0].date()} -> {prices.index[-1].date()})")

    # ---- 2. Select pairs (once) ----
    selected = select_pairs(prices, args.mode, args.top_k)

    # ---- 3. Sweep hold periods ----
    log.info(f"\n[Stage 2] Sweeping {len(HOLD_VALUES)} hold periods ...")
    ppy  = PERIODS_PER_YEAR["1D"]
    rows: list[dict] = []

    for hold in HOLD_VALUES:
        log.info(f"  min_hold_bars = {hold} ...")
        row = run_one_hold(prices, selected, hold, ppy)
        rows.append(row)
        log.info(
            f"    Net Sharpe={row['net_sharpe']:.3f}  "
            f"Trades/yr={row['trades_per_year']:.0f}  "
            f"Cost drag={row['cost_drag_pp']:.2f}pp"
        )

    # ---- 4. Find optimal (peak Net Sharpe) ----
    # Filter to rows where Net Sharpe is finite before picking the max
    valid = [r for r in rows if np.isfinite(r["net_sharpe"])]
    if valid:
        best = max(valid, key=lambda r: r["net_sharpe"])
        optimal_hold = int(best["min_hold_bars"])
    else:
        optimal_hold = HOLD_VALUES[-1]   # fallback: longest hold
        log.warning("No finite Net Sharpe found — defaulting to longest hold.")

    # ---- 5. Display & save ----
    print_sweep(rows, optimal_hold)

    run_id   = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"hold_period_sweep_{run_id}.json"
    payload  = {
        "experiment":    "E2_hold_period_sweep",
        "mode":          args.mode,
        "top_k":         args.top_k,
        "start":         MAIN_START,
        "end":           MAIN_END,
        "n_tickers":     len(prices.columns),
        "n_bars":        n_bars,
        "selected_pairs": [f"{p.a.split('.')[0]}-{p.b.split('.')[0]}" for p in selected],
        "hold_values":   HOLD_VALUES,
        "optimal_hold":  optimal_hold,
        "rows":          rows,
    }
    with open(out_path, "w") as fh:
        json.dump(payload, fh, indent=2, default=str)
    log.info(f"Results saved -> {out_path}")


if __name__ == "__main__":
    main()
