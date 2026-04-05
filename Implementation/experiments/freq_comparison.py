"""
experiments/freq_comparison.py
===============================
Frequency Comparison Experiment
---------------------------------
Runs the full two-stage pairs trading pipeline at daily (1D) and hourly (1H)
granularity on the SAME universe, date window, weights, and cost assumptions.
Computes a unified table of performance and spread-quality diagnostics to
empirically justify the choice of daily data over higher frequencies.

Thesis Section: "3.x — Data Frequency Selection"
Claim being tested: As sampling frequency increases (daily -> hourly), strategy
performance degrades due to (1) higher turnover-driven cost drag, (2) increased
microstructure noise in the spread signal (Hurst closer to 0.5), and (3) more
frequent spurious signal reversals.

Usage
-----
    # Full run (all 8 selectors, slow — ~20 min on CPU)
    python experiments/freq_comparison.py

    # Statistical + XGBoost only (fast — ~3 min on CPU)
    python experiments/freq_comparison.py --mode stat_ml

    # Statistical selectors only (fastest — ~1 min on CPU)
    python experiments/freq_comparison.py --mode stat_only

    # Change number of selected pairs
    python experiments/freq_comparison.py --top-k 15

Outputs
-------
    experiments/results/freq_comparison_<YYYYMMDD_HHMMSS>.json  — full numbers
    Console: formatted side-by-side comparison table
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

# Add repository root to path so this script runs from any working directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.backtest import BacktestConfig, BacktestResult, IndianCosts, backtest_pairs
from core.data import DataConfig, YFinanceNSESource
from core.ensemble import ensemble_pair_scores, ensemble_signals, normalize_weights
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
from core.selectors_base import _hurst_rs

from experiments.config import (
    DEFAULT_CAPITAL,
    DEFAULT_MAX_PAIRS,
    DEFAULT_MIN_HOLD,
    DEFAULT_PER_PAIR,
    DEFAULT_S1_WEIGHTS,
    DEFAULT_S2_WEIGHTS,
    DEFAULT_TOP_K,
    FREQ_COMPARISON_LOOKBACK_DAYS,
    NSE_UNIVERSE,
    PERIODS_PER_YEAR,
    STAT_ML_S1_WEIGHTS,
    STAT_ONLY_S1_WEIGHTS,
)

# ---------------------------------------------------------------------------
# Date window — computed at runtime so 1H data always stays within Yahoo's
# 730-day hard limit regardless of when the experiment is executed.
# Both frequencies use the same window for apples-to-apples comparison.
# ---------------------------------------------------------------------------
_TODAY = date.today()
FREQ_COMPARISON_END   = (_TODAY - timedelta(days=1)).strftime("%Y-%m-%d")
FREQ_COMPARISON_START = (_TODAY - timedelta(days=FREQ_COMPARISON_LOOKBACK_DAYS)).strftime("%Y-%m-%d")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Selector & model factories
# ---------------------------------------------------------------------------

_MODE_WEIGHTS = {
    "full":     DEFAULT_S1_WEIGHTS,
    "stat_ml":  STAT_ML_S1_WEIGHTS,
    "stat_only": STAT_ONLY_S1_WEIGHTS,
}


def _build_selectors(mode: str) -> dict:
    """Instantiate Stage-1 selectors according to the chosen speed mode."""
    weights = _MODE_WEIGHTS[mode]
    sel: dict = {}
    if weights.get("Correlation", 0):
        sel["Correlation"] = CorrelationSelector()
    if weights.get("Distance", 0):
        sel["Distance"] = DistanceSelector()
    if weights.get("Cointegration", 0):
        sel["Cointegration"] = CointegrationSelector()
    if weights.get("Combined", 0):
        sel["Combined"] = CombinedCriteriaSelector()
    if weights.get("ML", 0):
        sel["ML"] = MLSelector()
    if weights.get("LSTM", 0):
        sel["LSTM"] = LSTMSelector()
    if weights.get("Transformer", 0):
        sel["Transformer"] = TransformerSelector()
    if weights.get("GNN", 0):
        sel["GNN"] = GNNSelector()
    return sel


def _build_entry_models() -> dict:
    """Instantiate all four Stage-2 signal models."""
    return {
        "ZScore": ZScoreThreshold(),
        "OU":     OUThreshold(),
        "Kalman": KalmanHedge(),
        "ML":     MLSignal(),
    }


# ---------------------------------------------------------------------------
# Spread-quality diagnostics
# ---------------------------------------------------------------------------

def _hurst_stats(prices: pd.DataFrame, pairs: list[Pair]) -> dict[str, float]:
    """
    Compute the Hurst exponent of the log-price spread for each selected pair.

    H < 0.5  -> mean-reverting  (desirable for pairs trading)
    H ≈ 0.5  -> random walk     (no exploitable signal)
    H > 0.5  -> trending

    At higher sampling frequencies, microstructure noise pushes H toward 0.5.
    """
    h_vals: list[float] = []
    for p in pairs:
        if p.a not in prices.columns or p.b not in prices.columns:
            continue
        try:
            log_a = np.log(prices[p.a].dropna())
            log_b = np.log(prices[p.b].dropna())
            idx   = log_a.index.intersection(log_b.index)
            if len(idx) < 100:
                continue
            spread = (log_a - log_b).loc[idx]
            h = _hurst_rs(spread)
            if np.isfinite(h):
                h_vals.append(h)
        except Exception:
            pass
    return {
        "hurst_median": round(float(np.median(h_vals)), 4) if h_vals else float("nan"),
        "hurst_mean":   round(float(np.mean(h_vals)),   4) if h_vals else float("nan"),
        "hurst_pct_below_half": (
            round(float(np.mean(np.array(h_vals) < 0.5)), 4) if h_vals else float("nan")
        ),
    }


def _signal_reversal_rate(trades: pd.DataFrame) -> float:
    """
    Fraction of signal transitions (per pair) that are direct reversals
    (+1 -> -1 or -1 -> +1) rather than entries or exits.

    High reversal rate = whipsaw-prone strategy = microstructure noise trading.
    At higher frequencies this will be substantially larger.
    """
    if trades.empty or "signal" not in trades.columns:
        return float("nan")

    rates: list[float] = []
    for _, grp in trades.groupby("pair"):
        grp = grp.sort_index()
        sig = grp["signal"]
        non_zero = sig[sig != 0]
        if len(non_zero) < 2:
            continue
        # A reversal = consecutive non-zero signals with opposite sign
        reversals = (non_zero * non_zero.shift(1) < 0).sum()
        rates.append(float(reversals) / len(non_zero))

    return round(float(np.mean(rates)), 4) if rates else float("nan")


def _annualise_return(total_return: float, n_bars: int, ppy: int) -> float:
    """Convert a total-period return to an annualised percentage.

    Returns NaN if total_return ≤ -1 (strategy exceeded capital loss, the base
    of the exponent would be negative, producing a complex number).
    """
    if n_bars == 0 or not np.isfinite(total_return) or total_return <= -1.0:
        return float("nan")
    ann = (1.0 + total_return) ** (ppy / n_bars) - 1.0
    return round(float(ann * 100), 4)   # in percent


def _trades_per_year(total_trades: int, n_bars: int, ppy: int) -> float:
    """Normalise raw trade count to an annualised rate."""
    if n_bars == 0:
        return float("nan")
    return round(float(total_trades) / (n_bars / ppy), 2)


# ---------------------------------------------------------------------------
# Core experiment runner
# ---------------------------------------------------------------------------

def run_experiment(
    freq: str,
    universe: list[str],
    start: str,
    end: str,
    s1_weights: dict[str, float],
    s2_weights: dict[str, float],
    top_k: int,
    mode: str,
) -> dict:
    """
    Execute the full pipeline at a given data frequency and return a flat
    metrics dict suitable for JSON serialisation and tabular display.
    """
    log.info("=" * 64)
    log.info(f"  FREQUENCY: {freq}   |   universe={len(universe)} stocks   |   mode={mode}")
    log.info("=" * 64)

    wall_t0 = time.time()

    # ------------------------------------------------------------------
    # 1. Data Ingestion
    # ------------------------------------------------------------------
    log.info("[1/5] Fetching price data …")
    cfg = DataConfig(start=start, end=end, freq=freq)
    src = YFinanceNSESource()
    prices = src.get_prices(universe, cfg)

    if prices.empty:
        raise RuntimeError(
            f"No price data returned for freq={freq}. "
            "For 1H data yfinance enforces a 730-day hard limit — "
            "check that FREQ_COMPARISON_LOOKBACK_DAYS ≤ 728 in config.py."
        )

    # Drop tickers with >20% missing bars (unreliable at high frequency)
    max_missing = 0.20
    coverage = prices.notna().mean()
    valid_tickers = coverage[coverage >= (1 - max_missing)].index.tolist()
    prices = prices[valid_tickers]

    n_tickers = len(valid_tickers)
    n_bars    = len(prices)
    log.info(
        f"    {n_tickers} tickers × {n_bars} bars  "
        f"({prices.index[0].date()} -> {prices.index[-1].date()})"
    )
    if n_tickers < 4:
        raise RuntimeError(f"Too few tickers ({n_tickers}) after coverage filter for freq={freq}.")

    # ------------------------------------------------------------------
    # 2. Pair Candidate Generation
    # ------------------------------------------------------------------
    log.info("[2/5] Generating pair candidates …")
    candidates = [Pair(a, b) for a, b in combinations(valid_tickers, 2)]
    log.info(f"    {len(candidates)} candidates from {n_tickers} tickers")

    # ------------------------------------------------------------------
    # 3. Stage 1 — Pair Selection
    # ------------------------------------------------------------------
    log.info("[3/5] Stage 1 — pair selection …")
    selectors = _build_selectors(mode)
    active_weights = {k: v for k, v in s1_weights.items() if k in selectors and v > 0}

    scores_by_model: dict = {}
    selector_timing: dict[str, float] = {}

    for name, sel in selectors.items():
        if active_weights.get(name, 0) == 0:
            continue
        t0 = time.time()
        log.info(f"    [{name}] fitting + scoring …")
        try:
            sel.fit(prices)
            scores_by_model[name] = sel.score_pairs(prices, candidates)
            selector_timing[name] = round(time.time() - t0, 1)
            log.info(f"    [{name}] done in {selector_timing[name]}s")
        except Exception as exc:
            log.warning(f"    [{name}] FAILED — {exc}")

    if not scores_by_model:
        raise RuntimeError("All selectors failed — cannot select pairs.")

    aggregated   = ensemble_pair_scores(scores_by_model, active_weights, top_k=top_k)
    selected     = [ps.pair for ps in aggregated]
    top5_display = [f"{p.a.split('.')[0]}/{p.b.split('.')[0]}" for p in selected[:5]]
    log.info(f"    Top-{top_k} pairs selected. Top 5: {top5_display} …")

    # ------------------------------------------------------------------
    # 4. Stage 2 — Backtesting
    # ------------------------------------------------------------------
    log.info("[4/5] Stage 2 — backtest …")
    ppy = PERIODS_PER_YEAR[freq]
    bt_cfg = BacktestConfig(
        capital=DEFAULT_CAPITAL,
        max_concurrent_pairs=min(top_k, DEFAULT_MAX_PAIRS),
        per_trade_cap=DEFAULT_PER_PAIR,
        costs=IndianCosts(),
        periods_per_year=ppy,
        min_hold_bars=DEFAULT_MIN_HOLD,
    )
    entry_models = _build_entry_models()
    result: BacktestResult = backtest_pairs(
        prices, selected, entry_models, s2_weights, bt_cfg
    )

    # ------------------------------------------------------------------
    # 5. Diagnostics
    # ------------------------------------------------------------------
    log.info("[5/5] Computing spread diagnostics …")
    hurst = _hurst_stats(prices, selected)
    rev_rate = _signal_reversal_rate(result.trades)

    m = result.metrics
    total_return_gross = m["Gross.Return"]
    total_return_net   = m["Net.Return"]
    total_trades       = int(m["Turnover.Trades"])

    ann_ret_gross = _annualise_return(total_return_gross, n_bars, ppy)
    ann_ret_net   = _annualise_return(total_return_net,   n_bars, ppy)
    cost_drag_pp  = round(ann_ret_gross - ann_ret_net, 4)  # percentage points
    tpy           = _trades_per_year(total_trades, n_bars, ppy)

    elapsed = round(time.time() - wall_t0, 1)
    log.info(f"    Experiment complete in {elapsed}s")

    return {
        # Experiment metadata
        "freq":              freq,
        "mode":              mode,
        "start":             start,
        "end":               end,
        "n_tickers":         n_tickers,
        "n_bars":            n_bars,
        "n_candidates":      len(candidates),
        "n_selected":        len(selected),
        "selected_pairs":    [f"{p.a.split('.')[0]}-{p.b.split('.')[0]}" for p in selected],
        "elapsed_s":         elapsed,
        "selector_timing_s": selector_timing,

        # Performance — Gross (pre-cost)
        "gross_sharpe":      round(m["Gross.Sharpe"],      4),
        "gross_ann_ret_pct": ann_ret_gross,
        "gross_volatility":  round(m["Gross.Volatility"],  4),
        "gross_max_dd_pct":  round(m["Gross.MaxDrawdown"] * 100, 4),

        # Performance — Net (post-cost)
        "net_sharpe":        round(m["Net.Sharpe"],        4),
        "net_ann_ret_pct":   ann_ret_net,
        "net_volatility":    round(m["Net.Volatility"],    4),
        "net_max_dd_pct":    round(m["Net.MaxDrawdown"] * 100, 4),

        # Cost analysis
        "total_trades":      total_trades,
        "trades_per_year":   tpy,
        "cost_drag_pp":      cost_drag_pp,   # gross ann ret − net ann ret (pp)

        # Spread quality diagnostics
        "hurst_median":            hurst["hurst_median"],
        "hurst_mean":              hurst["hurst_mean"],
        "hurst_pct_below_half":    hurst["hurst_pct_below_half"],
        "signal_reversal_rate":    rev_rate,
    }


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

_TABLE_ROWS: list[tuple[str, str, str, str]] = [
    # (section_label, display_name, key, format)
    ("Performance (Gross)",  "Sharpe Ratio",           "gross_sharpe",      ".3f"),
    ("Performance (Gross)",  "Ann. Return (%)",         "gross_ann_ret_pct", ".2f"),
    ("Performance (Gross)",  "Volatility (ann.)",       "gross_volatility",  ".3f"),
    ("Performance (Gross)",  "Max Drawdown (%)",        "gross_max_dd_pct",  ".2f"),
    ("Performance (Net)",    "Sharpe Ratio",            "net_sharpe",        ".3f"),
    ("Performance (Net)",    "Ann. Return (%)",         "net_ann_ret_pct",   ".2f"),
    ("Performance (Net)",    "Volatility (ann.)",       "net_volatility",    ".3f"),
    ("Performance (Net)",    "Max Drawdown (%)",        "net_max_dd_pct",    ".2f"),
    ("Cost Analysis",        "Trades / Year",           "trades_per_year",   ".1f"),
    ("Cost Analysis",        "Cost Drag (ann. pp)",     "cost_drag_pp",      ".2f"),
    ("Spread Quality",       "Hurst (median)",          "hurst_median",      ".4f"),
    ("Spread Quality",       "Hurst < 0.5 (%pairs)",   "hurst_pct_below_half", ".1%"),
    ("Spread Quality",       "Signal Reversal Rate",    "signal_reversal_rate", ".3f"),
]


def print_comparison(results: dict[str, dict]) -> None:
    freqs   = list(results.keys())
    col_w   = 30
    val_w   = 14
    width   = col_w + val_w * len(freqs)
    sep     = "=" * width

    print(f"\n{sep}")
    print("  FREQUENCY COMPARISON RESULTS")
    print(f"  Period : {results[freqs[0]]['start']}  ->  {results[freqs[0]]['end']}")
    print(f"  Mode   : {results[freqs[0]]['mode']}")
    print(sep)
    print(f"{'Metric':<{col_w}}" + "".join(f"{f:>{val_w}}" for f in freqs))
    print("-" * width)

    current_section = None
    for section, label, key, fmt in _TABLE_ROWS:
        if section != current_section:
            print(f"\n  [{section}]")
            current_section = section
        row = f"  {label:<{col_w - 2}}"
        for freq in freqs:
            val = results[freq].get(key, float("nan"))
            try:
                if fmt.endswith("%"):
                    row += f"{val:>{val_w}{fmt}}"
                else:
                    row += f"{val:>{val_w}{fmt}}"
            except (TypeError, ValueError):
                row += f"{'N/A':>{val_w}}"
        print(row)

    print(f"\n{sep}")

    # Additional context rows
    print(f"\n  {'':30}" + "".join(f"{f:>{val_w}}" for f in freqs))
    print(f"  {'Tickers in universe':<30}" +
          "".join(f"{results[f]['n_tickers']:>{val_w}}" for f in freqs))
    print(f"  {'Pair candidates':<30}" +
          "".join(f"{results[f]['n_candidates']:>{val_w}}" for f in freqs))
    print(f"  {'Pairs selected':<30}" +
          "".join(f"{results[f]['n_selected']:>{val_w}}" for f in freqs))
    print(f"  {'Bars in backtest':<30}" +
          "".join(f"{results[f]['n_bars']:>{val_w}}" for f in freqs))
    print(f"  {'Wall time (s)':<30}" +
          "".join(f"{results[f]['elapsed_s']:>{val_w}.1f}" for f in freqs))
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Frequency comparison: daily vs hourly on NSE pairs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode", choices=["full", "stat_ml", "stat_only"], default="stat_ml",
        help=(
            "Selector set to use. "
            "'full' = all 8 selectors (slow); "
            "'stat_ml' = statistical + XGBoost (recommended); "
            "'stat_only' = statistical only (fastest)"
        ),
    )
    parser.add_argument(
        "--top-k", type=int, default=DEFAULT_TOP_K,
        help="Number of pairs to select per frequency run",
    )
    parser.add_argument(
        "--freqs", nargs="+", default=["1D", "1H"],
        choices=["1D", "1H"],
        help="Frequencies to compare (run in this order)",
    )
    args = parser.parse_args()

    s1_weights = _MODE_WEIGHTS[args.mode]
    run_ts     = datetime.now().strftime("%Y-%m-%d %H:%M")

    log.info(f"Frequency Comparison Experiment — {run_ts}")
    log.info(
        f"Universe : {len(NSE_UNIVERSE)} stocks | "
        f"Period   : {FREQ_COMPARISON_START} -> {FREQ_COMPARISON_END} | "
        f"Selectors: {args.mode} | "
        f"Top-K    : {args.top_k}"
    )

    results: dict[str, dict] = {}
    for freq in args.freqs:
        results[freq] = run_experiment(
            freq=freq,
            universe=NSE_UNIVERSE,
            start=FREQ_COMPARISON_START,
            end=FREQ_COMPARISON_END,
            s1_weights=s1_weights,
            s2_weights=DEFAULT_S2_WEIGHTS,
            top_k=args.top_k,
            mode=args.mode,
        )

    print_comparison(results)

    # Save to JSON for thesis data tables
    run_id   = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"freq_comparison_{run_id}.json"
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=2, default=str)
    log.info(f"Results saved -> {out_path}")


if __name__ == "__main__":
    main()
