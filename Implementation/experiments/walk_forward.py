"""
experiments/walk_forward.py
============================
Experiment E4 — Expanding-Window Walk-Forward Validation
----------------------------------------------------------
Replaces the single in-sample backtest with a rigorous out-of-sample
evaluation across 6 non-overlapping test folds (2018-2024).

WHY THIS IS CRITICAL FOR THE THESIS
-------------------------------------
A single in-sample backtest is academically inadmissible: any result —
good or bad — could be an artefact of the specific parameters being fit
and evaluated on the same data.  Walk-forward validation (WFV) separates
fit from evaluation at every fold, producing out-of-sample (OOS) Sharpe
ratios that reviewers can trust.

FOLD STRUCTURE  (expanding training window)
-------------------------------------------
Each fold trains on ALL available history up to the fold boundary, then
evaluates on the NEXT calendar year.  The training window grows by one
year per fold — this is the academic standard for financial time-series
(expanding > rolling because more data generally improves model quality).

    Fold 1:  Train 2015–2017  |  Test 2018
    Fold 2:  Train 2015–2018  |  Test 2019
    Fold 3:  Train 2015–2019  |  Test 2020
    Fold 4:  Train 2015–2020  |  Test 2021
    Fold 5:  Train 2015–2021  |  Test 2022
    Fold 6:  Train 2015–2022  |  Test 2023-2024

OOS EVALUATION DESIGN (no look-ahead bias)
-------------------------------------------
Each element of the pipeline is fit only on training data:

  Stage 1 (pair selection):
    All selectors are fit on train_prices only.
    Pairs selected in training may or may not remain cointegrated
    in the test window — that is exactly what we are measuring.

  Stage 2 (signal generation):
    A FRESH instance of each signal model is created per pair per fold.
    Each model is fit on (a_train, b_train) before generating signals.
    Signals are then generated on the FULL window (train + test) so that
    rolling lookback windows spanning the train/test boundary use real
    historical data — not future data.  Only the test-window portion of
    the signal is used for PnL computation.

    - ZScoreThreshold / OUThreshold / KalmanHedge:
        Rolling / sequential algorithms.  fit() is a no-op (they adapt
        on-the-fly).  trade_signals(a_full, b_full) is look-ahead-free
        because every computation references only past bars.

    - MLSignal:
        fit(a_train, b_train) trains the XGBoost/GBM classifier on the
        training window only.  trade_signals(a_full, b_full) then uses
        the pre-fitted model for inference — it does NOT retrain.
        This is look-ahead-free by construction.

  Minimum hold period:
    Applied after signal generation, before PnL computation.
    Same DEFAULT_MIN_HOLD (30 days) used in all folds — this is a
    structural parameter, not a per-fold tuned hyperparameter (see
    Decisions.md D5, D11).

OUTPUTS
-------
  Per-fold:   OOS Gross/Net Sharpe, Return, MaxDD, Trades/Year, Cost Drag
  Aggregate:  Mean ± Std of OOS metrics; % folds with Net Sharpe > 0
  Equity:     Cumulative OOS equity curve (stitched test windows)
  Files:      experiments/results/walk_forward_<YYYYMMDD_HHMMSS>.json

Usage
-----
    python experiments/walk_forward.py                   # stat_only, top-k 10
    python experiments/walk_forward.py --mode stat_ml
    python experiments/walk_forward.py --mode full       # slow; for final paper
    python experiments/walk_forward.py --top-k 15
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.backtest import BacktestConfig, IndianCosts, _apply_min_hold
from core.data import DataConfig, YFinanceNSESource
from core.ensemble import ensemble_pair_scores, ensemble_signals
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
    DEFAULT_MIN_HOLD,
    DEFAULT_PER_PAIR,
    DEFAULT_S1_WEIGHTS,
    DEFAULT_S2_WEIGHTS,
    DEFAULT_TOP_K,
    MAIN_END,
    MAIN_START,
    NSE_UNIVERSE,
    OU_ONLY_S2_WEIGHTS,
    PERIODS_PER_YEAR,
    STAT_ML_S1_WEIGHTS,
    STAT_ONLY_S1_WEIGHTS,
    STAT_S2_WEIGHTS,
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

_MODE_WEIGHTS = {
    "full":      DEFAULT_S1_WEIGHTS,
    "stat_ml":   STAT_ML_S1_WEIGHTS,
    "stat_only": STAT_ONLY_S1_WEIGHTS,
}

_S2_PRESETS = {
    "all":     DEFAULT_S2_WEIGHTS,    # all 4 signal models, equal weight
    "no_ml":   STAT_S2_WEIGHTS,       # ZScore + OU + Kalman (exclude MLSignal)
    "ou_only": OU_ONLY_S2_WEIGHTS,    # OUThreshold only — best from E3 ablation
    "rl_only": {"RL": 1.0},           # RLSignal only (E8 Experiment)
}

# ---------------------------------------------------------------------------
# Fold definitions  (expanding window, 1-year test each)
# ---------------------------------------------------------------------------

FOLDS = [
    {"name": "Fold1_2018", "test_start": "2018-01-01", "test_end": "2018-12-31"},
    {"name": "Fold2_2019", "test_start": "2019-01-01", "test_end": "2019-12-31"},
    {"name": "Fold3_2020", "test_start": "2020-01-01", "test_end": "2020-12-31"},
    {"name": "Fold4_2021", "test_start": "2021-01-01", "test_end": "2021-12-31"},
    {"name": "Fold5_2022", "test_start": "2022-01-01", "test_end": "2022-12-31"},
    {"name": "Fold6_2023-2024", "test_start": "2023-01-01", "test_end": "2024-12-31"},
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_selectors(weights: dict) -> dict:
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


def _zscore_local(x: pd.Series, lookback: int) -> pd.Series:
    """Local copy of zscore helper (avoids importing private from backtest)."""
    m = x.rolling(lookback, min_periods=lookback // 3).mean()
    s = x.rolling(lookback, min_periods=lookback // 3).std(ddof=0)
    return (x - m) / (s.replace(0, np.nan) + 1e-9)


def _annualise(total_ret: float, n_bars: int, ppy: int) -> float:
    if n_bars == 0 or not np.isfinite(total_ret) or total_ret <= -1.0:
        return float("nan")
    return round(float(((1 + total_ret) ** (ppy / n_bars) - 1) * 100), 4)


def _metrics_oos(
    pnl_gross: pd.Series,
    pnl_net: pd.Series,
    capital: float,
    ppy: int,
    total_trades: int,
) -> dict:
    """Recompute metrics from a PnL slice (OOS window)."""
    n = len(pnl_gross)
    if n == 0:
        return {}

    def _sharpe(pnl: pd.Series) -> float:
        r = (pnl / max(capital, 1)).replace([np.inf, -np.inf], np.nan).dropna()
        if len(r) < 2 or r.std(ddof=0) == 0:
            return 0.0
        return float((r.mean() / r.std(ddof=0)) * math.sqrt(ppy))

    def _maxdd(pnl: pd.Series) -> float:
        eq = capital + pnl.cumsum()
        mx = eq.cummax().replace(0, np.nan)
        return float(((mx - eq) / mx).max()) if len(eq) else 0.0

    def _total_ret(pnl: pd.Series) -> float:
        eq = capital + pnl.cumsum()
        return float((eq.iloc[-1] - capital) / capital)

    g_ret = _total_ret(pnl_gross)
    n_ret = _total_ret(pnl_net)
    g_ann = _annualise(g_ret, n, ppy)
    n_ann = _annualise(n_ret, n, ppy)

    return {
        "gross_sharpe":      round(_sharpe(pnl_gross), 4),
        "net_sharpe":        round(_sharpe(pnl_net),   4),
        "gross_ann_ret_pct": g_ann,
        "net_ann_ret_pct":   n_ann,
        "gross_maxdd_pct":   round(_maxdd(pnl_gross) * 100, 4),
        "net_maxdd_pct":     round(_maxdd(pnl_net)   * 100, 4),
        "total_trades":      total_trades,
        "trades_per_year":   round(total_trades / (n / ppy), 2) if n > 0 else float("nan"),
        "cost_drag_pp":      round(g_ann - n_ann, 4) if np.isfinite(g_ann) and np.isfinite(n_ann) else float("nan"),
        "n_bars_oos":        n,
    }


# ---------------------------------------------------------------------------
# Stage 1 — select pairs on training data
# ---------------------------------------------------------------------------

_CPU_SELECTORS = {"Correlation", "Distance", "Cointegration", "Combined", "ML"}
_DL_SELECTORS  = {"LSTM", "Transformer", "GNN"}


def select_pairs_train(
    train_prices: pd.DataFrame,
    top_k: int,
    s1_weights: dict,
) -> list[Pair]:
    """Run all selectors on TRAINING prices only and return top-k pairs.

    CPU-based selectors (Correlation, Distance, Cointegration, Combined, ML)
    run in parallel via ThreadPoolExecutor — each uses all available CPU cores
    internally (joblib/XGBoost n_jobs=-1) and releases the GIL for C extensions.

    DL selectors (LSTM, Transformer, GNN) run sequentially to avoid GPU memory
    contention; each model uses the full GPU for its training pass.
    """
    candidates = [Pair(a, b) for a, b in combinations(list(train_prices.columns), 2)]
    selectors  = _build_selectors(s1_weights)
    weights    = s1_weights
    scores_by_model: dict = {}

    def _run_selector(name: str):
        sel = selectors[name]
        sel.fit(train_prices)
        return name, sel.score_pairs(train_prices, candidates)

    # --- CPU selectors: run in parallel ---
    cpu_names = [n for n in selectors if n in _CPU_SELECTORS and s1_weights.get(n, 0) > 0]
    n_workers = min(len(cpu_names), os.cpu_count() or 4)
    if cpu_names:
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futs = {pool.submit(_run_selector, n): n for n in cpu_names}
            for fut in as_completed(futs):
                name = futs[fut]
                try:
                    _, scores = fut.result()
                    scores_by_model[name] = scores
                except Exception as exc:
                    log.warning(f"    [{name}] failed: {exc}")

    # --- DL selectors: sequential on GPU ---
    dl_names = [n for n in selectors if n in _DL_SELECTORS and s1_weights.get(n, 0) > 0]
    for name in dl_names:
        try:
            _, scores = _run_selector(name)
            scores_by_model[name] = scores
        except Exception as exc:
            log.warning(f"    [{name}] failed: {exc}")

    if not scores_by_model:
        raise RuntimeError("All selectors failed in this fold.")

    aggregated = ensemble_pair_scores(scores_by_model, weights, top_k=top_k)
    return [ps.pair for ps in aggregated]


# ---------------------------------------------------------------------------
# Stage 2 — signal generation per pair (fit on train, infer on full)
# ---------------------------------------------------------------------------

def _signals_for_pair(
    a_train: pd.Series,
    b_train: pd.Series,
    a_full: pd.Series,
    b_full: pd.Series,
    s2_weights: dict,
) -> pd.Series:
    """
    Generate ensemble signal for ONE pair with strict train/test separation.

    A FRESH model instance is created for each call so models from a
    previous pair never bleed into this pair's computation.

    Fit step:   uses a_train / b_train only.
    Infer step: uses a_full / b_full (includes test window) so rolling
                lookback windows spanning the boundary use real history.
    """
    # Fresh instances — never reuse across pairs or folds
    models = {
        "ZScore": ZScoreThreshold(),
        "OU":     OUThreshold(),
        "Kalman": KalmanHedge(),
        "ML":     MLSignal(),
        "RL":     __import__("importlib").import_module("core.entry_rl").RLSignal()  # lazy: gymnasium optional
    }

    idx_full = a_full.index.intersection(b_full.index)

    signals_by_model: dict[str, pd.Series] = {}
    for name, model in models.items():
        if s2_weights.get(name, 0) == 0:
            continue
        try:
            model.fit(a_train, b_train)                          # fit on TRAIN only
            sig = model.trade_signals(a_full, b_full)            # infer on FULL
            signals_by_model[name] = sig.reindex(idx_full).fillna(0)
        except Exception as exc:
            log.warning(f"      [{name}] signal failed: {exc}")

    if not signals_by_model:
        return pd.Series(0, index=idx_full)

    return ensemble_signals(signals_by_model, s2_weights)


# ---------------------------------------------------------------------------
# Fold runner
# ---------------------------------------------------------------------------

def run_fold(
    prices: pd.DataFrame,
    fold: dict,
    s1_weights: dict,
    top_k: int,
    s2_weights: dict,
    capital: float,
    per_pair_cap: float,
    costs: IndianCosts,
    min_hold: int,
    ppy: int,
    soft_stop_z: float = 3.0,
    soft_stop_decay: float = 0.5,
    soft_stop_persist: int = 5,
    soft_stop_lookback: int = 60,
) -> dict:
    """
    Execute one WFV fold and return OOS metrics.

    Parameters
    ----------
    prices : full price DataFrame (all years)
    fold   : dict with keys name, test_start, test_end
    """
    name       = fold["name"]
    test_start = fold["test_start"]
    test_end   = fold["test_end"]
    train_end  = str((pd.Timestamp(test_start) - pd.Timedelta(days=1)).date())

    log.info(f"  --- {name} | Train {MAIN_START} -> {train_end} | Test {test_start} -> {test_end} ---")

    train_prices = prices.loc[MAIN_START:train_end]
    full_prices  = prices.loc[MAIN_START:test_end]   # train + this test fold

    if len(train_prices) < 200:
        log.warning(f"  Skipping {name}: training window too short ({len(train_prices)} bars)")
        return {"name": name, "skipped": True}

    # ---- Stage 1: select pairs on TRAINING data ----
    t0 = time.time()
    selected = select_pairs_train(train_prices, top_k, s1_weights)
    display  = [f"{p.a.split('.')[0]}-{p.b.split('.')[0]}" for p in selected]
    log.info(f"    Pairs selected ({len(selected)}): {display[:5]} ...")
    t_sel = round(time.time() - t0, 1)

    # ---- Stage 2: signal + PnL across all selected pairs ----
    full_index   = full_prices.index
    pnl_gross    = pd.Series(0.0, index=full_index)
    cost_series  = pd.Series(0.0, index=full_index)
    turn_series  = pd.Series(0.0, index=full_index)
    notional     = float(min(capital / max(1, len(selected)), per_pair_cap))
    cost_frac    = float(costs.round_trip_cost_fraction())

    t0 = time.time()
    for p in selected:
        if p.a not in full_prices.columns or p.b not in full_prices.columns:
            continue

        a_full = full_prices[p.a].dropna()
        b_full = full_prices[p.b].dropna()
        idx    = a_full.index.intersection(b_full.index)
        if len(idx) < 200:
            continue
        a_full = a_full.reindex(idx).ffill()
        b_full = b_full.reindex(idx).ffill()

        # Training slices for fit
        a_train = a_full.loc[:train_end]
        b_train = b_full.loc[:train_end]
        if len(a_train) < 60:
            continue

        # Signal generation (fit on train, infer on full)
        sig = _signals_for_pair(a_train, b_train, a_full, b_full, s2_weights)
        sig = _apply_min_hold(sig, min_hold).astype(float)

        # Soft stop-loss (applied on full window)
        spread       = a_full - b_full
        z            = _zscore_local(spread, soft_stop_lookback)
        breach       = z.abs() > soft_stop_z
        b_persist    = breach.rolling(soft_stop_persist).sum().fillna(0) >= soft_stop_persist
        scale        = pd.Series(1.0, index=idx)
        scale.loc[breach] = soft_stop_decay
        sig_scaled   = (sig * scale).round().astype(int)
        sig_scaled.loc[b_persist] = 0

        # PnL computation
        r_spread     = a_full.pct_change().fillna(0) - b_full.pct_change().fillna(0)
        sig_prev     = sig_scaled.shift(1).fillna(0).astype(int)
        pair_turn    = (sig_scaled - sig_prev).abs()
        pair_gross   = sig_prev * r_spread * notional
        pair_costs   = pair_turn * (cost_frac / 2.0) * notional

        pnl_gross   = pnl_gross.add(pair_gross.reindex(full_index).fillna(0),  fill_value=0)
        cost_series = cost_series.add(pair_costs.reindex(full_index).fillna(0), fill_value=0)
        turn_series = turn_series.add(pair_turn.reindex(full_index).fillna(0),  fill_value=0)

    t_bt = round(time.time() - t0, 1)

    # ---- Slice to OOS test window ----
    pnl_gross_oos = pnl_gross.loc[test_start:test_end]
    pnl_net_oos   = (pnl_gross - cost_series).loc[test_start:test_end]
    trades_oos    = int(turn_series.loc[test_start:test_end].sum())

    oos_m = _metrics_oos(pnl_gross_oos, pnl_net_oos, capital, ppy, trades_oos)

    log.info(
        f"    OOS => Gross Sharpe={oos_m.get('gross_sharpe', 'N/A'):.3f}  "
        f"Net Sharpe={oos_m.get('net_sharpe', 'N/A'):.3f}  "
        f"Net Ret={oos_m.get('net_ann_ret_pct', 'N/A'):.2f}%  "
        f"Trades/yr={oos_m.get('trades_per_year', 'N/A'):.0f}  "
        f"[sel={t_sel}s bt={t_bt}s]"
    )

    return {
        "name":          name,
        "train_start":   MAIN_START,
        "train_end":     train_end,
        "test_start":    test_start,
        "test_end":      test_end,
        "selected_pairs": display,
        "sel_time_s":    t_sel,
        "bt_time_s":     t_bt,
        "pnl_gross_oos": pnl_gross_oos,   # keep Series for cumulative equity
        "pnl_net_oos":   pnl_net_oos,
        **oos_m,
    }


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_folds(fold_results: list[dict], capital: float) -> dict:
    """Compute mean, std, % positive across folds and stitch cumulative equity."""
    valid = [r for r in fold_results if not r.get("skipped")]
    if not valid:
        return {"error": "No valid folds"}

    def stat(key: str) -> dict:
        vals = [r[key] for r in valid if np.isfinite(r.get(key, float("nan")))]
        if not vals:
            return {"mean": float("nan"), "std": float("nan"), "pct_positive": float("nan")}
        return {
            "mean":         round(float(np.mean(vals)), 4),
            "std":          round(float(np.std(vals)),  4),
            "pct_positive": round(float(np.mean(np.array(vals) > 0)), 4),
        }

    # Stitch cumulative OOS equity (chains fold equity curves)
    all_gross = pd.concat([r["pnl_gross_oos"] for r in valid]).sort_index()
    all_net   = pd.concat([r["pnl_net_oos"]   for r in valid]).sort_index()
    cum_gross = (capital + all_gross.cumsum()).rename("cumulative_gross")
    cum_net   = (capital + all_net.cumsum()).rename("cumulative_net")

    # Full-OOS aggregate metrics (treating all test years as one continuous series)
    total_trades = sum(r.get("total_trades", 0) for r in valid)
    full_pnl_gross = all_gross
    full_pnl_net   = all_net
    n_total = len(full_pnl_gross)
    ppy     = PERIODS_PER_YEAR["1D"]
    full_m  = _metrics_oos(full_pnl_gross, full_pnl_net, capital, ppy, total_trades)

    return {
        "n_folds":                  len(valid),
        "oos_years":                [r["test_start"][:4] for r in valid],
        # Per-metric fold statistics
        "gross_sharpe":             stat("gross_sharpe"),
        "net_sharpe":               stat("net_sharpe"),
        "gross_ann_ret_pct":        stat("gross_ann_ret_pct"),
        "net_ann_ret_pct":          stat("net_ann_ret_pct"),
        "gross_maxdd_pct":          stat("gross_maxdd_pct"),
        "net_maxdd_pct":            stat("net_maxdd_pct"),
        "trades_per_year":          stat("trades_per_year"),
        "cost_drag_pp":             stat("cost_drag_pp"),
        # Full-OOS (all test years stitched) metrics
        "full_oos_metrics":         full_m,
        # Equity series for charting (as dict for JSON serialisation)
        "cumulative_gross":         cum_gross.to_dict(),
        "cumulative_net":           cum_net.to_dict(),
    }


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def print_results(fold_results: list[dict], agg: dict) -> None:
    sep = "=" * 100
    print(f"\n{sep}")
    print("  WALK-FORWARD VALIDATION RESULTS  (out-of-sample, expanding window)")
    print(sep)

    cols = [
        ("Fold",           "name",            "s",   16),
        ("Gross SR",       "gross_sharpe",    ".3f", 10),
        ("Gross Ret%",     "gross_ann_ret_pct",".2f",10),
        ("Net SR",         "net_sharpe",      ".3f", 10),
        ("Net Ret%",       "net_ann_ret_pct", ".2f", 10),
        ("Net MaxDD%",     "net_maxdd_pct",   ".2f", 10),
        ("Trd/Yr",         "trades_per_year", ".0f", 8),
        ("CostDrag",       "cost_drag_pp",    ".2f", 9),
    ]
    header = "".join(f"{lbl:>{w}}" for lbl, _, _, w in cols)
    print(header)
    print("-" * 100)

    for r in fold_results:
        if r.get("skipped"):
            print(f"  {r['name']:15s}  SKIPPED")
            continue
        row = ""
        for lbl, key, fmt, w in cols:
            val = r.get(key, "N/A")
            try:
                if fmt == "s":
                    row += f"{str(val):>{w}}"
                else:
                    row += f"{val:>{w}{fmt}}"
            except (TypeError, ValueError):
                row += f"{'N/A':>{w}}"
        print(row)

    print("-" * 100)
    # Aggregate row
    def _agg_str(key: str, fmt: str, w: int) -> str:
        s = agg.get(key, {})
        m, sd = s.get("mean", float("nan")), s.get("std", float("nan"))
        pct = s.get("pct_positive", float("nan"))
        try:
            return f"  {m:{fmt}} +/-{sd:{fmt}} [{pct:.0%} pos]"
        except (TypeError, ValueError):
            return "  N/A"

    print(f"\n  AGGREGATE ACROSS {agg['n_folds']} FOLDS:")
    print(f"  Gross Sharpe : {_agg_str('gross_sharpe', '.3f', 8)}")
    print(f"  Net   Sharpe : {_agg_str('net_sharpe',   '.3f', 8)}")
    print(f"  Gross Ret %  : {_agg_str('gross_ann_ret_pct', '.2f', 7)}")
    print(f"  Net   Ret %  : {_agg_str('net_ann_ret_pct',   '.2f', 7)}")
    print(f"  Net MaxDD %  : {_agg_str('net_maxdd_pct',     '.2f', 7)}")
    print(f"  Cost Drag pp : {_agg_str('cost_drag_pp',      '.2f', 7)}")

    fm = agg.get("full_oos_metrics", {})
    print(f"\n  FULL OOS (all test years stitched, {agg['n_folds']} yrs):")
    print(f"    Gross Sharpe  = {fm.get('gross_sharpe', 'N/A'):.3f}"
          f"  Net Sharpe = {fm.get('net_sharpe', 'N/A'):.3f}")
    print(f"    Gross Ret %   = {fm.get('gross_ann_ret_pct', 'N/A'):.2f}%"
          f"  Net Ret %  = {fm.get('net_ann_ret_pct', 'N/A'):.2f}%")
    print(f"    Net MaxDD %   = {fm.get('net_maxdd_pct', 'N/A'):.2f}%")
    print(sep + "\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="E4/E7: Walk-forward validation on NSE pairs trading strategy",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mode", choices=["full", "stat_ml", "stat_only"],
                        default="stat_only",
                        help="Predefined S1 weight preset. Ignored if --s1-weights is provided.")
    parser.add_argument("--s1-weights", type=str, default=None,
                        help='JSON dict of custom S1 selector weights, e.g. \'{"LSTM":3,"Correlation":2,"Distance":1,"Cointegration":1,"Combined":0.5,"ML":0,"Transformer":1,"GNN":0}\'. '
                             'Overrides --mode when provided. Zero-weight selectors are skipped.')
    parser.add_argument("--s2", choices=["all", "no_ml", "ou_only", "rl_only"],
                        default="no_ml",
                        help="Stage 2 signal model config. 'no_ml' excludes MLSignal (E3 finding). "
                             "'ou_only' uses OUThreshold only (best E3 ablation result). "
                             "'rl_only' uses PPO-based RLSignal (E8 experiment). "
                             "'all' uses equal-weight all 4 models.")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    args = parser.parse_args()

    # Resolve S1 weights: custom JSON overrides mode preset
    if args.s1_weights:
        try:
            s1_weights = json.loads(args.s1_weights)
        except json.JSONDecodeError as e:
            raise ValueError(f"--s1-weights is not valid JSON: {e}") from e
        # Fill missing keys with 0.0 so ensemble_pair_scores handles them
        all_keys = ["Correlation", "Distance", "Cointegration", "Combined", "ML", "LSTM", "Transformer", "GNN"]
        for k in all_keys:
            s1_weights.setdefault(k, 0.0)
        mode_label = f"custom({args.s1_weights[:60]})"
    else:
        s1_weights = _MODE_WEIGHTS[args.mode]
        mode_label = args.mode

    s2_weights = _S2_PRESETS[args.s2]

    log.info("=" * 64)
    log.info(f"Walk-Forward Validation  |  mode={mode_label}  |  s2={args.s2}  |  top_k={args.top_k}")
    log.info(f"S1 weights: { {k: v for k, v in s1_weights.items() if v > 0} }")
    log.info(f"Folds: {[f['name'] for f in FOLDS]}")
    log.info(f"min_hold_bars={DEFAULT_MIN_HOLD}  capital={DEFAULT_CAPITAL:,.0f}")
    log.info("=" * 64)

    # ---- Fetch all data once ----
    log.info("[Data] Fetching full 10-year daily prices ...")
    cfg    = DataConfig(start=MAIN_START, end=MAIN_END, freq="1D")
    prices = YFinanceNSESource().get_prices(NSE_UNIVERSE, cfg)
    coverage = prices.notna().mean()
    prices   = prices[coverage[coverage >= 0.80].index]
    log.info(f"  {len(prices.columns)} tickers x {len(prices)} bars  "
             f"({prices.index[0].date()} -> {prices.index[-1].date()})")

    costs = IndianCosts()

    # ---- Run each fold ----
    fold_results: list[dict] = []
    total_t0 = time.time()

    for fold in FOLDS:
        result = run_fold(
            prices=prices,
            fold=fold,
            s1_weights=s1_weights,
            top_k=args.top_k,
            s2_weights=s2_weights,
            capital=DEFAULT_CAPITAL,
            per_pair_cap=DEFAULT_PER_PAIR,
            costs=costs,
            min_hold=DEFAULT_MIN_HOLD,
            ppy=PERIODS_PER_YEAR["1D"],
        )
        fold_results.append(result)

    log.info(f"All folds done in {time.time() - total_t0:.0f}s")

    # ---- Aggregate ----
    agg = aggregate_folds(fold_results, DEFAULT_CAPITAL)

    # ---- Display ----
    print_results(fold_results, agg)

    # ---- Save ----
    run_id   = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"walk_forward_{run_id}.json"

    # Strip non-serialisable Series from fold results before saving
    saveable_folds = []
    for r in fold_results:
        clean = {k: v for k, v in r.items()
                 if not isinstance(v, pd.Series)}
        saveable_folds.append(clean)

    saveable_agg = {k: v for k, v in agg.items()
                    if k not in ("cumulative_gross", "cumulative_net")}
    saveable_agg["cumulative_gross"] = {str(k): v for k, v in agg.get("cumulative_gross", {}).items()}
    saveable_agg["cumulative_net"]   = {str(k): v for k, v in agg.get("cumulative_net",   {}).items()}

    payload = {
        "experiment":  "E4_walk_forward_validation",
        "mode":        mode_label,
        "s1_weights":  s1_weights,
        "s2":          args.s2,
        "top_k":       args.top_k,
        "min_hold":    DEFAULT_MIN_HOLD,
        "folds":       saveable_folds,
        "aggregate":   saveable_agg,
    }
    with open(out_path, "w") as fh:
        json.dump(payload, fh, indent=2, default=str)
    log.info(f"Results saved -> {out_path}")


if __name__ == "__main__":
    main()
