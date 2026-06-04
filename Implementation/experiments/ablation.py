"""
experiments/ablation.py
========================
Experiment E3 — Performance Attribution / Ablation Study
----------------------------------------------------------
Proves that the ensemble outperforms any individual component by running
each Stage 1 selector and each Stage 2 signal model in isolation, then
comparing against the full ensemble.

WHY THIS IS CRITICAL FOR THE THESIS
--------------------------------------
The central claim of this project is: "an ensemble of diverse pair-selection
and signal-generation algorithms produces better out-of-sample performance
than any single method alone."  Without this ablation, that claim is just an
assertion.  This experiment provides the empirical proof.

The ablation uses the SAME walk-forward fold structure as E4 (expanding window,
6 test years 2020-2025) so comparisons are apples-to-apples.

STAGE 1 ABLATION — Pair Selection
-----------------------------------
For each selector configuration (one selector active at a time, weight=1, all
others=0), run all 6 WFV folds.  Stage 2 always uses the full signal ensemble.

Configurations tested (stat_only mode):
  - Correlation_only     (Gatev et al. 2006 distance-style rolling correlation)
  - Distance_only        (Normalized SSD; Gatev et al.)
  - Cointegration_only   (Engle-Granger ADF test)
  - Combined_only        (Cointegration + Hurst exponent + half-life filter)
  - S1_Ensemble          (equal weights across all active selectors)

Stat_ml and full mode add MLSelector, LSTMSelector, TransformerSelector, GNNSelector
as additional individual configurations.

STAGE 2 ABLATION — Signal Generation
---------------------------------------
Pairs are selected ONCE per fold using the S1 ensemble (same pairs across all
S2 configs — this isolates the Stage 2 effect).  Then each signal model is run
individually.

Configurations tested:
  - ZScore_only    (classical z-score threshold entry/exit)
  - OU_only        (Ornstein-Uhlenbeck process)
  - Kalman_only    (Kalman filter dynamic hedge ratio)
  - ML_only        (XGBoost/GBM triclass classifier on spread features)
  - S2_Ensemble    (equal weights across all 4 models)

KEY CLAIM TO VERIFY
---------------------
    OOS Net Sharpe (Ensemble) > OOS Net Sharpe (best individual model)

If this does not hold, the complexity of the ensemble is not empirically
justified and the thesis claim must be revised.

OUTPUTS
--------
  Stage 1 table:  OOS metrics per selector config
  Stage 2 table:  OOS metrics per signal model config
  JSON file:      experiments/results/ablation_<YYYYMMDD_HHMMSS>.json

Usage
-----
    python experiments/ablation.py                   # stat_only selectors
    python experiments/ablation.py --mode stat_ml
    python experiments/ablation.py --mode full       # adds LSTM/Transformer/GNN
    python experiments/ablation.py --stage 1         # Stage 1 only
    python experiments/ablation.py --stage 2         # Stage 2 only
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.backtest import IndianCosts, _apply_min_hold
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
    DEFAULT_MIN_HOLD,
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

# ---------------------------------------------------------------------------
# Fold definitions  (identical to E4 walk_forward.py)
# ---------------------------------------------------------------------------

FOLDS = [
    {"name": "2020", "test_start": "2020-01-01", "test_end": "2020-12-31"},
    {"name": "2021", "test_start": "2021-01-01", "test_end": "2021-12-31"},
    {"name": "2022", "test_start": "2022-01-01", "test_end": "2022-12-31"},
    {"name": "2023", "test_start": "2023-01-01", "test_end": "2023-12-31"},
    {"name": "2024", "test_start": "2024-01-01", "test_end": "2024-12-31"},
    {"name": "2025", "test_start": "2025-01-01", "test_end": "2025-12-31"},
]

# ---------------------------------------------------------------------------
# Ablation configurations
# ---------------------------------------------------------------------------

def _s1_ablation_configs(mode: str) -> dict[str, dict[str, float]]:
    """
    Build Stage 1 ablation weight dicts: one selector active at a time + ensemble.
    Selectors included depend on the mode (stat_only / stat_ml / full).
    """
    base = {
        "stat_only": STAT_ONLY_S1_WEIGHTS,
        "stat_ml":   STAT_ML_S1_WEIGHTS,
        "full":      DEFAULT_S1_WEIGHTS,
    }[mode]

    active = [name for name, w in base.items() if w > 0]

    configs: dict[str, dict[str, float]] = {}
    for sel in active:
        single = {k: (1.0 if k == sel else 0.0) for k in base}
        configs[f"{sel}_only"] = single

    # Full ensemble of active selectors
    ensemble_cfg = {k: (1.0 if k in active else 0.0) for k in base}
    configs["S1_Ensemble"] = ensemble_cfg
    return configs


_S2_ABLATION_CONFIGS: dict[str, dict[str, float]] = {
    "ZScore_only": {"ZScore": 1.0, "OU": 0.0, "Kalman": 0.0, "ML": 0.0},
    "OU_only":     {"ZScore": 0.0, "OU": 1.0, "Kalman": 0.0, "ML": 0.0},
    "Kalman_only": {"ZScore": 0.0, "OU": 0.0, "Kalman": 1.0, "ML": 0.0},
    "ML_only":     {"ZScore": 0.0, "OU": 0.0, "Kalman": 0.0, "ML": 1.0},
    "S2_Ensemble": {"ZScore": 1.0, "OU": 1.0, "Kalman": 1.0, "ML": 1.0},
}

# ---------------------------------------------------------------------------
# Shared helpers  (mirrors walk_forward.py)
# ---------------------------------------------------------------------------

def _build_selectors_from_weights(weights: dict[str, float]) -> dict:
    sel_map = {
        "Correlation":   CorrelationSelector,
        "Distance":      DistanceSelector,
        "Cointegration": CointegrationSelector,
        "Combined":      CombinedCriteriaSelector,
        "ML":            MLSelector,
        "LSTM":          LSTMSelector,
        "Transformer":   TransformerSelector,
        "GNN":           GNNSelector,
    }
    return {name: cls() for name, cls in sel_map.items() if weights.get(name, 0) > 0}


def _zscore_local(x: pd.Series, lookback: int) -> pd.Series:
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
# Core fold logic
# ---------------------------------------------------------------------------

def _select_pairs(train_prices: pd.DataFrame, s1_weights: dict, top_k: int) -> list[Pair]:
    """Select top-k pairs using given Stage 1 weights. Fit on training data only."""
    candidates  = [Pair(a, b) for a, b in combinations(list(train_prices.columns), 2)]
    selectors   = _build_selectors_from_weights(s1_weights)
    scores_map  = {}

    for name, sel in selectors.items():
        try:
            sel.fit(train_prices)
            scores_map[name] = sel.score_pairs(train_prices, candidates)
        except Exception as exc:
            log.warning(f"      [{name}] selector failed: {exc}")

    if not scores_map:
        return []

    aggregated = ensemble_pair_scores(scores_map, s1_weights, top_k=top_k)
    return [ps.pair for ps in aggregated]


def _signals_for_pair(
    a_train: pd.Series,
    b_train: pd.Series,
    a_full: pd.Series,
    b_full: pd.Series,
    s2_weights: dict,
) -> pd.Series:
    """Fit on train, infer on full window. Fresh model instances per call."""
    models = {
        "ZScore": ZScoreThreshold(),
        "OU":     OUThreshold(),
        "Kalman": KalmanHedge(),
        "ML":     MLSignal(),
    }
    idx_full = a_full.index.intersection(b_full.index)
    signals_by_model: dict[str, pd.Series] = {}

    for name, model in models.items():
        if s2_weights.get(name, 0) == 0:
            continue
        try:
            model.fit(a_train, b_train)
            sig = model.trade_signals(a_full, b_full)
            signals_by_model[name] = sig.reindex(idx_full).fillna(0)
        except Exception as exc:
            log.warning(f"        [{name}] signal failed: {exc}")

    if not signals_by_model:
        return pd.Series(0, index=idx_full)
    return ensemble_signals(signals_by_model, s2_weights)


def _run_pairs_pnl(
    full_prices: pd.DataFrame,
    selected: list[Pair],
    train_end: str,
    s2_weights: dict,
    capital: float,
    per_pair_cap: float,
    costs: IndianCosts,
    min_hold: int,
    soft_stop_z: float = 3.0,
    soft_stop_decay: float = 0.5,
    soft_stop_persist: int = 5,
    soft_stop_lookback: int = 60,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Compute portfolio PnL (gross), cost series, and turnover series
    for a list of pairs over the full price window.
    Returns (pnl_gross, cost_series, turn_series) all on full_prices.index.
    """
    full_index  = full_prices.index
    pnl_gross   = pd.Series(0.0, index=full_index)
    cost_series = pd.Series(0.0, index=full_index)
    turn_series = pd.Series(0.0, index=full_index)
    notional    = float(min(capital / max(1, len(selected)), per_pair_cap))
    cost_frac   = float(costs.round_trip_cost_fraction())

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

        a_train = a_full.loc[:train_end]
        b_train = b_full.loc[:train_end]
        if len(a_train) < 60:
            continue

        sig          = _signals_for_pair(a_train, b_train, a_full, b_full, s2_weights)
        sig          = _apply_min_hold(sig, min_hold).astype(float)

        spread       = a_full - b_full
        z            = _zscore_local(spread, soft_stop_lookback)
        breach       = z.abs() > soft_stop_z
        b_persist    = breach.rolling(soft_stop_persist).sum().fillna(0) >= soft_stop_persist
        scale        = pd.Series(1.0, index=idx)
        scale.loc[breach] = soft_stop_decay
        sig_scaled   = (sig * scale).round().astype(int)
        sig_scaled.loc[b_persist] = 0

        r_spread     = a_full.pct_change().fillna(0) - b_full.pct_change().fillna(0)
        sig_prev     = sig_scaled.shift(1).fillna(0).astype(int)
        pair_turn    = (sig_scaled != sig_prev).astype(int)
        pair_gross   = sig_prev * r_spread * notional
        pair_costs   = pair_turn * cost_frac * notional

        pnl_gross   = pnl_gross.add(pair_gross.reindex(full_index).fillna(0),  fill_value=0)
        cost_series = cost_series.add(pair_costs.reindex(full_index).fillna(0), fill_value=0)
        turn_series = turn_series.add(pair_turn.reindex(full_index).fillna(0),  fill_value=0)

    return pnl_gross, cost_series, turn_series


def _aggregate_folds(fold_metrics: list[dict]) -> dict:
    """Aggregate per-fold OOS metrics: mean, std, % positive, full-OOS stitched."""
    valid = [m for m in fold_metrics if m.get("n_bars_oos", 0) > 0]
    if not valid:
        return {"error": "no valid folds"}

    def stat(key: str) -> dict:
        vals = [m.get(key) for m in valid if np.isfinite(m.get(key, float("nan")))]
        if not vals:
            return {"mean": float("nan"), "std": float("nan"), "pct_positive": float("nan")}
        return {
            "mean":         round(float(np.mean(vals)), 4),
            "std":          round(float(np.std(vals)),  4),
            "pct_positive": round(float(np.mean(np.array(vals) > 0)), 4),
        }

    # Full OOS (all test-year PnLs stitched)
    all_gross = pd.concat([m["_pnl_gross_oos"] for m in valid]).sort_index()
    all_net   = pd.concat([m["_pnl_net_oos"]   for m in valid]).sort_index()
    total_trades = sum(m.get("total_trades", 0) for m in valid)
    ppy = PERIODS_PER_YEAR["1D"]
    full_m = _metrics_oos(all_gross, all_net, DEFAULT_CAPITAL, ppy, total_trades)

    return {
        "n_folds":           len(valid),
        "gross_sharpe":      stat("gross_sharpe"),
        "net_sharpe":        stat("net_sharpe"),
        "gross_ann_ret_pct": stat("gross_ann_ret_pct"),
        "net_ann_ret_pct":   stat("net_ann_ret_pct"),
        "net_maxdd_pct":     stat("net_maxdd_pct"),
        "trades_per_year":   stat("trades_per_year"),
        "cost_drag_pp":      stat("cost_drag_pp"),
        "full_oos":          full_m,
    }


# ---------------------------------------------------------------------------
# Stage 1 Ablation
# ---------------------------------------------------------------------------

def run_stage1_ablation(
    prices: pd.DataFrame,
    mode: str,
    top_k: int,
    costs: IndianCosts,
) -> dict[str, dict]:
    """
    For each Stage 1 config (one selector active + ensemble):
    Run all 6 WFV folds and return aggregated OOS metrics.
    """
    configs = _s1_ablation_configs(mode)
    ppy     = PERIODS_PER_YEAR["1D"]
    results: dict[str, dict] = {}

    log.info(f"  Stage 1 ablation: {list(configs.keys())}")

    for cfg_name, s1_weights in configs.items():
        log.info(f"  [S1] Config: {cfg_name}")
        fold_metrics: list[dict] = []

        for fold in FOLDS:
            test_start = fold["test_start"]
            test_end   = fold["test_end"]
            train_end  = str((pd.Timestamp(test_start) - pd.Timedelta(days=1)).date())

            train_prices = prices.loc[MAIN_START:train_end]
            full_prices  = prices.loc[MAIN_START:test_end]
            if len(train_prices) < 200:
                continue

            # Select pairs using this single selector config
            t0 = time.time()
            selected = _select_pairs(train_prices, s1_weights, top_k)
            if not selected:
                log.warning(f"    [{cfg_name}] {fold['name']}: no pairs selected")
                continue

            # Compute PnL using full S2 ensemble
            pnl_gross, cost_series, turn_series = _run_pairs_pnl(
                full_prices, selected, train_end,
                DEFAULT_S2_WEIGHTS, DEFAULT_CAPITAL, DEFAULT_PER_PAIR,
                costs, DEFAULT_MIN_HOLD,
            )

            pnl_gross_oos = pnl_gross.loc[test_start:test_end]
            pnl_net_oos   = (pnl_gross - cost_series).loc[test_start:test_end]
            trades_oos    = int(turn_series.loc[test_start:test_end].sum())

            m = _metrics_oos(pnl_gross_oos, pnl_net_oos, DEFAULT_CAPITAL, ppy, trades_oos)
            m["_pnl_gross_oos"] = pnl_gross_oos
            m["_pnl_net_oos"]   = pnl_net_oos
            m["fold"]           = fold["name"]
            m["n_pairs"]        = len(selected)
            m["time_s"]         = round(time.time() - t0, 1)
            fold_metrics.append(m)

            if m:
                log.info(
                    f"    [{cfg_name}] {fold['name']} -> "
                    f"Gross SR={float(m['gross_sharpe']):.3f}  "
                    f"Net SR={float(m['net_sharpe']):.3f}  "
                    f"pairs={len(selected)}  t={m['time_s']}s"
                )
            else:
                log.warning(f"    [{cfg_name}] {fold['name']} -> empty metrics (skipped)")

        results[cfg_name] = _aggregate_folds(fold_metrics)

    return results


# ---------------------------------------------------------------------------
# Stage 2 Ablation
# ---------------------------------------------------------------------------

def run_stage2_ablation(
    prices: pd.DataFrame,
    s1_mode: str,
    top_k: int,
    costs: IndianCosts,
) -> dict[str, dict]:
    """
    Select pairs ONCE per fold using the S1 ensemble, then run each Stage 2
    signal model config in isolation to isolate the signal model effect.
    """
    # Determine which S1 weights to use for pair selection
    s1_weights = {
        "stat_only": STAT_ONLY_S1_WEIGHTS,
        "stat_ml":   STAT_ML_S1_WEIGHTS,
        "full":      DEFAULT_S1_WEIGHTS,
    }[s1_mode]

    ppy     = PERIODS_PER_YEAR["1D"]

    # Pre-select pairs per fold (done ONCE — shared across all S2 configs)
    log.info("  [S2] Pre-selecting pairs per fold ...")
    fold_selections: dict[str, tuple] = {}   # fold_name -> (selected, train_end, full_prices)

    for fold in FOLDS:
        test_start = fold["test_start"]
        test_end   = fold["test_end"]
        train_end  = str((pd.Timestamp(test_start) - pd.Timedelta(days=1)).date())

        train_prices = prices.loc[MAIN_START:train_end]
        full_prices  = prices.loc[MAIN_START:test_end]
        if len(train_prices) < 200:
            continue

        selected = _select_pairs(train_prices, s1_weights, top_k)
        if not selected:
            continue

        display = [f"{p.a.split('.')[0]}-{p.b.split('.')[0]}" for p in selected]
        log.info(f"    {fold['name']}: {len(selected)} pairs — {display[:4]} ...")
        fold_selections[fold["name"]] = (selected, train_end, full_prices)

    results: dict[str, dict] = {}

    for cfg_name, s2_weights in _S2_ABLATION_CONFIGS.items():
        log.info(f"  [S2] Config: {cfg_name}")
        fold_metrics: list[dict] = []

        for fold in FOLDS:
            if fold["name"] not in fold_selections:
                continue
            selected, train_end, full_prices = fold_selections[fold["name"]]
            test_start = fold["test_start"]
            test_end   = fold["test_end"]

            t0 = time.time()
            pnl_gross, cost_series, turn_series = _run_pairs_pnl(
                full_prices, selected, train_end,
                s2_weights, DEFAULT_CAPITAL, DEFAULT_PER_PAIR,
                costs, DEFAULT_MIN_HOLD,
            )

            pnl_gross_oos = pnl_gross.loc[test_start:test_end]
            pnl_net_oos   = (pnl_gross - cost_series).loc[test_start:test_end]
            trades_oos    = int(turn_series.loc[test_start:test_end].sum())

            m = _metrics_oos(pnl_gross_oos, pnl_net_oos, DEFAULT_CAPITAL, ppy, trades_oos)
            m["_pnl_gross_oos"] = pnl_gross_oos
            m["_pnl_net_oos"]   = pnl_net_oos
            m["fold"]           = fold["name"]
            m["time_s"]         = round(time.time() - t0, 1)
            fold_metrics.append(m)

            log.info(
                f"    [{cfg_name}] {fold['name']} -> "
                f"Gross SR={m.get('gross_sharpe','?'):.3f}  "
                f"Net SR={m.get('net_sharpe','?'):.3f}  t={m['time_s']}s"
            )

        results[cfg_name] = _aggregate_folds(fold_metrics)

    return results


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def _fmt_agg(agg: dict, key: str, fmt: str) -> str:
    """Format mean +/- std [%pos] for a metric."""
    s = agg.get(key, {})
    m   = s.get("mean", float("nan"))
    sd  = s.get("std",  float("nan"))
    pct = s.get("pct_positive", float("nan"))
    try:
        return f"{m:{fmt}} +/-{sd:{fmt}} [{pct:.0%}]"
    except (TypeError, ValueError):
        return "N/A"


def print_stage_results(
    stage: int,
    results: dict[str, dict],
    ensemble_key: str,
) -> None:
    """Print ablation table for one stage.  Highlights ensemble row and winner."""
    sep = "=" * 110
    print(f"\n{sep}")
    print(f"  STAGE {stage} ABLATION — Out-of-Sample Results (mean +/- std across 6 folds, 2020-2025)")
    print(sep)

    # Determine best individual (non-ensemble) Net Sharpe
    individual_keys = [k for k in results if k != ensemble_key]
    best_ind_net = max(
        (results[k].get("full_oos", {}).get("net_sharpe", float("-inf")) for k in individual_keys),
        default=float("-inf"),
    )
    ens_net = results.get(ensemble_key, {}).get("full_oos", {}).get("net_sharpe", float("-inf"))
    ensemble_wins = ens_net > best_ind_net

    header = (
        f"{'Config':>22}  {'GrossSharpeMean':>17}  {'NetSharpeMean':>17}  "
        f"{'NetRet% Mean':>14}  {'MaxDD% Mean':>12}  {'Trd/YrMean':>12}  "
        f"{'FullOOS GrossSR':>16}  {'FullOOS NetSR':>14}"
    )
    print(header)
    print("-" * 110)

    for cfg_name, agg in results.items():
        foos = agg.get("full_oos", {})
        g_sr = foos.get("gross_sharpe", float("nan"))
        n_sr = foos.get("net_sharpe",   float("nan"))

        marker = ""
        if cfg_name == ensemble_key:
            marker = " [ENSEMBLE]"
        elif np.isfinite(n_sr) and n_sr == best_ind_net:
            marker = " [BEST IND]"

        row = (
            f"{cfg_name + marker:>22}"
            f"  {_fmt_agg(agg, 'gross_sharpe', '.3f'):>17}"
            f"  {_fmt_agg(agg, 'net_sharpe',   '.3f'):>17}"
            f"  {_fmt_agg(agg, 'net_ann_ret_pct', '.2f'):>14}"
            f"  {_fmt_agg(agg, 'net_maxdd_pct',   '.2f'):>12}"
            f"  {_fmt_agg(agg, 'trades_per_year',  '.0f'):>12}"
            f"  {g_sr:>16.3f}"
            f"  {n_sr:>14.3f}"
        )
        print(row)

    print("-" * 110)
    verdict = "ENSEMBLE > BEST INDIVIDUAL" if ensemble_wins else "BEST INDIVIDUAL >= ENSEMBLE"
    ens_margin = ens_net - best_ind_net
    print(f"\n  Verdict: {verdict}  (ensemble full-OOS Net SR {ens_net:.3f}  vs  best individual {best_ind_net:.3f};  margin={ens_margin:+.3f})")
    print(sep + "\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="E3: Ablation study — ensemble vs individual components",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mode", choices=["full", "stat_ml", "stat_only"],
                        default="stat_only",
                        help="Which Stage 1 selectors to include")
    parser.add_argument("--stage", type=int, choices=[1, 2, 0], default=0,
                        help="Which ablation to run: 1=Stage1 only, 2=Stage2 only, 0=both")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    args = parser.parse_args()

    log.info("=" * 64)
    log.info(f"E3 Ablation  |  mode={args.mode}  |  stage={'both' if args.stage == 0 else args.stage}  |  top_k={args.top_k}")
    log.info(f"min_hold={DEFAULT_MIN_HOLD}  capital={DEFAULT_CAPITAL:,.0f}")
    log.info("=" * 64)

    # Fetch all data once
    log.info("[Data] Fetching full 10-year daily prices ...")
    cfg    = DataConfig(start=MAIN_START, end=MAIN_END, freq="1D")
    prices = YFinanceNSESource().get_prices(NSE_UNIVERSE, cfg)
    coverage = prices.notna().mean()
    prices   = prices[coverage[coverage >= 0.80].index]
    log.info(f"  {len(prices.columns)} tickers x {len(prices)} bars")

    costs = IndianCosts()

    run_s1 = args.stage in (0, 1)
    run_s2 = args.stage in (0, 2)

    s1_results: dict = {}
    s2_results: dict = {}

    total_t0 = time.time()

    if run_s1:
        log.info("\n--- Running Stage 1 Ablation ---")
        s1_results = run_stage1_ablation(prices, args.mode, args.top_k, costs)
        print_stage_results(1, s1_results, "S1_Ensemble")

    if run_s2:
        log.info("\n--- Running Stage 2 Ablation ---")
        s2_results = run_stage2_ablation(prices, args.mode, args.top_k, costs)
        print_stage_results(2, s2_results, "S2_Ensemble")

    log.info(f"Total ablation time: {time.time() - total_t0:.0f}s")

    # ---- Save ----
    run_id   = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"ablation_{run_id}.json"

    def _clean_agg(agg: dict) -> dict:
        """Strip internal PnL Series from fold data before JSON serialisation."""
        clean = {}
        for k, v in agg.items():
            if isinstance(v, dict):
                clean[k] = v
            else:
                clean[k] = v
        return clean

    payload = {
        "experiment": "E3_ablation",
        "mode":       args.mode,
        "top_k":      args.top_k,
        "min_hold":   DEFAULT_MIN_HOLD,
        "stage1":     {k: _clean_agg(v) for k, v in s1_results.items()},
        "stage2":     {k: _clean_agg(v) for k, v in s2_results.items()},
    }
    with open(out_path, "w") as fh:
        json.dump(payload, fh, indent=2, default=str)
    log.info(f"Results saved -> {out_path}")


if __name__ == "__main__":
    main()
