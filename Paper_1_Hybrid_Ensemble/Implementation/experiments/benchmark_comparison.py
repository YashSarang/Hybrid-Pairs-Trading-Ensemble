"""
experiments/benchmark_comparison.py
=====================================
Experiment E5 — Benchmark Comparison
--------------------------------------
Loads the OOS equity curves saved by walk_forward.py and compares them
against Indian market indices (Nifty 50, Nifty Bank, Nifty IT).

Metrics computed:
  - Cumulative return (strategy vs benchmark over full OOS period)
  - CAGR (compound annual growth rate)
  - Annualized volatility
  - Sharpe ratio
  - Max drawdown
  - Beta (market exposure — expected ~0 for market-neutral pairs trading)
  - Jensen's alpha (annualized)
  - Correlation with benchmark
  - Information ratio (tracking-error Sharpe vs benchmark)
  - Calmar ratio (CAGR / |Max DD|)

Usage
-----
    python experiments/benchmark_comparison.py
    python experiments/benchmark_comparison.py --wfv walk_forward_20260402_230753.json
    python experiments/benchmark_comparison.py --mode stat_only --s2 ou_only

Output
------
  experiments/results/benchmark_<timestamp>.json
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from experiments.config import MAIN_START, PERIODS_PER_YEAR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

_RESULTS_DIR = Path(__file__).parent / "results"
_RESULTS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Benchmark index tickers (yfinance)
# ---------------------------------------------------------------------------

BENCHMARKS = {
    "Nifty 50":   "^NSEI",
    "Nifty Bank": "^NSEBANK",
    "Nifty IT":   "^CNXIT",
}

PERIODS_PER_YEAR_1D = 252


# ---------------------------------------------------------------------------
# Helper: load OOS equity curve from WFV JSON
# ---------------------------------------------------------------------------

def _find_wfv_file(mode: str, s2: str) -> Path:
    """Find the most recent WFV result matching mode and s2."""
    candidates = sorted(_RESULTS_DIR.glob("walk_forward_*.json"), reverse=True)
    for p in candidates:
        try:
            d = json.loads(p.read_text())
            if d.get("mode") == mode and d.get("s2") == s2:
                return p
        except Exception:
            continue
    raise FileNotFoundError(
        f"No WFV result found for mode={mode!r} s2={s2!r} in {_RESULTS_DIR}"
    )


def load_oos_equity(wfv_path: Path) -> tuple[pd.Series, pd.Series, dict]:
    """Load gross and net OOS equity curves from a WFV JSON file.

    Returns (gross_equity, net_equity, metadata_dict).
    Equity is expressed as a normalized return series (starting at 0.0).
    """
    d = json.loads(wfv_path.read_text())
    agg = d["aggregate"]
    capital = d.get("capital", 1_000_000)

    raw_gross = agg["cumulative_gross"]
    raw_net   = agg["cumulative_net"]

    gross = pd.Series(raw_gross, dtype=float)
    net   = pd.Series(raw_net,   dtype=float)
    gross.index = pd.to_datetime(gross.index)
    net.index   = pd.to_datetime(net.index)

    # Normalize to fractional return (0 = start)
    gross_ret = gross / capital - 1.0
    net_ret   = net   / capital - 1.0

    meta = {
        "mode": d.get("mode"),
        "s2":   d.get("s2"),
        "top_k": d.get("top_k"),
        "min_hold": d.get("min_hold"),
        "wfv_file": wfv_path.name,
        "full_oos_gross_sr": agg["full_oos_metrics"]["gross_sharpe"],
        "full_oos_net_sr":   agg["full_oos_metrics"]["net_sharpe"],
    }
    return gross_ret, net_ret, meta


# ---------------------------------------------------------------------------
# Helper: fetch benchmark price series
# ---------------------------------------------------------------------------

def fetch_benchmark(ticker: str, start: str, end: str) -> pd.Series:
    """Download index close and return as a normalized return series."""
    raw = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if raw.empty:
        raise RuntimeError(f"yfinance returned empty data for {ticker}")
    close = raw["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = close.dropna()
    cum_ret = (close / close.iloc[0]) - 1.0
    cum_ret.name = ticker
    return cum_ret


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------

def _daily_returns(cum_ret: pd.Series) -> pd.Series:
    """Convert cumulative return series to daily log-returns."""
    price = 1.0 + cum_ret
    return np.log(price / price.shift(1)).dropna()


def compute_metrics(cum_ret: pd.Series, label: str = "Strategy") -> dict:
    """Compute performance metrics from a cumulative return series (start = 0)."""
    r = _daily_returns(cum_ret)
    n = len(r)
    years = n / PERIODS_PER_YEAR_1D

    total_ret  = float(cum_ret.iloc[-1])
    cagr       = float((1 + total_ret) ** (1 / years) - 1) if years > 0 else 0.0
    vol        = float(r.std() * np.sqrt(PERIODS_PER_YEAR_1D))
    sharpe     = float(cagr / vol) if vol > 0 else 0.0

    # Max drawdown
    price = 1.0 + cum_ret
    running_max = price.cummax()
    dd = (price - running_max) / running_max
    max_dd = float(dd.min())

    calmar = float(cagr / abs(max_dd)) if max_dd != 0 else 0.0

    return {
        "label":      label,
        "total_ret":  round(total_ret * 100, 2),    # %
        "cagr":       round(cagr * 100, 2),          # %
        "vol_ann":    round(vol * 100, 2),            # %
        "sharpe":     round(sharpe, 3),
        "max_dd":     round(max_dd * 100, 2),        # %
        "calmar":     round(calmar, 3),
        "n_days":     n,
        "n_years":    round(years, 2),
    }


def compute_relative_metrics(
    strategy_cum: pd.Series,
    benchmark_cum: pd.Series,
    bench_label: str = "Benchmark",
) -> dict:
    """Beta, alpha, correlation, information ratio vs a benchmark."""
    # Align on common dates
    common = strategy_cum.index.intersection(benchmark_cum.index)
    s = strategy_cum.reindex(common).ffill()
    b = benchmark_cum.reindex(common).ffill()

    rs = _daily_returns(s)
    rb = _daily_returns(b)

    # Align again after differencing
    idx = rs.index.intersection(rb.index)
    rs, rb = rs.reindex(idx), rb.reindex(idx)

    # Beta
    cov = np.cov(rs.values, rb.values)
    beta  = float(cov[0, 1] / cov[1, 1]) if cov[1, 1] > 0 else 0.0

    # Jensen's alpha (annualized, rf=0)
    alpha_daily = rs.mean() - beta * rb.mean()
    alpha_ann   = float(alpha_daily * PERIODS_PER_YEAR_1D)

    # Correlation
    corr = float(rs.corr(rb))

    # Information ratio (active return / tracking error)
    active   = rs - rb
    te       = float(active.std() * np.sqrt(PERIODS_PER_YEAR_1D))
    active_r = float(active.mean() * PERIODS_PER_YEAR_1D)
    ir       = float(active_r / te) if te > 0 else 0.0

    return {
        "vs_benchmark": bench_label,
        "beta":         round(beta, 4),
        "alpha_ann_pct": round(alpha_ann * 100, 2),
        "correlation":  round(corr, 4),
        "active_return_ann_pct": round(active_r * 100, 2),
        "tracking_error_ann_pct": round(te * 100, 2),
        "information_ratio": round(ir, 3),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="E5 Benchmark Comparison")
    parser.add_argument(
        "--wfv",
        default=None,
        help="WFV result filename (in experiments/results/). Auto-detected if omitted.",
    )
    parser.add_argument("--mode", default="stat_only", help="WFV mode (default: stat_only)")
    parser.add_argument(
        "--s2",
        default="ou_only",
        help="WFV s2 config (default: ou_only — the headline result)",
    )
    args = parser.parse_args()

    # --- Load equity curve ---
    if args.wfv:
        wfv_path = _RESULTS_DIR / args.wfv
    else:
        wfv_path = _find_wfv_file(args.mode, args.s2)
    log.info(f"Loading WFV result: {wfv_path.name}")

    gross_ret, net_ret, meta = load_oos_equity(wfv_path)

    oos_start = str(gross_ret.index[0].date())
    oos_end   = str(gross_ret.index[-1].date())
    log.info(f"OOS period: {oos_start} to {oos_end}  ({len(gross_ret)} bars)")
    log.info(f"Strategy: mode={meta['mode']!r}, s2={meta['s2']!r}")

    # --- Strategy metrics ---
    gross_m = compute_metrics(gross_ret, "Strategy (Gross)")
    net_m   = compute_metrics(net_ret,   "Strategy (Net)")

    # --- Benchmark metrics + relative ---
    bench_results = {}
    for name, ticker in BENCHMARKS.items():
        log.info(f"Fetching {name} ({ticker}) ...")
        try:
            bret = fetch_benchmark(ticker, oos_start, oos_end)
            bm   = compute_metrics(bret, name)
            rel_gross = compute_relative_metrics(gross_ret, bret, name)
            rel_net   = compute_relative_metrics(net_ret,   bret, name)
            bench_results[name] = {
                "benchmark_metrics": bm,
                "gross_vs_benchmark": rel_gross,
                "net_vs_benchmark":   rel_net,
            }
        except Exception as exc:
            log.warning(f"  {name}: {exc}")
            bench_results[name] = {"error": str(exc)}

    # --- Print summary ---
    print()
    print("=" * 72)
    print(f"  E5 Benchmark Comparison")
    print(f"  WFV: {wfv_path.name}  |  mode={meta['mode']}  |  s2={meta['s2']}")
    print(f"  OOS period: {oos_start} to {oos_end}")
    print("=" * 72)

    header = f"{'Metric':<30}{'Gross':>12}{'Net':>12}"
    print(header)
    print("-" * 54)
    for key, label in [
        ("total_ret",  "Total Return (%)"),
        ("cagr",       "CAGR (%)"),
        ("vol_ann",    "Ann. Vol (%)"),
        ("sharpe",     "Sharpe Ratio"),
        ("max_dd",     "Max Drawdown (%)"),
        ("calmar",     "Calmar Ratio"),
    ]:
        g = gross_m.get(key, "")
        n = net_m.get(key, "")
        print(f"  {label:<28}{g:>12}{n:>12}")

    for name, res in bench_results.items():
        if "error" in res:
            print(f"\n  {name}: ERROR — {res['error']}")
            continue
        bm  = res["benchmark_metrics"]
        rel = res["net_vs_benchmark"]
        print()
        print(f"  vs {name}:")
        print(f"    Benchmark CAGR:          {bm['cagr']:>8.2f}%")
        print(f"    Benchmark Sharpe:        {bm['sharpe']:>8.3f}")
        print(f"    Benchmark Max DD:        {bm['max_dd']:>8.2f}%")
        print(f"    Beta  (Net vs Bench):    {rel['beta']:>8.4f}")
        print(f"    Alpha (Net, Ann.):       {rel['alpha_ann_pct']:>8.2f}%")
        print(f"    Correlation (Net):       {rel['correlation']:>8.4f}")
        print(f"    Info Ratio (Net):        {rel['information_ratio']:>8.3f}")

    print()

    # --- Save results ---
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = {
        "experiment": "E5_benchmark_comparison",
        "wfv_file":   wfv_path.name,
        "mode":       meta["mode"],
        "s2":         meta["s2"],
        "oos_start":  oos_start,
        "oos_end":    oos_end,
        "strategy": {
            "gross": gross_m,
            "net":   net_m,
        },
        "benchmarks": bench_results,
    }
    out_path = _RESULTS_DIR / f"benchmark_{ts}.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    log.info(f"Results saved → {out_path.name}")


if __name__ == "__main__":
    main()
