"""
experiments/config.py
=====================
Single source of truth for all experiment parameters.

Every experiment script imports from here so results are fully reproducible:
the same universe, date ranges, weights, and cost assumptions appear across
every run. Change a value here and all experiments pick it up automatically.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# NSE Universe — 35 liquid large-cap stocks across 8 sectors
#
# Rationale for diversity:
#   - Same-sector pairs (e.g. HDFCBANK / ICICIBANK) are natural candidates
#     with genuine economic co-movement.
#   - Cross-sector pairs (e.g. TCS / INFY vs HDFCBANK / SBIN) test whether
#     the ensemble's ML/DL selectors find non-obvious relationships.
#   - All tickers are Nifty 100 constituents → liquid, well-covered by yfinance.
#
# Tickers are in yfinance format (.NS suffix for NSE).
# ---------------------------------------------------------------------------

NSE_UNIVERSE: list[str] = [
    # Banking & Financial Services (6)
    "HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS",
    "KOTAKBANK.NS", "AXISBANK.NS", "INDUSINDBK.NS",

    # Information Technology (5)
    "TCS.NS", "INFY.NS", "WIPRO.NS", "HCLTECH.NS", "TECHM.NS",

    # Automobiles & Components (5)
    "MARUTI.NS", "M&M.NS", "BAJAJ-AUTO.NS", "HEROMOTOCO.NS", "EICHERMOT.NS",

    # FMCG & Consumer Staples (4)
    "HINDUNILVR.NS", "ITC.NS", "NESTLEIND.NS", "BRITANNIA.NS",

    # Pharma & Healthcare (4)
    "SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS", "DIVISLAB.NS",

    # Energy & Oil & Gas (4)
    "RELIANCE.NS", "ONGC.NS", "IOC.NS", "BPCL.NS",

    # Metals & Mining (4)
    "TATASTEEL.NS", "JSWSTEEL.NS", "HINDALCO.NS", "COALINDIA.NS",

    # Cement & Infrastructure (3)
    "ULTRACEMCO.NS", "ACC.NS", "SHREECEM.NS",
]

SECTOR_MAP: dict[str, str] = {
    "HDFCBANK.NS":   "Banking",  "ICICIBANK.NS":  "Banking",  "SBIN.NS":       "Banking",
    "KOTAKBANK.NS":  "Banking",  "AXISBANK.NS":   "Banking",  "INDUSINDBK.NS": "Banking",
    "TCS.NS":        "IT",       "INFY.NS":       "IT",       "WIPRO.NS":      "IT",
    "HCLTECH.NS":    "IT",       "TECHM.NS":      "IT",
    "MARUTI.NS":     "Auto",     "M&M.NS":        "Auto",      "BAJAJ-AUTO.NS": "Auto",
    "HEROMOTOCO.NS": "Auto",     "EICHERMOT.NS":  "Auto",
    "HINDUNILVR.NS": "FMCG",    "ITC.NS":        "FMCG",     "NESTLEIND.NS":  "FMCG",
    "BRITANNIA.NS":  "FMCG",
    "SUNPHARMA.NS":  "Pharma",   "DRREDDY.NS":    "Pharma",   "CIPLA.NS":      "Pharma",
    "DIVISLAB.NS":   "Pharma",
    "RELIANCE.NS":   "Energy",   "ONGC.NS":       "Energy",   "IOC.NS":        "Energy",
    "BPCL.NS":       "Energy",
    "TATASTEEL.NS":  "Metals",   "JSWSTEEL.NS":   "Metals",   "HINDALCO.NS":   "Metals",
    "COALINDIA.NS":  "Metals",
    "ULTRACEMCO.NS": "Cement",   "ACC.NS":        "Cement",   "SHREECEM.NS":   "Cement",
}

# ---------------------------------------------------------------------------
# Date Ranges
# ---------------------------------------------------------------------------

# Frequency comparison experiment.
# yfinance hard limit for 60m / 1H data: last 730 calendar days from request time.
# We use a 700-day rolling window (30-day safety buffer) computed at runtime,
# so the experiment stays within the limit regardless of when it is run.
# The script (freq_comparison.py) computes actual start/end from today; these
# constants are kept only as a human-readable description of the intent.
FREQ_COMPARISON_LOOKBACK_DAYS = 700   # ≤ 730; safety buffer for timezone offsets
FREQ_COMPARISON_END   = "2026-03-31"  # overridden at runtime to yesterday

# Main experiments (walk-forward validation, ablation, etc.).
# 10-year window on daily data — maximum yfinance coverage for Nifty stocks.
MAIN_START = "2016-01-01"
MAIN_END   = "2026-03-31"

# ---------------------------------------------------------------------------
# Ensemble Weights
#
# Equal weights throughout for unbiased frequency comparison and ablation.
# The Streamlit UI allows interactive tuning on top of these defaults.
# ---------------------------------------------------------------------------

# All 8 Stage-1 selectors active
DEFAULT_S1_WEIGHTS: dict[str, float] = {
    "Correlation":   1.0,
    "Distance":      1.0,
    "Cointegration": 1.0,
    "Combined":      1.0,
    "ML":            1.0,
    "LSTM":          1.0,
    "Transformer":   1.0,
    "GNN":           1.0,
}

# Statistical + ML only (skip LSTM / Transformer / GNN for fast runs)
STAT_ML_S1_WEIGHTS: dict[str, float] = {
    "Correlation":   1.0,
    "Distance":      1.0,
    "Cointegration": 1.0,
    "Combined":      1.0,
    "ML":            1.0,
    "LSTM":          0.0,
    "Transformer":   0.0,
    "GNN":           0.0,
}

# Statistical only (fastest — useful for quick sanity checks)
STAT_ONLY_S1_WEIGHTS: dict[str, float] = {
    "Correlation":   1.0,
    "Distance":      1.0,
    "Cointegration": 1.0,
    "Combined":      1.0,
    "ML":            0.0,
    "LSTM":          0.0,
    "Transformer":   0.0,
    "GNN":           0.0,
}

# All 4 Stage-2 signal models active (equal weight)
DEFAULT_S2_WEIGHTS: dict[str, float] = {
    "ZScore": 1.0,
    "OU":     1.0,
    "Kalman": 1.0,
    "ML":     1.0,
}

# Statistical signal models only — excludes MLSignal.
# Used after E3 ablation revealed MLSignal overfit OOS (Net SR -0.401 vs OU +0.359).
STAT_S2_WEIGHTS: dict[str, float] = {
    "ZScore": 1.0,
    "OU":     1.0,
    "Kalman": 1.0,
    "ML":     0.0,
}

# OUThreshold only — the empirically dominant signal model from E3 ablation.
OU_ONLY_S2_WEIGHTS: dict[str, float] = {
    "ZScore": 0.0,
    "OU":     1.0,
    "Kalman": 0.0,
    "ML":     0.0,
}

# ---------------------------------------------------------------------------
# Backtest Defaults
# ---------------------------------------------------------------------------

DEFAULT_TOP_K      = 10    # pairs to select per run
DEFAULT_MIN_HOLD   = 30   # trading days; determined by E2 sweep (Research.md)
DEFAULT_CAPITAL    = 1_000_000   # INR 10 lakh — realistic retail/prop desk size
DEFAULT_PER_PAIR   = 100_000     # INR 1 lakh per pair leg
DEFAULT_MAX_PAIRS  = 10

# periods_per_year per frequency (for Sharpe & volatility annualization).
# NSE equity session: 09:15–15:30 = 6 h 15 min ≈ 6 complete 60-min bars/day.
PERIODS_PER_YEAR: dict[str, int] = {
    "1D": 252,
    "1H": 1512,   # 252 trading days × 6 hourly bars/day
}

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

RANDOM_SEED = 42
