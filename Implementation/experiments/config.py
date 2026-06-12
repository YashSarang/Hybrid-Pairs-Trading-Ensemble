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
# NSE Universe — 89 Nifty 100 stocks across 13 sectors
#
# Source: prices_2015-01-01_2024-12-31.parquet (89 columns confirmed 2026-06-04)
# Dropped from original ~95: TATAMOTORS (yfinance error, D9), BERGERPAINTS,
# VODAFONEIDEA, LTIM, ADANITRANS + 1 other (<5yr data filter in fetch_paper1_data.py)
#
# Tickers are in yfinance format (.NS suffix for NSE).
# This list must exactly match the parquet columns — do not modify without
# re-running fetch_paper1_data.py and regenerating the parquet.
# ---------------------------------------------------------------------------

NSE_UNIVERSE: list[str] = [
    # Banking & Financial Services (14)
    "HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "KOTAKBANK.NS", "AXISBANK.NS",
    "INDUSINDBK.NS", "BANDHANBNK.NS", "FEDERALBNK.NS", "IDFCFIRSTB.NS",
    "PNB.NS", "CANBK.NS", "BANKBARODA.NS", "BAJFINANCE.NS", "BAJAJFINSV.NS",

    # Information Technology (7)
    "TCS.NS", "INFY.NS", "WIPRO.NS", "HCLTECH.NS", "TECHM.NS",
    "MPHASIS.NS", "PERSISTENT.NS",

    # Automobiles & Auto Components (7)
    "MARUTI.NS", "M&M.NS", "BAJAJ-AUTO.NS", "HEROMOTOCO.NS", "EICHERMOT.NS",
    "BOSCHLTD.NS", "MOTHERSON.NS",

    # FMCG & Consumer Staples (7)
    "HINDUNILVR.NS", "ITC.NS", "NESTLEIND.NS", "BRITANNIA.NS",
    "DABUR.NS", "GODREJCP.NS", "MARICO.NS",

    # Pharma & Healthcare (8)
    "SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS", "DIVISLAB.NS",
    "APOLLOHOSP.NS", "TORNTPHARM.NS", "BIOCON.NS", "LUPIN.NS",

    # Energy, Oil & Gas (6)
    "RELIANCE.NS", "ONGC.NS", "IOC.NS", "BPCL.NS",
    "GAIL.NS", "POWERGRID.NS",

    # Metals, Mining & Materials (8)
    "TATASTEEL.NS", "JSWSTEEL.NS", "HINDALCO.NS", "COALINDIA.NS",
    "VEDL.NS", "SAIL.NS", "NATIONALUM.NS", "NMDC.NS",

    # Cement & Infrastructure (6)
    "ULTRACEMCO.NS", "ACC.NS", "SHREECEM.NS",
    "AMBUJACEM.NS", "JKCEMENT.NS", "RAMCOCEM.NS",

    # Capital Goods & Industrials (8)
    "LT.NS", "BHEL.NS", "SIEMENS.NS", "ABB.NS",
    "HAVELLS.NS", "CUMMINSIND.NS", "THERMAX.NS", "BEL.NS",

    # Telecom & Media (2)
    "BHARTIARTL.NS", "TATACOMM.NS",

    # Consumer Discretionary & Retail (5)
    "TITAN.NS", "ASIANPAINT.NS", "PIDILITIND.NS", "DMART.NS", "TRENT.NS",

    # Real Estate & Utilities (4)
    "DLF.NS", "GODREJPROP.NS", "OBEROIRLTY.NS", "PHOENIXLTD.NS",

    # Conglomerate / Diversified (3)
    "ADANIENT.NS", "ADANIPORTS.NS", "ADANIGREEN.NS",

    # Insurance & Asset Management (4)
    "HDFCLIFE.NS", "SBILIFE.NS", "ICICIPRULI.NS", "MUTHOOTFIN.NS",
]

SECTOR_MAP: dict[str, str] = {
    # Banking
    "HDFCBANK.NS":    "Banking",     "ICICIBANK.NS":   "Banking",     "SBIN.NS":        "Banking",
    "KOTAKBANK.NS":   "Banking",     "AXISBANK.NS":    "Banking",     "INDUSINDBK.NS":  "Banking",
    "BANDHANBNK.NS":  "Banking",     "FEDERALBNK.NS":  "Banking",     "IDFCFIRSTB.NS":  "Banking",
    "PNB.NS":         "Banking",     "CANBK.NS":       "Banking",     "BANKBARODA.NS":  "Banking",
    "BAJFINANCE.NS":  "Banking",     "BAJAJFINSV.NS":  "Banking",
    # IT
    "TCS.NS":         "IT",          "INFY.NS":        "IT",          "WIPRO.NS":       "IT",
    "HCLTECH.NS":     "IT",          "TECHM.NS":       "IT",          "MPHASIS.NS":     "IT",
    "PERSISTENT.NS":  "IT",
    # Auto
    "MARUTI.NS":      "Auto",        "M&M.NS":         "Auto",        "BAJAJ-AUTO.NS":  "Auto",
    "HEROMOTOCO.NS":  "Auto",        "EICHERMOT.NS":   "Auto",        "BOSCHLTD.NS":    "Auto",
    "MOTHERSON.NS":   "Auto",
    # FMCG
    "HINDUNILVR.NS":  "FMCG",        "ITC.NS":         "FMCG",        "NESTLEIND.NS":   "FMCG",
    "BRITANNIA.NS":   "FMCG",        "DABUR.NS":       "FMCG",        "GODREJCP.NS":    "FMCG",
    "MARICO.NS":      "FMCG",
    # Pharma
    "SUNPHARMA.NS":   "Pharma",      "DRREDDY.NS":     "Pharma",      "CIPLA.NS":       "Pharma",
    "DIVISLAB.NS":    "Pharma",      "APOLLOHOSP.NS":  "Pharma",      "TORNTPHARM.NS":  "Pharma",
    "BIOCON.NS":      "Pharma",      "LUPIN.NS":       "Pharma",
    # Energy
    "RELIANCE.NS":    "Energy",      "ONGC.NS":        "Energy",      "IOC.NS":         "Energy",
    "BPCL.NS":        "Energy",      "GAIL.NS":        "Energy",      "POWERGRID.NS":   "Energy",
    # Metals
    "TATASTEEL.NS":   "Metals",      "JSWSTEEL.NS":    "Metals",      "HINDALCO.NS":    "Metals",
    "COALINDIA.NS":   "Metals",      "VEDL.NS":        "Metals",      "SAIL.NS":        "Metals",
    "NATIONALUM.NS":  "Metals",      "NMDC.NS":        "Metals",
    # Cement
    "ULTRACEMCO.NS":  "Cement",      "ACC.NS":         "Cement",      "SHREECEM.NS":    "Cement",
    "AMBUJACEM.NS":   "Cement",      "JKCEMENT.NS":    "Cement",      "RAMCOCEM.NS":    "Cement",
    # Industrials
    "LT.NS":          "Industrials", "BHEL.NS":        "Industrials", "SIEMENS.NS":     "Industrials",
    "ABB.NS":         "Industrials", "HAVELLS.NS":     "Industrials", "CUMMINSIND.NS":  "Industrials",
    "THERMAX.NS":     "Industrials", "BEL.NS":         "Industrials",
    # Telecom
    "BHARTIARTL.NS":  "Telecom",     "TATACOMM.NS":    "Telecom",
    # Consumer
    "TITAN.NS":       "Consumer",    "ASIANPAINT.NS":  "Consumer",    "PIDILITIND.NS":  "Consumer",
    "DMART.NS":       "Consumer",    "TRENT.NS":       "Consumer",
    # Real Estate
    "DLF.NS":         "RealEstate",  "GODREJPROP.NS":  "RealEstate",  "OBEROIRLTY.NS":  "RealEstate",
    "PHOENIXLTD.NS":  "RealEstate",
    # Conglomerate
    "ADANIENT.NS":    "Conglomerate","ADANIPORTS.NS":  "Conglomerate","ADANIGREEN.NS":  "Conglomerate",
    # Insurance
    "HDFCLIFE.NS":    "Insurance",   "SBILIFE.NS":     "Insurance",   "ICICIPRULI.NS":  "Insurance",
    "MUTHOOTFIN.NS":  "Insurance",
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
MAIN_START = "2015-01-01"
MAIN_END   = "2024-12-31"

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

# ---------------------------------------------------------------------------
# Single-Selector Standalone Presets (E4.S experiments)
#
# Each preset activates exactly one Stage-1 selector (weight=1.0, all others 0).
# Used to establish individual selector baselines before ensemble comparisons.
# ---------------------------------------------------------------------------

_ZERO_S1: dict[str, float] = {
    "Correlation": 0.0, "Distance": 0.0, "Cointegration": 0.0, "Combined": 0.0,
    "ML": 0.0, "LSTM": 0.0, "Transformer": 0.0, "GNN": 0.0,
}

CORR_ONLY_S1:  dict[str, float] = {**_ZERO_S1, "Correlation":   1.0}  # E4.S1
DIST_ONLY_S1:  dict[str, float] = {**_ZERO_S1, "Distance":      1.0}  # E4.S2
COINT_ONLY_S1: dict[str, float] = {**_ZERO_S1, "Cointegration": 1.0}  # E4.S3
COMB_ONLY_S1:  dict[str, float] = {**_ZERO_S1, "Combined":      1.0}  # E4.S4
ML_ONLY_S1:    dict[str, float] = {**_ZERO_S1, "ML":            1.0}  # E4.S5
LSTM_ONLY_S1:  dict[str, float] = {**_ZERO_S1, "LSTM":          1.0}  # E4.S6
TRANS_ONLY_S1: dict[str, float] = {**_ZERO_S1, "Transformer":   1.0}  # E4.S7
GNN_ONLY_S1:   dict[str, float] = {**_ZERO_S1, "GNN":           1.0}  # E4.S8

# ---------------------------------------------------------------------------
# S1_PRESETS: unified lookup dict for CLI --s1-preset argument.
# Keys are the preset names accepted by walk_forward.py --s1-preset.
# ---------------------------------------------------------------------------
S1_PRESETS: dict[str, dict[str, float]] = {
    # Canonical 3-configuration ensemble presets
    "full":       DEFAULT_S1_WEIGHTS,
    "stat_ml":    STAT_ML_S1_WEIGHTS,
    "stat_only":  STAT_ONLY_S1_WEIGHTS,
    # Single-selector standalone benchmarks (E4.S)
    "corr_only":  CORR_ONLY_S1,
    "dist_only":  DIST_ONLY_S1,
    "coint_only": COINT_ONLY_S1,
    "comb_only":  COMB_ONLY_S1,
    "ml_only":    ML_ONLY_S1,
    "lstm_only":  LSTM_ONLY_S1,
    "trans_only": TRANS_ONLY_S1,
    "gnn_only":   GNN_ONLY_S1,
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
