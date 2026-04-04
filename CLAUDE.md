# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the Application

```bash
# First-time setup
python -m venv .venv
cd .venv/Scripts && ./activate && cd ../..
pip install -r requirements.txt

# Run (after activating venv)
streamlit run app.py
```

No build step, linting config, or test suite exists — `pytest` is listed in requirements but no tests are written yet. The prototype code lives in `Pairs-Trading-Ensemble-Prototype/` (git submodule, kept for reference only).

## Architecture Overview

A **two-stage ensemble pairs trading platform** for Indian NSE stocks, with a Streamlit UI (`app.py`) orchestrating a UI-agnostic `core/` library.

### Two-Stage Pipeline

**Stage 1 — Pair Selection** (`core/selectors_*.py`): Eight algorithms score stock pairs; scores are ensemble-averaged with user-set weights.
- `CorrelationSelector` — rolling correlation
- `DistanceSelector` — Gatev et al. (2006) normalized SSD
- `CointegrationSelector` — Engle-Granger ADF test
- `CombinedCriteriaSelector` — cointegration + Hurst exponent + half-life
- `MLSelector` — XGBoost with engineered spread features
- `LSTMSelector` — LSTM/BiLSTM multivariate sequence model
- `TransformerSelector` — Multi-head self-attention encoder
- `GNNSelector` — Graph Convolutional Network with link prediction

**Stage 2 — Entry/Exit Signals** (`core/entry.py`): Four signal models; signals are ensemble-averaged with user-set weights, then discretized to {+1, 0, −1}.
- `ZScoreThreshold` — entry |z| > 2, exit |z| < 0.5
- `OUThreshold` — Ornstein-Uhlenbeck process
- `KalmanHedge` — State-space Kalman Filter dynamic hedge ratio
- `MLSignal` — XGBoost/GBM classifier on spread features

### Core Modules

| File | Role |
|---|---|
| `core/data.py` | `DataConfig` dataclass, yfinance fetching, CSV/Parquet override |
| `core/selectors.py` | Re-exports all selectors from `selectors_base.py`, etc. |
| `core/entry.py` | Stage 2 signal classes |
| `core/ensemble.py` | Weighted score/signal combination utilities |
| `core/backtest.py` | Vectorized backtester, `BacktestConfig`, `IndianCosts` dataclass |
| `core/reports.py` | `ReportManager` (Repository pattern), `BenchmarkComparison` |
| `core/predictions.py` | Real-time prediction engine |
| `app.py` | All Streamlit UI and orchestration (~1,450 lines) |

### Key Design Patterns

- **Singleton**: `ReportManager` is cached in `st.session_state` via `get_report_manager()`
- **Strategy**: All selectors and entry models are interchangeable via shared interfaces
- **Repository**: `ReportManager` handles all disk I/O for reports

### Report Persistence

Every backtest run auto-saves to `reports/<YYYYMMDD_HHMMSS>/`:
- `metadata.json`, `metrics.json`, `params.json`
- `equity_gross.csv`, `equity_net.csv`, `pnl_gross.csv`, `pnl_net.csv`, `turnover.csv`, `trades.csv`

### Cost Model (`IndianCosts`)

Default NSE transaction costs: brokerage 3 bps, exchange 0.345 bps, SEBI 0.01 bps, STT 10 bps (sell only), stamp 1 bps (buy only), GST 18%, slippage 2 bps/leg. Toggle `intraday` flag for intraday vs. delivery STT rates.

### Adding a New Algorithm (Plugin Guide)

1. Implement a class in `core/selectors.py` (Stage 1) or `core/entry.py` (Stage 2) following the existing class interfaces.
2. Add any config as a `@dataclass`.
3. Wire it into the ensemble weight sliders in `app.py`.
4. Add a unit test in `tests/` against synthetic data.

### Data

- **Source**: Yahoo Finance via `yfinance`; NSE tickers use `.NS` suffix (e.g., `RELIANCE.NS`)
- **Override**: CSV or Parquet upload in the UI
- **Frequency**: Day-based (`1D`) in MVP; `DataConfig.freq` is designed to accept `1H`/`1min` for future intraday support

### Benchmarks

`BenchmarkComparison` fetches index data via yfinance: `^NSEI` (Nifty 50), `^BSESN` (Sensex), plus Nifty 100/200/500, Bank Nifty, Nifty IT.
