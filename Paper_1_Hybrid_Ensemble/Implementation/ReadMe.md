# README

A clean, developer-friendly blueprint for a pluggable, reproducible pairs-trading research & backtest platform.

Designed to compose stage-wise ensembles (Pairs Selection → Pairs Trading),
reproduce experiments, cache artifacts, and iterate fast.

This README explains repo structure, the order of files/units to implement, run examples, and the MVP roadmap.

## Quickstart (developer flow)

```bash

python -m venv .venv
cd .\.venv\Scripts
.\activate
cd ..\..
pip install -r requirements.txt
streamlit run .\app.py


Rerun:

cd .\.venv\Scripts
.\activate
cd ..\..
streamlit run .\app.py
```

## TL;DR (what this repo does)

- Stage-based pipeline: Selection (which pairs) → Trading (how to trade them).
- Every algorithm is a plugin (registry + metadata). Ensembles combine plugin outputs.
- **Persistent report storage**: All runs are automatically saved to disk with full parameters, metrics, and trade data.
- **Benchmark comparison**: Compare strategy returns with Indian indices (Nifty, Sensex, etc.) with toggle on/off.
- Single-command reproducible runs produce reports/<run_id>/ with trades CSV, equity curves, metrics, and full configuration.
- MVP target: day-based data using yfinance and parquet cache; future: intraday/intrahour support.

## New Features

### Comprehensive Report Management

Every simulation run is automatically saved to the `reports/` directory with:

- **Full parameter tracking**: Universe, data config, Stage 1/2 weights, backtest settings
- **Complete metrics**: Gross and Net returns, Sharpe ratios, volatility, max drawdown
- **Time series data**: Equity curves, P&L, turnover
- **Trade log**: Complete history of all trades with timestamps and signals

### Benchmark Comparison

Compare your strategy performance against major Indian indices:

- NIFTY 50, NIFTY 100, NIFTY 200, NIFTY 500
- SENSEX
- NIFTY BANK, NIFTY IT

Comparison includes:

- Excess returns
- Information ratio
- Tracking error
- Side-by-side return visualization

Toggle benchmark comparison on/off in the Reports page.

### Reports Page Features

- View all historical runs sorted by date
- Expandable parameter details for each run
- Performance metrics with visual indicators
- Benchmark comparison with selectable indices
- Download trades and equity curves as CSV
- Delete old reports to manage storage

### Defaults & decisions (MVP choices baked in)

- UI stack: Streamlit for MVP (fast dev). Next.js + FastAPI recommended for production UX.

- DL models: include LSTM as the first GPU model for stage-2 in MVP. Add Transformers/TCN later.

- Portfolio mode: Top-K per day (equal notional per pair) for MVP.

- Costs & slippage defaults: NSE IndianCosts model (16.3 bps round-trip: 0 bps brokerage for discount brokers, 0.322 bps exchange, 10 bps STT on sell, 1.5 bps stamp on buy, 2 bps slippage per leg).

- Sizing: fixed-notional in MVP; add volatility-targeted sizing later.

- Data cadence: day-based initially; design features & runner to be date-iterable so intraday fits later.

### Reproducibility & artifacts

Each run saves to `reports/<YYYYMMDD_HHMMSS>/`:

- **metadata.json**: Full configuration (universe, weights, parameters)
- **metrics.json**: Performance metrics (Gross/Net returns, Sharpe, volatility, drawdown)
- **params.json**: Detailed backtest parameters
- **equity_gross.csv**: Gross equity curve (before costs)
- **equity_net.csv**: Net equity curve (after costs)
- **pnl_gross.csv**: Per-period gross P&L
- **pnl_net.csv**: Per-period net P&L
- **turnover.csv**: Per-period turnover
- **trades.csv**: Complete trade log

### Plugin dev guide (how to add a new algo)

1. Implement algorithm class following interface in selection/ or trading/.
2. Add a small params_schema dataclass / pydantic model for UI-driven forms.
3. Register with @register_selection or @register_trading (adds metadata: name, gpu_capable, default_params).
4. Add unit tests in tests/ that run the algo on a small synthetic dataset.
5. Add example entry in configs/presets/.

### Notes about intraday / future work

- Design data/loader & features to accept freq param (day/1min/1s). MVP: day only.
- Backtest runner should be time-step agnostic — iterate over ordered timestamps. For intraday, ensure event ordering and partial fills/slippage model improvements.
- For GPU: keep RAPIDS wrappers behind gpu/ so code falls back to Pandas/Numpy when unavailable.

# Feature Summary: Report Management & Benchmark Comparison

### Comprehensive Report Saving [OK]

**Status:** Fully Implemented

Every simulation run now automatically saves:

- [OK] Complete parameter configuration (universe, weights, backtest settings)
- [OK] All selected data (tickers, date ranges, frequency)
- [OK] Complete trade log with timestamps and signals
- [OK] Performance metrics (Gross & Net)
- [OK] Time series data (equity curves, P&L, turnover)
- [OK] Selected pairs information

**Location:** `reports/<YYYYMMDD_HHMMSS>/`

**Files Saved:**

- `metadata.json` - All parameters and configuration
- `metrics.json` - Performance metrics
- `params.json` - Detailed backtest parameters
- `equity_gross.csv` - Gross equity curve
- `equity_net.csv` - Net equity curve
- `pnl_gross.csv` - Gross P&L time series
- `pnl_net.csv` - Net P&L time series
- `turnover.csv` - Position changes
- `trades.csv` - Complete trade log

### Repository Map

Hybrid-Pairs-Trading-Ensemble/ ├── app.py (1,452 lines) — Streamlit UI, NOT used for experiments  
 ├── core/ — the library; used by both app.py and experiments/ │ ├── data.py — yfinance fetching, DataConfig │ ├── selectors_base.py — base classes: Pair, PairScore, PairSelector │ ├── selectors_statistical.py — Correlation, Distance, Cointegration, Combined (4 classical) │ ├── selectors_ml.py — MLSelector, LSTMSelector, TransformerSelector, GNNSelector (4 ML/DL) │ ├── selectors.py — re-exports everything above (backward compat) │ ├── entry.py — ZScoreThreshold, OUThreshold, KalmanHedge, MLSignal │ ├── ensemble.py — ensemble_pair_scores(), ensemble_signals() │ ├── backtest.py — vectorized backtester, IndianCosts, BacktestConfig │ ├── reports.py — ReportManager, BenchmarkComparison │ └── predictions.py — real-time prediction engine (Streamlit only) └── experiments/ — reproducible thesis scripts; results saved as JSON ├── config.py — universe (35 NSE tickers), date ranges, seeds, mode weights  
 ├── freq_comparison.py — E1: daily vs hourly
├── hold_period_sweep.py — E2: min_hold_bars sweep
├── walk_forward.py — E4: expanding-window WFV (6 folds, 2020-2025)
├── ablation.py — E3: per-model isolation runs
├── benchmark_comparison.py — E5: vs Nifty/Sensex/BankNifty
├── significance_tests.py — E6: bootstrap CI, Bonferroni
└── results/ — 14 JSON result files already saved
