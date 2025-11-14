# pairs-trading — README

A clean, developer-friendly blueprint for a pluggable, reproducible pairs-trading research & backtest platform.
Designed to let product/ML/dev teams compose stage-wise ensembles (Pairs Selection → Pairs Trading), reproduce experiments, cache artifacts, and iterate fast. 

This README explains repo structure, the order of files/units to implement, run examples, and the MVP roadmap.

## TL;DR (what this repo does)

* Stage-based pipeline: Selection (which pairs) → Trading (how to trade them).
* Every algorithm is a plugin (registry + metadata). Ensembles combine plugin outputs.
* Single-command reproducible runs produce runs/<run_id>/ with trades CSV, charts, saved models, and summary.json.
* MVP target: day-based data using yfinance and parquet cache; future: intraday/intrahour support.

## Repo layout (top-level)
``` bash
pairs-trading/
├─ src/
│  ├─ app/                    # UI (Streamlit MVP + widgets). 
│  ├─ core/                   # Shared types, registries, metrics, utils
│  ├─ data/                   # Data loaders, cache (Parquet), feature builders, labelers
│  ├─ selection/              # Selection algorithms (plugins)
│  ├─ trading/                # Trading algorithms (plugins)
│  ├─ ensemble/               # Generic ensembling abstractions & implementations
│  ├─ backtest/               # Orchestrator: runner, positioning, simulator
│  ├─ eval/                   # Metrics, tables, plotting utilities
│  ├─ io/                     # RunConfig (pydantic), artifact writers, deterministic run IDs
│  ├─ gpu/                    # GPU toggles / RAPIDS wrappers & CPU fallbacks
│  └─ main.py                 # CLI entrypoint (already present)
├─ models/                    # Saved ML/DL models (per-run)
├─ reports/                   # Generated charts, tables
├─ runs/                      # Runs: <run_id>/ (artifacts, run.yaml, manifest)
├─ configs/
│  ├─ example_nse_banks.yaml
│  └─ presets/
├─ tests/
├─ README.md
└─ requirements*.txt
``` 

## Quickstart (developer flow)

``` bash
clone & create env
git clone <repo>
cd pairs-trading
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```


### CLI backtest (example)
Run a sample backtest (Streamlit UI is default MVP)

``` bash
python src/main.py --config configs/example_nse_banks.yaml
```

### run Streamlit UI (MVP)

``` bash
cd src
streamlit run app/streamlit_app.py
open http://localhost:8501
```

### Inspect run artifacts:

runs/<YYYYMMDD_HHMMSS_hash>/
  ├─ trades.csv
  ├─ equity.png
  ├─ pnl_by_pair.png
  ├─ summary.json
  ├─ run.yaml
  └─ models/

## Example RunConfig (pydantic-validated YAML)

Use this as the UI⇄backend contract. Backend will compute a deterministic run.id from the config hash.
``` bash
run:
  id: null
  seed: 42
  output_dir: runs/
  gpu: true

data:
  source: yfinance
  tickers: ["HDFCBANK.NS","ICICIBANK.NS","SBIN.NS","PNB.NS"]
  start: "2018-01-01"
  end: "2025-09-01"
  cache: true

features:
  window: 60
  min_periods: 60
  use_gpu: true

selection:
  ensemble:
    name: "rank_avg"
    members:
      - { algo: "cointegration", params: { adf_p: 0.05, half_life_min: 3, half_life_max: 250 } }
      - { algo: "distance_gatev", params: { lookback: 252 } }
      - { algo: "cluster_partial_corr", params: { pca_k: 5, min_cluster: 2 } }
  daily_top_k: 5

trading:
  ensemble:
    name: "agreement"
    members:
      - { algo: "zscore_bands", params: { entry_z: 1.5, exit_z: 0.3, max_holding: 20, capital_per_leg: 10000 } }
      - { algo: "lstm_forecast", params: { horizon: 1, lookback: 120, hidden: 64, epochs: 25, batch: 64 } }

backtest:
  capital: 1000000
  parallel_pairs: true
  costs_bps: 10
  slippage_bps: 5

evaluate:
  metrics: ["cum_return","sharpe","max_dd","win_rate"]
  table_by: ["selection.ensemble","trading.ensemble"]
```

Interfaces / Contracts (short)

These are the canonical plugin interfaces (implementations live in selection/ and trading/):

SelectionAlgo
```bash
class SelectionAlgo:
    name: str
    def fit(self, prices: pd.DataFrame, features: pd.DataFrame | None) -> "SelectionAlgo": ...
    def select(self, asof_date: pd.Timestamp, universe: list[str]) -> list[tuple[str,str]]: ...
    def diagnostics(self) -> dict: ...  # optional
```

TradingAlgo
```bash
class TradingAlgo:
    name: str
    def fit(self, prices: pd.DataFrame, pair: tuple[str,str], **ctx) -> "TradingAlgo": ...
    def signals(self, pair: tuple[str,str], history: pd.DataFrame) -> pd.DataFrame
    def trade_once(self, state, market_row) -> dict  # optional
```

Ensemble
```bash
class Ensemble:
    name: str
    def combine(self, outputs: list, meta: dict) -> any
    # selection: returns ranked pairs
    # trading: returns final signal/size
```

Registry pattern
* core/registry.py exposes register_selection, register_trading, register_ensemble.
* Plugins annotate/register on import; UI queries registries for cards/forms.

## MVP: prioritized order of work (files / units to implement)

Implement in this order so devs can iterate and test end-to-end quickly:

1. core/registry.py + core/types.py (interfaces + simple plugin loader)

2. io/run_config.py (pydantic schemas + config hashing → deterministic run_id)

3. data/loader.py (yfinance downloader + Parquet caching) & data/universe.py

4. data/features.py (day-based features; GPU toggle placeholder)

5. selection/ — baseline algos:
    correlation_topk.py
    distance_gatev.py
    cointegration.py (Engle-Granger)

6. trading/ — baseline algos:

    zscore_bands.py (statistical baseline)

    ou_thresholds.py (half-life aware)

    rf_classifier.py (simple ML baseline)

7. ensemble/rank_avg.py and ensemble/agreement.py

8. backtest/runner.py + backtest/positioning.py (loop wiring + trades.csv output)

9. eval/metrics.py, eval/plots.py, eval/tables.py

10. app/streamlit_app.py (MVP), src/main.py CLI wrapper

11. LSTM model (GPU) as optional MVP stage if GPU available: trading/lstm_forecast.py

12. Tests & CI for the above

### Defaults & decisions (MVP choices baked in)

* UI stack: Streamlit for MVP (fast dev). Next.js + FastAPI recommended for production UX.

* DL models: include LSTM as the first GPU model for stage-2 in MVP. Add Transformers/TCN later.

* Portfolio mode: Top-K per day (equal notional per pair) for MVP.

* Costs & slippage defaults: 10 bps cost per leg, 5 bps slippage.

* Sizing: fixed-notional in MVP; add volatility-targeted sizing later.

* Data cadence: day-based initially; design features & runner to be date-iterable so intraday fits later.

### Reproducibility & artifacts

Each run must save:

* run.yaml (full resolved config), manifest.json (git commit, python pkg versions, seed, GPU info)

* trades.csv (timestamp, pair, action, size, price, pnl)

* plots: equity.png, pnl_by_pair.png, timeline.png

* saved models in models/<run_id>/ (joblib or torch)

* summary.json with top-level metrics (Sharpe, cum return, maxdd, win_rate)

Deterministic run_id = sha1(run_config_yaml + git_commit + seed).

### Plugin dev guide (how to add a new algo)

1. Implement algorithm class following interface in selection/ or trading/.

2. Add a small params_schema dataclass / pydantic model for UI-driven forms.

3. Register with @register_selection or @register_trading (adds metadata: name, gpu_capable, default_params).

4. Add unit tests in tests/ that run the algo on a small synthetic dataset.

5. Add example entry in configs/presets/.


### Notes about intraday / future work

* Design data/loader & features to accept freq param (day/1min/1s). MVP: day only.

* Backtest runner should be time-step agnostic — iterate over ordered timestamps. For intraday, ensure event ordering and partial fills/slippage model improvements.

* For GPU: keep RAPIDS wrappers behind gpu/ so code falls back to Pandas/Numpy when unavailable.


### Run CLI backtest: 
```bash
python src/main.py --config configs/example_nse_banks.yaml
```

### Start Streamlit: 
```bash
streamlit run src/app/streamlit_app.py
```