# GAP_ANALYSIS.md

Tracks planned architecture (from DevNotes.md) against implementation status. Update as work progresses.

**Legend:** [OK] Done |  Partial | [Error] Missing

---

## Section 1 — Data Ingestion

| # | Component | Status | Notes |
|---|---|---|---|
| 1.1 | Market Data (yfinance, CSV/Parquet upload) | [OK] | `core/data.py` — `YFinanceNSESource`, `CSVUploadSource` |
| 1.2 | Alternative / Fundamental Data | [Error] | Not started |
| 1.3 | Data Storage / Caching | [Error] | No Parquet cache; all data re-fetched every run |

---

## Section 2 — Data Processing

| # | Component | Status | Notes |
|---|---|---|---|
| 2.1 | Data Cleaning | [OK] | ffill, dropna, NaN/Inf handling |
| 2.2 | Normalization / Alignment | [OK] | DatetimeIndex normalization, frequency resampling |
| 2.3 | Spread Construction | [OK] | Unit hedge (A−B) and rolling-OLS beta hedge in `entry.py` |
| 2.4 | Feature Engineering |  | Only exists inside `MLSelector._pair_features()`; no shared feature pipeline |
| 2.5 | Regime Detection |  | Basic vol + correlation regime in `predictions.py`; no formal HMM/Markov model |
| 2.6 | Data Storage | [Error] | No dedicated data layer; reports saved, raw prices not cached |

---

## Section 3 — Candidate Pair Generation

| # | Component | Status | Notes |
|---|---|---|---|
| 3.1 | Universe Filtering (liquidity/ADV) |  | `estimate_adv()` exists in `data.py` but never wired into the pipeline |
| 3.2 | Similarity / Sector Screening | [Error] | Not started |
| 3.3 | Candidate Pair Creation |  | Brute-force combinatorial only; no principled pre-screening |

---

## Section 4 — Pairs Selection Ensemble

| # | Component | Status | Notes |
|---|---|---|---|
| 4.1.1 | Correlation Selector (Rolling Pearson) | [OK] | `CorrelationSelector` in `selectors.py` |
| 4.1.2 | Distance Selector (Gatev et al. 2006) | [OK] | `DistanceSelector` in `selectors.py` |
| 4.1.3 | Cointegration Selector (Engle-Granger + ADF) | [OK] | `CointegrationSelector` in `selectors.py` |
| 4.1.4 | Combined Criteria Selector (Cointegration + Hurst) | [OK] | `CombinedCriteriaSelector` in `selectors.py` |
| 4.2.1 | XGBoost Selector | [OK] | `MLSelector` in `selectors.py` (XGBoost primary, GB fallback) |
| 4.2.2 | Gradient Boosting Selector | [OK] | Fallback inside `MLSelector` |
| 4.3.1 | Graph Neural Network Selector | [OK] | `GNNSelector` in `core/selectors.py`. Two-layer GCN with correlation-weighted adjacency, 6 node features, multi-snapshot training via GradientTape, link-prediction head [hᵢ‖hⱼ‖hᵢ⊙hⱼ]. Sources: Kipf & Welling (2017), Zhang & Chen (2018), Matsunaga et al. (2019). |
| 4.3.2 | LSTM / BiLSTM Selector | [OK] | `LSTMSelector` in `core/selectors.py`. 6-feature multivariate time series, sliding-window training, BiLSTM toggle. Sources: Hochreiter & Schmidhuber (1997), Schuster & Paliwal (1997), Fischer & Krauss (2018). |
| 4.3.3 | Transformer Selector | [OK] | `TransformerSelector` in `core/selectors.py`. Multi-head self-attention encoder, sinusoidal positional encoding, GlobalAveragePooling1D head. Sources: Vaswani et al. (2017), Zerveas et al. (2021), Wen et al. (2023). |
| 4.4 | Ensemble Scoring / Ranking | [OK] | `ensemble_pair_scores()` in `ensemble.py` — weighted average |

---

## Section 5 — Signal Generation / Trading Models

| # | Component | Status | Notes |
|---|---|---|---|
| 5.1a | Z-Score Threshold Model | [OK] | `ZScoreThreshold` in `entry.py` |
| 5.1b | OU Model | [OK] | `OUThreshold` in `entry.py` |
| 5.1c | Kalman Filter Hedge | [OK] | Real state-space KF implemented in `KalmanHedge` (`core/entry.py`). State = [β, α], random-walk transition, time-varying H_t. Sources: Kalman (1960), Elliott et al. (2005), Pole (2007). |
| 5.2 | ML Signal Models | [OK] | `MLSignal` in `core/entry.py`. 11 spread features (z-score, lags, velocity, momentum, corr, vol_ratio), triclass label (+1/0/−1) from forward spread return, XGBoost primary ({-1,0,1}→{0,1,2} label remapping) / sklearn GBM fallback / ZScore fallback. Sources: Friedman (2001), Chen & Guestrin (2016), Krauss et al. (2017). |
| **5.3** | **Reinforcement Learning Models** | [Error] | Not started (e.g., PPO/DQN agent on spread environment) |
| 5.4 | Signal Ensemble / Meta-Decision Layer | [OK] | `ensemble_signals()` in `ensemble.py` |

---

## Section 6 — Execution & Trade Management

| # | Component | Status | Notes |
|---|---|---|---|
| 6.1 | Entry / Exit Logic | [OK] | Signal-based execution in `backtest.py` |
| 6.2 | Position Sizing |  | Fixed notional per pair only; volatility-targeted sizing not implemented |
| 6.3 | Transaction Cost & Slippage | [OK] | `IndianCosts` dataclass in `backtest.py` with full NSE cost model |
| 6.4 | Order Execution Logic (live) | [Error] | No broker integration; Streamlit-only |

---

## Section 7 — Portfolio & Risk Management

| # | Component | Status | Notes |
|---|---|---|---|
| 7.1 | Portfolio Optimization | [Error] | Equal weighting only; no mean-variance or risk-parity optimization |
| 7.2 | Exposure Control |  | `max_concurrent_pairs`, `per_trade_cap` in `BacktestConfig` |
| 7.3 | Risk Limits / Stop Rules |  | Soft z-score stop-loss only; no hard stops, drawdown limits, or VaR |
| **7.4** | **Performance Attribution** | [Error] | Not started — critical for thesis; need to decompose returns by model contribution |

---

## Section 8 — Backtesting & Evaluation

| # | Component | Status | Notes |
|---|---|---|---|
| 8.1 | Historical Simulation | [OK] | `backtest_pairs()` in `backtest.py` — vectorized, gross + net equity |
| **8.2** | **Walk-Forward Validation** | [Error] | Not started — major academic gap; single in-sample backtest is not credible for a thesis |
| **8.3** | **Stress Testing** | [Error] | Not started |
| 8.4 | Benchmark Comparison | [OK] | `BenchmarkComparison` in `reports.py` — 7 Indian indices |

---

## Section 9 — Deployment & Monitoring

| # | Component | Status | Notes |
|---|---|---|---|
| 9.1 | Production Deployment |  | Streamlit MVP only |
| 9.2 | Live Monitoring / Predictions |  | `PredictionEngine` in `predictions.py` — functional but basic |
| 9.3 | Drift Detection | [Error] | Not started |
| 9.4 | Retraining / Maintenance | [Error] | Not started |

---

## Experiment Framework (`experiments/`)

All reproducible experiment scripts live in `experiments/`. Results are written to `experiments/results/` as JSON. Do **not** use the Streamlit `reports/` directory for thesis experiments — those are interactive-session artefacts.

| Script | Status | Purpose |
|---|---|---|
| `experiments/config.py` | [OK] | Single source of truth — universe (35 NSE tickers, 8 sectors), date ranges, weights, seeds |
| `experiments/freq_comparison.py` | [OK] | Daily vs hourly: Sharpe, cost drag, Hurst, signal reversal rate |
| Walk-forward validation script | [Error] | Rolling OOS folds — see Phase 3 |
| Ablation / attribution script | [Error] | Per-model isolation runs — see Phase 3 |

Run: `python experiments/freq_comparison.py --mode stat_ml`

---

## Build Priority

### Phase 1 — Complete the Hybrid Core (thesis contribution)
1. [OK] **Real Kalman Filter** — state-space KF with dynamic [β, α] state, implemented in `core/entry.py`
2. [OK] **LSTM / BiLSTM Selector** — `LSTMSelector` in `core/selectors.py`; BiLSTM toggle, 6 temporal features, sliding-window training
3. [OK] **Transformer Selector** — `TransformerSelector` in `core/selectors.py`; sinusoidal positional encoding, stacked multi-head self-attention encoder blocks
4. [OK] **GNN Selector** — `GNNSelector` in `core/selectors.py`; GCN with GradientTape training, inductive weights applicable to any universe size

### Phase 2 — Signal Layer Completion
5. [OK] **ML Signal Model** — `MLSignal` in `core/entry.py`; XGBoost / GBM triclass classifier on 11 spread features
6. [Error] **RL Signal Model** — PPO/DQN agent trained on spread environment

### Phase 3 — Academic Rigour
7. [Error] **Walk-Forward Validation** — rolling train/test windows; without this backtests lack credibility for a thesis
8. [Error] **Performance Attribution** — decompose returns by model contribution; show what the ensemble gains vs. each component alone

### Phase 4 — Pipeline Hardening
9.  **Universe Filtering** — wire `estimate_adv()` into pair generation; add sector-based screening
10. [Error] **Feature Engineering Module** — extract shared `core/features.py` used by ML selector and ML signal model
11.  **Volatility-Targeted Position Sizing** — replace fixed notional
12. [Error] **Data Cache** — Parquet cache layer to avoid re-fetching on every run

### Phase 5 — Post-Thesis / Deployment
13. [Error] Drift Detection
14. [Error] Live order execution / broker integration
15. [Error] Retraining pipeline

---

## Code Quality & Technical Debt

Issues found during codebase audit (March 2026). Tracked here so they don't get lost.

### Fixed
| # | Issue | Fix Applied |
|---|---|---|
| Q1 | `backtest.py` had duplicate `BacktestResult` dataclass — first definition silently overridden by second | Removed first definition; single canonical `BacktestResult` with `equity_gross / equity_net / pnl_gross / pnl_net / turnover / trades / metrics / params` |
| Q2 | `backtest.py` had duplicate import block (`from dataclasses import ...` etc.) and dead comment scaffold from an old copy-paste | Removed entire duplicate block |
| Q3 | `backtest.py` `_annualize_sharpe()` function defined but never called — `_metrics_from_pnl()` does its own inline Sharpe | Removed dead function |
| Q4 | `predictions.py` imported `streamlit as st` but `st` was never used — architecture violation (core modules must be UI-agnostic) | Removed import |
| Q5 | `predictions.py` used `print()` for all logging (8 call sites) | Replaced with `logging` (`_log.warning`, `_log.error`, `_log.debug`) |
| Q6 | `predictions.py` had 2× bare `except:` (swallows all errors silently including KeyboardInterrupt) | Changed to `except Exception:` |
| Q7 | `predictions.py` had stale `import traceback` + `traceback.print_exc()` debugging artifact | Removed |
| Q8 | `reports.py` re-imported `yfinance as yf` inside `fetch_index_returns()` despite top-level import | Removed redundant inner import |
| Q9 | `reports.py` used `print()` for logging | Replaced with `_log.warning` |
| Q10 | `selectors.py` is 1,400+ lines — all 8 selectors in one file | Split into `selectors_statistical.py` + `selectors_ml.py` + `selectors_base.py` with re-export |

### Known / Pending
| # | Issue | Severity | Notes |
|---|---|---|---|
| Q11 | `_zscore()` helper defined independently in `entry.py` and `backtest.py` (slightly different `min_periods` / denominator guard) | Low | Both versions intentionally different; document the difference or extract to `core/_math.py` |
| Q12 | `MLSelector._pair_features()` and `MLSignal._build_features()` compute overlapping spread features independently | Medium | Blocked on Phase 4 item 10 (shared `core/features.py`) |
| Q13 | Selector loop in `app.py` and `predictions.py` runs 8 selectors sequentially — with 50 stocks this can take 4+ minutes | High | Add `concurrent.futures.ThreadPoolExecutor` around selector loop; each selector is independent |
| Q14 | Cointegration selector runs a full Engle-Granger test per pair — O(n²) pairs × O(T) per test | High | Cache results keyed on `(pair, data_hash)`; or pre-screen with rolling-correlation filter before cointegration |
| Q15 | `BenchmarkComparison.fetch_index_returns()` downloads from Yahoo on every call; no caching | Medium | Add `@functools.lru_cache` or session-level cache keyed on `(index_name, start, end)` |
| Q16 | Protobuf version warning spam at startup (TF 2.20 + protobuf 6.x mismatch) | Low | `pip install protobuf==5.29.4` pins to compatible version; cosmetic only |
| Q17 | `estimate_adv()` downloads a second yfinance batch independently from `get_prices()` | Medium | Pass already-downloaded price frame into `estimate_adv()` instead of re-downloading |

---

## Suggestions — Understanding & Thesis Validation

Things that would most improve thesis credibility and codebase understanding, in priority order.

### S1 — Walk-Forward Validation (critical for thesis)
Single in-sample backtest is the biggest academic gap. Implement rolling out-of-sample evaluation:
- Split data into expanding windows (e.g. train on years 1–3, test on year 4; roll forward)
- Re-fit all selectors and signal models on each training window
- Report OOS Sharpe / Return / MaxDD — these are the numbers the thesis reviewers will scrutinize
- Maps to **Section 8.2** in the architecture table

### S2 — Performance Attribution (core thesis contribution)
To justify the ensemble, you must show it outperforms any single model alone. Concrete approach:
- Run `backtest_pairs()` once per model with that model's weight = 1.0 and all others = 0.0
- Compare Sharpe ratio: Correlation-only vs Distance-only vs Cointegration-only vs ... vs Full Ensemble
- Same for Stage 2: ZScore-only vs OU-only vs Kalman-only vs ML-only vs Full Ensemble
- Show ensemble Sharpe > best individual model Sharpe
- Maps to **Section 7.4**

### S3 — Universe Filtering (wire `estimate_adv()`)
`data.py` has `estimate_adv()` fully implemented but it is never called. With 50 stocks the combinatorial pair count is 1,225 — all fed to 8 selectors. Filtering to the top-30 by liquidity (ADV) cuts this to 435 pairs, roughly 3× faster. Steps:
- Call `estimate_adv()` after `get_prices()` in `app.py` and `predictions.py`
- Drop tickers below a configurable ADV threshold before generating candidates
- Maps to **Section 3.1**

### S4 — Shared Feature Engineering (`core/features.py`)
`MLSelector` and `MLSignal` both compute spread z-score, correlation windows, vol-ratio, momentum independently. Before adding the RL agent (Phase 2 item 5.3) extract these to a shared module:
- `build_pair_features(a, b, lookback) -> pd.DataFrame`
- Used by `MLSelector`, `MLSignal`, and future RL state encoder
- Maps to **Section 2.4** and **Phase 4 item 10**

### S5 — RL Signal Model (next implementation item)
The remaining [Error] in Phase 2. Suggested approach:
- Environment: state = feature vector from `build_pair_features()` at bar t; action = {−1, 0, +1}; reward = realized PnL net of costs
- Agent: PPO (stable-baselines3) is easiest to get working; DQN as alternative
- Training: same temporal split as LSTM (last 252 bars held out)
- Integrate as `RLSignal` class implementing `EntryExitModel` interface (`fit()` trains the agent; `trade_signals()` runs inference)
- Maps to **Section 5.3**

### S6 — Parallelise Selector Loop
Currently selectors run sequentially. With GNN + LSTM + Transformer each taking 10–30s on CPU, the full Stage 1 pass can exceed 4 minutes. Each selector is stateless during `score_pairs()` so parallelisation is safe:
```python
from concurrent.futures import ThreadPoolExecutor
with ThreadPoolExecutor(max_workers=4) as ex:
    futures = {name: ex.submit(sel.fit(prices).score_pairs, prices, candidates)
               for name, sel in selectors.items()}
    scores_by_model = {name: f.result() for name, f in futures.items()}
```
- Maps to **Q13** in Code Quality table above

### S7 — `selectors.py` Split (maintainability)
At 1,400+ lines the file is difficult to navigate. Proposed split:
- `core/selectors_base.py` — `Pair`, `PairScore`, `PairSelector` base class
- `core/selectors_statistical.py` — `CorrelationSelector`, `DistanceSelector`, `CointegrationSelector`, `CombinedCriteriaSelector`
- `core/selectors_ml.py` — `MLSelector`, `LSTMSelector`, `TransformerSelector`, `GNNSelector`
- `core/selectors.py` — re-exports everything for backward compatibility (`from .selectors_base import *` etc.)
No changes to any other file needed.
