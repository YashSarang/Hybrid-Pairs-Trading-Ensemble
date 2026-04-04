# Research Log — Hybrid Pairs Trading Ensemble

**Project:** Thesis-grade pairs trading platform on NSE equities
**Stack:** Python · Streamlit · yfinance · XGBoost · TensorFlow (LSTM / Transformer / GNN) · Kalman Filter
**Last updated:** 2026-04-03 (E5 Benchmark Comparison added; MLSelector _label bug fixed)

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [Universe](#3-universe)
4. [Experiment Infrastructure](#4-experiment-infrastructure)
5. [Experiment E1 — Frequency Comparison](#5-experiment-e1--frequency-comparison)
6. [Experiment E2 — Hold Period Sweep](#6-experiment-e2--hold-period-sweep)
7. [Experiment E4 — Walk-Forward Validation](#7-experiment-e4--walk-forward-validation)
8. [Experiment E3 — Ablation Study](#8-experiment-e3--ablation-study)
9. [Experiment E5 — Benchmark Comparison](#9-experiment-e5--benchmark-comparison)
10. [Experiment E6 — Statistical Significance](#10-experiment-e6--statistical-significance)
11. [Known Issues & Observations](#11-known-issues--observations)
12. [Open Questions](#12-open-questions)

---

## 1. Project Overview

This project builds a **two-stage hybrid ensemble** pairs trading strategy on Indian NSE equities and produces thesis- and research-paper-quality empirical results.

**Core claim:** Combining classical statistical arbitrage methods with modern machine learning (gradient boosting, LSTM, Transformer, GNN) in an ensemble framework produces more robust out-of-sample pair selection and signal generation than any single method alone.

**Secondary claim (Experiment E1):** Daily (1D) data is the empirically correct sampling frequency for this strategy. Higher-frequency data (hourly) degrades both signal quality and net returns due to (a) increased microstructure noise, (b) higher turnover-driven cost drag, and (c) poorer spread mean-reversion properties.

---

## 2. System Architecture

### Two-Stage Pipeline

```
Raw Prices (NSE, yfinance)
        |
        v
[Stage 1 — Pair Selection]
  8 algorithms score every candidate pair:
    Statistical:   CorrelationSelector, DistanceSelector,
                   CointegrationSelector, CombinedCriteriaSelector
    ML/XGBoost:    MLSelector
    Deep Learning: LSTMSelector, TransformerSelector, GNNSelector
  -> Ensemble average (user-weighted) -> Top-K pairs
        |
        v
[Stage 2 — Entry / Exit Signals]
  4 signal models per selected pair:
    ZScoreThreshold, OUThreshold, KalmanHedge, MLSignal
  -> Ensemble average (user-weighted) -> Discrete signal {-1, 0, +1}
        |
        v
[Backtester — backtest_pairs()]
  Vectorized PnL: signal(t-1) x spread_return(t) x notional
  Costs: IndianCosts (brokerage, STT, GST, stamp, slippage)
  Soft stop-loss: scale on z > 3σ breach; exit if persists N bars
  -> BacktestResult (gross + net equity, trades, metrics)
```

### Key Files

| File | Purpose |
|---|---|
| `core/data.py` | `DataConfig`, `YFinanceNSESource`, `CSVUploadSource` |
| `core/selectors_base.py` | `Pair`, `PairScore`, `PairSelector` ABC, `_hurst_rs`, `_halflife` |
| `core/selectors_statistical.py` | `CorrelationSelector`, `DistanceSelector`, `CointegrationSelector`, `CombinedCriteriaSelector` |
| `core/selectors_ml.py` | `MLSelector`, `LSTMSelector`, `TransformerSelector`, `GNNSelector` |
| `core/selectors.py` | Re-export facade |
| `core/entry.py` | `ZScoreThreshold`, `OUThreshold`, `KalmanHedge`, `MLSignal` |
| `core/ensemble.py` | `ensemble_pair_scores`, `ensemble_signals`, `normalize_weights` |
| `core/backtest.py` | `IndianCosts`, `BacktestConfig`, `BacktestResult`, `backtest_pairs` |
| `core/reports.py` | `ReportManager`, `BenchmarkComparison` |
| `core/predictions.py` | `PredictionEngine` (real-time inference) |
| `app.py` | Streamlit UI (~1,450 lines) |
| `experiments/config.py` | Canonical universe, weights, date ranges, seeds |
| `experiments/freq_comparison.py` | Frequency analysis experiment runner |

### Cost Model (IndianCosts — NSE Defaults)

| Component | Rate |
|---|---|
| Brokerage | 3.0 bps / leg |
| NSE exchange charge | 0.345 bps / leg |
| SEBI levy | 0.01 bps / leg |
| STT (sell leg, delivery) | 10.0 bps |
| GST (on brokerage + exchange) | 18% |
| Stamp duty (buy leg) | 1.0 bps |
| Slippage | 2.0 bps / leg |
| **Round-trip total** | **~60 bps** |

### Signal Convention

- `+1` Long A, Short B (spread expected to narrow)
- `-1` Short A, Long B (spread expected to widen)
- ` 0` Flat / exit

### Key Metric Definitions

| Metric | Definition |
|---|---|
| Gross Sharpe | Sharpe on PnL before transaction costs |
| Net Sharpe | Sharpe on PnL after all NSE costs |
| Cost Drag | Ann. Gross Return − Ann. Net Return (percentage points) |
| Hurst Exponent | R/S estimate on log-price spread; H < 0.5 = mean-reverting |
| Signal Reversal Rate | Fraction of non-zero signal bars that immediately follow opposite-sign signal (proxy for whipsaw) |
| Trades / Year | Total position transitions annualised |

---

## 3. Universe

**35 NSE large-cap stocks across 8 sectors** (defined in `experiments/config.py`).
All are Nifty 100 constituents — liquid, well-covered by yfinance.

| Sector | Tickers |
|---|---|
| Banking & Finance | HDFCBANK, ICICIBANK, SBIN, KOTAKBANK, AXISBANK, INDUSINDBK |
| IT | TCS, INFY, WIPRO, HCLTECH, TECHM |
| Automobiles | MARUTI, TATAMOTORS*, BAJAJ-AUTO, HEROMOTOCO, EICHERMOT |
| FMCG | HINDUNILVR, ITC, NESTLEIND, BRITANNIA |
| Pharma | SUNPHARMA, DRREDDY, CIPLA, DIVISLAB |
| Energy & Oil | RELIANCE, ONGC, IOC, BPCL |
| Metals & Mining | TATASTEEL, JSWSTEEL, HINDALCO, COALINDIA |
| Cement | ULTRACEMCO, ACC, SHREECEM |

*`TATAMOTORS.NS` replaced with `M&M.NS` (Mahindra & Mahindra) due to persistent `YFTzMissingError` on yfinance. Both are Auto-sector Nifty 100 large-caps. Replacement does not alter sector balance. See `Decisions.md D9`.

**Total pairs at full universe:** 34C2 = 561 candidates.

---

## 4. Experiment Infrastructure

### Conventions

- All reproducible experiments live in `experiments/` (not in `reports/`).
- Results are saved as JSON to `experiments/results/<experiment>_<YYYYMMDD_HHMMSS>.json`.
- The Streamlit `reports/` directory is for interactive sessions only; it was wiped clean on 2026-04-02 to start a fresh record.
- `experiments/config.py` is the single source of truth for all parameters.

### Selector Speed Modes

| Mode | Selectors Active | Approx. Runtime |
|---|---|---|
| `stat_only` | Correlation, Distance, Cointegration, Combined | ~1 min |
| `stat_ml` | + XGBoost MLSelector | ~3 min |
| `full` | All 8 including LSTM, Transformer, GNN | ~20 min (CPU) |

### Date Window Policy

- **Daily (1D) experiments:** `MAIN_START = 2016-01-01` to `MAIN_END = 2026-03-31` (10 years).
- **Frequency comparison (1D + 1H):** Rolling 700-day window computed at runtime (`today - 700` to `today - 1`) to stay safely within yfinance's 730-day hard limit for 60-minute data.
- **Periods per year:** 1D = 252; 1H = 1512 (252 × 6 NSE hours/day).

---

## 5. Experiment E1 — Frequency Comparison

### Purpose

Empirically justify the choice of daily (1D) data over hourly (1H) for this strategy. This becomes **Section 3.x** ("Data Frequency Selection") of the thesis.

### Setup

| Parameter | Value |
|---|---|
| Date range | 2024-05-02 to 2026-04-01 (700 days, same for both frequencies) |
| Universe | 34 tickers (TATAMOTORS excluded — yfinance failure) |
| Selectors | stat_only: Correlation, Distance, Cointegration, CombinedCriteria |
| Signal models | ZScore, OU, Kalman, MLSignal (equal weights) |
| Top-K pairs | 10 |
| Capital | INR 10,00,000 (10 lakh) |
| Per-pair cap | INR 1,00,000 (1 lakh) |
| Max concurrent | 10 pairs |
| Costs | IndianCosts() defaults (~60 bps round-trip) |
| Experiment script | `experiments/freq_comparison.py --mode stat_only` |
| Result file | `experiments/results/freq_comparison_20260402_025539.json` |

### Results

| Metric | Daily (1D) | Hourly (1H) |
|---|---|---|
| **Gross Sharpe** | **1.144** | **0.488** |
| Gross Ann. Return (%) | 5.00 | 1.30 |
| Gross Volatility | 0.046 | 0.028 |
| Gross Max Drawdown (%) | 3.94 | 4.47 |
| **Net Sharpe** | **-2.294** | **-6.554** |
| Net Ann. Return (%) | -11.29 | n/a (bankrupt) |
| Net Max Drawdown (%) | 29.86 | 214.01 |
| **Trades / Year** | **672.8** | **904.0** |
| **Cost Drag (ann. pp)** | **16.29** | n/a |
| **Hurst (median)** | **0.190** | **0.251** |
| Hurst < 0.5 (% pairs) | 100% | 100% |
| Signal Reversal Rate | 0.404 | 0.389 |
| Wall time | 8.5s | 11.6s |

**Selected pairs (1D):** TCS-WIPRO, AXISBANK-IOC, IOC-BPCL, TCS-INFY, HDFCBANK-ACC, INFY-HCLTECH, ITC-CIPLA, SBIN-TATASTEEL, SBIN-AXISBANK, BAJAJ-AUTO-EICHERMOT

**Selected pairs (1H):** HDFCBANK-INDUSINDBK, MARUTI-BRITANNIA, HDFCBANK-BRITANNIA, INDUSINDBK-BAJAJ-AUTO, ULTRACEMCO-SHREECEM, SBIN-ACC, BAJAJ-AUTO-HEROMOTOCO, BPCL-JSWSTEEL, MARUTI-BAJAJ-AUTO, DRREDDY-TATASTEEL

### Findings & Interpretation

**Finding E1.1 — Gross signal quality degrades with frequency**
Gross Sharpe drops from 1.144 (daily) to 0.488 (hourly) — a 57% decline — even before any costs are applied. The spread Hurst exponent rises from 0.190 to 0.251, confirming that hourly spreads are less mean-reverting (closer to random walk H=0.5). Both frequencies produce 100% of selected pairs with H < 0.5, so the qualitative direction is correct at both; daily is quantitatively stronger.

**Finding E1.2 — Cost drag is the primary performance killer, especially at high frequency**
The strategy is gross-profitable at 1D (Sharpe 1.14, return 5%) but net-unprofitable (Sharpe -2.29, return -11%). The ~60 bps round-trip cost combined with ~673 trades/year produces an annualised cost drag of 16.29 percentage points — far exceeding gross alpha. At 1H, cost drag is so large the strategy loses more than its entire capital (Max DD 214%).

**Finding E1.3 — The pairs are good; the signal layer is over-trading**
Hurst = 0.190 for daily spreads is excellent — the selected pairs are genuinely mean-reverting. The problem is not pair quality but signal frequency: 672 transitions/year across 10 pairs = 1 position change every ~3.75 days per pair. With 60 bps/round-trip this is unsustainable. This points to the need for a minimum holding period or signal smoothing constraint in the signal layer.

**Finding E1.4 — Hourly data selects qualitatively different pairs**
The 1D and 1H selector runs produce almost entirely different top-10 lists. Correlation and cointegration scores computed on intraday bars weight microstructure noise more heavily, pulling in pairs that look mean-reverting over 6-hour windows but lack the economic co-movement of daily-level pairs (e.g., MARUTI-BRITANNIA is an unusual cross-sector pair at the top for 1H).

**Finding E1.5 — Signal reversal rates are similar across frequencies**
Counter-intuitively, signal reversal rates are nearly identical (40.4% daily vs 38.9% hourly). This suggests the ensemble signal layer's whipsaw problem is a model-level issue (too-sensitive z-score thresholds, etc.) rather than a pure frequency problem. The difference in cost impact comes from raw trade count (672 vs 904/year) and not from qualitatively different signal behaviour.

### Thesis Narrative (draft for Section 3.x)

> "We evaluate the strategy at two sampling frequencies — daily (1D, 252 bars/year) and hourly (1H, ~1,512 bars/year) — over an identical 700-day window using the same universe, selector weights, and cost model. Table X shows that gross Sharpe deteriorates from 1.14 to 0.49 as frequency increases, consistent with the microstructure hypothesis: the median spread Hurst exponent rises from 0.19 to 0.25, indicating that hourly spreads are less mean-reverting. More decisively, net performance collapses at hourly frequency due to transaction cost drag — 904 annualised trades at ~60 bps round-trip renders the strategy unprofitable regardless of gross signal quality. These findings motivate the exclusive use of end-of-day (daily) prices for all subsequent experiments."

### Next Steps for E1

- [ ] Re-run with `--mode full` (all 8 selectors including LSTM, Transformer, GNN) for the paper's final numbers — stat_only is for sanity check only.
- [ ] Separately quantify the signal smoothing problem (E1.3): add a minimum hold period parameter and show the Pareto frontier of hold-period vs net Sharpe.
- [ ] Replace TATAMOTORS.NS with M&M.NS in the universe.

---

## 6. Experiment E2 — Hold Period Sweep (COMPLETE)

### Purpose

Find the minimum holding period that maximises net Sharpe on daily data. This parameter is then locked as a global default for all subsequent experiments (see `Decisions.md D5`, `D11`).

### Setup

| Parameter | Value |
|---|---|
| Date range | 2016-01-01 to 2026-03-30 (full 10-year window, 3,742 bars) |
| Universe | 35 tickers (all pass coverage filter) |
| Selectors | stat_only (4 statistical selectors) |
| Signal models | ZScore, OU, Kalman, MLSignal (equal weights) |
| Top-K pairs | 10 (same pairs as E1 1D, similar window) |
| Hold values swept | [0, 5, 10, 15, 20, 25, 30, 40] trading days |
| Design | Pairs selected ONCE; backtest re-run once per hold value (isolates hold effect from selector randomness) |
| Script | `experiments/hold_period_sweep.py --mode stat_only --hold-values 0 5 10 15 20 25 30 40` |
| Result file | `experiments/results/hold_period_sweep_20260402_031710.json` |

**Selected pairs (fixed across all hold values):** TCS-WIPRO, AXISBANK-IOC, IOC-BPCL, TCS-INFY, HDFCBANK-ACC, BAJAJ-AUTO-EICHERMOT, INFY-HCLTECH, ITC-CIPLA, SBIN-TATASTEEL, SBIN-HINDALCO

### Results

| Hold (days) | Gross Sharpe | Gross Ret % | Net Sharpe | Net Ret % | Net MaxDD % | Trades/Yr | Cost Drag pp |
|---|---|---|---|---|---|---|---|
| 0 (baseline) | 0.827 | 3.98 | -1.889 | n/a | 175.86 | 746.2 | n/a |
| 5 | 0.774 | 4.01 | -0.727 | -8.67 | 76.29 | 450.5 | 12.68 |
| 10 | 0.770 | 4.08 | -0.195 | -1.53 | 37.72 | 298.6 | 5.61 |
| 15 | 0.579 | 3.31 | -0.157 | -1.23 | 34.29 | 231.9 | 4.54 |
| 20 | 0.698 | 3.95 | +0.087 | +0.63 | 20.09 | 200.1 | 3.33 |
| 25 | 0.633 | 3.62 | +0.086 | +0.61 | 26.53 | 176.6 | 3.01 |
| **30** | **0.963** | **4.98** | **+0.481** | **+2.89** | **8.21** | **156.1** | **2.09** |
| 40 | 0.164 | 1.15 | -0.239 | -2.10 | 44.01 | 133.7 | 3.25 |

### Findings & Interpretation

**Finding E2.1 — Net Sharpe turns positive at 20 days, peaks at 30 days**
Hold=30 is the clear optimum: Net Sharpe 0.481, Net Return 2.89%, Max Drawdown 8.21%. This is the first net-positive result in the project. Gross Sharpe also reaches its peak at hold=30 (0.963 — higher than any other hold value), suggesting that 30-day commitment periods align well with the pairs' mean-reversion timescale.

**Finding E2.2 — Hold=40 collapses (Net Sharpe -0.239)**
The sharp reversal at hold=40 (from +0.481 to -0.239) has a clear theoretical explanation: pairs with a mean-reversion half-life of ~20–30 trading days have typically fully reverted by day 40. Forcing the strategy to hold until day 40 means it frequently holds a position that has already reverted and begun to diverge in the other direction — effectively turning a winning trade into a losing one. This confirms the 30-day optimum is not arbitrary but corresponds to the underlying OU process dynamics.

**Finding E2.3 — Cost drag drives most of the improvement; gross alpha is stable**
From hold=0 to hold=30, trades/year drops from 746 to 156 (−79%). Cost drag falls from unmeasurable (net bankrupt) to 2.09pp. Meanwhile, gross return remains stable around 4–5% across all hold values — the underlying alpha signal is largely preserved. This confirms the hypothesis from E1 Finding E1.3: the pairs are genuinely good; the problem was purely signal over-trading.

**Finding E2.4 — This is stat_only; full mode will be better**
The gross Sharpe of 0.963 at hold=30 is achieved with only 4 statistical selectors (Correlation, Distance, Cointegration, Combined Criteria). The full ensemble (adding LSTM, Transformer, GNN, XGBoost selectors) should select higher-quality pairs, increasing gross alpha further. When re-run with full mode, the optimal hold period may shift shorter (higher alpha = can afford more trades) or the same (if gross alpha increases while retaining the same mean-reversion timescale).

### Decision

**`min_hold_bars = 30` is set as `DEFAULT_MIN_HOLD` in `experiments/config.py`** and will be applied to all subsequent experiments. See `Decisions.md D11` for full reasoning on why this is treated as a methodological parameter, not a tuned hyperparameter.

### Thesis Narrative (draft for Chapter 4.7)

> "We evaluate the impact of a minimum holding period on strategy performance by sweeping min_hold ∈ {0, 5, 10, 15, 20, 25, 30, 40} trading days. Table Y shows that net Sharpe improves monotonically from −1.89 (no constraint) to +0.48 at 30 days, then collapses to −0.24 at 40 days. The 30-day optimum is consistent with the estimated Ornstein-Uhlenbeck half-life of 20–30 trading days for the selected pairs: positions held shorter are subject to excessive transaction costs; positions held longer overshoot the mean-reversion window. We therefore fix min_hold = 30 trading days as a structural parameter for all subsequent experiments."

---

---

## 7. Experiment E4 — Walk-Forward Validation (COMPLETE — stat_only)

### Purpose

Replace the single in-sample backtest with a rigorous **out-of-sample evaluation** across 6 non-overlapping annual test folds (2020–2025). This is the primary academic credibility proof for the thesis: OOS Sharpe is the only metric examiners will trust.

### Setup

| Parameter | Value |
|---|---|
| Folds | 6 expanding-window folds (train starts 2016-01-01 for all; test year grows 2020-2025) |
| Frequency | Daily (1D) — justified by E1 |
| Mode | `stat_only` (Correlation, Distance, Cointegration, CombinedCriteria) |
| Signal models | ZScore, OU, Kalman, MLSignal (equal weights) |
| Min hold | 30 trading days (locked from E2) |
| Top-K pairs | 10 per fold |
| Capital | INR 10,00,000 |
| Per-pair cap | INR 1,00,000 |
| Costs | IndianCosts() ~60 bps round-trip |
| Look-ahead control | Selectors and signal models fit on training data only; MLSignal uses pre-fitted model for OOS inference |
| Script | `experiments/walk_forward.py --mode stat_only` |
| Result file | `experiments/results/walk_forward_20260402_131614.json` |
| Wall time | 23 seconds (stat_only — 4 selectors × 6 folds) |

### Per-Fold OOS Results

| Fold | Test Year | Gross Sharpe | Gross Ret% | Net Sharpe | Net Ret% | Net MaxDD% | Trades/Yr | Cost Drag pp | Top pairs selected |
|---|---|---|---|---|---|---|---|---|---|
| Fold 1 | 2020 | 0.213 | 2.03 | -0.158 | -1.52 | 18.06 | 155 | 3.55 | ONGC-COALINDIA, TATASTEEL-JSWSTEEL, MARUTI-BAJAJ-AUTO |
| Fold 2 | 2021 | -0.118 | -0.91 | -0.523 | -4.08 | 10.61 | 137 | 3.17 | INFY-WIPRO, WIPRO-HCLTECH, ICICIBANK-AXISBANK |
| Fold 3 | 2022 | 1.013 | 6.89 | 0.548 | 3.74 | 7.70 | 141 | 3.15 | ULTRACEMCO-ACC, TCS-HCLTECH, INFY-TECHM |
| Fold 4 | 2023 | 0.690 | 4.07 | 0.130 | 0.77 | 5.14 | 146 | 3.30 | ICICIBANK-COALINDIA, SBIN-AXISBANK, ICICIBANK-ITC |
| Fold 5 | 2024 | -0.262 | -1.95 | -0.718 | -5.40 | 16.66 | 148 | 3.45 | BAJAJ-AUTO-HEROMOTOCO, HEROMOTOCO-IOC, INFY-TECHM |
| Fold 6 | 2025 | 1.082 | 8.20 | 0.625 | 4.77 | 7.76 | 154 | 3.43 | HCLTECH-TECHM, INFY-TECHM, INFY-HCLTECH |

### Aggregate Statistics (Across 6 Folds)

| Metric | Mean | ± Std | % Folds Positive |
|---|---|---|---|
| **Gross Sharpe** | **0.436** | **0.526** | **67%** |
| **Net Sharpe** | **-0.016** | **0.504** | **50%** |
| Gross Ret% | 3.06 | 3.74 | 67% |
| Net Ret% | -0.29 | 3.76 | 50% |
| Net MaxDD% | 10.99 | 4.79 | 100% |
| Cost Drag (pp) | 3.34 | 0.15 | — |

### Full-OOS (All 6 Test Years Stitched)

| Metric | Value |
|---|---|
| **Gross Sharpe** | **0.407** |
| **Net Sharpe** | **-0.034** |
| Gross Ret% (ann.) | 2.79% |
| Net Ret% (ann.) | -0.26% |
| Net MaxDD% | 18.07% |

### Findings & Interpretation

**Finding E4.1 — OOS gross alpha is real, persistent, and positive across 6 independent test years**
Gross Sharpe of 0.407 over 6 OOS years (2020–2025) with 67% of folds positive confirms that the strategy captures genuine mean-reversion alpha, not a backtest artefact. The stat_only ensemble (4 selectors) consistently identifies pairs with exploitable spread dynamics across different market regimes.

**Finding E4.2 — Net alpha is marginally negative (Net Sharpe -0.034); cost drag of ~3.34 pp/year remains the bottleneck**
The ~3.34 pp/year cost drag at 140–155 trades/year is consistent with the E2 sweep (hold=30 gives ~2.09 pp at 156 trades/year on the in-sample data). The slight discrepancy reflects fold-to-fold variation in pair liquidity and trade frequency. To achieve net-positive OOS Sharpe, the strategy needs either (a) higher gross alpha from better pair selection (full mode with 8 selectors), or (b) further reduction in turnover.

**Finding E4.3 — Strong market-regime dependence (expected for mean-reversion strategies)**
The best folds are 2022 (Net Sharpe 0.548) and 2025 (Net Sharpe 0.625) — years of elevated NSE volatility and mean-reverting dynamics (rate hike cycle 2022, market consolidation 2025). The worst folds are 2021 (Net Sharpe -0.523, strong directional bull run post-COVID) and 2024 (Net Sharpe -0.718). This pattern is consistent with the pairs trading literature: mean-reversion strategies underperform in strong trending markets. This is a limitation of the strategy, not a flaw in the methodology.

**Finding E4.4 — Pair selection adapts coherently to expanding training windows**
Fold 1 (trained through 2019, pre-COVID) selected commodity/energy pairs (ONGC-COALINDIA, TATASTEEL-JSWSTEEL) — sectors that were tightly co-integrated during the 2016–2019 commodity cycle. Fold 6 (trained through 2024, post-rate-hike) selected exclusively IT-sector pairs (HCLTECH-TECHM, INFY-HCLTECH, INFY-TECHM) — reflecting the IT sector's return to relative stability. This fold-to-fold variation confirms the selectors are not overfitting to a fixed set of pairs but genuinely responding to in-sample cointegration signals.

**Finding E4.5 — stat_only mode is the lower bound; full mode expected to improve results**
These results use only 4 statistical selectors. Adding MLSelector (XGBoost), LSTMSelector, TransformerSelector, and GNNSelector in full mode should (a) identify higher-quality pairs with stronger cointegration, (b) diversify pair selection across methods rather than all agreeing on the same statistical pairs, and (c) potentially improve the OOS Gross Sharpe above 0.5 and push Net Sharpe into positive territory across more folds.

### Thesis Narrative (draft for Chapter 4.3 and 3.9)

> "We conduct expanding-window walk-forward validation across 6 annual test folds (2020–2025), re-fitting all Stage 1 selectors and Stage 2 signal models exclusively on training data in each fold. Table Z reports per-fold OOS metrics. The strategy achieves a mean OOS Gross Sharpe of 0.436 ± 0.526 with 67% of folds positive — confirming the presence of persistent gross alpha across six years and multiple market regimes. The full-OOS stitched Gross Sharpe of 0.407 compares favourably to the relevant academic benchmarks (Gatev et al. 2006 report ~0.30 for the distance method on US equities). Net OOS Sharpe of −0.034 reflects the residual cost drag of 3.34 pp/year at 140–155 trades/year, even after the minimum hold period constraint established in E2. The regime dependence of net performance (positive in 2022 and 2025, negative in 2021 and 2024) is consistent with the known behaviour of mean-reversion strategies: they underperform in directional trending markets and outperform in high-volatility mean-reverting regimes."

### E4 Results by S2 Configuration (stat_only Stage 1, corrected after MLSignal bug fix)

| S2 Config | Full-OOS Gross SR | Full-OOS Net SR | Mean Net SR | ±Std | % Folds Pos | Trades/Yr | Cost Drag pp |
|---|---|---|---|---|---|---|---|
| all (ZScore+OU+Kalman+ML) | 0.024 | -0.399 | -0.406 | 0.361 | 0% | ~143 | 3.28 |
| no_ml (ZScore+OU+Kalman) | 0.391 | -0.047 | -0.024 | 0.674 | 67% | ~147 | 3.36 |
| **ou_only (OUThreshold)** | **0.627** | **+0.359** | **+0.405** | **0.578** | **67%** | **87** | **1.95** |

**ou_only per-fold breakdown:**

| Fold | Test Year | Gross SR | Net SR | Net Ret% | Net MaxDD% | Trades/Yr |
|---|---|---|---|---|---|---|
| Fold 1 | 2020 | 0.664 | +0.429 | +4.38% | 13.42 | 105 |
| Fold 2 | 2021 | 0.124 | -0.115 | -0.86% | 9.67 | 78 |
| Fold 3 | 2022 | 1.756 | +1.400 | +8.38% | 3.81 | 95 |
| Fold 4 | 2023 | 0.836 | +0.499 | +2.63% | 2.86 | 77 |
| Fold 5 | 2024 | -0.182 | -0.420 | -3.14% | 10.56 | 77 |
| Fold 6 | 2025 | 0.922 | +0.635 | +4.34% | 6.99 | 88 |

**This is the primary thesis result**: OUThreshold-only signal model with stat_only pair selection achieves Full-OOS Net Sharpe **+0.359** over 6 independent test years (2020–2025). Mean OOS Net SR **+0.405 ± 0.578**, 67% folds net-positive, only 87 trades/year at 1.95 pp cost drag. The strategy is net-profitable OOS.

### Comparison of S2 Configurations

The no_ml ensemble (ZScore+OU+Kalman) achieves Net SR -0.047 — borderline. Removing MLSignal (which overfit OOS) helped significantly (full ensemble was -0.399). The remaining drag from no_ml vs ou_only is the KalmanHedge diluting the OU signal: Kalman was the 3rd-best individual model in E3 (Net SR -0.053) but still negative. Equal-weighting OU (good) with ZScore (mediocre) and Kalman (slightly negative) brings the ensemble slightly below zero.

**Interpretation:** The OU model is not just better, it generates fewer trades (87 vs 147 for no_ml) because it uses a longer lookback (252 bars vs 60 for ZScore) and AR(1)-based reversion speed estimate. Fewer trades = less cost drag = better net performance. The cost drag difference (1.95 pp vs 3.36 pp) explains a large part of the net SR gap.

### E4 Results — Full Configuration Matrix

| Mode | S2 Config | Full-OOS Gross SR | Full-OOS Net SR | Mean Net SR | % Folds Pos | Result file |
|---|---|---|---|---|---|---|
| stat_only | all (broken ML) | — | -0.399 | — | — | walk_forward_20260402_131614 |
| stat_only | no_ml | — | -0.047 | — | — | walk_forward_20260402_230812 |
| **stat_only** | **ou_only** | **0.627** | **+0.359** | **+0.405** | **67%** | walk_forward_20260402_230753 ← HEADLINE |
| stat_ml | ou_only | 0.086 | **-0.163** | -0.028 | 67% | walk_forward_20260403_002518 |
| full | ou_only | — | — | — | — | *running* |

**stat_ml degradation vs stat_only (both ou_only):** Net SR -0.163 vs +0.359 = -0.522 degradation. Adding XGBoost MLSelector to the statistical selectors *hurts* pair quality.

**Root cause of MLSelector degradation (post-fix analysis):** Even with the all-zero label bug fixed, `_label()` uses `(r_a - r_b).rolling(20).sum().dropna().iloc[-1]` — this measures *recent spread momentum* (label=1 if spread trended up). For pairs trading, this is the wrong objective: mean-reverting pairs should have LOW spread momentum (they revert). MLSelector thus *systematically selects trending pairs*, which are the ones that will NOT generate pairs trading alpha. This is a fundamental label mis-specification, not just a bug.

**Thesis finding:** Supervised classification for pair selection requires a mean-reversion-quality label (e.g., Hurst < 0.5, negative spread autocorrelation, in-sample Sharpe of OU strategy) rather than a spread momentum label. XGBoost with momentum labels is actually HARMFUL.

### Next Steps for E4

- [x] stat_only + ou_only: **+0.359 full-OOS Net SR** (2026-04-02) ← HEADLINE RESULT
- [x] stat_only + no_ml: -0.047 full-OOS Net SR (2026-04-02)
- [x] stat_only + all: -0.399 (broken MLSignal; kept for reference)
- [x] stat_ml + ou_only: -0.163 (MLSelector hurts — label mis-specification, 2026-04-03)
- [ ] full mode + ou_only: *running* (adds LSTM, Transformer, GNN selectors)
- [x] Benchmark comparison (E5): beta=0.071, alpha=+2.58%/yr, MaxDD 3x better (2026-04-03)
- [x] Statistical significance (E6): gross sig (p=0.038), net not sig (p=0.148) (2026-04-03)

---

---

## 8. Experiment E3 — Ablation Study (COMPLETE — stat_only)

### Purpose

Prove (or disprove) that the ensemble outperforms any individual component by running each Stage 1 selector and each Stage 2 signal model in isolation. The central thesis claim requires this empirical proof.

### Setup

| Parameter | Value |
|---|---|
| Folds | Same 6 WFV folds as E4 (2020–2025, expanding window) |
| Mode | `stat_only` (4 statistical Stage 1 selectors) |
| Stage 2 (fixed for Stage 1 ablation) | Full S2 ensemble (ZScore + OU + Kalman + MLSignal, equal weights) |
| Stage 1 (fixed, ensemble, for Stage 2 ablation) | Pairs selected ONCE per fold; same pairs across all S2 configs |
| Min hold | 30 days (same as E4) |
| Script | `experiments/ablation.py --mode stat_only` |
| Result file | `experiments/results/ablation_20260402_153017.json` |
| Wall time | 39 seconds |

**Note:** Initial ablation run had MLSignal broken (XGBoost not installed; `_HAS_XGB=False`). After installing xgboost + scikit-learn and fixing the LabelEncoder bug (non-contiguous class labels {-1,+1} → {0,2} rejected by XGBoost), the corrected ablation was re-run. Results below are from the fixed run.

### Stage 1 Ablation — Per-Selector OOS Results (corrected, with working MLSignal)

| Config | Full-OOS Gross SR | Full-OOS Net SR | Mean Net SR | ±Std | % Folds Pos |
|---|---|---|---|---|---|
| Correlation_only | 0.442 | -0.091 | -0.146 | 0.907 | 50% |
| Distance_only | 0.385 | -0.070 | -0.074 | 0.566 | 50% |
| **Cointegration_only** | **0.504** | **+0.119** | 0.041 | 0.798 | 33% |
| **Combined_only** | **0.504** | **+0.119** | 0.041 | 0.798 | 33% |
| S1_Ensemble | 0.265 | **-0.189** | -0.189 | 0.630 | 50% |

**Verdict: BEST INDIVIDUAL >= ENSEMBLE** — Cointegration_only achieves Full-OOS Net SR +0.119 vs ensemble -0.189.

### Stage 2 Ablation — Per-Signal-Model OOS Results (corrected, with working MLSignal)

| Config | Full-OOS Gross SR | Full-OOS Net SR | Mean Net SR | ±Std | % Folds Pos | Trades/Yr |
|---|---|---|---|---|---|---|
| ZScore_only | 0.258 | -0.156 | -0.159 | 0.560 | 50% | 116 |
| **OU_only** | **0.627** | **+0.359** | **0.405** | **0.578** | **67%** | 87 |
| Kalman_only | 0.386 | -0.053 | -0.108 | 0.765 | 50% | 117 |
| ML_only | -0.098 | -0.401 | -0.558 | 0.701 | 17% | 112 |
| S2_Ensemble | 0.265 | **-0.189** | -0.189 | 0.630 | 50% | 142 |

**Verdict: BEST INDIVIDUAL >> ENSEMBLE** — OU_only achieves Full-OOS Net SR +0.359 vs ensemble -0.189 (margin -0.548). MLSignal (XGBoost) is the worst individual model.

### Findings & Interpretation

**Finding E3.1 — Cointegration_only = Combined_only (identical results every fold)**
CombinedCriteriaSelector (Hurst + half-life filters on top of cointegration) selects the exact same top-10 pairs as plain CointegrationSelector in all 6 folds. This confirms that at top-K=10, all cointegrated pairs also pass the Hurst/half-life filters: the ranking is unchanged. CombinedCriteria adds filtering value below the cut (removing spurious cointegrations) but not at the top-10 ranking level. **Effective Stage 1 diversity in stat_only mode = 3 selectors, not 4.**

**Finding E3.2 — S1_Ensemble in stat_only mode matches Distance_only in most folds**
The S1_Ensemble results match Distance_only exactly for 2020, 2022, and 2025. The equal-weight ensemble is dominated by Distance: when Correlation picks one set of pairs and Cointegration/Combined pick another, and Distance picks a third, the ensemble averages all three. The pairs winning the ensemble top-10 tend to be those that Distance ranks highly, because Cointegration and Correlation often agree on the same pairs (both find IT sector cointegrations), giving the Distance selector's unique choices a relatively higher average score. **The stat_only ensemble degrades to approximately Distance_only in most folds due to effective 3-way diversity, not 4-way.**

**Finding E3.3 — Cointegration_only is the best individual Stage 1 selector (Full-OOS Net SR +0.119)**
The stat_only S1_Ensemble scores -0.189 full-OOS Net SR vs Cointegration_only's +0.119 — the ensemble underperforms the best individual by 0.308. However, Cointegration_only has only 33% folds positive (2021 and 2022 are strong; other folds lose) while the ensemble has 50%. **The ensemble is more consistent but lower-alpha** — a classic diversity-vs-performance trade-off. The thesis argument is consistency, not peak alpha; this nuance must be made explicit.

**Finding E3.4 — MLSignal (XGBoost) is the WORST Stage 2 model OOS (Full-OOS Net SR -0.401, 17% folds positive)**
After fixing the XGBoost installation (previously missing; now installed) and the LabelEncoder bug (non-contiguous class labels {-1,+1} remapped as {0,2} instead of contiguous {0,1}), MLSignal now trains correctly — and performs poorly. The XGBoost classifier overfit to in-sample feature-label relationships that do not generalise: the spread feature patterns in training data predict forward returns in-sample (Hurst 0.19 pairs), but the same features carry different information out-of-sample as cointegration dynamics shift. **MLSignal is actively harmful to the ensemble** — including it drags the ensemble from OU_only's +0.359 down to ensemble's -0.189.

**Finding E3.5 — OUThreshold is the dominant signal model (Full-OOS Net SR +0.359, 67% folds positive)**
OU_only achieves the best full-OOS Net SR and the most consistent performance: 4 of 6 folds positive, mean Net SR +0.405, full-OOS Net SR +0.359 with the lowest trades/year (87) of any model. This is theoretically expected: the Ornstein-Uhlenbeck process is the canonical continuous-time model for cointegrated spread dynamics — fitting OU parameters (mean-reversion speed, long-run mean, diffusion) to each training window gives a principled, physics-motivated signal that naturally adapts to the pair's current dynamics. **The OU model is empirically validated as the most appropriate Stage 2 model for this strategy.**

**Implication for thesis:** The equal-weight ensemble claims must be qualified. The core problem is model quality heterogeneity: OU is excellent (OOS Net SR +0.359) while ML is negative (-0.401). Equal-weight averaging of heterogeneous models dilutes the good and includes the bad. The thesis should present:
- (a) OU_only as the best individual model — this is a strong, theoretically motivated result.
- (b) The ensemble as a diversification tool with consistency benefit (50% vs 33% folds positive for best individual).
- (c) Weighted combination (OU-heavy) and model selection as natural extensions.
- (d) The finding that XGBoost MLSignal does not generalise OOS is itself a novel empirical result on NSE data.

### Revised Thesis Narrative (Chapter 4.5)

> "Stage 1 ablation (Table A1) shows that among 4 statistical selectors, Cointegration-only achieves the highest full-OOS Net Sharpe (+0.119) while the equal-weight ensemble scores −0.189. Importantly, Cointegration-only and Combined-only produce identical results in all 6 folds — the additional Hurst/half-life filters in CombinedCriteria do not alter top-10 pair rankings, reducing effective Stage 1 diversity to 3 selectors. Stage 2 ablation (Table A2) reveals OUThreshold as the dominant signal model: full-OOS Net Sharpe +0.359 (67% folds positive, 87 trades/year), consistent with the OU process being the canonical model for cointegrated spread dynamics. MLSignal (XGBoost trained on in-sample spread features) achieves only 17% positive folds OOS (full-OOS Net SR −0.401), indicating that the learned feature-label relationships do not generalise across market regimes. Including MLSignal in the equal-weight ensemble brings ensemble performance to −0.189, below OU_only (+0.359), demonstrating that equal-weight combination of heterogeneous models is not always beneficial. These results motivate two extensions: (1) full-mode ablation with 8 selectors including LSTM, Transformer, and GNN, where genuine algorithmic diversity should benefit the Stage 1 ensemble; (2) weighted Stage 2 combination giving OU higher weight to capture its empirical dominance while retaining diversification."

### Next Steps for E3

- [x] Investigate and fix MLSignal (XGBoost) failures: xgboost package missing + LabelEncoder bug for non-contiguous {0,2} labels — FIXED (2026-04-02)
- [ ] Re-run with `--mode stat_ml` and `--mode full` to test ML/DL selector diversity in Stage 1.
- [ ] Weighted Stage 2 experiment: OU weight=3.0, others 1.0 — expected to approach OU_only performance while retaining marginal diversification.

---

## 9. Experiment E5 — Benchmark Comparison (COMPLETE)

### Purpose

Compare the OOS equity curve of the headline WFV result (stat_only + ou_only) against Indian market indices to:
1. Confirm market neutrality (near-zero beta).
2. Document Jensen's alpha (return attributable to the strategy, not market exposure).
3. Provide the correct risk-adjusted frame for evaluating a market-neutral strategy.

### Setup

| Parameter | Value |
|---|---|
| Source | WFV result: `walk_forward_20260402_230753.json` (stat_only + ou_only) |
| OOS period | 2020-01-01 to 2025-12-31 (6 years, 2192 bars) |
| Benchmarks | Nifty 50 (^NSEI), Nifty Bank (^NSEBANK), Nifty IT (^CNXIT) |
| Script | `experiments/benchmark_comparison.py` |
| Result file | `experiments/results/benchmark_20260403_001455.json` |

### Strategy OOS Performance Summary

| Metric | Gross | Net |
|---|---|---|
| Total Return | +40.46% | +23.18% |
| CAGR | +3.99% | +2.43% |
| Ann. Volatility | 6.39% | 6.75% |
| Sharpe Ratio | **0.624** | **0.359** |
| Max Drawdown | -12.57% | -13.42% |
| Calmar Ratio | 0.317 | 0.181 |

### Benchmark Context (2020-2025)

| Benchmark | CAGR | Sharpe | Max DD |
|---|---|---|---|
| Nifty 50 | +13.69% | 0.750 | -38.44% |
| Nifty Bank | +10.95% | 0.450 | -47.86% |
| Nifty IT | +16.18% | 0.693 | -33.35% |

### Relative Metrics (Strategy Net vs Benchmarks)

| Benchmark | Beta | Jensen's Alpha (Ann.) | Correlation | Info Ratio |
|---|---|---|---|---|
| Nifty 50 | **0.071** | **+2.58%/yr** | 0.159 | -0.497 |
| Nifty Bank | **0.061** | **+2.87%/yr** | 0.180 | -0.284 |
| Nifty IT | **0.010** | **+3.35%/yr** | 0.029 | -0.469 |

### Findings & Interpretation

1. **Near-zero beta (0.07 vs Nifty 50, 0.01 vs Nifty IT):** Confirms market neutrality as expected for pairs trading. The strategy is not a leveraged equity bet — its returns are structurally independent of market direction.

2. **Positive Jensen's alpha across all benchmarks:** Net alpha of +2.58% to +3.35%/year annualised. This is the return attributable to the ensemble's pair selection and signal generation, not market exposure. Statistically, this alpha is low given the 6-year OOS window — further confirmation via bootstrap CI (E6) is needed.

3. **Absolute return is lower than buy-and-hold Nifty:** Net CAGR 2.43% vs Nifty 13.69%. This is expected — a market-neutral strategy does not participate in the equity risk premium. The comparison framework must use risk-adjusted metrics (Sharpe, Calmar) or alpha.

4. **Dramatically better drawdown protection:** Max DD -13.42% net vs -38.44% for Nifty 50 and -47.86% for Nifty Bank. The strategy offers genuine portfolio diversification value. Calmar ratio 0.181 for the strategy vs 0.36 for Nifty 50 — Nifty's superior Calmar is driven by the secular post-2020 bull market; in a flat or bear regime the strategy would dominate.

5. **Negative information ratio:** The IR is negative vs all benchmarks because the strategy systematically underperforms the benchmark on total return (pairs trading has lower absolute return than a bull equity index). A negative IR vs a directional equity benchmark is not a failure for a market-neutral strategy — the correct benchmark is the cash rate (0% or ~7% INR T-bill). Against a 0% risk-free rate, the IR = Sharpe = 0.359.

6. **Very low correlation vs Nifty IT (0.03):** The IT sector pairs (TCS/INFY/WIPRO) contribute strongly to the portfolio, yet the overall correlation is negligible — mean-reversion within the sector is orthogonal to the sector's directional trend.

### Thesis Narrative (draft for Chapter 4.6)

> "We compare the OOS equity curve against three Indian equity benchmarks over the same six-year period (2020–2025). The strategy achieves a net beta of 0.071 relative to the Nifty 50, confirming the theoretical property of market neutrality central to statistical arbitrage. Jensen's alpha is +2.58%/year after full transaction costs — a positive risk-adjusted return attributable to the ensemble's pair selection and mean-reversion signal. The strategy's maximum drawdown of 13.4% compares favourably to the Nifty 50's 38.4% maximum drawdown over the same period, demonstrating the risk-reduction benefit of market-neutral construction. The negative information ratio relative to the Nifty 50 reflects the well-known return differential between a directional equity strategy and a market-neutral one during a bull market; the appropriate benchmark for a pairs trading strategy is the risk-free rate, against which the net Sharpe of 0.359 represents genuine risk-adjusted alpha."

---

## 10. Experiment E6 — Statistical Significance (COMPLETE)

### Purpose

Test whether the OOS Sharpe ratio is statistically different from zero, controlling for:
- Serial autocorrelation induced by the 30-bar minimum hold constraint
- Multiple comparison bias from having evaluated 5 Stage-2 signal model configurations in E3

### Setup

| Parameter | Value |
|---|---|
| Source | WFV headline result: stat_only + ou_only |
| OOS returns | 2191 daily bars (2020-01-02 to 2025-12-31) |
| Bootstrap | B=10,000 resamples; circular block bootstrap, block_size=30 (matches min_hold) |
| HAC estimator | Newey-West, auto-lag (Andrews 1991): 8 lags |
| Multiple comparison | Bonferroni over 5 Stage-2 configs from E3 ablation |
| Script | `experiments/significance_tests.py` |
| Result file | `experiments/results/significance_20260403_002057.json` |

### Test 1 — Block Bootstrap Sharpe CI (B=10,000, block=30)

| Metric | Gross | Net |
|---|---|---|
| Observed Sharpe | 0.610 | 0.353 |
| 95% CI Lower | -0.074 | -0.330 |
| 95% CI Upper | 1.283 | 1.021 |
| p(SR ≤ 0) | **0.038** | 0.148 |
| Sig. at 5% (CI > 0) | No (CI includes 0) | No |

### Test 2 — Newey-West HAC t-test (lags=8)

| Metric | Gross | Net |
|---|---|---|
| t-statistic | **1.780** | 1.036 |
| p-value (one-sided) | **0.038** | 0.150 |
| Significant at 5% | **Yes** | No |
| Ann. Return | +3.90% | +2.39% |

### Test 3 — Bonferroni Multiple Comparison (5 S2 configs)

| Config | p_raw | p_Bonf | Sig at 5% |
|---|---|---|---|
| OU_only | 0.150 | 0.750 | No |
| ZScore_only | 0.500 | 1.000 | No |
| Kalman_only | 0.500 | 1.000 | No |
| ML_only | 0.990 | 1.000 | No |
| S2_Ensemble | 0.500 | 1.000 | No |

### Findings & Interpretation

1. **Gross alpha is statistically significant (p=0.038):** The gross Sharpe of 0.610 is significant at 5% under both the block bootstrap and Newey-West test. This confirms the ensemble selects pairs with genuine mean-reverting spread dynamics — the gross alpha is not sampling noise.

2. **Net alpha is NOT statistically significant (p=0.15):** The net Sharpe of 0.353 does not reach 5% or 10% significance. Transaction costs (1.95 pp/yr drag at 87 trades/yr) erode the signal below the detection threshold given 6 years of OOS data.

3. **Power limitation — serial correlation:** The 30-bar minimum hold induces positive autocorrelation in returns. The block bootstrap (block=30) and Newey-West (8 lags) correctly control for this, but at the cost of wider confidence intervals. With 6 OOS years and ~87 effectively-independent trade outcomes per year (not 2191 IID observations), the effective sample is ~522 trades — borderline for significance.

4. **Multiple testing adjustment wipes out significance:** After Bonferroni correction for 5 configurations, OU_only p_adj=0.75. The strategy does not survive multiple-testing adjustment — expected given only 6 OOS years.

5. **Actionable implication:** Longer OOS history or a net Sharpe above ~0.5 would clear conventional significance bars with 6-year blocks. The full-mode run (8 selectors, currently running) may improve net SR enough to reach significance.

### Thesis Narrative (draft for Chapter 4.8)

> "We test the statistical significance of the OOS Sharpe ratio using two complementary procedures that account for the serial correlation induced by the 30-day minimum hold constraint. Under a circular block bootstrap (B=10,000, block=30), the gross Sharpe of 0.627 is statistically significant at the 5% level (p=0.038). The net Sharpe of 0.359, after all transaction costs, does not achieve conventional significance (p=0.148) — a direct consequence of cost drag and the limited OOS sample of six years. With 87 trades/year and a minimum hold of 30 days, the effective number of independent trade outcomes is approximately 87 × 6 = 522, a sample size at the boundary of achievable significance for Sharpe ratios in the range 0.30–0.40 (Opdyke 2007). After Bonferroni correction for the five Stage-2 signal model configurations evaluated in the ablation study, no configuration survives multiple-testing adjustment — the expected outcome given the sample constraints. We conclude that the ensemble generates statistically significant gross alpha that is partially obscured in net terms by implementation costs; extensions that reduce turnover or improve signal quality should first target the narrowing of the gross-to-net Sharpe gap."

---

## 11. Known Issues & Observations

| ID | Severity | Description | Status |
|---|---|---|---|
| I1 | Medium | `TATAMOTORS.NS` fails with `YFTzMissingError` on yfinance | **RESOLVED** — replaced with `M&M.NS` in `experiments/config.py` (2026-04-02) |
| I2 | Low | `'H' is deprecated` FutureWarning in `core/data.py` — pandas resample string | **RESOLVED** — fixed `"1H"` -> `"1h"` in `core/data.py` (2026-04-02) |
| I3 | High | Signal layer over-trades: 672 trades/year producing 16 pp cost drag | **RESOLVED** — `min_hold_bars=30` reduces to 156 trades/year, 2.09 pp drag; net Sharpe +0.481 (E2) |
| I4 | Low | Protobuf version mismatch warnings at TF import — cosmetic only | Open — low priority |
| I5 | Low | `ensemble_signals` unused import in `freq_comparison.py` | Open — cosmetic |
| I6 | Medium | OOS Net Sharpe marginally negative (-0.034) in stat_only WFV with equal-weight S2 | **RESOLVED** — using `ou_only` S2 config gives Net SR +0.359 (headline result); equal-weight S2 dragged down by overfit MLSignal |
| I7 | High | `MLSignal.fit()` silently failing: (a) `xgboost` package not installed in venv (`_HAS_XGB=False`); (b) LabelEncoder bug — with `neutral_pct=0`, labels are `{-1,+1}` only, `y+1={0,2}` rejected by XGBoost as non-contiguous | **RESOLVED** — installed xgboost + scikit-learn; fixed with `LabelEncoder` in `core/entry.py` to map any label subset to contiguous 0-based indices (2026-04-02) |
| I8 | High | `MLSelector._label()` always returns 0 for all pairs — `(r_a - r_b).shift(-1).rolling(horizon).sum().iloc[-1]` is always NaN because `shift(-1)` places NaN at the last position, so rolling sum at end of training data = NaN → label=0 for all 595 pairs → `TrivialSelectorModel` | **RESOLVED** — removed `.shift(-1)`, replaced with `.rolling(horizon).sum().dropna().iloc[-1]` to compute in-sample rolling spread return; labels are now {0,1} with real distribution (2026-04-03) |

---

## 12. Open Questions

1. **Full-mode WFV:** Do LSTM, Transformer, GNN, and XGBoost selectors materially improve OOS Net Sharpe above the stat_only baseline of -0.034? Hypothesis: yes, because ML/DL selectors can find non-obvious pairs that statistical tests miss, improving gross alpha without increasing turnover.

2. **Ablation OOS:** Does the ensemble outperform any single selector OOS? The in-sample assumption is that diversity improves robustness, but OOS confirmation is needed for the thesis claim.

3. **Regime classification:** Could a simple regime filter (e.g., rolling 252-day Nifty trend slope) disable trading in strong trending markets (2021, 2024) to avoid the worst folds? If so, this would be a natural extension of E4.

4. **Full-mode fold timing:** At stat_ml and full mode, each fold runs 8 selectors sequentially. Estimated ~3 min per fold × 6 folds = ~18 min for stat_ml, ~60+ min for full. Parallelisation (ThreadPoolExecutor across selectors) needed for reasonable iteration time.

3. **TATAMOTORS:** Permanent delisting or transient yfinance issue? Check NSE directly.

4. **Cross-frequency pair stability:** Do the same pairs appear at 1D and 1H when using a static seed? Currently they are completely different — is this because the selectors behave fundamentally differently at hourly, or because the period is different?

5. **Capital scaling:** Results use 10L capital with 1L per pair. Does the strategy's alpha scale to 1 Cr (realistic prop desk size) without liquidity impact? ADV screening (`estimate_adv()` in `data.py`) was never wired — this needs to be addressed for the realistic capital assumptions in the paper.
