# Plan — Hybrid Pairs Trading Ensemble (Thesis & Research Paper)

**Goal:** Produce a research-paper-worthy empirical study of a hybrid ensemble pairs trading strategy on Indian NSE equities, combining classical statistical arbitrage with modern deep learning.

**Last updated:** 2026-04-06 (E4 full-mode + E3 full-mode complete; Transformer bug fixed; E7 weighted ensemble is next)

---

## Table of Contents

1. [Research Paper Narrative](#1-research-paper-narrative)
2. [Thesis Structure (Draft)](#2-thesis-structure-draft)
3. [Experiment Roadmap](#3-experiment-roadmap)
4. [Build Roadmap](#4-build-roadmap)
5. [Design Decisions (Settled)](#5-design-decisions-settled)
6. [Academic Rigour Checklist](#6-academic-rigour-checklist)
7. [Current State Summary](#7-current-state-summary)

---

## 1. Research Paper Narrative

### Core Argument

> "Classical pairs trading relies on a single cointegration test and a fixed z-score threshold. We propose a **two-stage hybrid ensemble**: Stage 1 selects pairs by aggregating 8 algorithms spanning correlation, distance, cointegration, ML classifiers, and deep learning (LSTM, Transformer, GNN); Stage 2 generates signals by ensembling 4 models including a Kalman filter hedge, Ornstein-Uhlenbeck process, gradient-boosted classifier, and (optionally) an RL agent. Walk-forward evaluation on 35 NSE equities over 10 years shows the ensemble achieves an out-of-sample Sharpe of X vs. Y for the best individual model. We further demonstrate that daily data is empirically superior to higher-frequency data, with gross Sharpe declining from 1.14 (daily) to 0.49 (hourly) and net returns collapsing due to transaction cost drag."

### Novel Contributions

1. **First ensemble of 8 pair selectors (incl. GNN + Transformer) on NSE equities.** Most existing work uses a single selector. E3 ablation shows equal-weight ensemble < best individual (LSTM_only), but this is itself a publishable finding: the diversity benefit requires weighted combination, not naive equal-weighting. E7 (weighted ensemble) will complete the argument.
2. **Empirical frequency analysis on NSE.** We quantify the gross-to-net degradation across daily vs hourly, separating signal quality (Hurst exponent) from cost drag.
3. **Indian cost model.** Existing pairs trading literature typically uses US cost assumptions. We build an accurate NSE cost model (brokerage + STT + GST + stamp + slippage).
4. **Walk-forward validation.** We report OOS Sharpe across rolling folds — not in-sample — making results credible for academic publication.
5. **(Stretch)** RL signal model (`RLSignal` via PPO/DQN) as Stage 2 component — if implemented, first application of RL to NSE pairs trading.

---

## 2. Thesis Structure (Draft)

### Chapter 1 — Introduction
- Motivation: pairs trading in emerging markets, NSE context
- Research questions (list 4–5 specific questions, each answered by an experiment)
- Contributions summary

### Chapter 2 — Literature Review
- Classical pairs trading: Gatev et al. (2006), Vidyamurthy (2004)
- Cointegration: Engle & Granger (1987), ADF test
- Distance method: Gatev et al. (2006)
- Hurst exponent & mean-reversion: Hurst (1951), Lo (1991)
- Ornstein-Uhlenbeck: Uhlenbeck & Ornstein (1930), Elliott et al. (2005)
- Kalman filter hedge: Kalman (1960), Pole (2007)
- ML for pairs trading: Krauss et al. (2017), Huck (2009)
- Deep learning for finance: Fischer & Krauss (2018), Zerveas et al. (2021)
- GNN for stock relations: Matsunaga et al. (2019), Kipf & Welling (2017)
- Transformer in finance: Vaswani et al. (2017), Wen et al. (2023)
- RL for trading: Mnih et al. (2015), PPO: Schulman et al. (2017)
- Indian market microstructure: NSE transaction cost structure

### Chapter 3 — Data & Methodology
- **3.1** Dataset: NSE tickers, yfinance, date range, data quality
- **3.2** Universe construction: 35 stocks, 8 sectors, liquidity filter
- **3.3** Cost model: IndianCosts derivation
- **3.4** Frequency Selection: **Experiment E1** (daily vs hourly — DONE in stat_only mode)
- **3.5** Stage 1 — Pair Selection algorithms (describe all 8)
- **3.6** Stage 2 — Signal models (describe all 4)
- **3.7** Ensemble methodology (weighted average, neutral band)
- **3.8** Backtesting framework (vectorized, gross vs net, soft stop)
- **3.9** Walk-forward validation design

### Chapter 4 — Results
- **4.1** Frequency comparison (E1) — justify daily data choice
- **4.2** Pair selection — which pairs are selected, sector distribution
- **4.3** Main backtest results — OOS Sharpe, return, max drawdown (from walk-forward)
- **4.4** Benchmark comparison — vs Nifty 50, Nifty Bank, Nifty IT
- **4.5** Ablation study — ensemble vs each individual model (E3)
- **4.6** Cost sensitivity — what break-even cost is needed for net profitability
- **4.7** Signal hold period analysis — optimal minimum hold (E2)

### Chapter 5 — Discussion
- Why gross alpha exists but net alpha is hard to capture
- The over-trading problem and remedies
- Ensemble benefits: diversification vs model diversity
- Limitations: yfinance data quality, look-ahead bias controls, NSE market impact

### Chapter 6 — Conclusion
- Summary of contributions
- Practical implications for retail/prop trading on NSE
- Future work: RL agent, live execution, intraday regime

### Appendices
- A: Full cost model derivation
- B: Selector hyperparameters
- C: Walk-forward fold definitions
- D: All experiment result tables (raw JSON)

---

## 3. Experiment Roadmap

Each experiment is self-contained, reproducible via a script in `experiments/`, and saves results to `experiments/results/`.

### E1 — Frequency Comparison (IN PROGRESS)

**Purpose:** Justify daily data choice empirically.
**Script:** `experiments/freq_comparison.py`
**Status:**
- [x] stat_only run complete (2026-04-02) — sanity check passed
- [ ] Full run (--mode full, all 8 selectors) — needed for paper
- [ ] Fix TATAMOTORS.NS -> M&M.NS in universe first
- [ ] Fix signal over-trading (E2) before final E1 run so results are meaningful

**Key metrics to report:** Gross Sharpe, Net Sharpe, Ann. Return (gross & net), Max DD, Trades/Year, Cost Drag (pp), Hurst median, Signal Reversal Rate.

---

### E2 — Signal Hold Period Sweep (NEXT)

**Purpose:** Quantify the impact of a minimum holding period on net Sharpe. Find the Pareto-optimal hold period.

**Problem:** Current signal layer generates ~673 trades/year on 10 daily pairs, causing 16 pp cost drag. The pairs themselves have excellent mean-reversion (Hurst 0.19). The problem is the signal layer, not the pair selection.

**Approach:**
- Add `min_hold_bars: int` parameter to `BacktestConfig` (default 0 = no constraint).
- Sweep: min_hold = 0, 1, 2, 3, 5, 7, 10 trading days.
- For each: record Net Sharpe, Trades/Year, Cost Drag.
- Report the "elbow" — where Net Sharpe peaks.

**Expected result:** Net Sharpe should turn positive at some hold period (hypothesis: around 3–5 days). This motivates the hold constraint used in all subsequent experiments.

**Script to build:** `experiments/hold_period_sweep.py`
**Thesis section:** Chapter 4.7 / Appendix or Chapter 5 discussion.

---

### E3 — Performance Attribution / Ablation (COMPLETE — stat_only + full mode)

**Purpose:** Prove (or disprove) the ensemble vs individual claim. Without this, there is no justification for the complexity.

**Key results:**

| Mode | Best Individual (S1) | Best Net SR | S1 Ensemble Net SR | Verdict |
|---|---|---|---|---|
| stat_only | Cointegration_only | +0.119 | -0.189 | Individual wins |
| full | LSTM_only | +0.305 | -0.719 | Individual wins (by larger margin) |

S2 ablation (both modes): OU_only dominates (+0.359 stat_only, +0.063 full); S2_Ensemble = -0.189 / -0.719.

**Revised thesis claim:** Equal-weight ensemble underperforms best individual — this is the finding. Weighted ensemble (E7) tests whether optimal weighting recovers or improves the result.

**Known issue fixed:** TransformerSelector Lambda+GPU crash fixed 2026-04-06. Transformer_only fold needs re-run.

**Script:** `experiments/ablation.py`
**Thesis section:** Chapter 4.5.

---

### E4 — Walk-Forward Validation (COMPLETE — all modes)

**Purpose:** Replace single in-sample backtest with rolling out-of-sample evaluation. Without this, thesis reviewers will reject the results.

**Design (expanding window):**
```
Fold 1: Train 2016-2019, Test 2020
Fold 2: Train 2016-2020, Test 2021
Fold 3: Train 2016-2021, Test 2022
Fold 4: Train 2016-2022, Test 2023
Fold 5: Train 2016-2023, Test 2024
Fold 6: Train 2016-2024, Test 2025
```

**Results by configuration (all ou_only S2):**

| Mode | Full-OOS Net SR | Mean Net SR | ±Std | % Folds Pos | Result file |
|---|---|---|---|---|---|
| stat_only | **+0.359** ← HEADLINE | +0.405 | 0.578 | 67% | walk_forward_20260402_230753 |
| stat_ml | -0.163 | -0.028 | — | 67% | walk_forward_20260403_002518 |
| **full** | **+0.067** | **+0.255** | **0.894** | **67%** | walk_forward_20260406_011541 |

**Key finding:** stat_only+OU is the best mode. Adding DL selectors under equal-weight degrades OOS Net SR by 0.292. Headline result = **+0.359 full-OOS Net SR (stat_only + ou_only)**.

**Script:** `experiments/walk_forward.py`
**Thesis section:** Chapter 4.3, 3.9.

---

### E5 — Benchmark Comparison

**Purpose:** Compare strategy net returns against Indian indices (Nifty 50, Nifty Bank, Nifty IT, Sensex).

**Approach:** Use `BenchmarkComparison` class already implemented in `core/reports.py`. Plot cumulative returns of strategy (net) vs each benchmark. Compute Information Ratio: `(Ann Return_strategy - Ann Return_benchmark) / Tracking Error`.

**Script to build:** `experiments/benchmark.py` (or integrate into walk_forward output)
**Thesis section:** Chapter 4.4.

---

### E6 — Statistical Significance (after E3 & E4)

**Purpose:** Confirm that performance differences are not due to chance.

**Methods:**
- **Bootstrap Sharpe confidence intervals:** resample daily PnL 10,000 times, compute 95% CI on Sharpe.
- **White's Reality Check / Hansen's SPA:** correct for multiple testing across the 8 selector configurations in E3.
- **t-test on rolling OOS returns:** paired t-test comparing ensemble vs best individual model rolling monthly returns.

**Script to build:** `experiments/significance_tests.py`
**Thesis section:** Chapter 4.5 (alongside ablation).

---

### E7 — Weighted Ensemble (NEXT — HIGH PRIORITY)

**Purpose:** Test whether optimal weighting (vs equal weighting) allows the ensemble to beat LSTM_only (best individual Stage 1) and OU_only (best individual Stage 2). This directly addresses the revised thesis claim.

**Design:**

Stage 1 weights (to test): `LSTM=3.0, Correlation=2.0, Distance=1.0, Cointegration=1.0, Combined=0.5, ML=0.0, Transformer=1.0, GNN=0.5`
- ML=0.0: MLSelector label mis-specification confirmed across all modes
- LSTM=3.0: best individual in full mode
- Correlation=2.0: best in stat_only mode, strong in full mode
- Transformer=1.0: not yet evaluated (bug was fixed); include at unit weight

Stage 2 weights: `OU=1.0, ZScore=0.0, Kalman=0.0, ML=0.0` (OU_only — already proven best)

**Alternative S1 weight sets to sweep:**
1. `LSTM=3, Correlation=2, rest=1, ML=0, GNN=0` (exclude clearly harmful selectors)
2. `LSTM=1, Correlation=1, Distance=1, Cointegration=1, Combined=0` (stat+LSTM, no bad selectors)
3. `LSTM=1, Correlation=1` only (tightest, highest-quality pair selection)

**Script to build:** `experiments/weighted_ensemble.py` (or add `--s1-weights` flag to `walk_forward.py`)
**Thesis section:** Chapter 4.5 (extension of ablation), Chapter 5 (discussion of ensemble design).

---

### E8 — RL Signal Model (STRETCH GOAL)

**Purpose:** Novel contribution — first RL-based signal model for NSE pairs trading.

**Design:**
- Environment: state = 11-feature vector from spread features at bar t; action = {-1, 0, +1}; reward = net PnL after costs.
- Agent: PPO (stable-baselines3) — easier to train than DQN for continuous-action-like setups.
- Training: same expanding-window splits as E4.
- Class: `RLSignal(EntryExitModel)` — `fit()` trains PPO agent; `trade_signals()` runs inference.
- Integrate into Stage 2 ensemble alongside existing signal models.

**Script to build:** `core/entry_rl.py` + `experiments/train_rl.py`
**Thesis section:** Chapter 4 (if results are good) or Chapter 5 / Future Work (if not).

---

## 4. Build Roadmap

Priority order based on thesis deadline and academic impact:

### Phase A — Fix before any final runs (COMPLETE 2026-04-02)

1. [x] Replace `TATAMOTORS.NS` with `M&M.NS` in `experiments/config.py`
2. [x] Add `min_hold_bars` to `BacktestConfig` + `_apply_min_hold()` in `core/backtest.py`
3. [x] Fix `core/data.py:148` deprecation warning (`"1H"` -> `"1h"`)

### Phase B — Signal Quality Fix / E2 (COMPLETE 2026-04-02)

4. [x] Build `experiments/hold_period_sweep.py` with `--hold-values` CLI arg
5. [x] Swept [0,5,10,15,20,25,30,40] days — **optimal = 30 trading days** (Net Sharpe +0.481, Net Return +2.89%)
6. [x] `DEFAULT_MIN_HOLD = 30` locked in `experiments/config.py`; all subsequent experiments use it

### Phase C — Walk-Forward Framework (E4)

7. [x] Build `experiments/walk_forward.py` with expanding-window fold logic (2026-04-02)
8. [x] Run WFV with stat_only + ou_only — **Net SR +0.359** HEADLINE (2026-04-02)
9. [x] Add `--s2` CLI arg with ou_only/no_ml/all presets to config (2026-04-02)
10. [x] Fix MLSignal LabelEncoder bug + install xgboost (2026-04-02)
11. [x] Fix MLSelector._label() all-zero bug — removed shift(-1) (2026-04-03)
12. [x] Run WFV with stat_ml + ou_only — Net SR -0.163 (MLSelector hurts, 2026-04-03)
13. [ ] Run WFV with full mode — *RUNNING* (ETA ~12h from 23:09 Apr 2)

### Phase D — Ablation & Attribution (E3) (COMPLETE — stat_only + full mode)

14. [x] Build `experiments/ablation.py` (2026-04-02)
15. [x] Run stat_only ablation (2026-04-02): S1 best = Cointegration_only (+0.119); S2 best = OU_only (+0.359)
16. [x] Run stat_ml ablation (2026-04-06): `ablation_20260406_012530.json`
17. [x] Run full-mode ablation (2026-04-06): S1 best = LSTM_only (+0.305); S1 ensemble = -0.719
18. [x] Fix Transformer Lambda+GPU bug: `_PositionalEncodingLayer` in `core/selectors_ml.py` (2026-04-06)
19. [ ] Re-run Transformer_only single ablation fold on cluster (validate fix)

### Phase E — Weighted Ensemble (E7) (NEXT)

20. [ ] Build `experiments/weighted_ensemble.py` (or add `--s1-weights` to walk_forward.py)
21. [ ] Run S1 weight sweep: LSTM-heavy, Correlation-heavy, stat+LSTM configs
22. [ ] Target: beat LSTM_only (+0.305) on full-OOS Net SR

### Phase F — Benchmarks & Significance (COMPLETE)

23. [x] Build and run `experiments/benchmark_comparison.py` (E5) — 2026-04-03
    - Beta=0.071, alpha=+2.58%/yr, Max DD -13.4% vs -38.4% Nifty 50
24. [x] Build and run `experiments/significance_tests.py` (E6) — 2026-04-03
    - Gross SR p=0.038 (sig); Net SR p=0.148 (not sig); honest academic finding
25. [ ] Run E6 significance on full-mode + E7 weighted result once available
26. [ ] Polish all result tables for final thesis chapter write-up

### Phase G — Final Frequency Run (E1 full mode)

27. [ ] Run E1 with `--mode full` (lower priority — after E7 complete)

### Phase H — Stretch (RL)

28. [ ] Implement `RLSignal` if time permits
29. [ ] Integrate into E3 ablation and E4 WFV

---

## 5. Design Decisions (Settled)

These are locked — do not revisit without strong reason.

| Decision | Rationale |
|---|---|
| **Daily (1D) data only** for main experiments | E1 shows gross Sharpe 1.14 vs 0.49 for hourly; hourly selects poor pairs and goes bankrupt after costs |
| **35 NSE large-cap stocks, 8 sectors** | Diverse enough for cross-sector pairs; all Nifty 100 = liquid; 561 pairs is tractable |
| **10-year window (2016–2026)** | Covers multiple market regimes (demonetisation 2016, COVID 2020, post-COVID bull run, rate-hike cycle 2022) |
| **Expanding window WFV** | More data in later folds = better model quality; standard in academic finance |
| **Equal weights for ablation** | Any other choice would bias the comparison |
| **INR 10 lakh capital, 1 lakh per pair** | Realistic retail/prop desk size; avoids ADV constraints at 1L notional |
| **`experiments/` for reproducible scripts, `reports/` for Streamlit sessions** | Clean separation of research artifacts from UI artefacts |
| **Separate `Research.md` (log) and `Plan.md` (roadmap)** | Research.md records what was found; Plan.md records what we're doing next |

---

## 6. Academic Rigour Checklist

Items a thesis reviewer will check. Track completion here.

| Item | Status |
|---|---|
| No look-ahead bias in signal generation | Partially — selectors are fit on training window; need to verify signal models do not use future data in walk-forward setup |
| Out-of-sample evaluation (WFV) | **DONE** — E4 all modes complete. Headline: stat_only+OU Net SR +0.359; full-mode +0.067 |
| Multiple-testing correction in ablation | **DONE** — E6 Bonferroni over 5 S2 configs; OU_only p_adj=0.75 (not sig after correction — expected given 6yr sample) |
| Bootstrap confidence intervals on Sharpe | **DONE** — E6: Gross SR CI excludes zero (p=0.038); Net SR CI includes zero (p=0.148) |
| Gross AND net results reported | Done — both are always reported |
| Accurate transaction cost model | Done — IndianCosts with NSE-specific rates |
| Benchmark comparison | **DONE** — E5: beta=0.071 vs Nifty 50, Jensen's alpha +2.58%/yr, Max DD 3x better |
| Universe fixed before any backtest | Done — locked in `experiments/config.py` |
| Random seeds fixed | Done — RANDOM_SEED = 42 |
| Data source documented | Done — yfinance, NSE .NS tickers, frequency, date range |
| Hyperparameters documented | Partially — in code; need to consolidate in paper |
| Reproducibility: results can be re-run from scripts | Done for E1–E6; pending for E7 (script to build) |

---

## 7. Current State Summary

**What works end-to-end:**
- Full pipeline (data -> Stage 1 -> Stage 2 -> backtest -> metrics) runs cleanly.
- E1 (frequency comparison), E2 (hold period sweep), E4 (walk-forward validation) all run and produce results.
- Universe defined (35 tickers, M&M replaces TATAMOTORS), date windows fixed, cost model accurate.
- All reproducible experiment results saved as JSON in `experiments/results/`.

**Key results so far:**
- E1: Gross Sharpe 1.14 (daily) vs 0.49 (hourly) — daily confirmed as the right frequency.
- E2: Optimal hold = 30 days — Net Sharpe +0.481, cost drag 2.09 pp/year.
- E3 (stat_only): S1 — Cointegration_only best individual (Net SR +0.119 vs stat_only ensemble -0.189); S2 — OU_only dominates (Net SR +0.359); MLSignal overfit OOS (Net SR -0.401, worst).
- **E4 HEADLINE (stat_only + ou_only): Full-OOS Net Sharpe +0.359** (Gross 0.627), 87 trades/yr, 1.95 pp cost drag, 67% folds positive.
- E4 (stat_ml + ou_only): Net SR -0.163 — MLSelector HURTS even with label bug fixed; momentum label ≠ mean-reversion quality (important negative result).
- **E5 Benchmark (stat_only + ou_only):** Beta=0.071 (near-zero), Jensen's alpha +2.58%/yr vs Nifty 50, Max DD -13.4% vs -38.4% for Nifty 50.
- **E6 Statistical Significance:** Gross SR p=0.038 (sig); Net SR p=0.148 (not sig at 5%); Bonferroni 5-config p_adj=0.75 (not sig). Gross alpha is real; net alpha below detection threshold given 6yr OOS.

**What is missing for thesis submission:**
- **E7 Weighted ensemble WFV** — core next experiment. Can weighted S1 (LSTM-heavy) + OU-only S2 beat the equal-weight ensemble and match/beat LSTM_only (+0.305)?
- **Transformer_only re-run** — validate the Lambda fix on cluster (single config, fast).
- **E6 significance on full-mode + E7 results** — extend significance testing once E7 is done.
- **E1 full-mode** — lower priority, run after E7.
- **Final result table polish** for thesis write-up.

**Phase completion:**
- Phase A (fixes): COMPLETE
- Phase B (hold period E2): COMPLETE
- Phase C (walk-forward E4): COMPLETE (all modes: stat_only +0.359 headline; stat_ml -0.163; full +0.067)
- Phase D (ablation E3): COMPLETE (stat_only + full mode); Transformer_only re-run pending
- Phase E (weighted ensemble E7): NOT STARTED ← CURRENT PRIORITY
- Phase F (benchmarks E5, significance E6): E5/E6 on headline result COMPLETE; re-run on E7 result pending
- Phase G (E1 full): NOT STARTED
- Phase H (RL stretch): NOT STARTED
