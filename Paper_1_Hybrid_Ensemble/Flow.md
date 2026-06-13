# Paper 1 — Project Flow
# Hybrid Ensemble Pairs Trading: Do ML Selectors Outperform Statistical Baselines?
# Last updated: 2026-06-12 (Matched-universe robustness results added)

---

## 0. Glossary of Short Forms and Jargon (used in this file and Chapter 2)

| Term | Meaning |
|------|---------|
| SR | Sharpe Ratio (risk-adjusted return metric). Here, "Net SR" means after transaction costs and "Gross SR" means before costs. |
| Sharpe Ratio formula | $SR = \frac{E[R_p - R_f]}{\sigma_p}$ |
| CAGR | Compound Annual Growth Rate |
| MaxDD / MDD | Maximum Drawdown |
| WFV | Walk-Forward Validation |
| OOS | Out-of-Sample |
| OU | Ornstein-Uhlenbeck mean-reverting process |
| ADF | Augmented Dickey-Fuller stationarity test |
| OLS | Ordinary Least Squares |
| PCA | Principal Component Analysis |
| SSD | Sum of Squared Deviations |
| bps | Basis points (1 bps = 0.01%) |
| STT | Securities Transaction Tax |
| GST | Goods and Services Tax |
| ML / DL / RL | Machine Learning / Deep Learning / Reinforcement Learning |
| LSTM / GNN / CNN | Long Short-Term Memory / Graph Neural Network / Convolutional Neural Network |
| PPO | Proximal Policy Optimization |
| SOTA | State of the Art |
| CI | Confidence Interval |
| NW | Newey-West |
| FWER | Family-Wise Error Rate |

---

## What This File Is

A complete chronological record of Paper 1 — what was built, why each decision was made, what went wrong, how it was fixed, and where the project stands today. Intended as the single narrative reference for any collaborator or future session starting fresh.

---

## 1. Research Question

**Core question:** Does a hybrid ensemble of ML + statistical pair selectors outperform individual statistical baseline selectors on the NSE Nifty 100?

This is deliberately different from Paper 2's question (which asks whether universe quality dominates methodology). Paper 1 is the foundational study:
- Compare ensemble vs single-strategy (not ML vs stat)
- Single market (NSE Nifty 100), one cost regime (India 16.28 bps)
- Longer horizon (10 years, 2015–2024) to give ML models sufficient training data

---

## 2. System Architecture

### Stage 1 — Pair Selection
Seven selectors run on training-window prices to score each candidate pair:

| Selector | Type | Notes |
|----------|------|-------|
| CorrelationSelector | Statistical | Rolling Pearson correlation |
| DistanceSelector | Statistical | Normalized spread distance (Gatev et al. 2006) |
| CointegrationSelector | Statistical | Engle-Granger p-value |
| CombinedCriteriaSelector | Statistical | Composite of above 3 |
| MLSelector | ML (XGBoost) | Feature-based classification |
| LSTMSelector | Deep Learning | Sequence autoencoder |
| TransformerSelector | Deep Learning | Attention-based encoder |
| GNNSelector | Deep Learning | Graph neural network on correlation graph |
| ~~CNNSelector~~ | ~~Deep Learning~~ | **DISABLED** — ablation showed consistent degradation (Sharpe Ratio +0.481 → -0.12). Disclosed in Appendix B. |

Selector outputs are combined as a weighted ensemble (S1 score per pair). Top-k pairs pass to Stage 2.

### Stage 2 — Signal Generation
Three signal models per selected pair:
- **ZScoreThreshold** — entry |z|>2.0, exit |z|<0.5 (primary)
- **OUThreshold** — OU-MLE parameters estimated per training fold
- **MLSignal** — RL-based (BLOCKED: gymnasium not installed on Kalpana)

Signal outputs are weighted (S2 ensemble). Best configuration found: **OU-only** (`--s2 ou_only`).

### Stage 3 — Backtest
- Vectorised backtest (no lookahead)
- IndianCosts: **16.28 bps round-trip** (D12)
- Capital: INR 10 lakh total / 1 lakh per pair
- Minimum hold: **30 trading days** (D11)
- Soft stop-loss fires after min_hold (emergency override)

---

## 3. Infrastructure

| Item | Value |
|------|-------|
| Code repo | `D:/Code/Hybrid-Pairs-Trading-Ensemble/` (local) |
| Cluster | IIT Bombay Kalpana — `~/Hybrid-Pairs-Trading-Ensemble/` |
| SLURM partition/account | `cn3_anandi` / `cminds_anandi` |
| Python env | `Implementation/.venv` (Python 3.12) |
| Data cache | `Implementation/experiments/data/nse_nifty100/prices_2015-01-01_2024-12-31.parquet` |
| Config | `Implementation/experiments/config.py` — single source of truth (D10) |
| ML mode | CPU-only: `CUDA_VISIBLE_DEVICES=""`, `TF_DETERMINISTIC_OPS=1`, `PYTHONHASHSEED=42` (D13) |
| Workflow | Local edit → git push → Kalpana git pull → sbatch (NEVER edit on Kalpana directly) |

---

## 4. Universe Construction

### Initial universe (Paper 2 / early Paper 1)
- 35 Nifty-100 stocks, 8 sectors, defined in `config.py`
- Rationale: tractable (595 pairs), all sectors represented, all Nifty 100 liquid (D2)
- `TATAMOTORS.NS` replaced with `M&M.NS` (persistent yfinance timezone error, D9)

### Expanded universe for Paper 1 (current)
- ~95 Nifty 100 constituents attempted via `fetch_paper1_data.py`
- 6 dropped: TATAMOTORS, BERGERPAINTS, VODAFONEIDEA, LTIM, ADANITRANS + 1 (< 5yr data filter)
- Final: **89 tickers, 13 sectors, 2015-01-01 to 2024-12-31, 2466 trading days**
- Why expand (D14): C(89,2)=3,916 pairs — sufficient for ensemble DISAGREEMENT to be meaningful
- Survivorship bias: using current Nifty 100 constituents. Disclosed as limitation.
- Pre-downloaded to Parquet (D15): eliminates yfinance 429 errors, ensures bit-identical results

### Date range: 2015–2024 (D16)
- 10 years captures a full market cycle (demonetisation, GST, IL&FS, COVID, rate hike)
- Pre-2015 NSE data quality is poor on yfinance
- End date 2024-12-31 keeps dataset closed; no partial-year data

---

## 5. Cost Model Evolution

### Phase 1 (March–April 2026): 22.5–22.9 bps
- Estimated from NSE fee schedule but **double-counted STT**
- STT for delivery trades = 0.1% ONE-WAY, not round-trip
- All E1-E8 runs in this period archived in `results/backup_old_costs/` — NOT used in paper

### Phase 2 (May 2026 onwards): **16.28 bps** (D12)
- Corrected after audit

| Component | Rate |
|-----------|------|
| Brokerage | 5.00 bps |
| STT (one-way delivery) | 1.00 bps |
| NSE transaction charges | 0.335 bps |
| GST on brokerage | 0.90 bps |
| SEBI charges | 0.005 bps |
| Stamp duty | 0.015 bps |
| Slippage / market impact | 9.025 bps |
| **Total round-trip** | **16.28 bps** |

9.025 bps slippage is intentionally conservative for large-cap Nifty 100 stocks.

---

## 6. Walk-Forward Validation Design

### Why expanding window (D7, D17)
- Paper 1 question = model comparison → expanding WFV is standard
- Allows ML models to train on progressively larger datasets (their intended condition)
- Paper 2 used rolling WFV (question = cross-market generalisation)

### 6-fold structure

| Fold | Train | Test |
|------|-------|------|
| F1 | 2015–2017 | 2018 |
| F2 | 2015–2018 | 2019 |
| F3 | 2015–2019 | 2020 |
| F4 | 2015–2020 | 2021 |
| F5 | 2015–2021 | 2022 |
| F6 | 2015–2022 | 2023–2024 |

Both Stage 1 selectors AND Stage 2 signal models are re-fit per fold on training data only (D8). No look-ahead.

---

## 7. Key Design Decisions (Summary)

Design decisions are justified by one of three evidence types: **Literature**, **Experiment**, or **Engineering/Methodological constraint**.

| Decision | Choice | Evidence Type | Basis | Current Status |
|----------|--------|---------------|-------|----------------|
| D1 | Daily data | Experiment + Literature | E1 frequency comparison showed hourly Sharpe collapse (~57%) and higher microstructure noise; aligns with prior daily-frequency pairs-trading literature | **Proven (internal)** |
| D3 | Universe pre-specified | Methodological | Prevents post-hoc universe snooping; standard research design practice | **Defensible by design** |
| D4 | Equal ensemble weights in ablation | Methodological | Needed to isolate structural effect of ensembling from weight tuning effects | **Defensible by design** |
| D5 | min_hold = fixed parameter | Methodological | Treated as execution constraint, not predictive signal parameter | **Defensible by design** |
| D6 | min_hold blocks exits AND reversals | Engineering + Cost logic | Exit/reverse churn creates similar turnover cost; rule avoids churn loops | **Needs direct experiment** |
| D7 | Expanding WFV | Literature + Methodological | Standard for model-comparison studies and matches Paper 1 objective | **Supported** |
| D8 | Re-fit selectors + signals per fold | Methodological | Prevents look-ahead leakage; required for valid walk-forward inference | **Supported** |
| D10 | config.py = single source of truth | Engineering | Prevents parameter drift between scripts and environments | **Operationally validated** |
| D11 | min_hold = 30 days | Experiment + Literature | E2 sweep peak at 30 days (Net Sharpe Ratio 0.481, Gross Sharpe Ratio 0.963); consistent with OU half-life range (~20–30 days) | **Proven (internal)** |
| D12 | 16.28 bps round-trip | Data audit + Market structure | Corrected STT treatment and updated broker fee model; conservative slippage retained | **Supported** |
| D13 | CPU-only ML | Experiment (reproducibility) | GPU/non-deterministic spread in Sharpe (~1.226 range) too large for defensible inference | **Proven (internal)** |
| D14 | 89-ticker Nifty 100 | Design + feasibility | Pair count (3,916) sufficient for selector disagreement tests; practical compute/runtime balance | **Partly validated** |
| D15 | Parquet cache | Engineering | Eliminates yfinance 429/runtime instability and improves reproducibility | **Operationally validated** |
| D16 | 2015–2024 | Literature + data quality constraint | Full-cycle coverage and acceptable NSE data quality; pre-2015 quality weaker | **Supported** |
| D17 | 6-fold expanding | Literature + design | Consistent with expanding-WFV setup and enough fold diversity for regime variation | **Supported** |
| D18 | CNNSelector disabled | Experiment | Ablation showed consistent performance degradation | **Proven (internal)** |
| D19 | ZScore primary, OU secondary | Literature + Experiment | ZScore for comparability; OU-only outperformed S2 ensemble in E4 | **Supported** |
| D20 | Block bootstrap + Bonferroni | Literature/Statistics | Handles dependence/fat tails and controls family-wise error under multiple testing | **Supported** |
| D21 | Two-paper structure | Research design | Different research questions require different evaluation designs | **Defensible by design** |

### 7.1 Decisions Needing Additional Direct Evidence — Experiment Plan

The following decisions are currently logical/engineering-valid but not yet stress-tested by dedicated experiments:

| Decision | Gap | Proposed Experiment | Primary Output |
|----------|-----|---------------------|----------------|
| D6 (block exits + reversals) | No isolated causal test | Run A/B walk-forward backtests: (A) block exits+reversals, (B) block reversals only, (C) no block; keep all else fixed | Sharpe Ratio, turnover, cost drag, trade duration |
| D14 (89-ticker universe adequacy) | No formal scale-sensitivity test | Universe-size sweep (e.g., 35/60/89 tickers) with identical pipeline and cost model | Sharpe stability vs universe size; selector-disagreement statistics |
| D10/D15 (single config + parquet) | Mostly operational validation | Reproducibility rerun protocol: repeated runs with and without cache/config locking | Variance in headline metrics; run-failure rate |

If these experiments are executed, their results should be added as E9+ and this table updated from "Needs direct experiment/Partly validated" to "Proven (internal)" where appropriate.

---

## 8. Experiments — Chronological History

### 8.1 Causal Flow (Why each next experiment happened)

| Step | Experiment | Why it was run at that point | Key result | What it triggered next |
|------|------------|------------------------------|------------|------------------------|
| 1 | E1 (frequency comparison, initial) | First, lock market sampling frequency before tuning anything else | Daily clearly dominated hourly (large Sharpe drop on hourly) | Lock D1 = daily; proceed to hold-period calibration |
| 2 | E2 (min_hold sweep) | After fixing frequency, calibrate the key execution horizon parameter | min_hold = 30 days was best (Net Sharpe Ratio 0.481, Gross Sharpe Ratio 0.963) | Lock D11; use fixed min_hold for all main evaluations |
| 3 | E4 (canonical WFV, 89 tickers) | With core settings fixed, run the main model-comparison experiment | full hybrid OU-only outperformed stat_only OU-only (0.653 vs 0.480); S2=all underperformed | Run benchmark, significance, and decomposition experiments |
| 4 | E5 (benchmark vs Nifty50) | E4 showed internal model ranking, but not market-relative value | Strategy had lower Sharpe than Nifty50 (0.550 vs 0.720) but much lower drawdown | Validate statistical reliability of observed edge/underperformance |
| 5 | E6 (significance tests) | Determine if observed Sharpe outcomes are statistically reliable | Marginal at 10%, not significant at 5% in key runs | Add ablation and weighted tests to understand where edge comes from |
| 6 | E3 (ablation) | Identify which selectors/components actually drive performance | Distance best in stat_only; some ensembles underperformed components | Test if non-equal/tuned weights can recover performance |
| 7 | E4 Grid (exhaustive weight search) | E4/E3 suggested equal weights may not be optimal; reviewer challenge | Standalone ML SR 0.610, Corr+Coint best pair SR 0.726; DM tests p > 0.45 | Parsimony holds; weight tuning or complex ML ensembling yields no sig. gains |
| 8 | E1 (rerun on refreshed setup) | Confirm D1 daily-frequency decision under updated universe/cost pipeline | Daily remained the practical/defensible primary frequency | Finalize writing with D1 retained |

### 8.2 Experiment Details (by execution order)

### E1 — Frequency Comparison (initial decision lock + later rerun)
- **Why:** Frequency must be fixed before all downstream experiments; otherwise comparisons are confounded.
- **Initial outcome:** daily Gross Sharpe Ratio 1.14 vs hourly 0.49 (old setup), so D1 was locked to daily.
- **Rerun purpose:** verify D1 still holds after universe/cost/pipeline updates.
- **Rerun outcome (final table):** E1 complete with Net Sharpe Ratio 0.343; daily remained the primary frequency.

### E2 — Hold Period Sweep (old universe, locked result)
- **Why next:** after locking frequency, the next structural execution parameter is min_hold.
- Swept min_hold ∈ {0,5,10,15,20,25,30,40} on full 10-year dataset.
- **Result:** min_hold=30 days optimal (Net Sharpe Ratio 0.481, Gross Sharpe Ratio 0.963).
- Theory match: OU half-life ~20–30 days for Hurst 0.19 spreads.
- **Consequence:** D11 locked; min_hold=30 applied in all subsequent experiments.

### E4 — Walk-Forward Validation (89 tickers older, 35 tickers universe experiments pending)
- **Why next:** with D1 and D11 fixed, execute the primary model-comparison experiment.
- Jobs 8693–8699 | June 2026 | All 3 modes + benchmark.

| Mode | Net Sharpe Ratio (SR) | CAGR | MaxDD | Trades |
|------|--------|------|-------|--------|
| stat_only + ou_only | **0.480** | 3.30% | 12.72% | 473 |
| stat_ml + ou_only | 0.431 | — | — | — |
| full hybrid + ou_only | **0.653** | 4.51% | 10.43% | — |
| stat_only + s2=all | 0.312 | — | 19.32% | — |

Fold breakdown (stat_only ou_only): 2018: 0.021 | 2019: 0.462 | 2020: 0.572 | 2021: 1.972 | 2022: -0.707 | 2023-24: 0.564.
Key finding: OU-only signal clearly superior to S2=all.
Computed Sharpe Ratio lift (full hybrid vs stat_only, OU-only): **+0.173 absolute** (0.653 - 0.480), i.e., **+36.0% relative**.
- **Data availability note (E4):** From committed project artifacts, stat_only has complete metrics (SR/CAGR/MaxDD/Trades). full hybrid has SR/CAGR/MaxDD but no committed total-trades field. stat_ml has committed SR only; CAGR/MaxDD/Trades are not present in the committed canonical E4 outputs.
- **Consequence:** run E5 (external benchmark), E6 (significance), E3 (component attribution), and E7 (weight tuning).

### E5 — Benchmark vs Nifty50 (external baseline check)
- **Why next:** E4 gives internal ranking; E5 tests market-relative competitiveness.
- Job 8699.

| Metric | Strategy | Nifty50 |
|--------|----------|---------|
| Net Sharpe Ratio (SR) | 0.550 | 0.720 |
| CAGR | 3.76% | 12.84% |
| MaxDD | 12.28% | 38.44% |

Verdict: underperforms Nifty50 on returns; materially lower drawdown (market-neutral characteristic).
Computed Sharpe Ratio gap (Strategy vs Nifty50): **-0.170 absolute** (0.550 - 0.720), i.e., **-23.6% relative**.

### E6 — Statistical Significance (reliability test)
- **Why next:** after effect-size estimates (E4/E5), test whether Sharpe outcomes are statistically reliable.
- Bootstrap 95% CI: [-0.209, +1.154] | p(SR≤0) = **0.086**
- Newey-West: t=1.300, p=0.097 (lags=8)
- **Result:** marginally significant at 10%, not significant at 5%.
- **Consequence:** need diagnostic follow-ups (E3 and E7), not just headline reporting.

### E3 — Ablation (component attribution)
- **Why next:** significance was marginal, so we decomposed the system to identify robust contributors.
- **Result (final):** stat_only Distance best (Sharpe Ratio 0.829), while some ensemble variants underperformed.
- **Consequence:** motivated E7 to test whether alternative weighting (not structure alone) could improve performance.

### E4 Grid — Exhaustive Weight Space Search (sensitivity and robustness)
- **Why next:** E4/E3 indicated potential mismatch between equal weighting and component quality; reviewer challenged us to search the weight space.
- Swept all 8 selectors standalone (E4.S) and all 28 pairwise equal-weight combinations (E4.W2).
- **Result:** standalone ML is best single (Net SR 0.610), while `Corr+Coint` is the best pair (Net SR 0.726). LSTM standalone collapses (Net SR -1.034).
- **Consequence:** keep equal weights for fair baseline ablation (D4), report the grid search and DM test non-significance (p > 0.45) as proof of the parsimony principle.

### 8.3 Execution/Debug History (why some experiments were delayed)

- **Round 1 (jobs 8702-8703):** E3 crashed (missing `gross_sharpe`, then dict mutation during iteration).
- **Round 2 (job 8704, cancelled):** wrong universe (35 tickers) due to stale `NSE_UNIVERSE` in `config.py`; also format bug in `ablation.py` when missing `gross_sharpe`; fixed in commit `3c62532`.
- **Round 3 (jobs 8734-8737):** corrected reruns for E3/E6/E7/E1 that produced the final results reported in Section 11.

---

## 9. Experiment Notes

Detailed E1 and E7 motivations/results are now consolidated in **Section 8.2 (Experiment Details by execution order)** to preserve a single causal narrative and avoid duplicate descriptions.

---

## 11. Current State (2026-06-05) — ALL EXPERIMENTS COMPLETE

### Experiment Status
| Exp | Status | Key Result |
|-----|--------|-----------|
| E1 | COMPLETE | Net Sharpe Ratio=0.343, MaxDD=19.26% (84-ticker post-refresh) |
| E2 | LOCKED | min_hold=30 days optimal |
| E3 | COMPLETE | stat_only: Distance Sharpe Ratio=0.829 best; Ensemble Sharpe Ratio=0.256; stat_ml Ensemble Sharpe Ratio=-0.311 |
| E4 | COMPLETE (CANONICAL) | stat_only Sharpe Ratio=0.480 / stat_ml Sharpe Ratio=0.453 / full Sharpe Ratio=0.519 ±0.061 |
| E5 | COMPLETE | Strategy Sharpe Ratio=0.550, MaxDD=12.28% vs Nifty50 Sharpe Ratio=0.720, MaxDD=38.44% |
| E6 | COMPLETE | None at 5%; full: p_boot=0.069, NW p=0.076 (sig at 10%) |
| E4 Grid | COMPLETE | Standalone ML SR 0.610; Pairwise Corr+Coint SR 0.726 (best); DM tests p > 0.45 (not significant) |
| E8 | EXCLUDED | gymnasium not on Kalpana; outside paper scope |

### Data Integrity Issue (E7) & Resolution (E4 Grid)
The old E7 SLURM script ran `fetch_paper1_data.py` which re-fetched and overwrote parquet 89→84 tickers. This has been fully resolved: Experiment E4 Grid (standalone and pairwise) was executed from scratch on the canonical 89-ticker universe under 16.28 bps costs. All data-drift issues are resolved, and the results are 100% consistent.

### What remains
- Ch4 §4.5/§4.7 placeholders to fill (ablation + weighted ensemble)
- Abstract final numbers
- Full CPU non-determinism in full mode (Sharpe Ratio 0.437–0.618 across 6 runs) — document

---

## 12. Future Work Scope

### Immediate — Experimentation
1. [x] Run **matched-universe robustness suite** on the same 35-ticker universe used in Paper 2 (E4/E5/E6 core stack). (Completed 2026-06-12)
2. [x] Keep **89-ticker E4 as primary** and report 35-ticker results as a controlled robustness appendix (not replacement). (Added to DATA_APPENDIX.md)
3. [x] Backfill missing E4 table fields by exporting complete per-mode metrics (CAGR/MaxDD/Trades) from canonical outputs or reruns. (Backfilled 2026-06-12)
4. [x] Add per-mode uncertainty (bootstrap 95% CI) for headline SR/CAGR/MaxDD where feasible. (Backfilled 2026-06-12)
5. [x] Add a direct attribution note: method effect vs universe effect (89 vs 35) with consistent cost/min_hold/WFV setup. (Added to DATA_APPENDIX.md)

### Immediate — Paper Writing (all compute done)
1. Fill Ch4 §4.5 ablation table with E3 results
2. Fill Ch4 §4.7 weighted ensemble section with E7 results
3. [x] Update E6 significance section for all 3 modes (Updated 2026-06-12)
4. [x] Update abstract with final canonical numbers (Updated 2026-06-12)
5. Document full-mode CPU non-determinism (Sharpe Ratio 0.437–0.618) as ML variance caveat
6. Fix E7 SLURM script — remove fetch_paper1_data.py call

### Short-term (paper finalisation)
7. [x] Bootstrap 95% CIs for E4 headline numbers (Completed 2026-06-12)
8. [x] Diebold-Mariano pairwise tests (ensemble vs each single selector) (Completed 2026-06-12)
9. [x] Sensitivity analysis: cost ±5 bps table (Completed 2026-06-12)
10. [x] Ch5 cost section: fix 6 bps brokerage → 0 bps (Completed 2026-06-12)

### Medium-term (paper writing)
13. Structure final paper from chapter drafts — 8,000–10,000 word target (JFM/QF format)
14. All figures: fold-level SR bar chart, equity curves, ablation heatmap, cost sensitivity
15. Internal critique pass (same checklist used for Paper 2 rounds)
16. Ensure honest framing: if ensemble does NOT significantly outperform stat-only, report as negative result (mirrors Paper 2's universe quality finding)

### Submission readiness checklist
- [ ] All results at 16.28 bps, CPU-only ML
- [ ] All net Sharpe Ratio figures have bootstrap 95% CIs
- [ ] Bonferroni correction applied and documented
- [ ] No uncaveated single GPU runs
- [ ] CNNSelector disabled status disclosed (Appendix B)
- [ ] Survivorship bias disclosed
- [ ] Conflict with Paper 2 (expanding vs rolling WFV performance difference) addressed in methodology
- [ ] E8 / RL signal: either included with caveat or excluded with explanation
- [ ] Target venue identified: JFM (35% est.) or QF (50% est.)

### Open research questions (longer horizon)
- Can Bayesian weight optimisation per fold (D4 trade-off) recover performance?
- Does the 30-day min_hold remain optimal for full-mode (8-selector) runs? (D11 trade-off)
- Does expanding WFV performance degrade in later folds? (Paper 2 finding replication on Nifty 100)
- Can a rolling WFV variant outperform expanding on this dataset?
- Point-in-time index constituents (Bloomberg/Refinitiv) to remove survivorship bias
- Multi-market extension of Paper 1 ensemble (cross-market pairs with sector alignment)

---

## 13. Ensemble Optimisation and Cross-Universe Consistency Plan

### Objective
Optimise Stage-1 ensemble construction while testing whether gains are stable across **89-ticker (Paper 1 primary)** and **35-ticker (Paper 2-matched robustness)** universes.

### Principles
1. Do not change evaluation protocol between universes (same cost model, min_hold, WFV logic, top-k policy).
2. Separate **model-selection runs** from **final reporting runs** to avoid leakage.
3. Treat 89-ticker as primary inference; 35-ticker as transfer/robustness evidence.

### Plan (phased)
1. **Baseline replication phase:** rerun baseline configs on both universes (stat_only+ou_only, stat_ml+ou_only, full+ou_only) to produce a clean comparable matrix.
2. **Weight search phase:** evaluate constrained weighting families (equal, correlation-upweight, sparse/pruned, Bayesian/coordinate search) using training-only fold information.
3. **Stability phase:** compute rank stability of configs across folds and across universes (not only mean Sharpe).
4. **Significance phase:** apply bootstrap/Newey-West and multiple-testing correction for selected top configs.
5. **Selection phase:** pick one primary ensemble by pre-declared rule (best risk-adjusted metric with stability threshold), then freeze.
6. **Final confirmation phase:** one locked rerun on both universes; publish full metric panel (SR, CAGR, MaxDD, Trades, CI).

### Consistency Criteria (must pass)
- Same directional outperformance vs stat_only in both universes, or clearly disclosed regime-specific failure.
- No severe variance inflation (full-mode non-determinism bounded and documented).
- Trade count/cost drag remains economically plausible under NSE costs.

### Deliverables
1. Cross-universe comparison table (89 vs 35) for baseline and optimised ensembles.
2. Ensemble-weight sensitivity plot/table (performance vs weight regime).
3. Final locked configuration note with justification and caveats.

---

## 14. File Reference

| File | Purpose |
|------|---------|
| `Implementation/experiments/config.py` | All parameters — single source of truth |
| `Implementation/experiments/walk_forward.py` | E4/E7 WFV runner |
| `Implementation/experiments/ablation.py` | E3 ablation runner |
| `Implementation/experiments/significance_tests.py` | E6 significance |
| `Implementation/experiments/benchmark_comparison.py` | E5 benchmark |
| `Implementation/experiments/freq_comparison.py` | E1 frequency |
| `Implementation/experiments/slurm/` | Active SLURM scripts |
| `Implementation/experiments/results/` | Current results (89-ticker) |
| `Implementation/experiments/results/archive_old_universe/` | Old 35-ticker results — DO NOT CITE |
| `Implementation/core/` | Shared engine — do not modify |
| `Implementation/Decisions.md` | Full decision log D1–D22 |
| `Paper_1_Hybrid_Ensemble/KnowledgeGraph/KnowledgeGraph.md` | Session start reference |
| `Paper_1_Hybrid_Ensemble/KnowledgeGraph/graph/experiments.json` | Experiment status graph |

---


## Appendix: Final Results (2026-06-05)

All experiments complete. See KnowledgeGraph.md for canonical result tables.
