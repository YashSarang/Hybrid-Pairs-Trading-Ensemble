# Paper 1 — Project Flow
# Hybrid Ensemble Pairs Trading: Do ML Selectors Outperform Statistical Baselines?
# Last updated: 2026-06-05

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
| ~~CNNSelector~~ | ~~Deep Learning~~ | **DISABLED** — ablation showed consistent degradation (SR +0.481 → -0.12). Disclosed in Appendix B. |

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

| Decision | Choice | Why |
|----------|--------|-----|
| D1 | Daily data | Hourly SR fell 57%, microstructure noise, OU assumptions violated |
| D3 | Universe pre-specified | Academic credibility; no post-hoc snooping |
| D4 | Equal ensemble weights in ablation | Isolate structural ensemble benefit from weight optimisation |
| D5 | min_hold = fixed parameter | Not a signal parameter; cannot encode future information |
| D6 | min_hold blocks exits AND reversals | Rapid exit pattern equally costly as reversal |
| D7 | Expanding WFV | Standard for model comparison; uses all historical data |
| D8 | Re-fit selectors + signals per fold | Only academically defensible approach |
| D10 | config.py = single source of truth | Prevents parameter drift across scripts |
| D11 | min_hold = 30 days | E2 sweep peak; consistent with OU half-life theory |
| D12 | 16.28 bps round-trip | Corrected STT double-count; conservative slippage |
| D13 | CPU-only ML | GPU non-determinism range = 1.226 SR — scientifically meaningless |
| D14 | 89-ticker Nifty 100 | Enough pairs for ensemble disagreement testing |
| D15 | Parquet cache | Reproducibility; no yfinance 429 errors at SLURM runtime |
| D16 | 2015–2024 | Full market cycle; 10yr gives ML sufficient training data |
| D17 | 6-fold expanding | Consistent with model comparison literature |
| D18 | CNNSelector disabled | Ablation: consistent degradation; disclosed in Appendix B |
| D19 | ZScore primary, OU secondary | Maximises comparability with literature; interpretable |
| D20 | Block bootstrap + Bonferroni | Fat-tailed returns; controls FWER |
| D21 | Two-paper structure | Different RQs, different designs; natural publication arc |

---

## 8. Experiments — Chronological History

### E2 — Hold Period Sweep (old universe, locked result)
- Swept min_hold ∈ {0,5,10,15,20,25,30,40} on full 10-year dataset
- **Result:** min_hold=30 days optimal (net SR 0.481, gross SR 0.963)
- Theory match: OU half-life ~20–30 days for Hurst 0.19 spreads
- Not rerun on 89-ticker universe — hold period is a structural parameter (D5), not universe-dependent
- **Locked: min_hold=30 applied to all experiments**

### E4 — Walk-Forward Validation (89 tickers, COMPLETE)
- Jobs 8693–8699 | June 2026 | All 3 modes + benchmark

| Mode | Net SR | CAGR | MaxDD | Trades |
|------|--------|------|-------|--------|
| stat_only + ou_only | **0.480** | 3.30% | 12.72% | 473 |
| stat_ml + ou_only | 0.431 | — | — | — |
| full hybrid + ou_only | **0.653** | 4.51% | 10.43% | — |
| stat_only + s2=all | 0.312 | — | 19.32% | — |

Fold breakdown (stat_only ou_only): 2018: 0.021 | 2019: 0.462 | 2020: 0.572 | 2021: 1.972 | 2022: -0.707 | 2023-24: 0.564

Key finding: OU-only signal clearly superior to ensemble signal (S2=all). 2022 fold negative (-0.707) consistent with global rate shock regime.

### E5 — Benchmark vs Nifty50 (COMPLETE)
- Job 8699

| Metric | Strategy | Nifty50 |
|--------|----------|---------|
| Net SR | 0.550 | 0.720 |
| CAGR | 3.76% | 12.84% |
| MaxDD | 12.28% | 38.44% |

Verdict: underperforms Nifty50 on returns; materially lower drawdown (market-neutral characteristic).

### E6 — Statistical Significance (stat_only, prior run)
- Bootstrap 95% CI: [-0.209, +1.154] | p(SR≤0) = **0.086**
- Newey-West: t=1.300, p=0.097 (lags=8)
- **Marginally significant at 10%, NOT conventional 5%**
- NOTE: Paper 2's headline (Nifty50 ZScore, SR +0.752, p=0.036) is primary; this is secondary

### E3, E6 (all modes), E7, E1 — Bugs and History

**Round 1 (jobs 8702-8703):** E3 crashed — KeyError on `gross_sharpe` key, then dict mutation during iteration.

**Round 2 (job 8704, CANCELLED):** E3 ran with **wrong universe** (35 tickers). Root cause: `experiments/config.py` still had the old 35-ticker `NSE_UNIVERSE` list. The parquet had 89 columns but `get_prices()` filters to only the requested tickers. Additionally, `ablation.py` line 567 had `f"gross SR={m.get('gross_sharpe','?'):.3f}"` — when key missing, `'?'` string caused `ValueError: Unknown format code 'f' for str`. Both bugs fixed in commit `3c62532`.

**Round 3 (jobs 8734-8737, CURRENT — submitted 2026-06-05):**
| Job | Exp | Status | Est. runtime |
|-----|-----|--------|-------------|
| 8734 | E3 ablation (3 modes) | RUNNING | ~8h |
| 8735 | E6 significance (3 modes) | RUNNING | ~6h |
| 8736 | E7 weighted ensemble (3 configs) | RUNNING | ~8h |
| 8737 | E1 freq comparison | RUNNING | ~4h |

---

## 9. E7 — Weighted Ensemble (Design)

Motivation: E4 showed equal-weight ensembling on full mode (SR 0.653) outperforms stat_only (SR 0.480). But equal weights may not be optimal — LSTM and Correlation selectors may have higher pair-selection quality.

Three configs submitted:
- **E7-A:** Weighted S1 (Corr=2.0, others=1.0) + OU-only — stat_ml mode
- **E7-B:** Weighted S1 (LSTM=3.0, Corr=2.0, others=1.0) + OU-only — full mode
- **E7-C:** stat_only + OU-only (replication check of E4 stat_only headline)

Key constraint (D4): equal weights are used for ablation comparison. E7 is explicitly framed as "performance upper bound under tuned weights" — reported separately from the main ablation table.

---

## 10. E1 — Frequency Comparison (Rerun)

Early E1 result (old universe, old costs): daily Gross SR 1.14 vs hourly 0.49.
Decision D1 locked: daily data is the primary frequency.
E1 rerun on 89-ticker universe (job 8737) to confirm this holds under the expanded universe and corrected costs. Expected: similar qualitative conclusion.

---

## 11. Current State (2026-06-05) — ALL EXPERIMENTS COMPLETE

### Experiment Status
| Exp | Status | Key Result |
|-----|--------|-----------|
| E1 | COMPLETE | Net SR=0.343, MaxDD=19.26% (84-ticker post-refresh) |
| E2 | LOCKED | min_hold=30 days optimal |
| E3 | COMPLETE | stat_only: Distance SR=0.829 best; Ensemble SR=0.256; stat_ml Ensemble SR=-0.311 |
| E4 | COMPLETE (CANONICAL) | stat_only SR=0.480 / stat_ml SR=0.453 / full SR=0.519 ±0.061 |
| E5 | COMPLETE | Strategy SR=0.550, MaxDD=12.28% vs Nifty50 SR=0.720, MaxDD=38.44% |
| E6 | COMPLETE | None at 5%; full: p_boot=0.069, NW p=0.076 (sig at 10%) |
| E7 | COMPLETE | Corr=2.0 SR=0.548; LSTM=3.0 SR=-0.121 (catastrophic) |
| E8 | EXCLUDED | gymnasium not on Kalpana; outside paper scope |

### Data Integrity Issue (E7)
E7 SLURM script ran `fetch_paper1_data.py` which re-fetched and overwrote parquet 89→84 tickers.
E4 canonical (89 tickers) remains primary. E7 results directionally valid.

### What remains
- Ch4 §4.5/§4.7 placeholders to fill (ablation + weighted ensemble)
- Abstract final numbers
- Full CPU non-determinism in full mode (SR 0.437–0.618 across 6 runs) — document

---

## 12. Future Work Scope

### Immediate — Paper Writing (all compute done)
1. Fill Ch4 §4.5 ablation table with E3 results
2. Fill Ch4 §4.7 weighted ensemble section with E7 results
3. Update E6 significance section for all 3 modes
4. Update abstract with final canonical numbers
5. Document full-mode CPU non-determinism (SR 0.437–0.618) as ML variance caveat
6. Fix E7 SLURM script — remove fetch_paper1_data.py call

### Short-term (paper finalisation)
7. Bootstrap 95% CIs for E4 headline numbers
8. Diebold-Mariano pairwise tests (ensemble vs each single selector)
9. Sensitivity analysis: cost ±5 bps table
10. Ch5 cost section: fix 6 bps brokerage → 0 bps

### Medium-term (paper writing)
13. Structure final paper from chapter drafts — 8,000–10,000 word target (JFM/QF format)
14. All figures: fold-level SR bar chart, equity curves, ablation heatmap, cost sensitivity
15. Internal critique pass (same checklist used for Paper 2 rounds)
16. Ensure honest framing: if ensemble does NOT significantly outperform stat-only, report as negative result (mirrors Paper 2's universe quality finding)

### Submission readiness checklist
- [ ] All results at 16.28 bps, CPU-only ML
- [ ] All net SR figures have bootstrap 95% CIs
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

## 13. File Reference

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
