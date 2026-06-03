# Project Research Log: Hybrid Pairs Trading Ensemble

## Overview

**Research Question:** Do hybrid ensemble pair selectors improve NSE Nifty 50 pairs trading profitability? Does universe quality (Nifty 50 vs Nifty 100) dominate methodology improvements?

This project investigates statistical and machine-learning-based pair selection methods for equity pairs trading on the Indian National Stock Exchange (NSE), with multi-market walk-forward validation across India, the United States, Brazil, and the United Kingdom.

**Markets covered:**
- India: NSE Nifty 50 and Nifty 100
- United States: S&P 500
- Brazil: B3
- United Kingdom: FTSE 100

**Pair selectors implemented:**
- Statistical: Correlation, Distance, Cointegration, Combined (ensemble of above three)
- Machine learning: LSTM, Transformer, Graph Neural Network (GNN)

**Signal models:** ZScore (standard Bollinger-band spread reversion), Ornstein-Uhlenbeck (OU process calibration)

**Transaction costs applied:**
- India: 16.28 bps round-trip
- United States: 2.74 bps round-trip
- Brazil: 8.4–16.4 bps round-trip (sensitivity tested up to 30 bps)
- United Kingdom: 8.0 bps round-trip

**Infrastructure:** Python codebase, yfinance data sourcing, SLURM cluster (Kalpana HPC, IIT Bombay, cminds_anandi partition).

---

## Phase 1: Initial Thesis (Chapter 3) — NSE Baseline

### Setup

The initial Chapter 3 experiment used an expanding-window walk-forward methodology over 6 folds spanning 2020–2025, applied to NSE Nifty 100 stocks (35 tickers available with sufficient history). The ensemble combined correlation, distance, and cointegration selectors with a ZScore signal model.

### Results

- Mean Net Sharpe: -0.409 ± 0.738
- Positive folds: 2 out of 6
- Total trades: 1096
- Cost drag: approximately -0.526 Sharpe per fold

### Root Cause Analysis

The expanding window methodology accumulated pairs and trades across folds, resulting in high average turnover (183 trades per fold). Transaction cost drag at 16.28 bps eliminated any gross alpha.

### Fix Attempted: Rolling Window Methodology (Section 3.6)

A rolling window design was substituted for the expanding window. Each fold used a fixed training and test window, preventing turnover accumulation.

**Rolling window result:** Mean Net Sharpe +0.052, 293 total trades, cost drag substantially reduced.

### Ablation Study

An ablation across individual selectors was run. The correlation selector performed best as a standalone. The full ensemble added only marginal improvement over the best individual selector.

### ML Non-Determinism Discovery

During GPU-accelerated runs of the LSTM and Transformer selectors, Sharpe variance of 1.226 was observed across three otherwise identical runs (results: +0.398, -0.386, +0.840). This rendered the GPU-based ML results irreproducible and scientifically indefensible.

**Fix:** CPU-only deterministic execution enforced via environment variables:
- `CUDA_VISIBLE_DEVICES=""`
- `TF_DETERMINISTIC_OPS=1`

**CPU deterministic range:** +0.353 to +0.484 across replicates (difference 0.131, compared to GPU variance of 1.226). This was accepted as an upper bound on remaining numerical instability.

---

## Phase 2: Multi-Market Expansion (Chapter 4)

### Setup

The rolling window methodology established in Phase 1 was extended to three additional markets: US (S&P 500), Brazil (B3), and UK (FTSE 100). All markets used the same selector and signal architecture.

### Results by Market

**India (NSE Nifty 50, ZScore):**
- Best GPU run (originally reported headline): +0.840 Sharpe
- 3-run mean: +0.284 ± 0.631
- CPU deterministic range: +0.353 to +0.484

**United States (S&P 500, ZScore, n=1 exploratory):** +0.774, dominated by 2022 bear market fold (+2.147 in that fold alone)

**United States (OU, 3 runs):** -0.085 mean

**Brazil (OU, 3 runs):** +0.107 mean, best single run +0.321

**Brazil (ZScore):** -0.312 mean

**United Kingdom (ZScore):** -0.245

**United Kingdom (OU):** -0.405

### Initial Interpretation

The India Nifty 50 result of +0.840 was approximately 5.5x the Nifty 100 rolling baseline (+0.052). The initial claim formulated was that "universe quality dominates methodology" — that selecting a cleaner, more liquid universe (Nifty 50) produced more alpha than any methodological improvement. A secondary claim of "geographic alpha" was advanced for the India outperformance relative to other markets.

---

## Phase 3: Critique and Salvage

**Artefacts:** CRITIQUE.md and SALVAGE_PLAN.md written to document identified problems before submission.

### Six Fatal Flaws Identified

1. **Confound unresolved:** Nifty 50 vs Nifty 100 comparison conflates universe quality with geography. The two universes were never tested with all other variables held constant.
2. **ML non-determinism:** Three GPU runs produced +0.398, -0.386, +0.840. Results are irreproducible. The headline +0.840 may be a lucky seed outcome.
3. **P-hacking via rolling window selection:** The rolling window was chosen post-hoc after observing that expanding windows produced negative results. This is a form of methodological selection bias.
4. **Multiple runs without transparency:** Best-of-3 was reported without disclosing the other two runs.
5. **Small sample n=4:** One outlier fold drives any apparent statistical significance.
6. **No control experiment:** NSE Nifty 50 had never been run standalone as a baseline; the comparison was always Nifty 50 (multi-market) versus Nifty 100 (India-only baseline).

### Control Experiment: NSE Nifty 50 Standalone

To isolate the universe quality effect, NSE Nifty 50 was run standalone using the rolling window ZScore methodology.

**Result:** +0.752 Sharpe (4-fold, 2021–2024), 95% CI [+0.422, +1.082], one-sample t-test p=0.036.

This confirmed **Scenario A**: the Nifty 50 outperformance is attributable to universe quality (liquid, coherent large-cap index), not to geography or the multi-market methodology.

**Implied uplift:** +0.700 Sharpe units from Nifty 50 (+0.752) over Nifty 100 (+0.052).

### Additional Discoveries

- A UK positive run (Run 1: +0.265) was found in the archive. It had not been reported. This confirmed selective reporting had occurred.
- A lookback bug was identified: lookback=252 trading days was exhausting the available test window length in several folds, artificially reducing trade counts. Fixed by switching to lookback=126.

---

## Phase 4: Figures and Publication Preparation

Following the salvage plan, the paper was restructured for submission to the Journal of Financial Markets (JFM).

**Figures generated:**
- Figure 4.1: Fold-by-fold Sharpe across selectors with bootstrap 95% CI error bars
- Figure 4.3: Universe quality comparison (Nifty 50 vs Nifty 100) with bootstrap CI

**Statistical additions:**
- Bootstrap Difference-in-Differences (DiD) test: P(Universe Quality > Methodology) = 0.811
- HAC Newey-West autocorrelation correction: lag=1, HAC-corrected p=0.012 (compared to naive p=0.037)

**Benchmark comparison added:**
- Fold-by-fold returns compared to equal-weight Nifty 50 buy-and-hold
- Raw alpha was negative in 3 of 4 folds (strategy is market-neutral at approximately 5% annualised volatility vs index at approximately 15%)
- Calmar ratio: mean 1.844; maximum drawdown: 3.1%

**Transparency report written:** Documented all 33+ experimental runs, including failed runs, best/worst outcomes, and methodology evolution.

---

## Phase 5: JFM Submission — Round 1 Critique (11 Items)

The paper was submitted to JFM. The desk review returned 11 critique items.

### Item 1: Data Period Too Short
An 8-fold extension was submitted (SLURM job 8543), covering 2017–2024.

**8-fold result:** Mean Sharpe +0.242, p=0.473. Not statistically significant.

Key observation: 2021–2024 was a favourable regime. Adding 2019 (Sharpe -0.835) and 2020 (Sharpe -0.876) diluted the signal.

**Multiple testing corrections applied:**
- m_eff = 13.42 (eigenvalue decomposition, Nyholt 2004 method) — Bonferroni-corrected p moved from 0.936 to 0.491
- BH-FDR threshold: 0.00373 — primary p=0.036 fails this threshold
- Sign test (4/4 positive folds): p=0.0625 (borderline, not significant at 5%)

### Item 2: Title Claim Untested
Bootstrap DiD test added (P=0.811 in favour of universe quality). Addressed.

### Item 3: Concentration Risk
Intra-sector pair analysis conducted: 9.1% of 595 pairs are intra-sector. No extreme concentration found. Partially addressed.

### Item 4: Benchmark Comparison
Fold-by-fold buy-and-hold benchmark added. Addressed.

### Item 5: Fold Independence
HAC Newey-West correction applied. Addressed.

### Item 6: Risk Metrics
Calmar ratio, maximum drawdown, and cost drag metrics added. Partially addressed.

### Item 7: Fama-French Factor Attribution
An EM (emerging market) Fama-French proxy was run. Result: alpha +9.84% per annum, t-statistic = 9.01 (4-fold).

**Subsequently found to be circular:** All trading days within a fold were assigned identical synthetic monthly factor returns, so the t-statistic is a function of the number of days in the fold, not a genuine factor relationship. Section removed entirely in Phase 6.

### Item 8: CVaR
CVaR was computed using 2024–2026 deployment run data.

**Subsequently found to be wrong period:** CVaR for a 2021–2024 backtest must be derived from 2021–2024 data. Fixed in Phase 6.

### Item 9: Cross-Market Universe Test
Acknowledged as future work at this stage. Addressed in Phase 7 via 16-fold paired test.

### Item 10: Survivorship Bias
A Sharpe × 0.92 adjustment was cited from Elton et al. (1996).

**Subsequently found to be invalid:** Elton et al. measures mutual fund survivorship bias, not pairs trading or index membership survivorship. No valid derivation exists for this specific adjustment. Removed in Phase 6.

### Item 11: Brazil Transaction Costs
Full cost sensitivity analysis provided: at 16.4 bps, Brazil Sharpe = -0.216; at 30 bps, Sharpe = -0.768.

---

## Phase 6: Round 2 Critique — Errors Introduced During Round 1 Response

A second review pass identified three fatal flaws and five major concerns introduced or uncorrected during the Round 1 response.

### Fatal Flaw 3: Circular Fama-French Attribution
The FF alpha computation assigned all trading days in a fold a single monthly factor return value. The t-statistic therefore scaled with the number of daily observations (approximately 250 per fold), not the quality of the factor relationship. A t-statistic of 9.01 is a mathematical artefact.

**Fix:** Section removed entirely from the paper. No FF attribution claim made.

### CVaR Period Error
CVaR had been computed using 2024–2026 out-of-sample deployment data, not the 2021–2024 backtest period.

**Fix:** Recomputed from fold-level daily pnl_net over 2021–2024. Corrected values: CVaR@95% = -0.549%/day, CVaR@99% = -1.123%/day (pooled across folds).

### Survivorship Bias Adjustment Removed
The Elton et al. ×0.92 citation is not applicable to index membership survivorship in equity pairs trading. No quantitative adjustment is defensible without a valid derivation.

**Fix:** Replaced with a qualitative disclosure noting that the Nifty 50 universe represents the current index composition and that delisted constituents are excluded from the backtest.

### Brazil Cost Sensitivity Arithmetic Error
The original sensitivity analysis mixed series: gross returns from the best run were combined with net cost adjustments derived from the mean run. This produced inconsistent cost drag estimates.

**Fix:** Recomputed using a consistent basis. drag_per_bps = 0.00154 (best run), 0.000474 (mean run). Conclusion unchanged: Brazil has near-zero gross alpha regardless of cost assumptions.

### Structural Fix: Section Misplacement
Sections 4.4.9 and 4.4.10 were located after the Chapter Conclusions (§4.7). This placed key empirical analysis after the summary of findings.

**Fix:** Sections relocated to their correct position before §4.5.

### Other Fixes
- HAC claim corrected: text had implied HAC Newey-West strengthened inference. Corrected to describe HAC as a robustness check, not an inferential strengthening.
- CVaR summary table values brought into consistency with text.
- Abstract cleaned: acceptance likelihood language removed, committee recommendation language removed, date inconsistencies resolved.

### Additions
- **Pre-registration framework (§5.5.4):** Study formally classified as exploratory. H1 stated as a pre-registration candidate with a paired Wilcoxon protocol for any future confirmatory study.
- **Quaternary contribution (§5.1.1):** Regime-conditionality of NSE pairs trading alpha formalised as a contribution. Sharpe ratios cycle between strongly positive and negative years; 2021–2024 was an anomalously favourable mean-reversion environment.

---

## Phase 7: Long-Run Validation (16-Fold)

### Motivation

Two unresolved problems remained after Phase 6:
1. **n=4 (Fatal Flaw 2 from Round 2):** Four folds are insufficient to distinguish signal from noise or regime artefact.
2. **No 20-year sample:** All experiments covered at most 8 years, preventing assessment of structural robustness.

### Implementation

**NSE Nifty 50 Long-Run:**
- 31 NSE tickers identified with continuous data from 2004 (TATAMOTORS, COALINDIA, and POWERGRID excluded for data continuity reasons)
- Config file created: `nse_nifty50_longrun.yaml` (31 tickers, 2004–2024, 16 annual folds)
- Config patched to add `selectors.weights` block (missing block caused a KeyError on execution)
- SLURM job 8653 submitted: ZScore signal, 4 selectors (correlation, distance, cointegration, combined), 16 folds

**NSE Nifty 100 Long-Run Paired Control:**
- 47 tickers identified (31 Nifty 50 tickers plus 16 mid-cap additions with data from 2004)
- Config file created: `nse_nifty100_longrun.yaml`
- SLURM job 8654 submitted: identical 16-fold protocol on broader universe

**Data:** Prices cached as `prices_2004-01-01_2024-12-31.parquet` for both configurations.

---

## Final Results Summary

### NSE Nifty 50 — 16-Fold (2005–2024)

Fold-by-fold Sharpe ratios: 0.622, 0.270, 1.273, -0.898, 0.033, 1.537, -1.017, 0.240, -0.210, -2.076, 0.315, 0.133, 0.579, 0.224, 0.642, -0.053

- Mean: +0.101
- Std: 0.874
- t-statistic: 0.462
- p-value: 0.651
- 95% CI: [-0.365, +0.566]
- Positive folds: 11 of 16
- Cohen's d: 0.115

### NSE Nifty 100 — 16-Fold (2005–2024)

Fold-by-fold Sharpe ratios: 0.484, 0.432, 1.067, -0.651, -0.201, 1.542, -1.073, -0.760, 0.603, -0.784, -0.555, 0.821, 0.457, -0.400, 1.554, 0.058

- Mean: +0.162
- Std: 0.835
- t-statistic: 0.777
- p-value: 0.449
- 95% CI: [-0.283, +0.607]
- Positive folds: 9 of 16
- Cohen's d: 0.194

### Paired Difference (Nifty 50 minus Nifty 100)

- Mean difference: -0.061
- Paired t-statistic: -0.389
- p-value: 0.703
- Wilcoxon signed-rank p: 0.860

### Interpretation

The 4-fold result (+0.752, p=0.036) was a regime artefact. The 2021–2024 period was an anomalously favourable mean-reversion environment for NSE large-cap equities. Over 20 years, NSE Nifty 50 pairs trading produces a near-zero mean Sharpe with high year-to-year variance.

The universe quality hypothesis is not supported at the 20-year horizon. Nifty 100 marginally outperforms Nifty 50 (mean +0.162 vs +0.101), and the difference is not statistically significant at any conventional threshold (Wilcoxon p=0.860).

The genuine finding from the full project is **regime-conditionality**: the strategy Sharpe is strongly time-varying, with some years producing Sharpe above +1.5 and others below -1.0. Any short-window evaluation risks confounding regime effects with strategy quality.

### Summary Table: All Experiments

**NSE Nifty100 Expanding Window** — 2020–2025, 6-fold — Mean Sharpe: -0.409 — Not significant (negative)

**NSE Nifty100 Rolling Window** — 2020–2025 — Mean Sharpe: +0.052 — p=0.32 — Not significant

**NSE Nifty50 Rolling Window** — 2021–2024, 4-fold — Mean Sharpe: +0.752 — p=0.036 — Significant, but regime artefact

**NSE Nifty50 8-fold** — 2017–2024 — Mean Sharpe: +0.242 — p=0.473 — Not significant

**NSE Nifty50 16-fold** — 2005–2024 — Mean Sharpe: +0.101 — p=0.651 — Not significant

**NSE Nifty100 16-fold** — 2005–2024 — Mean Sharpe: +0.162 — p=0.449 — Not significant

**US ZScore** — 2021–2024, n=1 — Mean Sharpe: +0.774 — Exploratory only

**Brazil OU** — 2021–2024 — Mean Sharpe: +0.107 — Not significant

**UK ZScore** — 2021–2024 — Mean Sharpe: -0.245 — Negative

---

## Critique Resolution Status

### Round 1 (11 Items)

- **Items 2, 4, 5:** Fully addressed (DiD test, benchmark, HAC correction).
- **Items 1, 3, 6:** Partially addressed (8-fold extended data; sector concentration quantified; Calmar and MaxDD added).
- **Item 7 (Fama-French):** Addressed by complete removal (result was circular).
- **Items 8, 10, 11:** Fixed (CVaR recomputed from correct period; survivorship ×0.92 removed; Brazil cost arithmetic corrected).
- **Item 9 (cross-market universe test):** Addressed in Phase 7 — Nifty 50 vs Nifty 100 16-fold paired test completed.

### Round 2 (3 Fatal Flaws + 5 Major Concerns)

- **FF2 (n=4):** Fixed — 16-fold run completed.
- **FF3 (circular FF):** Fixed — section removed.
- **FF1 (no significant result):** Remains open. The 16-fold result p=0.651 provides no significant finding at any correction level. The primary hypothesis (universe quality dominance) is not supported at 20 years.
- **All 5 major concerns:** Fixed — CVaR period, US fold outlier disclosure, Brazil arithmetic, survivorship removal, and section placement.

---

## Venue Assessment

**Journal of Financial Markets (JFM):** Not viable. The paper has no statistically significant primary result. The original hypothesis is contradicted by 20-year data. JFM expects confirmatory findings with adequate power.

**Quantitative Finance:** Viable after reframing as a null result study with a regime-conditionality finding as the primary contribution. Audience is receptive to negative and conditional results if methodology is rigorous.

**Emerging Markets Review:** Best current fit. The NSE and India-market scope aligns directly with the journal's remit. Accepts shorter samples and exploratory framing. The regime-conditionality finding has direct policy relevance for practitioners in emerging equity markets.

**Finance Research Letters:** Viable as a focused 4,000-word note centred on the regime-conditionality finding, with the 16-fold fold-by-fold results as the empirical core.

---

## Remaining Work

1. **Section 4.4.11:** Fill with actual 16-fold results. Currently contains placeholder TBD values.
2. **Abstract update:** Incorporate 16-fold numbers and replace the universe quality framing with regime-conditionality framing.
3. **Central contribution reframe:** Shift from "universe quality dominates methodology" to "NSE pairs trading alpha is strongly regime-conditional; 2021–2024 was an anomalously favourable period."
4. **JFM_CRITIQUE_RESPONSE.md:** Update to reflect final resolution status for all Round 1 and Round 2 items.
5. **Venue decision:** Select one of Emerging Markets Review, Quantitative Finance, or Finance Research Letters and prepare the appropriate cover letter and manuscript format.
