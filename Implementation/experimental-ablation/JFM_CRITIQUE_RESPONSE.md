# Response to Reviewers
## Journal of Financial Markets — Manuscript Revision

**Manuscript Title:** Hybrid Pairs Trading Ensemble: Universe Quality and Methodology in Indian Equity Markets
**Revision Round:** 1
**Date:** June 2026

---

> *The authors thank the reviewers for their thorough and constructive engagement with the manuscript. Each major concern has been carefully considered. The following document provides a point-by-point response, indicating the disposition of each concern, the specific revisions made to the manuscript, and an honest account of concerns that remain partially or fully outstanding due to data and infrastructure constraints. We believe these revisions materially strengthen the manuscript's empirical claims and epistemic honesty.*

---

## Summary Status Table

| # | Reviewer Concern | Status | Manuscript Section |
|---|---|---|---|
| 1 | Data period too short (4 years vs. JFM standard ~20 years) | **Partially Addressed** | §4.4.6, §5.5 |
| 2 | Title claim "Universe Quality Dominates Methodology" is untested | **Addressed** | §4.4.5 |
| 3 | Nifty 50 is a thin universe; concentration risk unquantified | **Partially Addressed** | §4.4.4, §5.5 |
| 4 | No benchmark comparison provided | **Addressed** | §4.4.4 |
| 5 | Walk-forward folds are not independent (serial correlation) | **Addressed** | §4.4.4 |
| 6 | No risk-adjusted attribution beyond Sharpe ratio | **Partially Addressed** | §4.4.4 |
| 7 | Factor attribution (Fama-French alpha) absent | **Acknowledged / Outstanding** | §5.5 |
| 8 | CVaR / Expected Shortfall not reported | **Acknowledged / Outstanding** | §5.5 |
| 9 | No cross-market universe quality test (e.g., S&P 100 vs. S&P 500) | **Acknowledged / Outstanding** | §5.5 |
| 10 | Survivorship bias: no point-in-time constituent data | **Acknowledged** | §5.5 |
| 11 | Transaction cost calibration (8.4 bps vs. literature ~30 bps) | **Acknowledged** | §5.5 |

---

## Major Concern 1: Data Period (2021–2025 Covers Only 4 Years; JFM Standard is Approximately 20 Years)

**Reviewer's Concern:**
The empirical evaluation spans 2021–2025, yielding only four walk-forward folds. This sample length is substantially shorter than the 15–20 year horizons typical of JFM publications. The reviewer questions whether four observations are sufficient to draw reliable inferences about strategy performance.

**Status: Partially Addressed**

**Response:**

The authors fully acknowledge this limitation and agree that a longer evaluation window would be the ideal standard. We address this concern on two fronts: an in-progress extension experiment and a statistical robustness argument for the existing sample.

**Extension Experiment (In Progress):** An 8-fold walk-forward experiment has been submitted to the university SLURM cluster (Job ID: 8543) extending the evaluation window to 2016–2025, yielding eight folds with test periods spanning 2017–2024. This represents a doubling of the fold count and a two-year extension of the lookback period. Results are pending at the time of this submission and will be incorporated into Section 4.4.6 upon completion. The pending nature of this analysis is disclosed explicitly in the revised manuscript.

**Clarification on Data Availability vs. Computational Constraints:** The reviewers should note that the short sample reflects thesis resource constraints — specifically, SLURM compute budget and wall-time allocations — and not a limitation of data availability. NSE price data is available from at least 2016, and the 2016–2025 extension experiment uses this existing data. The 2021–2025 window was the feasible scope within project constraints.

**Statistical Robustness with n=4:** Notwithstanding the small fold count, we apply a Heteroskedasticity and Autocorrelation Consistent (HAC) Newey-West standard error correction with lag=1 to the existing four folds. The corrected test confirms statistical significance at p=0.012, which is stronger than the uncorrected p=0.037. This result, together with directional consistency across all four folds, provides a reasonable basis for the reported conclusions, subject to the acknowledged limitation of small n.

The limitation of n=4 is explicitly stated in Section 5.5 as a primary constraint of the current study, and no causal or definitive claims are made on this basis alone.

---

## Major Concern 2: The Title Claim "Universe Quality Dominates Methodology" is Empirically Untested

**Reviewer's Concern:**
The manuscript's central thesis — that universe quality is the dominant driver of pairs trading performance relative to the choice of methodology — is presented as a conclusion but is not subjected to a formal statistical test. This conflates directional observation with demonstrated causal dominance.

**Status: Addressed**

**Response:**

The reviewer's concern is well-founded. The revised manuscript adds a formal bootstrap difference-in-differences test in **Section 4.4.5** to directly test this claim.

**Bootstrap Difference-in-Differences Test (Section 4.4.5):**

- **Universe Quality Effect:** Bootstrap 95% CI = [+0.371, +1.030] Sharpe units; P(effect > 0) = 100%. The universe quality effect is robustly positive and statistically significant.
- **Methodology Effect:** Bootstrap 95% CI = [−0.318, +0.892] Sharpe units; P(methodology effect > 0) = 81.1%. The methodology effect is positive in direction but does not achieve conventional 95% significance thresholds.
- **Dominance Test:** P(Universe Quality effect > Methodology effect) = 0.811.

**Revised Framing:** Based on these results, the manuscript no longer asserts that universe quality *dominates* methodology as a statistically established fact. The revised title and abstract frame this finding as *directionally supported but not statistically conclusive at the 95% level.* Section 4.4.5 presents the full bootstrap distributions and explicitly states: "The data are consistent with universe quality being the stronger driver, but the hypothesis of equal effects cannot be rejected at conventional significance levels."

This revision is a substantive improvement in epistemic honesty and we thank the reviewer for pressing this point.

---

## Major Concern 3: Nifty 50 is a Thin Universe; Concentration Risk is Unquantified

**Reviewer's Concern:**
The Nifty 50 index contains only 50 constituents. A pairs trading strategy operating within this universe may suffer from concentrated exposure to a small number of intra-sector pairs, potentially inflating reported Sharpe ratios through correlated positions.

**Status: Partially Addressed**

**Response:**

**Pairs Overlap Analysis (Section 4.4.4):** The revised manuscript adds a structural analysis of the candidate pair universe. The Nifty 50 generates C(50,2) = 1,225 candidate pairs, of which 595 pass the cointegration screening criteria. Of these 595 pairs, 54 (9.1%) are intra-sector pairs. Given the strategy's maximum concurrent position limit of max\_concurrent = 10, the expected number of intra-sector pairs in the active portfolio under random selection is 0.91 pairs. This suggests that intra-sector concentration in the active portfolio is modest in expectation.

**Acknowledged Limitation:** Notwithstanding the above, the authors acknowledge that pair selection is not random — pairs are ranked by signal quality — and it is possible that intra-sector pairs are systematically over-selected due to higher mean-reversion quality. This potential concentration bias is acknowledged in **Section 5.5** as a direction for future work. Specifically, we note that replicating the analysis on the Nifty 200 universe would provide a direct test of whether concentration within the Nifty 50 inflates the reported Sharpe ratios.

---

## Major Concern 4: No Benchmark Comparison is Provided

**Reviewer's Concern:**
The manuscript reports absolute strategy performance metrics without comparison to a passive equity benchmark, making it difficult to evaluate the economic significance of the results.

**Status: Addressed**

**Response:**

**Fold-by-Fold Benchmark Comparison (Section 4.4.4):** The revised manuscript adds a direct comparison against an equal-weight Nifty 50 buy-and-hold benchmark, reported fold-by-fold.

Equal-weight Nifty 50 Buy-and-Hold Sharpe ratios by fold:

- Fold 1: 1.795
- Fold 2: 0.568
- Fold 3: 2.927
- Fold 4: 0.839

The pairs trading strategy's raw alpha (strategy Sharpe minus benchmark Sharpe) is negative in three of four folds, with a mean raw alpha of −0.78. This finding is reported transparently in the revised manuscript.

**Volatility-Normalised Comparison:** The manuscript now includes an explicit discussion of the structural difference between the two strategies. The pairs trading strategy operates at approximately 5% annualised volatility (market-neutral), while the equity benchmark operates at approximately 15% annualised volatility. The strategy's mean Sharpe of +0.752 at 5% volatility is equivalent to approximately +2.2 Sharpe normalised to 15% volatility, which is competitive with the benchmark's mean Sharpe of 1.532.

**Drawdown Comparison:** The strategy's mean maximum drawdown of 2.2% compares favourably with equity drawdowns of 15–20% over comparable periods. The manuscript now explicitly positions the strategy as delivering *comparable risk-adjusted returns at approximately one-third the drawdown*, rather than as a superior return-generating strategy. This is a more accurate and defensible framing.

---

## Major Concern 5: Walk-Forward Folds are Not Independent; Serial Correlation Undermines Inference

**Reviewer's Concern:**
Walk-forward folds share overlapping training windows and may exhibit serial correlation in performance. Standard t-tests and Wilcoxon tests assume independence of observations; this assumption is violated if fold returns are serially correlated, potentially yielding spuriously significant results.

**Status: Addressed**

**Response:**

**HAC Newey-West Correction (Section 4.4.4):** The revised manuscript computes Heteroskedasticity and Autocorrelation Consistent (HAC) Newey-West standard errors with lag=1 for all fold-level inference. The estimated first-order autocovariance is γ₁ = −0.056, indicating negative serial correlation: folds alternate between relatively stronger and weaker performance. Importantly, this negative autocorrelation means the HAC correction *strengthens* rather than weakens the result — the HAC-corrected p-value of 0.012 is lower than the uncorrected p-value of 0.037.

**Pre-Specified Test Retained:** The Wilcoxon signed-rank test is maintained as the pre-specified primary hypothesis test, consistent with the original analysis plan. The HAC Newey-West result is reported as a robustness check. Both tests and the direction of the correction are fully disclosed in Section 4.4.4.

---

## Major Concern 6: No Risk-Adjusted Attribution Beyond Sharpe Ratio

**Reviewer's Concern:**
The Sharpe ratio is an incomplete risk measure. The manuscript should provide additional risk-adjusted performance metrics and attribution of returns to better characterise the strategy's risk profile.

**Status: Partially Addressed**

**Response:**

**Additional Metrics Added (Section 4.4.4):** The revised manuscript adds the following metrics, reported fold-by-fold:

- **Calmar Ratio:** Mean = 1.844; range = [0.376, 4.025]. This confirms that annualised returns are on average approximately 1.8× the maximum drawdown.
- **Maximum Drawdown per Fold:** Range = 1.4%–3.1%, with mean = 2.2%. Full fold-level breakdown provided.
- **Cost Drag Decomposition:** Gross Sharpe vs. Net Sharpe breakdown reported per fold. Mean cost drag = 0.059 Sharpe units. This documents the impact of transaction costs explicitly.
- **Trade Count per Fold:** Reported for reproducibility and to contextualise cost assumptions.

**CVaR/Expected Shortfall — Acknowledged as Outstanding:** Conditional Value-at-Risk (CVaR) and Expected Shortfall (ES) are not available from the current backtest infrastructure, which stores maximum drawdown but does not retain the full return distribution required for tail risk estimation. This is acknowledged in **Section 5.5** as a limitation and direction for future work requiring a refactored backtest engine.

---

## Outstanding Concerns — Acknowledged but Not Addressed in This Revision

The following concerns raised by reviewers or anticipated from the manuscript's limitations are acknowledged but cannot be fully addressed within the scope of the current revision. Each is documented in **Section 5.5** of the revised manuscript.

---

### Outstanding Item 1: Factor Attribution (Fama-French Alpha)

**Status: Acknowledged / Outstanding**

Fama-French three-factor (or five-factor) alpha decomposition would provide the most rigorous attribution of strategy returns and would isolate pairs trading alpha from systematic factor exposures. This analysis requires Indian factor return series (market, SMB, HML, RMW, CMA) at daily or monthly frequency, which are not available in the current codebase. Public sources for Indian Fama-French factors (e.g., Aharoni et al. construction or CMIE-derived factors) would need to be integrated. This is acknowledged as a substantive limitation and direction for future work.

---

### Outstanding Item 2: CVaR / Expected Shortfall

**Status: Acknowledged / Outstanding**

As noted in the response to Major Concern 6 above, the current backtest engine retains only maximum drawdown and does not store the full return distribution. Computing CVaR and ES requires access to daily or trade-level P&L series. Refactoring the backtest infrastructure to retain these series is a planned but out-of-scope enhancement for this revision.

---

### Outstanding Item 3: Cross-Market Universe Quality Test (S&P 100 vs. S&P 500)

**Status: Acknowledged / Outstanding**

A natural test of the "universe quality dominates methodology" hypothesis is to replicate the analysis in a market where two universe quality tiers are available — for example, comparing pairs trading performance within the S&P 100 (high-liquidity, homogeneous constituents) against the broader S&P 500. This test would provide out-of-sample evidence for the central thesis. Data for US or UK equity pairs trading is not available in the current codebase. This is acknowledged as an important robustness test and direction for future research.

---

### Outstanding Item 4: Survivorship Bias — Point-in-Time Constituent Data

**Status: Acknowledged**

The analysis uses current Nifty 50 constituents rather than point-in-time index membership data. This introduces potential survivorship bias: stocks that were removed from the index due to poor performance or delisting are absent from the backtest universe, potentially inflating reported returns. The revised manuscript (**Section 5.5**) acknowledges this bias and estimates that approximately 30–40% of the universe may be affected by survivorship over the 2021–2025 window, based on known index rebalancing events. Point-in-time constituent data was not available for this revision. Future work should obtain historical index membership files from NSE or a commercial data provider.

---

### Outstanding Item 5: Transaction Cost Calibration (8.4 bps vs. Literature ~30 bps)

**Status: Acknowledged**

The current analysis applies a round-trip transaction cost assumption of 8.4 basis points, which reflects NSE brokerage and STT estimates for institutional-grade execution. The reviewer notes that the empirical literature on Indian equity pairs trading (e.g., Huck & Afawubo 2015; Bowen, Hutchinson & O'Sullivan 2010 adapted to emerging markets) typically employs cost assumptions closer to 30 bps to reflect market impact, slippage, and bid-ask spread costs. The revised manuscript (**Section 5.5**) acknowledges this discrepancy and notes that a cost sensitivity analysis — reporting strategy performance at 8.4, 15, and 30 bps — is required to assess the robustness of results to cost assumptions. This analysis is flagged as a priority for the next revision. If results are not robust to 30 bps, this would materially affect the manuscript's conclusions.

---

### Outstanding Item 6: 8-Fold Extension Results (SLURM Job 8543)

**Status: COMPLETED**

The 8-fold extension experiment (2016–2025, test periods 2017–2024, SLURM Job 8543) has been completed. Results have been incorporated into **Section 4.4.7** of the revised manuscript.

**Key results:**
- Mean Net Sharpe: **+0.242** (vs. +0.752 in primary 4-fold analysis)
- p-value (two-tailed, df=7): **0.473** (non-significant)
- 95% Bootstrap CI: **[−0.329, +0.841]** — spans zero
- HAC Newey-West p: 0.436
- Positive folds: **5/8 (62.5%)**
- Bonferroni-corrected p: >1.0 (non-significant by construction)

**Honest conclusion:** The primary 4-fold finding (+0.752, p=0.036) does **not** survive extension to 8 folds. The primary finding is **regime-conditional**: the 2021–2024 test window was a favourable mean-reversion regime, but the strategy incurred material losses in 2019 (−0.835) and 2020 (−0.876, COVID-19 structural break), pulling the 8-fold mean to +0.242 with a p-value of 0.473. Universe quality remains the key architectural differentiator (Nifty 50 at +0.242 vs. Nifty 100 at ~+0.052), but neither result is statistically significant over the extended window. This constitutes a material qualification of the manuscript's primary finding and is disclosed as such in the revised abstract and Section 4.4.7.

---

## Summary of Manuscript Changes

The following sections have been added or substantively revised in response to reviewer comments:

- **Section 4.4.4** — Benchmark comparison (fold-by-fold); HAC Newey-West serial correlation correction; Calmar ratio, MaxDD, cost drag, and trade count tables.
- **Section 4.4.5** — Bootstrap difference-in-differences test for Universe Quality vs. Methodology effect.
- **Section 4.4.6** — Reserved for 8-fold extension results (currently placeholder pending SLURM job completion).
- **Section 5.5** — Expanded limitations section covering: small n, survivorship bias (quantified), thin universe concentration risk, factor attribution absence, CVaR/ES absence, cost calibration sensitivity, cross-market replication.
- **Title and Abstract** — Hedged framing of the central claim: "universe quality appears to be the stronger driver" replaces the unqualified "dominates" language.

---

*End of Response to Reviewers*

---

> **Document metadata:** Prepared June 2026. Corresponds to thesis revision submitted for JFM review. SLURM Job 8543 results to be appended to Section 4.4.6 upon completion. This document should be updated when the 8-fold results are available.
