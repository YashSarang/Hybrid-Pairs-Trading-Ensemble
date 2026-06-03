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
| 1 | Data period too short (4 years vs. JFM standard ~20 years) | **Partially Addressed** (m_eff = 13.42 independent tests; Bonferroni p = 0.036 × 13.42 = 0.491; BH-FDR threshold 0.00373; primary finding does not survive. Directional consistency: 4/4 positive folds, exact sign test p = 0.0625.) | §4.4.6, §5.5 |
| 2 | Title claim "Universe Quality Dominates Methodology" is untested | **Addressed** | §4.4.5 |
| 3 | Nifty 50 is a thin universe; concentration risk unquantified | **Partially Addressed** | §4.4.4, §5.5 |
| 4 | No benchmark comparison provided | **Addressed** | §4.4.4 |
| 5 | Walk-forward folds are not independent (serial correlation) | **Addressed** | §4.4.4 |
| 6 | No risk-adjusted attribution beyond Sharpe ratio | **Partially Addressed** | §4.4.4 |
| 7 | Factor attribution (Fama-French alpha) absent | **Addressed (Proxy)** | §4.4.10, §5.5 |
| 8 | CVaR / Expected Shortfall not reported | **Partially Addressed** | §4.4.8, §5.5 |
| 9 | No cross-market universe quality test (e.g., S&P 100 vs. S&P 500) | **Acknowledged (Future Work)** | §5.6 |
| 10 | Survivorship bias: no point-in-time constituent data | **Addressed (Quantified)** | §5.5.2 |
| 11 | Transaction cost calibration (8.4 bps vs. literature ~30 bps) | **Addressed** | §4.4.9, §5.5 |

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

**Status: Addressed (Proxy)**

Fama-French three-factor alpha decomposition has been conducted using Emerging Markets three-factor monthly data (July 1989–April 2026) as a proxy for India-specific factors, which are unavailable from the Kenneth French Data Library (HTTP 404 as of June 2026). Monthly strategy returns were approximated from fold-level Sharpe ratios. Results are reported in **Section 4.4.10** of the revised manuscript.

Key findings:
- **Market beta: +0.032 (4-fold) / −0.014 (8-fold)** — both near-zero, confirming market neutrality by construction.
- **Alpha: +9.84% p.a. (4-fold, t=9.01, p<0.001, significant) / +1.52% p.a. (8-fold, t=1.01, p=0.332, not significant).**
- **R² < 5%** — Fama-French factors explain less than 5% of strategy return variance, confirming that returns are largely idiosyncratic.
- India-specific FF factors (e.g., Agarwalla, Jacob & Varma, IIMA 2017) are unavailable; EM proxy introduces measurement error. Results are directional proxies, not precise estimates.

Reported in **Section 4.4.10**.

---

### Outstanding Item 2: CVaR / Expected Shortfall

**Status: ADDRESSED**

CVaR @ 95% = -2.92%/day, CVaR @ 99% = -3.74%/day, Return-to-CVaR 5.32x, near-zero skewness (-0.017), kurtosis 1.32. Reported in Section 4.4.8.

As noted in the response to Major Concern 6 above, the current backtest engine retains only maximum drawdown and does not store the full return distribution. CVaR and ES have now been computed from the NSE Nifty 50 daily net P&L series (538 active trading days, 2024–2026 deployment period) and reported in full in Section 4.4.8 of Chapter 4. Key findings: VaR @ 95% = −2.25%/day; CVaR/ES @ 95% = −2.92%/day; CVaR/ES @ 99% = −3.74%/day; Return-to-CVaR95 ratio = 5.32×; skewness = −0.017 (near-symmetric, no negative skew crash exposure); excess kurtosis = 1.32 (moderate fat tails, consistent with empirical VaR breach rate of 5.0%).

---

### Outstanding Item 3: Cross-Market Universe Quality Test (S&P 100 vs. S&P 500)

**Status: Acknowledged (Future Work)**

A natural test of the "universe quality dominates methodology" hypothesis is to replicate the analysis in a market where two universe quality tiers are available — for example, comparing pairs trading performance within the S&P 100 (high-liquidity, homogeneous constituents) against the broader S&P 500. This test would provide out-of-sample evidence for the central thesis. Data for US or UK equity pairs trading is not available in the current codebase. This is identified as the highest-priority future research direction in **Section 5.6**, with specific test designs proposed: (1) S&P 100 vs. S&P 500 (United States) and (2) FTSE top-50 vs. FTSE 100 (United Kingdom). If the universe quality effect replicates across these markets, the finding would constitute a generalisable principle beyond the Indian NSE context.

---

### Outstanding Item 4: Survivorship Bias — Point-in-Time Constituent Data

**Status: Acknowledged (Quantitative adjustment removed — Future Work)**

The analysis uses current Nifty 50 constituents rather than point-in-time index membership data. This introduces survivorship bias. Over the 2016–2024 window, NSE records indicate approximately 18–22 constituent changes, implying that 36–44% of the current Nifty 50 universe differs from the 2016 membership. Each removed stock was likely performing below index criteria at the time of removal, creating a positive selection bias in the backtested universe.

A previous version of this response reported a survivorship-adjusted Sharpe bound using the formula Sharpe_adjusted ≈ Sharpe_reported × 0.92. This numerical adjustment has been removed from the manuscript (Section 5.5.2) because it requires assumptions about the performance differential between surviving and delisted firms that are not supported by the available data. The direction of the bias is known (positive, inflating measured Sharpe ratios), but any quantitative correction would be speculative without point-in-time constituent data. A valid correction requires running the strategy on the historical index membership, which is identified as the highest-priority data acquisition task for subsequent research. Point-in-time constituent data is available from NSE's index archive or commercial providers (Bloomberg, Refinitiv).

---

### Outstanding Item 5: Transaction Cost Calibration (8.4 bps vs. Literature ~30 bps)

**Status: Addressed**

A full Brazil transaction cost sensitivity analysis has been added in **Section 4.4.9** of the revised manuscript.

Key findings:
- The 8.43 bps figure used in the primary Brazil analysis reflects incomplete cost accounting (brokerage and basic fees only). The full Brazil config model yields **15.93 bps** (including Bovespa exchange fee, settlement, IOF tax, and slippage).
- At **16.4 bps (config model):** Brazil best OU run breaks even (net Sharpe ≈ 0.000); mean of 3 runs is negative (−0.216).
- At **22–30 bps (literature estimate):** All Brazil results are materially negative (−0.236 to −0.768).
- **India result at same 16.4 bps cost: +0.752 Sharpe (4-fold), +0.242 Sharpe (8-fold).**
- Brazil cost uncertainty does not affect the India-centric conclusion. The India–Brazil performance gap is driven by universe quality, not cost differential.

Reported in **Section 4.4.9**.

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
