# JFM Rejection Letter — Round 2 Assessment

**Manuscript:** Universe Quality Dominates Methodology: A Multi-Market Ensemble Pairs Trading Study
**Date:** June 2026
**Decision:** Reject without invitation to revise.

---

## Summary Assessment

The Round 1 revisions added five new sections and addressed several peripheral concerns. They did not address the paper's central problem: it has no statistically significant finding after multiple testing correction (p_corrected = 0.936, self-reported). Revisions that add CVaR tables, Fama-French approximations, and Brazil cost extrapolations cannot substitute for a paper with an actual statistically significant result. The manuscript also introduced new methodological errors in its revisions — a Fama-French regression that is circular by construction, CVaR estimates from a time period outside the backtest window, and a cost sensitivity analysis using inconsistent series. The paper is not suitable for the Journal of Financial Markets in its current form.

---

## Fatal Flaws

### Fatal Flaw 1: No Statistically Significant Result After Multiple Testing Correction

> **Update:** m_eff = 13.42 (eigenvalue correction); Bonferroni p = 0.491, not 0.936. Result still does not survive. Required fix remains: pre-register single hypothesis or obtain longer sample.

**The flaw:** The paper self-reports that its headline finding does not survive Bonferroni correction: p_corrected = 0.036 × 26 = 0.936. The 8-fold extension — correctly identified as the more reliable window — yields p = 0.473 (HAC: p = 0.436), which is not significant under any correction. The paper therefore contains zero statistically significant findings after appropriate multiple-testing adjustment.

**Why it matters to JFM:** JFM requires that empirical claims be supported by results that survive correction for multiple comparisons. A paper that openly states its headline result has p_corrected = 0.936 is a paper that openly states it cannot reject the null hypothesis. Honest reporting of this fact is appreciated but does not change the editorial decision: the result is not significant and cannot be presented as a contribution to the literature.

**Required fix:** Either (a) pre-register a single hypothesis and report one test — the universe quality hypothesis — removing all other market/signal combinations from primary analysis, or (b) apply Benjamini-Hochberg FDR correction instead of Bonferroni and identify which, if any, hypotheses survive at q < 0.05. If no result survives FDR correction, the paper should be framed as an exploratory study with null findings, not submitted to JFM.

---

### Fatal Flaw 2: Primary Analysis Rests on 4 Observations

**The flaw:** The headline result (p = 0.036, t = 2.75) is computed on n = 4 fold-level Sharpe ratios with 3 degrees of freedom. The implied Cohen's d is approximately 2.74 — larger than nearly any effect size reported in the empirical finance literature. An effect size this large from n = 4 is the expected outcome of data mining, not a genuine empirical finding. A t-test on 4 observations has 80% power to detect only effects with d ≥ 2.0; the test is structurally incapable of distinguishing genuine effects from noise for the effect sizes typical of pairs trading strategies (d ≈ 0.3–0.7).

**Why it matters to JFM:** JFM's stated minimum data requirement is 10 years of data for strategy evaluation papers. The primary analysis covers 4 years (2021–2024). The 8-fold extension covers 8 years (2017–2024) and is still below the minimum. The primary finding cannot be accepted at JFM on 4 data points regardless of its nominal p-value.

**Required fix:** Obtain a minimum 15-year data sample. If NSE Nifty 50 data is unavailable pre-2016, explicitly acknowledge this as a hard constraint that prevents JFM submission and redirect to a shorter-sample-tolerant venue (Quantitative Finance, Emerging Markets Review).

---

### Fatal Flaw 3: Fama-French Attribution Is Circular by Construction

**The flaw:** Section 4.4.10 reports a Fama-French 3-factor alpha of +9.84% p.a. with t = 9.01 for the 4-fold window. The methodology approximates monthly strategy returns as (Sharpe / 12) × σ where σ = 5% (constant, assumed). Under this approximation, monthly return r_t = (Sharpe_fold / 12) × 0.05 for each month in a fold. This means all months within a fold have identical returns. When these are regressed on monthly FF factors, the intercept mechanically equals the fold mean return regardless of what the FF factors do (R² < 5% confirms the factors explain nothing). The t-statistic of 9.01 is a function of the number of monthly observations generated, not of the statistical relationship between strategy returns and factor loadings. This is not Fama-French attribution — it is a circular transformation of the Sharpe ratio dressed in regression notation.

**Why it matters to JFM:** Reporting a t = 9.01 alpha that is a mathematical artefact of the approximation method, not an empirical result, is a material misrepresentation of the analysis. A reviewer who reads the methodology will identify this immediately. It destroys credibility for the entire paper.

**Required fix:** Remove Section 4.4.10 entirely, or rewrite it using actual daily P&L time series from the backtest period (2021–2024). Constant-volatility approximation from fold Sharpes is not an acceptable input to a factor regression.

---

## Major Concerns

### Major Concern 1: CVaR Estimate Is from the Wrong Time Period

**The flaw:** Section 4.4.8 reports CVaR computed from a 2024–2026 deployment run (538 active days). The backtest whose results are the subject of this paper runs from 2021–2024. These are different market regimes. The 2024–2026 period is post-backtest. Presenting tail risk estimates from a period not covered by the reported Sharpe ratios is methodologically incoherent: the CVaR characterises a strategy performance window that is not the one being evaluated.

**Required fix:** Compute CVaR from the 2021–2024 backtest period P&L data, not from the post-backtest deployment run. If the backtest engine does not retain daily P&L, state this explicitly as a limitation and do not substitute out-of-period data.

---

### Major Concern 2: US ZScore Result Driven by a Single Fold

**The flaw:** The US ZScore strategy reports mean Sharpe +0.774, fold results [−0.335, +2.147, +0.626, +0.656]. Fold 2 (+2.147) alone contributes the majority of the signal. Without fold 2, mean ≈ +0.316. Fold 2 spans 2022 — the US equity bear market — a period of abnormally high volatility and dislocated pair relationships. There is no analysis of whether the +2.147 fold reflects genuine mean-reversion exploitation or spurious correlation during an extreme volatility regime. The paper presents this as "exploratory, n=1" but uses the number in comparative discussions without this caveat being consistently applied.

**Required fix:** Decompose the US result by fold and provide a regime-conditional analysis. If fold 2 is an outlier driven by 2022 volatility, report the US result with and without it. The paper should not use the +0.774 mean as a data point in any comparison without flagging that it is dominated by a single observation.

---

### Major Concern 3: Brazil Cost Sensitivity Uses Inconsistent Series

**The flaw:** Section 4.4.9 computes cost drag per basis point as (0.449 − 0.107) / 8.43 = 0.0406, using 0.449 as the "best gross" and 0.107 as the "mean net." These come from different runs. The best OU run's own cost drag is (0.334 − 0.321) / 8.43 = 0.0015 per bps — a 27× difference from the 0.0406 figure used in the sensitivity table. The sensitivity analysis mixes best-run gross with mean-run net, producing a cost drag estimate that corresponds to neither the best run nor the mean. The resulting table (net = −0.768 at 30 bps) is arithmetic derived from an internally inconsistent calculation.

**Required fix:** Compute sensitivity separately for (a) the best OU run using its own gross/net pair and (b) the mean of OU runs using the mean gross/mean net pair. Report both. Do not mix series.

---

### Major Concern 4: Survivorship Bias "Correction" Is Not a Correction

**The flaw:** Section 5.5.2 presents Sharpe × 0.92 as a survivorship-bias-adjusted estimate, citing Elton, Gruber and Blake (1996). The Elton et al. result measures survivorship bias in mutual fund performance persistence studies (return series from funds that survived vs. all funds including dead funds). It does not measure the effect of applying ex-post index membership retroactively to a cointegration-based pairs strategy. The 0.92 adjustment factor has no methodological derivation specific to this paper's data structure. It is a number applied to make the result look like it accounts for survivorship bias when it does not.

**Required fix:** Either (a) obtain point-in-time NSE Nifty 50 constituent history and re-run the strategy — this is the only valid correction — or (b) remove the Sharpe × 0.92 figure entirely and retain only the qualitative acknowledgement that survivorship bias direction is positive and magnitude is unknown.

---

### Major Concern 5: Structural Error — Core Empirical Sections After Chapter Conclusions

**The flaw:** Sections 4.4.9 (Brazil cost sensitivity) and 4.4.10 (Fama-French attribution) are placed at lines 728 and 766 of the chapter file, after Section 4.7 (Chapter Conclusions) at line 673. These sections contain primary empirical results. A paper that presents empirical findings after its own conclusions section is structurally broken and cannot be submitted to any journal.

**Required fix:** Relocate Sections 4.4.9 and 4.4.10 to their correct position within Section 4.4 (before §4.5 Implementation Considerations). Regenerate Section 4.7 to incorporate references to all subsections including the cost sensitivity and FF attribution results.

---

## Newly Introduced Problems (from Round 1 Revisions)

### New Problem 1: HAC Result Misrepresented

The text implies HAC Newey-West correction strengthens inference. In the 8-fold window, HAC t = 0.825 vs naive t = 0.758. The HAC result is marginally weaker, not stronger. The academic-journal-submission-critique skill notes that HAC can strengthen results when fold outcomes exhibit negative serial autocorrelation. The 8-fold results ([0.501, 1.268, −0.835, −0.876, 0.510, −0.231, 1.587, 0.011]) do show some alternating pattern, but the HAC correction is still producing a weaker t. The text must not imply HAC strengthens the result when the numbers show it does not.

### New Problem 2: FF Alpha Significance Claim in Summary Table

The JFM_CRITIQUE_RESPONSE.md summary table marks item 7 (Factor Attribution) as "Addressed (Proxy)". The alpha t = 9.01 should not appear anywhere in the paper as evidence of significance — it is a circular artefact. If the critique response document will be reviewed, calling this item "Addressed" with a t = 9.01 claim will be identified as a methodological error in the response itself.

### New Problem 3: CVaR Status Inconsistency

The summary table still shows item 8 (CVaR) as "Acknowledged / Outstanding" (line 25 of JFM_CRITIQUE_RESPONSE.md) even though the body text marks it as "ADDRESSED." The summary table is what reviewers read first. This inconsistency needs to be fixed.

---

## Minor Issues

- Abstract reports "Acceptance Likelihood: 70-75%." This must not appear in any submitted document. Remove.
- "Thesis Committee Recommendation" section in abstract.md is internal planning material, not manuscript content.
- Abstract says 2021-2025 in one place and 2021-2024 in another. Pick one and be consistent throughout.
- Section 5.5 is numbered twice — appears as both "Addressing Principal Methodological Limitations" and "Future Work" with the same section number.
- Brazil gross Sharpe = 0.449 appears in the sensitivity narrative but the actual best Brazil OU run shows gross = 0.334. The 0.449 figure needs sourcing or correction.
- The plain-language abstract says "IIT Bombay Researcher Discovers Profitable Stock Trading Strategy" — the strategy is not robustly profitable after multiple testing correction. This headline is misleading.

---

## What Has Been Successfully Addressed Since Round 1

- Calmar ratio and drawdown profile reported (§4.4.5.2) — adequate
- HAC Newey-West formally applied and reported — adequate
- Bootstrap DiD honestly reports P(UQ > Meth) = 0.811 with CI including zero — appropriate hedging
- 8-fold extension conducted and reported as primary result — correct prioritisation
- Brazil marked as non-viable market in multiple places — correct framing
- Transaction cost model for India cited with actual NSE fee structure — adequate
- Survivorship bias direction and qualitative magnitude acknowledged — acceptable acknowledgement (minus the invalid Sharpe × 0.92 number)
- Cross-market test proposed as future work with specific designs — adequate for limitations section

---

## Revision Roadmap (if major revision were hypothetically invited)

1. **Remove Section 4.4.10 (FF attribution) entirely** or replace with real daily P&L regression. ~1 week if daily P&L data is recovered from backtest logs.
2. **Fix CVaR to use 2021–2024 backtest P&L**, not post-backtest deployment data. ~2 days.
3. **Restructure Chapter 4**: move §4.4.9 and §4.4.10 to correct position before §4.5, regenerate §4.7. ~1 day.
4. **Fix Brazil sensitivity** to use internally consistent series per run, not mixed best/mean. ~1 day.
5. **Remove Sharpe × 0.92 survivorship adjustment** — retain qualitative acknowledgement only. ~30 minutes.
6. **Address multiple testing**: either pre-register one hypothesis and report one test, or apply BH-FDR and report adjusted q-values. This is the hardest fix — it requires a philosophical reframe of what the paper claims. ~1 week.
7. **Fix HAC claim** — remove implication that HAC strengthens the 8-fold result. ~30 minutes.
8. **Fix summary table inconsistency** for CVaR status. ~10 minutes.
9. **Remove acceptance likelihood and committee recommendation** from abstract. ~10 minutes.
10. **Fix section numbering** for the duplicate §5.5. ~30 minutes.

**Total: 2–3 weeks for a legitimate revision.**

---

## Venue Assessment

**JFM (current state):** Reject. Primary result does not survive multiple testing correction. Sample too short. FF section contains a methodological error. Structural problems indicate manuscript is not submission-ready.

**Quantitative Finance:** Conditional accept possible after items 1–5 and 6–10 are fixed. QF is tolerant of shorter samples and methodology-focused papers. The honest hedging (8-fold p = 0.473, Bonferroni non-significant) combined with a strong positive framing of the regime-conditionality finding could work. Remove FF section entirely first.

**Emerging Markets Review:** Best current fit. NSE/India angle is a direct scope match. Shorter samples accepted. Exploratory framing works. Requires structural fixes (§4.4.9/§4.4.10 placement) and FF section removal but not the sample-size fix. Realistically 3–4 weeks of revision away from a submittable manuscript.

**Finance Research Letters:** Viable for a 4,000-word focused note on the universe quality finding alone. Strip everything except the NSE Nifty 50 vs Nifty 100 result with proper hedging. n=4 is borderline but accepted at FRL if framed as exploratory. Do not include FF attribution or Brazil cost sensitivity.

**Recommended action:** Fix structural errors and remove the FF section (2–3 days), then submit to Emerging Markets Review. In parallel, prepare a short FRL note on the universe quality finding using only the Nifty 50 vs Nifty 100 result.
