---
status: FULLY RESOLVED
round: 2
verdict: MAJOR REVISION REQUIRED
resolved_date: 2026-06-04
note: All surviving R2 items were resolved. DO NOT re-read for active issues — see INDEX.md.
---

# REVISED CRITIQUE — Round 2

**Reviewer:** Anonymous Reviewer #2 (same reviewer)
**Date:** June 2, 2026
**Prior Recommendation:** Reject with Invitation to Resubmit
**Revised Recommendation:** **MAJOR REVISION REQUIRED** (downgraded from Reject — progress is real but incomplete)

---

## OVERALL ASSESSMENT

The authors have made substantive progress. The fatal confound (Nifty 50 vs Nifty 100 universe comparison) has been resolved with proper control experiments. Statistical machinery (bootstrap CI, Bonferroni correction, ML reproducibility testing) is now present. The core narrative has been reframed from "geographic alpha" to "universe quality alpha."

However, the manuscript remains internally inconsistent. Multiple revisions were applied piecemeal across sections, leaving contradictions between them. A reviewer reading linearly will notice the paper argues two incompatible theses simultaneously in different sections. Several fixes were applied to some locations but not all. This is not a polished resubmission — it is a partially patched draft.

**Recommendation:** Major Revision. The core finding is now scientifically defensible. The presentation is not yet coherent.

---

## ISSUES THAT WERE FIXED

### RESOLVED: Fatal Confound (#1)
NSE Nifty 50 control experiments are present. Rolling +0.752, Expanding +1.064. Table 4.2.1 includes these rows with mean ± std. The universe quality explanation is stated prominently in Chapter 4. This was the most critical fix and it was done correctly.

### RESOLVED: Bonferroni Correction (#3)
All four occurrences of p=0.320 in Chapter 3 now carry the Bonferroni-corrected value (p_corrected=0.640). The standalone note at the end of Section 3.6 is appropriate.

### RESOLVED: OUThreshold Specification (#15)
Chapter 3 now correctly states "rolling AR(1)" instead of "MLE." The note clarifying that ADF is a ranking criterion — not an exclusion filter for OU — is present and accurate.

### RESOLVED: Transaction Cost Breakdown (#6)
16.28 bps with itemized components (STT: 10.0 bps, NSE exchange fee: 0.322 bps, SEBI: 0.01 bps, stamp duty: 1.5 bps, slippage: 2.0 bps/leg) is now in Chapter 3.

### RESOLVED: ML Non-Determinism (#2)
CPU-only reproducibility run (Job 8465) documented. Mean-level variance reduced 9.4x (1.226 → 0.131 Sharpe). 4/4 fold sign concordance confirmed. Disclosure in Chapter 3 Section 3.3.2 is scientifically honest.

### RESOLVED: UK Failure Analysis (#8)
VIX regime table present (Chapter 4 Section 4.4.3a). Cointegration pass rates computed: UK 7.5% ≈ India 8.3% — UK failure is macro regime sensitivity, not structural cointegration deficit. This is data-driven, not speculative.

### RESOLVED: Deployment Disclaimer (#14)
Tier 1/2/3 capital allocation table replaced with academic disclaimer citing CI [-0.207, +0.758], backtest limitations, and four deployment prerequisites.

---

## FATAL ISSUES — NOT YET RESOLVED

### FATAL 1: Abstract Is Internally Contradictory

The abstract was patched section-by-section but never rewritten as a coherent whole. It now contains four sections arguing mutually contradictory theses.

**Evidence (exact quotes from the current document):**

- Structured Abstract Findings (patched): *"Universe quality dominates methodology optimization and geographic diversification."* — Correct.
- Plain-Language Abstract (unpatched): *"Trading pairs in India's Nifty 50 index produced a Sharpe ratio of +0.840 (meaning you earn 84 cents of profit per unit of risk) in the best run"* — Leads with the cherry-picked result, not the mean.
- Contribution Statement (unpatched): *"Quantifies geographic alpha (16.2x India/NSE multiplier) vs methodological alpha"* — Directly contradicts the corrected narrative. If universe quality is the finding, there is no geographic alpha to quantify.
- Three-Sentence Summary (unpatched): *"16 times better returns"* — The 16x claim was explicitly removed from Chapter 4 but survives here verbatim.
- Impact Projection (unpatched): *"Will become reference for 'geographic alpha' concept in pairs trading literature"* — The paper's own finding refutes geographic alpha. This sentence asserts the opposite of the paper's conclusion.
- Fold count contradiction: Structured Abstract says "6-fold walk-forward (2014-2025)." Executive Summary says "4-fold walk-forward (2021-2025)." These are different experimental designs. Both cannot be true.

**Verdict:** A reviewer reading the abstract from top to bottom will encounter the corrected thesis, then the original discredited thesis, then the corrected thesis again. This signals that the authors patched rather than revised. Fatal for journal submission.

---

### FATAL 2: Capital Allocation Recommendation Survives in Chapter 5

The Tier-based deployment block was replaced with an academic disclaimer, but the following line was not removed (Chapter 5, line 175):

*"Cap India at 50% (concentration risk despite dominance)"*

This is a specific capital allocation recommendation. It now coexists with the disclaimer two paragraphs above it that explicitly states: *"Real-money deployment based solely on these results would be premature."* Both cannot be true in the same section.

A practitioner reading selectively will find actionable advice (50% capital to India) in a section that claims to give none. This is the same legal and academic liability identified in the original critique.

---

## MAJOR ISSUES — NOT YET RESOLVED

### MAJOR 1: +0.284 vs +0.840 Gap Is Stated But Never Explained

Chapter 4 states both numbers side by side: *"Multi-market India (Nifty 50): +0.284 mean Sharpe (best run +0.840 across 3 runs)"* — but no sentence explains why the mean is 0.556 below the best run.

The explanation exists in Chapter 3 (ML non-determinism, GPU randomness causing different pair selections). It is never connected back to Chapter 4. A reader who encounters this discrepancy has no framework to interpret it. They will conclude it is unexplained variance, which is worse than non-determinism because it looks like concealment.

**Required fix:** One paragraph in Section 4.2 bridging the Chapter 3 ML non-determinism finding to the Chapter 4 India run variance.

---

### MAJOR 2: Flag Emojis Survive in Chapter 4

Chapter 4 lines 13–16 (Markets Tested list in the chapter overview) still contain:

*"🇺🇸 United States … 🇮🇳 India … 🇧🇷 Brazil … 🇬🇧 United Kingdom"*

The fix was applied to tables and some sections but not this list. Journal of Financial Markets mandates plain text throughout. One missed instance still causes desk rejection.

---

### MAJOR 3: Avellaneda Not Connected to NSE Results

Chapter 2 cites Avellaneda & Lee (2010) only for US declining profitability. It does not mention that the authors explicitly tested NSE constituents and found 0% of their 35-stock universe passed the stationarity filter — the same size universe this thesis uses.

This thesis achieves positive Sharpe on NSE Nifty 50. That is a direct empirical refutation of Avellaneda's NSE finding. This tension must be explicitly acknowledged and resolved in the literature review or methodology. Not doing so will guarantee a question from any reviewer who knows the Avellaneda paper, which is most reviewers in this field.

The resolution is available in the existing work: different lookback window (126-day AR(1) vs Avellaneda's multi-year MLE), different ADF threshold, NSE market structure changes post-2010. These arguments exist implicitly — they need to appear explicitly in Chapter 2 or Chapter 3.

---

### MAJOR 4: Figure Error Bars Still Not Present

`figure_ci_data.json` was generated correctly. The chapter text references confidence intervals. The actual figure PNG files have not been regenerated. A reviewer sees figures. They do not read JSON files. Critique item #13 is unresolved at the only level that matters for journal submission.

---

## MODERATE ISSUES — NEWLY IDENTIFIED

### MODERATE 1: Fold Count Inconsistency Across Chapters

Abstract Structured section: "6-fold walk-forward validation (2014-2025)."
Abstract Executive Summary: "4-fold walk-forward testing (2021-2025)."
Chapter 3 baseline: 6 folds (2020-2025).
Chapter 4 multi-market results: 4 folds.
NSE Nifty 50 control: 4 folds.

The Chapter 3 baseline (6 folds) is being compared to Chapter 4 multi-market results (4 folds). This is an undisclosed apples-to-oranges comparison. Even if both are internally valid, comparing a 6-fold expanding baseline against a 4-fold rolling multi-market result inflates the apparent advantage of multi-market without controlling for this design difference.

### MODERATE 2: "16.4 bps" Survives in the Plain-Language Abstract

The Plain-Language Abstract section was not patched for the cost figure. It still reads "16.4 basis points for India." Chapter 3 and the Structured Abstract say 16.28 bps. The abstract now contains both figures simultaneously.

### MODERATE 3: "Universe Quality Dominates" Is One Data Point

The thesis concludes that universe quality dominates. This conclusion rests on a single comparison (Nifty 50 vs Nifty 100 on NSE). No equivalent comparison was made for:
- S&P 50 vs S&P 500 (US)
- FTSE 50 vs FTSE 100 (UK)
- IBOV top-20 vs IBOV full (Brazil)

Without cross-market generalizability, the finding is: "universe quality matters on NSE." The claim "universe quality dominates" as a general principle is not established. The thesis should explicitly bound its claim to NSE and flag cross-market replication as future work.

---

## MINOR ISSUES — NEWLY IDENTIFIED

### MINOR 1: Brazil OU Narrative Still Incomplete

Table 4.2.1 correctly shows Brazil OU mean = +0.107 (3-run mean). However, the Chapter 4 narrative text around Brazil OU does not explicitly note that the originally-reported +0.321 was a best-run cherry-pick and the honest mean is +0.107. The table is corrected but the surrounding explanation is not.

### MINOR 2: Chapter 5 Closing Is Now Generic

The replacement for the informal "Wall Street/Mumbai" closing — *"These findings suggest that universe selection and market microstructure — rather than algorithmic sophistication — are the primary determinants of pairs trading profitability in emerging markets"* — is accurate but reads as a placeholder. It does not name a specific finding, cite a number, or point toward future work. A thesis conclusion should end with a precise statement of contribution, not a general suggestion.

---

## SUMMARY OF REQUIRED CHANGES

### FATAL (Must Fix Before Any Submission):

1. **Rewrite abstract top-to-bottom as a single coherent document.** One finding: universe quality drives alpha. Remove all references to geographic alpha, 16x multiplier, and the original confounded narrative. Fix the 6-fold vs 4-fold contradiction. Fix 16.4 vs 16.28 bps contradiction.

2. **Remove "Cap India at 50%" from Chapter 5.** The disclaimer and the recommendation cannot coexist in the same section.

### MAJOR (Must Fix for Journal Acceptance):

3. **Add bridge paragraph in Chapter 4 Section 4.2** explaining that the +0.284 vs +0.840 gap in India ZScore runs is attributable to ML non-determinism documented in Chapter 3 Section 3.3.2.

4. **Remove flag emojis from Chapter 4 lines 13–16** (Markets Tested overview list).

5. **Add Avellaneda NSE context to Chapter 2** — cite their 0% stationarity finding on NSE; explain how this thesis's methodology achieves positive results (different lookback, AR(1) vs MLE, 12 years of post-2010 market development).

6. **Regenerate Figure 4.1 and Figure 4.3 with error bars** using data in `figure_ci_data.json`.

### MODERATE:

7. **Acknowledge the 6-fold vs 4-fold design difference** between Chapter 3 baseline and Chapter 4 multi-market experiments. Either rerun Chapter 3 with 4 folds for a clean comparison, or explicitly disclose the discrepancy as a limitation.

8. **Bound the "universe quality dominates" claim to NSE.** State explicitly that cross-market generalizability (S&P 50 vs 100, FTSE 50 vs 100) is future work.

---

## FINAL RECOMMENDATION

**MAJOR REVISION REQUIRED** (elevated from Reject)

The scientific core is now defensible. The control experiment is done correctly. The statistics are applied correctly. The key finding — universe quality, not geography — is correct and novel enough for journal consideration.

What remains is a coherence problem caused by piecemeal patching. The abstract alone now contains three different theses simultaneously. A complete top-to-bottom consistency pass is required. The list of fatal fixes above is achievable in one focused week of rewriting.

**Estimated time to resolve remaining issues:** 1–2 weeks.
**Confidence in acceptance after fixes:** 65% (Journal of Financial Markets), 80% (Quantitative Finance).

The paper has moved from "reject" to "likely accept with major revisions." That is real and substantial progress.

---

**Reviewer Signature:** Anonymous Reviewer #2
**Specialty:** Quantitative Finance, Statistical Arbitrage, Pairs Trading
**Papers Reviewed:** 120+ for JFM, QF, JFQA, JFE

---

