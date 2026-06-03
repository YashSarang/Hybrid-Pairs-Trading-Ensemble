---
status: FULLY RESOLVED
round: 3
verdict: REJECT with invitation to resubmit (26 open items on arrival)
resolved_date: 2026-06-04
note: All 26 R3 items were resolved. DO NOT re-read for active issues — see INDEX.md.
---

## Round 3 Critique
**Date:** 2026-06-09  
**Reviewer:** Anonymous Reviewer #2 (third pass)  
**Prior Recommendations:** Round 1 — Reject; Round 2 — Major Revision Required  
**Current Status on Arrival:** Partial revision; Round 2 FATAL issues partially addressed  

### OVERALL ASSESSMENT

Progress from Round 2 is acknowledged but insufficient. 14 new defects found beyond surviving Round 2 issues.

**Key finding resolved:** The 'US / unknown' result (+0.774 Sharpe) is NOT suppressed — the `signal_model` field was missing from the JSON so the statistics script labelled it 'unknown'. Fold-level sharpes: [−0.335, +2.147, +0.626, +0.656], mean = +0.774. However, Chapter 4 still incorrectly states 'No result (data issue?)' — US ZScore works and must be reported.

---

### PART I — SURVIVING ISSUES FROM ROUND 2

**[FATAL-R2-A]** Chapter 4 Tier 1/2/3 deployment table survives intact (Section 4.5.1–4.5.2). '50% India' capital allocation still present. Round 2 only patched Chapter 5 line 175.

**[FATAL-R2-B]** Chapter 2 Sections 2.4–2.5 still say '16.2x geographic alpha', 'RQ3: YES, 16.2x multiplier proves dominance'. The corrected universe-quality narrative has NOT been applied to Chapter 2.

**[MAJOR-R2-C]** Figure PNGs still lack CI error bars except newly created Figure 4.1.

**[MAJOR-R2-D]** Chapter 4 still builds all multiplier comparisons on +0.840 cherry-picked best run; the +0.284 mean was disclosed but not used to recalculate headline multipliers.

**[MODERATE-R2-E]** Flag emojis still in Chapter 5 Appendix A Table A.1.

---

### PART II — NEW ISSUES

**[FATAL-NEW-1] US ZScore result unreported** — Chapter 4 says 'No result (data issue?)' but US ZScore 4-fold run exists with mean +0.774 Sharpe (folds: −0.335, +2.147, +0.626, +0.656). This is a real result that must be reported. It is comparable in magnitude to NSE Nifty 50 rolling (+0.752). File: results/us/wfv_4folds_20260529_025102.json (signal_model field missing from JSON caused mislabelling).

**[FATAL-NEW-2] Look-ahead bias undisclosed + false 'no survivorship bias' claim** — Section 3.2.1 states 'no survivorship bias' but uses 2024/2025 Nifty 50 constituents retroactively for 2021-2022 training folds. Correct statement: 'Mild look-ahead bias present; point-in-time constituent lists not used.'

**[FATAL-NEW-3] 'Geography 1.7x methodology' claim inverts under honest arithmetic** — computed as India best-run (+0.840) − NSE rolling (+0.052) = +0.788. Honest: India mean (+0.284) − period-matched NSE rolling 2021-2024 (mean −0.084) = +0.368. Methodology improvement = +0.461. Honest geography = +0.368. Thesis claim reverses sign.

**[MAJOR-NEW-4] CNNSelector disabled — ensemble is 7 selectors, not 8** — Section 3.3.1 admits 'disabled due to data requirements'. All '8-selector ensemble' claims throughout the thesis are false.

**[MAJOR-NEW-5] Brazil transaction cost contradiction** — Chapter 2 cites ~30 bps for Brazil (consistent with Martins & dos Santos 2021); Chapter 4 uses 8.4 bps. At 30 bps, all Brazil results go negative.

**[MAJOR-NEW-6] Section 4.3.4 missing** — Chapter 2 Section 2.3.3 promises 'we investigate why in Section 4.3.4'. The section does not exist. Liew & Wu (2013) contradiction (smaller caps more profitable) is unresolved.

**[MAJOR-NEW-7] 12-month ML training window** — ~190 usable sequences to train LSTMs/Transformers with thousands of parameters. No overfitting diagnostics.

**[MAJOR-NEW-8] No selector ablation** — statistical-only achieves +0.752, ML ensemble achieves +0.284. ML selectors may be net-negative. No ablation table exists.

**[MAJOR-NEW-9] Period-confounded methodology vs geography comparison** — methodology comparison covers 2020-2025 (6 folds), geography comparison covers 2021-2024 (4 folds). Not apples-to-apples.

**[MODERATE-NEW-10]** Krauss (2017) misattributed for LSTM autoencoders — paper does return prediction, not pair selection.

**[MODERATE-NEW-11]** 'No survivorship bias' + 'complete price history filter' are internally contradictory.

**[MODERATE-NEW-12]** Plain-language abstract says '2014-2025' — no dataset starts in 2014 (earliest is 2016).

**[MODERATE-NEW-13]** VIX regime table self-contradicts: 2024 VIX = 15.5 (lower than 2023 = 17.5) but UK Sharpe = −0.677 (worse than 2023's +0.967).

**[MODERATE-NEW-14]** Chapter 2 Gap 3 has a third India cost figure: 16.5 bps (wrong derivation missing slippage). Document now has 16.28, 16.4, and 16.5 bps.

**[MINOR-NEW-15]** OU folds pattern [0, 0, 0, X] across all markets — n=1 effective observation. OU CIs not meaningful.

**[MINOR-NEW-16]** 'results / ou' — 5 runs, all zero mean Sharpe. Unexplained in chapters.

**[MINOR-NEW-17]** Gatev citation: 2006 (correct) vs 1999 (NBER) used inconsistently across chapters.

**[MINOR-NEW-18]** Conflicting gross Sharpe profitability thresholds: +0.60 (Section 3.5.1) vs +0.90 (Chapter 5 RQ2).

---

### SUMMARY TABLE

| Severity | Open Count |
|---|---|
| FATAL | 5 (2 surviving R2 + 3 new) |
| MAJOR | 9 (2 surviving R2 + 5 new; was FATAL-NEW-1 resolved as mislabelling) |
| MODERATE | 7 (2 surviving R2 + 5 new) |
| MINOR | 5 (1 surviving R2 + 4 new) |
| **Total Open** | **26** |

### ACCEPTANCE PROBABILITY (Updated)

| Venue | Round 2 | Round 3 |
|---|---|---|
| JFM | 65% | 35% |
| Quantitative Finance | 80% | 50% |

**Recommendation:** Reject with Invitation to Resubmit. Core scientific finding (NSE Nifty 50 statistical-only: +0.752 Sharpe, CI [+0.422, +1.082], t=3.60, p=0.036) is real and publishable. Current framing leads with non-significant ML result (+0.284, CI crosses zero). Invert framing: make statistical-only Nifty 50 the headline, ML as exploratory negative finding.

