# CRITIQUE AUDIT — What's Fixed vs Still Broken
**Date:** 2026-06-01  
**Purpose:** Map every critique item to current status. No sugarcoating.

---

## FATAL ISSUES (Items 1–6)

| # | Issue | Status | Evidence |
|---|-------|--------|----------|
| 1 | Universe cherry-picking — Nifty 50 vs 100 confound | ✅ **FIXED (experiment)** | NSE Nifty50 Rolling +0.752, Expanding +1.064. Control experiment run. |
| 2 | ML non-determinism — India ZScore 3 runs (+0.398/−0.386/+0.840) | ❌ **NOT FIXED** | Control used statistical selectors only. ML variance NOT resolved. Need CPU-only TF run. |
| 3 | P-hacking via post-hoc rolling window optimization | ❌ **NOT FIXED in writing** | No Bonferroni correction added. Abstract still claims rolling is "optimized methodology". |
| 4 | Multiple runs without reporting protocol | ✅ **FIXED (data)** | TRANSPARENCY_REPORT.md documents all 25 runs. But NOT yet reflected in thesis chapters. |
| 5 | Sample size (n=4 folds, outlier fold +1.996) | ❌ **NOT FIXED** | Still 4 folds. No bootstrap CI. No outlier analysis in chapters. |
| 6 | Transaction cost inconsistency (16.355 vs 16.4 bps) | ⚠️ **PARTIALLY** | NSE cost corrected in code per Documentation/NSE_Trading_Costs_Research_2024.md. US/Brazil/UK cost tables not written. |

---

## MAJOR ISSUES (Items 7–10)

| # | Issue | Status | Evidence |
|---|-------|--------|----------|
| 7 | Regime analysis is post-hoc (circular reasoning) | ❌ **NOT FIXED** | No VIX/volatility labels added. Chapter 4 still uses post-hoc year labels. |
| 8 | UK underperformance dismissed casually | ❌ **NOT FIXED** | No correlation matrix, cointegration rates, or sector comparison for UK in any chapter. |
| 9 | Literature review weak — missing Avellaneda, Huck, Vidyamurthy | ⚠️ **PARTIALLY** | Chapter 2 has Avellaneda's NEGATIVE RESULT documented (it fails on NSE). Huck 2015, Vidyamurthy 2004 not confirmed cited in Ch2. |
| 10 | Missing NSE Nifty 50 baseline experiments | ✅ **FIXED** | Both rolling (+0.752) and expanding (+1.064) now done. |

---

## MAJOR ISSUES (Items 11–15 from critique — numbered differently)

| # | Issue | Status | Evidence |
|---|-------|--------|----------|
| 11 | Abstract present? | ✅ Abstract exists (~1,900 words) | abstract.md is complete. But uses OLD narrative (geographic alpha, +0.840 headline). MUST be rewritten. |
| 12 | Literature review — missing key papers | ⚠️ **PARTIALLY** | Ch2 has foundational papers. Avellaneda cited (failure on NSE). Huck/Vidyamurthy/recent ML: UNCONFIRMED. |
| 13 | Figures lack confidence intervals / error bars | ❌ **NOT FIXED** | No CI on figures. No bootstrap analysis done. |
| 14 | Deployment recommendation (Tier 1: 50% capital) | ❌ **NOT FIXED** | Chapter 5 still says "Deploy India+ZScore at 50% capital". Critique calls this irresponsible. |
| 15 | OU model under-specified (no ADF, no MLE window) | ❌ **NOT FIXED** | Code uses rolling AR(1) not MLE. No ADF filter in OUThreshold. Chapter 3 likely doesn't disclose this accurately. |

---

## MINOR ISSUES (Items 16–19)

| # | Issue | Status | Evidence |
|---|-------|--------|----------|
| 16 | Emoji in tables (flags, ★, ✅/❌) | ❌ **NOT FIXED** | MULTI_MARKET_RESULTS.md and Ch4 still use flag emojis and ★. |
| 17 | Inconsistent terminology | ❌ **NOT FIXED** | "Multi-market India" / "India Nifty 50" / "Geographic India" still mixed in chapters. |
| 18 | Section numbering broken (3.6 written before 3.1–3.5) | ⚠️ **PARTIALLY** | chapter_3_integrated.md exists with all sections. Need to verify 3.1–3.5 are substantive. |
| 19 | Submission timeline unrealistic | N/A | Not a writing fix — timeline acknowledgment only. |

---

## SUMMARY: What Must Be Fixed Before Writing New Chapters

### Experiments Still Needed
- [ ] **ML non-determinism fix**: Run NSE Nifty 50 with CPU-only TF (full 8 selectors), get deterministic result. Until this is done, can only report "statistical selectors only" caveat.

### Statistical Analysis Needed (can be done now)
- [ ] **Bootstrap CI** on all 4 new Nifty 50 results (rolling + expanding, ZScore + OU)
- [ ] **Bonferroni correction** on rolling vs expanding comparison (2 tests → p × 2)
- [ ] **Outlier analysis** of India fold 3 (+1.996) — does removing it collapse the multi-market claim?
- [ ] **UK analysis** — cointegration pass rate, correlation matrix, sector composition table

### Thesis Chapter Rewrites Needed
- [ ] **Abstract** — rewrite around Scenario A ("universe quality dominates"), drop "+0.840 headline", add mean±std
- [ ] **Chapter 4** — add Nifty 50 control results, remove "geographic alpha" framing, add CI on all figures, fix emoji tables, add Bonferroni
- [ ] **Chapter 5** — remove deployment recommendation (or add heavy disclaimers), fix conclusion to match Scenario A
- [ ] **Chapter 3** — add OU parameter specification (AR(1) not MLE, no ADF filter disclosure), add cost breakdown table for all markets
- [ ] **Chapter 2** — confirm Huck 2015 and Vidyamurthy 2004 are cited; add recent ML papers if missing

### Document Cleanup
- [ ] **MULTI_MARKET_RESULTS.md** — remove flags/emojis, replace +0.840 headline with mean±std table
- [ ] **Standardise terminology** across all chapters: pick ONE name per concept

---

## Priority Order
1. Bootstrap CI + Bonferroni (1–2 hours, code)
2. UK analysis (correlation/cointegration, 1 hour)  
3. Abstract rewrite (1 hour)
4. Chapter 4 rewrite (3–4 hours)
5. Chapter 5 fix — remove deployment rec (30 min)
6. Chapter 3 — OU specification fix (1 hour)
7. Chapter 2 — confirm missing citations (30 min)
8. Terminology pass + emoji removal (1 hour)
9. ML CPU-only experiment (submit job, 2 hours runtime) — can run in background
