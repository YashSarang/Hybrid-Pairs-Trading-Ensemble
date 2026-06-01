# THESIS SALVAGE PLAN — ADDRESSING FATAL CONFOUNDS

**Date:** June 1, 2026  
**Status:** CRITICAL PATH — EXPERIMENTS REQUIRED BEFORE SUBMISSION  
**Estimated Time:** 3-5 days for experiments + 1 week for analysis/rewrite

---

## PROBLEM SUMMARY

The current thesis claims **"geographic alpha dominates methodology optimization"** based on India multi-market (+0.840 Sharpe) being 16x better than NSE rolling (+0.052 Sharpe).

**FATAL CONFOUND:** India multi-market uses **Nifty 50** (35 tickers) while NSE baseline uses **Nifty 100** (35 tickers with 71% overlap but different quality tiers).

**You changed BOTH geography AND universe simultaneously, making the conclusion invalid.**

---

## CRITICAL EXPERIMENTS REQUIRED

### **EXPERIMENT 1: NSE Nifty 50 + Rolling Windows** ⚠️ **HIGHEST PRIORITY**

**Purpose:** Isolate universe quality effect from geographic effect

**Config:** `experimental-ablation/configs/nse_nifty50.yaml` (already created)

**Methodology:**
- Same 35 tickers as "India multi-market"
- Same rolling 12-month windows (optimized methodology)
- Same 4 folds (2021-2024)
- Same signals (ZScore + OU)

**Expected Outcomes & Narrative:**
1. **If NSE Nifty 50 ≈ +0.75-0.85:** 
   - Universe quality is THE driver
   - Reframe: "Nifty 50 blue chips outperform Nifty 100 diluted portfolio"
   - NEW CONTRIBUTION: Universe selection > methodology tuning
   
2. **If NSE Nifty 50 ≈ +0.05-0.15:**
   - Multi-market diversification IS real (but smaller than claimed)
   - Reframe: "Geographic diversification adds +0.7 Sharpe beyond universe quality"
   - CONTRIBUTION: Both universe AND geography matter
   
3. **If NSE Nifty 50 ≈ -0.3 to 0:**
   - Current "India" result is unreliable (ML non-determinism?)
   - Reframe: "Rolling windows help but results remain unstable"
   - PROBLEM: May need to downgrade to M.S. thesis only (not publishable)

---

### **EXPERIMENT 2: NSE Nifty 50 + Expanding Windows**

**Purpose:** Complete 2×2 matrix (universe × methodology)

**Config:** Same as Experiment 1, but change `train_months: 48` (expanding)

**Expected Outcome:**
- Completes the full comparison table:

| Universe | Methodology | Sharpe | Trades | Status |
|----------|-------------|--------|--------|--------|
| Nifty 100 | Expanding | -0.409 | 1,096 | ✅ Done (Chapter 3) |
| Nifty 100 | Rolling | +0.052 | 293 | ✅ Done (Section 3.6) |
| Nifty 50 | Expanding | **???** | **???** | ❌ Missing |
| Nifty 50 | Rolling | **???** | **???** | ❌ Missing (CRITICAL) |

---

### **EXPERIMENT 3: Document ALL Existing Runs**

**Purpose:** Fix cherry-picking suspicion

**Action Required:**
1. Load ALL JSON files from `experimental-ablation/results/`
2. For each (market, signal) pair, report:
   - Number of runs (timestamps)
   - Mean ± std across runs
   - Which run was reported in the paper
3. Create transparency table:

```markdown
| Market | Signal | Run 1 | Run 2 | Run 3 | Reported | Selection Method |
|--------|--------|-------|-------|-------|----------|------------------|
| India | ZScore | +0.398 (289 trades) | -0.386 (279 trades) | +0.840 (123 trades) | +0.840 | Last chronological |
| India | OU | 0.000 (0 trades) | 0.000 (0 trades) | — | 0.000 | Failed experiment |
| Brazil | OU | ??? | ??? | ??? | +0.321 | ??? |
| ... | ... | ... | ... | ... | ... | ... |
```

**If you reported the BEST run, you MUST:**
- Disclose this fact prominently
- Apply Bonferroni correction (multiply p-values by number of runs)
- Recalculate "16x multiplier" using median, not best

---

## REFRAMING STRATEGIES BASED ON RESULTS

### **SCENARIO A: Nifty 50 Rolling ≈ +0.75-0.85 (Universe Quality Drives Results)**

**New Thesis Title:**
> "Universe Quality Dominates Methodology in Pairs Trading: Evidence from NSE Nifty 50 vs Nifty 100"

**New Abstract (sketch):**
> We demonstrate that stock universe selection (Nifty 50 blue chips vs Nifty 100 diversified) has 15x larger impact on pairs trading profitability than methodology optimization (expanding vs rolling windows). NSE Nifty 50 achieves +0.80 Sharpe vs Nifty 100 +0.05 Sharpe with identical methodology, costs, and market. This challenges the literature's focus on signal model tuning and suggests practitioners should prioritize asset selection over algorithm design.

**Key Changes:**
- Chapter 4 becomes: "Universe Quality Validation" (not multi-market)
- Multi-market results (US, Brazil, UK) become **supporting evidence** that framework generalizes
- Main contribution: "We found the wrong problem was being optimized"

**Strength:** CLEAR, NOVEL, DEFENSIBLE

---

### **SCENARIO B: Nifty 50 Rolling ≈ +0.05-0.15 (Small Geographic Effect)**

**New Thesis Title:**
> "Multi-Market Validation of Ensemble Pairs Trading: Geographic Diversification Beyond Universe Selection"

**New Abstract (sketch):**
> We demonstrate that multi-market diversification improves pairs trading profitability by +0.7 Sharpe beyond universe quality alone. Controlling for stock universe (Nifty 50), Indian multi-market portfolios achieve +0.84 Sharpe vs single-exchange NSE +0.10 Sharpe. This effect persists after controlling for methodology (rolling windows) and transaction costs (16.4 bps), suggesting cross-market correlation exploitation offers structural alpha.

**Key Changes:**
- Acknowledge confound explicitly in introduction
- Report NSE Nifty 50 as "within-market baseline"
- Multi-market results show **incremental value** beyond universe selection
- Contribution: "Geographic diversification matters, but less than expected"

**Strength:** HONEST, CREDIBLE, MODERATE IMPACT

---

### **SCENARIO C: Nifty 50 Rolling ≈ -0.3 to 0 (ML Non-Determinism Problem)**

**New Thesis Title:**
> "Challenges in Ensemble Pairs Trading: Sensitivity Analysis of Universe, Methodology, and ML Robustness"

**New Abstract (sketch):**
> We investigate why ensemble pairs trading fails on NSE Nifty 100 (-0.41 Sharpe) despite promising methodology. Rolling windows improve results modestly (+0.46 Sharpe, non-significant p=0.32). Universe selection (Nifty 50 vs 100) shows high sensitivity to ML selector randomness, with observed Sharpe ranging from -0.4 to +0.8 across runs (seed=42 insufficient). We conclude that ML-based pair selection requires deterministic CPU-only training for academic reproducibility.

**Key Changes:**
- Downgrade to "negative result" paper
- Focus on **methodological lessons learned**
- Contribution: "We identified why ensemble methods are unreliable"
- Target venue: Workshop (not journal) or M.S. thesis only

**Strength:** HONEST FAILURE ANALYSIS (publishable in workshops, not top journals)

---

## EXECUTION PLAN

### **PHASE 1: RUN CRITICAL EXPERIMENTS (3-4 days)**

**Day 1-2: Experiment 1 (NSE Nifty 50 + Rolling)**
```bash
cd /d/Code/Hybrid-Pairs-Trading-Ensemble/Implementation/experimental-ablation

# Fetch data (if not cached)
python scripts/fetch_market_data.py --market nse_nifty50

# Run ZScore signal
python scripts/run_multi_market_wfv.py \
    --market nse_nifty50 \
    --signal zscore \
    --n_folds 4 \
    --lookback 126

# Run OU signal
python scripts/run_multi_market_wfv.py \
    --market nse_nifty50 \
    --signal ou \
    --n_folds 4 \
    --lookback 126
```

**Day 2-3: Experiment 2 (NSE Nifty 50 + Expanding)**
- Modify config: `train_months: 48` (expanding)
- Re-run both signals

**Day 3-4: Experiment 3 (Document All Runs)**
```python
# Create comprehensive run analysis
python scripts/analyze_all_runs.py --output transparency_report.md
```

---

### **PHASE 2: ANALYZE & DECIDE NARRATIVE (1 day)**

**Decision Tree:**
1. Load NSE Nifty 50 Rolling result
2. Compare to India multi-market (+0.840)
3. Choose Scenario A, B, or C
4. Draft new abstract + chapter titles

---

### **PHASE 3: REWRITE THESIS (5-7 days)**

**Scenario A (Universe Quality):**
- Rewrite Chapter 1: Focus on "wrong optimization target"
- Rewrite Chapter 4: "Universe Quality Validation"
- Add Section 4.5: "Why Blue Chips Outperform"

**Scenario B (Small Geographic Effect):**
- Rewrite Section 4.1: "Controlling for Confounds"
- Add Section 4.2: "Incremental Value of Multi-Market"
- Tone down "16x" claims to "2-3x after controlling for universe"

**Scenario C (ML Robustness Issues):**
- Rewrite Chapter 5: "Lessons Learned"
- Add Section 5.6: "Reproducibility Challenges"
- Target: Workshop submission (NeurIPS ML in Finance)

---

## IMMEDIATE NEXT STEPS

### ✅ **COMPLETED:**
1. Created `configs/nse_nifty50.yaml`
2. Wrote `CRITIQUE.md` (brutal review)
3. Wrote `SALVAGE_PLAN.md` (this document)

### 🔄 **IN PROGRESS:**
4. Run Experiment 1: NSE Nifty 50 + Rolling + ZScore

### ❌ **PENDING:**
5. Run Experiment 1: NSE Nifty 50 + Rolling + OU
6. Run Experiment 2: NSE Nifty 50 + Expanding (both signals)
7. Document all existing runs (transparency report)
8. Analyze results → choose scenario A/B/C
9. Rewrite thesis based on chosen narrative

---

## TIMELINE ADJUSTMENT

**Original Target:** July 15, 2026 (JFM submission)

**Realistic Target:**
- **Best case (Scenario A):** August 15, 2026 (4 weeks for experiments + rewrite)
- **Medium case (Scenario B):** September 1, 2026 (6 weeks with deeper analysis)
- **Worst case (Scenario C):** October 15, 2026 (Workshop track, not journal)

**Buffer before JFM deadline (July 15):** **IMPOSSIBLE** without control experiments

**Recommendation:** Push back submission target to September 2026 (8 weeks) to do this RIGHT.

---

## RISK MITIGATION

### **IF EXPERIMENT 1 FAILS (e.g., 0 trades generated):**
- Debug lookback/signal issues
- Test on smaller date ranges (2022-2023 only)
- Worst case: Use expanding windows (more data = less risk of zero trades)

### **IF ML NON-DETERMINISM PERSISTS:**
- Switch to CPU-only TensorFlow (slow but deterministic)
- OR: Exclude ML selectors entirely (use 4 statistical selectors only)
- Report: "8-selector ensemble is unstable; 4-selector statistical-only is robust"

### **IF NSE NIFTY 50 RESULT IS WORSE THAN MULTI-MARKET:**
- Check data quality (same tickers, same dates?)
- Verify transaction costs are identical
- If confirmed: Scenario B narrative (small geographic effect is real)

---

## SUCCESS CRITERIA

**Minimum Viable Thesis:**
- Control experiment (NSE Nifty 50) run successfully
- Confound acknowledged explicitly in paper
- Honest reporting of all runs (no cherry-picking)
- Clear narrative (A, B, or C)

**Strong Thesis:**
- All 4 cells of 2×2 matrix filled
- Transparency report shows no cherry-picking
- Confidence intervals on all figures
- Literature review expanded (Avellaneda, Huck, etc.)

**Publishable Paper (JFM-level):**
- Scenario A (universe quality) with clean results
- All statistical tests robust (Bonferroni, bootstrap CI)
- Deterministic ML (CPU-only) for reproducibility
- Out-of-sample validation (2025+ data)

---

## FINAL WORD

**You have a thesis worth saving.** The experimental work is solid, the figures are excellent, the scope is ambitious.

**But you CANNOT claim geographic alpha without running the control experiment.**

Run NSE Nifty 50 + Rolling **THIS WEEK**. The result will determine your narrative.

**If +0.75:** You have a BETTER story (universe quality > everything)  
**If +0.10:** You have a HONEST story (small geographic effect)  
**If -0.30:** You have a LEARNING story (ML robustness challenges)

All three are defensible M.S. theses. Only Scenario A is a top-journal paper.

**DO NOT SUBMIT to JFM without this experiment. You WILL be rejected, and the rejection will cite this exact confound.**

---

**Next Action:** Run Experiment 1 NOW. Report back when results are in.
