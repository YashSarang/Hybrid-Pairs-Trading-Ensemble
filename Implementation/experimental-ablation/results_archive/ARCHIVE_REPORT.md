# PRE-SALVAGE ARCHIVE REPORT

**Date:** June 1, 2026  
**Archive Location:** `results_archive/2026-06-01_pre-salvage/`  
**Total Files:** 33 JSON files  
**Total Size:** 24 MB  

---

## CRITICAL FINDINGS FROM ARCHIVED RESULTS

### **🚨 HIGH VARIANCE ACROSS RUNS (NON-DETERMINISM CONFIRMED)**

| Market | Signal | Runs | Sharpe Range | Trade Range | Notes |
|--------|--------|------|--------------|-------------|-------|
| **India** | **ZScore** | **3** | **-0.386 to +0.840** | 123 to 289 | **1.2 Sharpe variance!** |
| **India** | OU | 2 | 0.000 to +0.200 | 0 to 26 | Run 1 failed (0 trades) |
| UK | ZScore | 2 | -0.245 to +0.265 | 111 to 268 | 0.5 Sharpe variance |
| US | Unknown | 3 | 0.000 to +0.774 | 0 to 302 | 0.77 Sharpe variance |
| Brazil | ZScore | 2 | -0.400 to -0.225 | 115 to 261 | Both negative |
| Brazil | OU | 3 | 0.000 to +0.321 | 0 to 32 | 2 runs failed |
| UK | OU | 3 | -0.405 to 0.000 | 0 to 42 | 2 runs failed |
| US | OU | 3 | -0.254 to 0.000 | 0 to 39 | 2 runs failed |

---

## DETAILED INVENTORY

### **INDIA (Multi-Market, Nifty 50)**

#### ZScore Signal: 3 runs
1. **Run 070512**: +0.398 Sharpe, 289 trades
2. **Run 083100**: **-0.386 Sharpe**, 279 trades ← NEGATIVE!
3. **Run 104009**: **+0.840 Sharpe**, 123 trades ← **REPORTED IN THESIS**

**Analysis:**
- Variance: 1.226 Sharpe units (from -0.386 to +0.840)
- Mean: +0.284 ± 0.631
- **The "headline result" (+0.840) is the BEST of 3 runs, not the mean**
- **Run 2 was NEGATIVE despite identical config**
- Trade count variation: 123 to 289 (2.4x difference)

**Transparency issue:** Thesis reports +0.840 without disclosing:
1. Two other runs with different results
2. One run was negative
3. Selection criteria (best? last? median?)

#### OU Signal: 2 runs
1. **Run 085647**: 0.000 Sharpe, 0 trades ← FAILED (no trades generated)
2. **Run 104015**: +0.200 Sharpe, 26 trades ← **REPORTED IN THESIS**

**Analysis:**
- First run failed completely (lookback issue?)
- Second run succeeded after parameter fix
- Only 26 trades across 4 years (6.5 trades/year = very sparse)

---

### **BRAZIL (B3 Ibovespa)**

#### OU Signal: 3 runs
1. **Run 074037**: 0.000 Sharpe, 0 trades ← FAILED
2. **Run 090411**: 0.000 Sharpe, 0 trades ← FAILED
3. **Run 101431**: +0.321 Sharpe, 32 trades ← **SUCCESS (REPORTED?)**

**Analysis:**
- 2 out of 3 runs failed (lookback issue identified and fixed)
- Only the last run succeeded
- Reported result (+0.321) is from the ONLY successful run

#### ZScore Signal: 2 runs
1. **Run 084830**: -0.400 Sharpe, 261 trades
2. **Run 101426**: -0.225 Sharpe, 115 trades ← **REPORTED?**

**Analysis:**
- Both negative, but Run 2 is "less bad" (-0.225 vs -0.400)
- Trade reduction (261 → 115) suggests parameter change
- If Run 2 reported, it's the "better failure"

---

### **UK (FTSE 100)**

#### ZScore Signal: 2 runs
1. **Run 092241**: **+0.265 Sharpe**, 268 trades ← POSITIVE!
2. **Run 110559**: -0.245 Sharpe, 111 trades ← **REPORTED IN THESIS**

**Analysis:**
- Run 1 was POSITIVE (+0.265), Run 2 was NEGATIVE (-0.245)
- **Thesis reports the NEGATIVE run, not the positive one**
- Trade reduction (268 → 111) suggests parameter change
- **WHY was the negative run reported instead of the positive?**

#### OU Signal: 3 runs
1. **Run 074531**: 0.000 Sharpe, 0 trades ← FAILED
2. **Run 092934**: 0.000 Sharpe, 0 trades ← FAILED
3. **Run 110551**: -0.405 Sharpe, 42 trades ← **REPORTED**

---

### **US (S&P 500 Subset)**

#### OU Signal: 3 runs
1. **Run 070452**: 0.000 Sharpe, 0 trades ← FAILED
2. **Run 083228**: 0.000 Sharpe, 0 trades ← FAILED
3. **Run 113145**: -0.254 Sharpe, 39 trades ← **REPORTED**

#### Unknown Signal (ZScore?): 3 runs
1. **Run 020824**: 0.000 Sharpe, 0 trades ← FAILED
2. **Run 023302**: +0.116 Sharpe, 106 trades
3. **Run 025102**: **+0.774 Sharpe**, 302 trades ← HIGH VARIANCE!

**Analysis:**
- US results show same pattern: early runs failed, later runs succeeded
- Unknown signal shows 0.658 Sharpe variance (0.116 → 0.774)
- If this is ZScore, Run 3 (+0.774) is VERY STRONG (not reported?)

---

## PATTERN ANALYSIS

### **Common Pattern: Early Runs Failed, Later Runs Succeeded**

**Timeline reconstruction:**
- **~02:00-07:00** (timestamps 020824-070512): Many 0-trade failures
- **~08:00-09:00** (timestamps 083100-092934): Mixed success/failure
- **~10:00-11:00** (timestamps 101426-113145): Mostly successful

**Hypothesis:** Lookback parameter bug identified during May 29 experiments:
- Initial: `lookback=252` (12 months) exhausted test windows → 0 trades
- Fixed: `lookback=126` (6 months) left sufficient test data → trades generated

**Evidence from logs:**
> "Issue: Initial runs with lookback=252 consumed entire test window, leaving no data for signal generation → zero trades. Solution: Reduced to lookback=126."

**Implication:**
- Early failures (0 trades) are due to known bug, not signal failure
- Only post-fix runs (lookback=126) are valid
- BUT: Post-fix runs still show high variance (India -0.386 to +0.840)

---

### **Non-Determinism CONFIRMED**

**Evidence:**
1. **India ZScore:** +0.398 → -0.386 → +0.840 (same config, 1.2 Sharpe variance)
2. **UK ZScore:** +0.265 → -0.245 (0.5 Sharpe variance, SIGN FLIP!)
3. **US Unknown:** +0.116 → +0.774 (0.66 Sharpe variance)

**Cause:** ML selectors (LSTM, Transformer, GNN) are non-deterministic on GPU despite `seed=42`

**From thesis Section 3.6.7:**
> "Run 1 vs Run 2 comparison shows TensorFlow GPU randomness despite seed=42"

**Impact on thesis:**
- **The +0.840 result may be within-run noise**, not a true signal
- **Reproducibility is BROKEN** (reviewer cannot replicate result)
- **Cherry-picking suspicion** (was best run intentionally selected?)

---

## REPORTING PROTOCOL ANALYSIS

### **Which Runs Were Reported?**

| Market | Signal | Reported Sharpe | Run Used | Other Runs | Selection Criteria? |
|--------|--------|----------------|----------|------------|---------------------|
| India | ZScore | +0.840 | 104009 | +0.398, -0.386 | Last? Best? |
| India | OU | +0.200 | 104015 | 0.000 (failed) | Only successful |
| Brazil | OU | +0.321 | 101431 | 0.000, 0.000 (failed) | Only successful |
| Brazil | ZScore | -0.225 | 101426 | -0.400 | Less negative |
| UK | ZScore | -0.245 | 110559 | **+0.265** | NEGATIVE over POSITIVE?! |
| UK | OU | -0.405 | 110551 | 0.000, 0.000 (failed) | Only successful |
| US | OU | -0.254 | 113145 | 0.000, 0.000 (failed) | Only successful |

**Observations:**
1. **For most markets:** Last run (by timestamp) was reported
2. **For India ZScore:** Last run (+0.840) also happened to be the best
3. **For UK ZScore:** Last run (-0.245) was WORSE than first run (+0.265) — why not report the positive one?

**Two possible interpretations:**
1. **Chronological protocol:** "Always report the last (most recent) run" → defensible
2. **Result-dependent:** "Report best for India, report last for others" → cherry-picking

**Without explicit documentation, suspicion remains.**

---

## TRANSPARENCY RECOMMENDATIONS

### **What Should Have Been Reported:**

| Market | Signal | Mean ± Std | Median | Best | Worst | Reported |
|--------|--------|------------|--------|------|-------|----------|
| India | ZScore | +0.284 ± 0.631 | +0.398 | **+0.840** | -0.386 | +0.840 |
| UK | ZScore | +0.010 ± 0.361 | +0.010 | +0.265 | -0.245 | -0.245 |

**Honest reporting would state:**
> "India ZScore achieved +0.840 Sharpe in the final run (Run 3 of 3), with prior runs yielding +0.398 and -0.386. Mean across runs: +0.284 ± 0.631. High variance attributed to ML selector non-determinism (TensorFlow GPU randomness despite seed=42)."

**Instead, thesis reports:**
> "India multi-market achieves +0.840 Sharpe" (no mention of other runs or variance)

---

## ACTION ITEMS FOR TRANSPARENCY

1. **Document ALL runs explicitly in thesis:**
   - Table showing all runs per (market, signal)
   - Report mean ± std, not just single best run
   - Disclose selection criteria (chronological? best? median?)

2. **Add confidence intervals to ALL figures:**
   - Error bars = ±1 std across runs
   - If only 1 successful run, note "insufficient replication"

3. **Fix ML non-determinism:**
   - Option A: Use CPU-only mode (`export CUDA_VISIBLE_DEVICES=-1`)
   - Option B: Exclude ML selectors (use 4 statistical selectors only)
   - Option C: Report high variance explicitly as "methodological limitation"

4. **Re-run critical experiments with deterministic config:**
   - India ZScore: 5 runs on CPU-only → report mean ± std
   - If variance persists on CPU, ML selector weight = 0 (use statistical only)

5. **Create supplementary materials:**
   - All JSON files published alongside paper
   - Reproducibility script: `run_all_experiments.sh` with exact seeds/config

---

## ARCHIVE INTEGRITY

✅ **All 33 JSON files backed up to `results_archive/2026-06-01_pre-salvage/`**

**File structure:**
```
results_archive/2026-06-01_pre-salvage/
├── brazil/
│   ├── wfv_4folds_ou_20260529_074037.json
│   ├── wfv_4folds_ou_20260529_090411.json
│   ├── wfv_4folds_ou_20260529_101431.json
│   ├── wfv_4folds_zscore_20260529_084830.json
│   └── wfv_4folds_zscore_20260529_101426.json
├── india/
│   ├── wfv_4folds_ou_20260529_085647.json
│   ├── wfv_4folds_ou_20260529_104015.json
│   ├── wfv_4folds_zscore_20260529_070512.json
│   ├── wfv_4folds_zscore_20260529_083100.json
│   └── wfv_4folds_zscore_20260529_104009.json (HEADLINE +0.840)
├── uk/ (5 files)
├── us/ (6 files)
└── [various other files from root results/ directory]
```

**Verification checksum:** [TO BE ADDED IF NEEDED]

---

## NEXT STEPS

1. ✅ Archive complete (33 files, 24 MB)
2. 🔄 Run NSE Nifty 50 control experiments (RUN 001, RUN 002)
3. ⏳ Compare NSE Nifty 50 vs India multi-market (isolate geographic effect)
4. ⏳ Create full transparency report with ALL runs documented
5. ⏳ Decide on narrative (Scenario A, B, or C) based on control results

---

**Archive Status:** ✅ COMPLETE AND DOCUMENTED  
**Next Action:** Execute `run_control_experiment.sh`
