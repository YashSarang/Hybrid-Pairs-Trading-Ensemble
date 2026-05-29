# Methodology Comparison: Expanding vs Rolling Windows

**Date:** May 29, 2026  
**Purpose:** Validate whether thesis baseline (expanding window) is comparable to multi-market experiments (rolling window)

---

## **Background**

The thesis contains **two sets of experiments with different methodologies:**

### **Thesis Baseline (E1-E6):**
- **File:** `experiments/results/walk_forward_20260506_104613.json`
- **Methodology:** Expanding-window WFV
- **Training:** 4-6 years (grows each fold)
- **Test:** 12 months per fold
- **Signal params:** ZScore lookback=60, OU lookback=252
- **Result:** Net Sharpe **-0.409 ± 0.774**
- **Trades:** 1,096 total (149-203 per fold)

### **Multi-Market Experiments:**
- **Files:** `experimental-ablation/results/*/wfv_4folds_*.json`
- **Methodology:** Rolling-window WFV
- **Training:** 12 months (fixed)
- **Test:** 12 months per fold
- **Signal params:** Both signals lookback=126
- **Result (India):** Net Sharpe **+0.840 ± 0.748** (ZScore), **+0.200 ± 0.346** (OU)
- **Trades:** 123 (ZScore), 26 (OU) per experiment

---

## **The Problem**

**Can't compare directly** — different training windows, different signal parameters, dramatically different results.

**Options:**
1. Keep both, explain difference in thesis (Option A)
2. Re-run thesis with rolling windows to match multi-market (Option B)

**We're doing BOTH:** Add multi-market as new chapter + re-run thesis baseline as validation

---

## **Experiment: Rolling-Window Validation**

### **Objective:**
Re-run thesis E1-E6 experiments using **rolling 12-month windows** matching multi-market methodology.

### **Methodology:**
- **Fixed 12-month training** (2020 → test 2021, 2021 → test 2022, etc.)
- **lookback=126** for both ZScore and OU (matches multi-market)
- **Same ensemble:** 8 selectors, equal-weight voting
- **Same costs:** IndianCosts (16.4 bps per trade)
- **Same universe:** NSE Nifty 50

### **Fold Structure:**
```
Fold 1: Train 2020 → Test 2021
Fold 2: Train 2021 → Test 2022
Fold 3: Train 2022 → Test 2023
Fold 4: Train 2023 → Test 2024 (partial)
```

### **Comparison Matrix:**

| Aspect | Expanding (Thesis) | Rolling (Validation) | Multi-Market India |
|--------|-------------------|----------------------|-------------------|
| Training Window | 4-6 years | 12 months | 12 months |
| ZScore lookback | 60 | 126 | 126 |
| OU lookback | 252 | 126 | 126 |
| Net Sharpe | -0.409 | **TBD** | +0.840 (ZScore) |
| Trades/Fold | 149-203 | **TBD** | 123 (ZScore), 26 (OU) |

---

## **Expected Outcomes**

### **Scenario 1: Rolling > Expanding** (most likely)
- **Result:** Rolling-window NSE gets positive Sharpe (close to multi-market +0.84)
- **Interpretation:** Shorter training windows adapt faster to regime changes
- **Thesis Decision:** **REPLACE** expanding with rolling in Chapter 3
- **Narrative:** "Rolling windows match deployment reality and outperform expanding"

### **Scenario 2: Rolling ≈ Expanding**
- **Result:** Both negative or both weakly positive
- **Interpretation:** NSE 2020-2024 period is challenging regardless of methodology
- **Thesis Decision:** **KEEP** expanding (academic standard) in Ch3, add rolling multi-market as Ch4
- **Narrative:** "Academic rigor (Ch3) + deployment validation (Ch4)"

### **Scenario 3: Expanding > Rolling** (unexpected)
- **Result:** Expanding -0.41 is better than rolling (e.g., rolling gets -0.8)
- **Interpretation:** Longer training provides more stable parameter estimates
- **Thesis Decision:** **KEEP** expanding in Ch3, explain multi-market uses different approach
- **Narrative:** "Thesis validates methodology rigorously; multi-market explores market dependence"

---

## **Decision Criteria**

### **IF Rolling NSE Sharpe > +0.5:**
✅ **REPLACE** expanding with rolling in Chapter 3  
✅ Multi-market becomes "natural extension" to other markets  
✅ Unified methodology throughout thesis  

### **IF Rolling NSE Sharpe ∈ [-0.2, +0.5]:**
⚠️ **KEEP** expanding in Chapter 3 (marginal difference not worth changing)  
✅ Multi-market as separate Chapter 4 with methodology explanation  

### **IF Rolling NSE Sharpe < -0.5:**
❌ **KEEP** expanding in Chapter 3 (rolling is worse)  
⚠️ Multi-market requires careful framing ("deployment-focused study")  

---

## **Files & Locations**

### **Thesis Baseline (Expanding):**
- Script: `experiments/walk_forward.py`
- Results: `experiments/results/walk_forward_20260506_104613.json`
- Status: ✅ Complete, preserved

### **Rolling Validation (NEW):**
- Script: `experiments/walk_forward_rolling.py`
- Results: `experiments/results/rolling_window_validation_20260529/walk_forward_rolling_<timestamp>.json`
- Log: `experiments/results/rolling_window_validation_20260529/run_log.txt`
- Status: 🔄 Running

### **Multi-Market:**
- Scripts: `experimental-ablation/scripts/run_multi_market_wfv.py`
- Results: `experimental-ablation/results/*/wfv_4folds_*.json`
- Documentation: `experimental-ablation/MULTI_MARKET_RESULTS.md`
- Status: ✅ Complete (7/7 experiments)

---

## **Next Steps**

1. ⏳ **Wait for rolling validation** to complete (~30-60 minutes)
2. 📊 **Compare** expanding vs rolling NSE results
3. 🎯 **Decide** which to use in Chapter 3 (based on criteria above)
4. 📝 **Write comparison section** for thesis
5. 📝 **Write Chapter 4** using multi-market results (regardless of decision)

---

## **Timeline**

- **2026-05-29 13:00:** Rolling validation started
- **2026-05-29 14:00:** Expected completion
- **2026-05-29 14:30:** Analysis & decision
- **2026-05-29 15:00-21:00:** Write Chapter 4 (6-8 hours)

---

**Status: 🔄 ROLLING VALIDATION IN PROGRESS**
