# Thesis Chapter Structure & Experiment Organization

**Last Updated:** 2026-05-29  
**Status:** Rolling-window validation in progress

---

## **Current Thesis Structure (Planned)**

### **Chapter 1: Introduction**
- Pairs trading background
- Research motivation
- Contributions

### **Chapter 2: Literature Review** ✅ COMPLETE
- **File:** `reports/chapter2_literature_review.md`
- 228 lines, 18 papers cited
- Gap analysis: `GAP_ANALYSIS.md`

### **Chapter 3: Methodology & NSE Baseline** ⚠️ UNDER REVIEW
Two candidate versions (choose one after validation completes):

#### **Option 3A: Expanding Window (Current)**
- **Files:** `experiments/walk_forward.py`, `results/walk_forward_20260506_104613.json`
- **Methodology:** Expanding-window WFV (academic standard)
- **Training:** 4-6 years per fold
- **Result:** Net Sharpe **-0.409 ± 0.774**
- **Trades:** 1,096 total
- **Status:** ✅ Complete, academically rigorous

#### **Option 3B: Rolling Window (Validation)** 🔄 RUNNING
- **Files:** `experiments/walk_forward_rolling.py`, `results/rolling_window_validation_20260529/`
- **Methodology:** Rolling 12-month windows (deployment realism)
- **Training:** 12 months per fold
- **Signal params:** lookback=126 (matches multi-market)
- **Result:** **TBD** (running now)
- **Status:** 🔄 In progress

**Decision criteria:** See `METHODOLOGY_COMPARISON.md`

### **Chapter 4: Multi-Market Generalization Study** ✅ READY TO WRITE
- **Files:** `experimental-ablation/` directory
- **Documentation:** `MULTI_MARKET_RESULTS.md` (13KB), `SUMMARY_QUICK_REF.md` (4KB)
- **Experiments:** 7 complete (4 markets × 2 signals, minus US ZScore)
- **Key Result:** India+ZScore Sharpe +0.840, signal fit > transaction costs
- **Status:** ✅ Experiments complete, ready for thesis writing

### **Chapter 5: Discussion & Limitations**
- UK market failure analysis
- Transaction cost sensitivity
- Regime detection needs
- Future work

### **Chapter 6: Conclusion**
- Contributions summary
- Deployment recommendations

---

## **Experiment Directory Structure**

```
Implementation/
├── experiments/                          # Chapter 3 (NSE Baseline)
│   ├── walk_forward.py                  # Original expanding-window WFV
│   ├── walk_forward_rolling.py          # NEW: Rolling-window validation
│   ├── results/
│   │   ├── walk_forward_20260506_104613.json        # Expanding: Sharpe -0.409
│   │   └── rolling_window_validation_20260529/      # Rolling: TBD
│   │       ├── METHODOLOGY_COMPARISON.md            # Decision criteria
│   │       ├── run_log.txt                          # Execution log
│   │       └── walk_forward_rolling_<timestamp>.json  # Results (pending)
│   └── [other ablation studies...]
│
├── experimental-ablation/               # Chapter 4 (Multi-Market)
│   ├── configs/                         # 4 market YAMLs (india, us, brazil, uk)
│   ├── scripts/
│   │   └── run_multi_market_wfv.py      # Main experiment script
│   ├── results/
│   │   ├── india/wfv_4folds_*.json      # India results (2 experiments)
│   │   ├── us/wfv_4folds_*.json         # US results (1 experiment)
│   │   ├── brazil/wfv_4folds_*.json     # Brazil results (2 experiments)
│   │   └── uk/wfv_4folds_*.json         # UK results (2 experiments)
│   ├── MULTI_MARKET_RESULTS.md          # Full analysis (13KB)
│   ├── SUMMARY_QUICK_REF.md             # Quick reference (4KB)
│   └── KALPANA_QUICKSTART.md            # Cluster usage guide
│
└── reports/                             # Thesis chapters
    ├── chapter2_literature_review.md    # ✅ Complete
    └── GAP_ANALYSIS.md                  # ✅ Complete
```

---

## **Results Summary**

### **NSE Baseline (India Market)**

| Experiment | Methodology | Net Sharpe | Trades | Status |
|------------|-------------|------------|--------|--------|
| Expanding WFV | 4-6 year train | **-0.409** | 1,096 | ✅ Complete |
| Rolling WFV | 12-month train | **TBD** | TBD | 🔄 Running |
| Multi-Market (India+ZScore) | Rolling 12-month | **+0.840** | 123 | ✅ Complete |
| Multi-Market (India+OU) | Rolling 12-month | **+0.200** | 26 | ✅ Complete |

### **Cross-Market Comparison (Rolling 12-month)**

| Market | Signal | Net Sharpe | Trades | Cost (bps) | Status |
|--------|--------|------------|--------|------------|--------|
| 🇮🇳 India | ZScore | **+0.840** | 123 | 16.4 | ⭐ Best |
| 🇧🇷 Brazil | OU | **+0.321** | 32 | 8.4 | ✅ Good |
| 🇮🇳 India | OU | **+0.200** | 26 | 16.4 | ✅ Good |
| 🇧🇷 Brazil | ZScore | -0.225 | 115 | 8.4 | ❌ Negative |
| 🇬🇧 UK | ZScore | -0.245 | 111 | 8.0 | ❌ Negative |
| 🇺🇸 US | OU | -0.254 | 39 | 2.7 | ❌ Negative |
| 🇬🇧 UK | OU | -0.405 | 42 | 8.0 | ❌ Worst |

---

## **Key Findings (Chapter 4)**

1. **Market Dependence Dominates:** India wins (+0.84), UK fails (-0.41) — signal fit > costs
2. **Signal Comparison:** ZScore aggressive (100+ trades), OU conservative (26-42 trades)
3. **Cost Paradox:** India profitable despite highest costs (16.4 bps), US unprofitable despite lowest (2.7 bps)
4. **Ensemble Robustness:** Pair selection worked across all 4 markets (100% execution rate)
5. **Lookback Critical:** lookback=126 required for 12-month test windows (252 exhausted them)

---

## **Decision Timeline**

| Time | Event | Action Required |
|------|-------|----------------|
| **2026-05-29 13:00** | Rolling validation started | ⏳ Wait |
| **2026-05-29 14:00** | Expected completion | 📊 Analyze results |
| **2026-05-29 14:30** | Compare expanding vs rolling | 🎯 Decide Chapter 3 version |
| **2026-05-29 15:00** | Start Chapter 4 writing | 📝 6-8 hours |
| **2026-05-30 AM** | Chapter 4 draft complete | 📤 Ready for review |

---

## **Writing Priority**

1. ✅ **Chapter 2** — Complete
2. ⏳ **Chapter 3** — Awaiting rolling validation results
3. 🎯 **Chapter 4** — Ready to write (start after rolling results analyzed)
4. ⏸️ **Chapters 5-6** — After 3 & 4 complete

---

## **Git Commits**

- `b311fd0` — Multi-market results documented (MULTI_MARKET_RESULTS.md)
- `cc8a3bc` — Fixed lookback=126 for signal models
- `2c7f584` — Added rolling-window validation script (current HEAD)

---

**Status: 🔄 ROLLING VALIDATION IN PROGRESS**  
**Next Action: Wait for results, then write Chapter 4**
