# EXPERIMENT ARCHIVE & DOCUMENTATION LOG

**Date Created:** June 1, 2026  
**Purpose:** Track all experimental runs with full provenance and audit trail  
**Status:** Active logging of thesis salvage experiments

---

## ARCHIVE STRUCTURE

```
Implementation/
├── experimental-ablation/
│   ├── results/                    # Current/active results
│   ├── results_archive/            # Historical results (timestamped backups)
│   │   └── 2026-06-01_pre-salvage/ # Backup BEFORE control experiments
│   └── logs/                       # Execution logs
│       └── experiment_log.md       # This file (master log)
```

---

## PRE-SALVAGE BASELINE (June 1, 2026)

### **Context**
Before running NSE Nifty 50 control experiments, we archive ALL existing results to preserve the original thesis state.

**Reason for control experiments:**
- Current thesis compares India multi-market (Nifty 50) vs NSE baseline (Nifty 100)
- Confound: Different stock universes (71% overlap, but different quality tiers)
- Need: NSE Nifty 50 results to isolate universe quality from geographic effects

### **Archived Results Location**
`Implementation/experimental-ablation/results_archive/2026-06-01_pre-salvage/`

**Preserved experiments:**
1. **Brazil** (4 markets tested)
   - OU signal: 3 runs (timestamps: 074037, 090411, 101431)
   - ZScore signal: 2 runs (timestamps: 084830, 101426)

2. **India** (multi-market, Nifty 50)
   - OU signal: 2 runs (timestamps: 085647, 104015)
   - ZScore signal: 3 runs (timestamps: 070512, 083100, 104009)

3. **UK** (4 markets tested)
   - OU signal: 3 runs (timestamps: 074531, 092934, 110551)
   - ZScore signal: 2 runs (timestamps: 092241, 110559)

4. **US** (4 markets tested)
   - OU signal: 3 runs (timestamps: 070452, 083228, 113145)
   - ZScore signal: 1 run (timestamp: 025102)

### **Key Results (as reported in thesis_drafts/)**

| Market | Signal | Runs | Reported Sharpe | Reported Trades | File Used | Notes |
|--------|--------|------|----------------|-----------------|-----------|-------|
| India | ZScore | 3 | **+0.840** | 123 | 104009 | Headline result; variance across runs: [+0.398, -0.386, +0.840] |
| India | OU | 2 | +0.200 | 26 | 104015 | First run (085647) had 0 trades |
| Brazil | OU | 3 | +0.321 | 32 | 090411 | Need to verify which run was reported |
| Brazil | ZScore | 2 | -0.225 | 115 | 084830 | Need to verify |
| UK | ZScore | 2 | -0.245 | 111 | 092241 | Need to verify |
| UK | OU | 3 | -0.405 | 42 | 074531 | Need to verify |
| US | OU | 3 | -0.254 | 39 | 070452 | Need to verify |

**⚠️ CRITICAL OBSERVATION:**
India ZScore has **3 runs** with drastically different results:
- Run 1 (070512): +0.398 Sharpe, 289 trades
- Run 2 (083100): **-0.386 Sharpe**, 279 trades
- Run 3 (104009): +0.840 Sharpe, 123 trades ← **REPORTED IN THESIS**

**Questions for transparency:**
1. Why 3 runs? (Bug fixes? Parameter changes? Random re-runs?)
2. Which run was intentionally selected? (Last chronological? Best result?)
3. If best result, was this disclosed in thesis?

---

## EXPERIMENT RUN LOG

### **RUN 001: NSE Nifty 50 + Rolling + ZScore** (June 1, 2026)

**Purpose:** Control experiment to isolate universe quality from geographic effects

**Configuration:**
- Market: NSE (National Stock Exchange of India)
- Universe: Nifty 50 (same 35 tickers as "India multi-market")
- Methodology: Rolling 12-month windows (optimized methodology from Section 3.6)
- Signal: ZScoreThreshold (lookback=126, entry_z=2.0, exit_z=0.5)
- Folds: 4 (test years 2021-2024)
- Transaction costs: 16.355 bps (identical to multi-market)

**Hypothesis:**
- If Sharpe ≈ +0.75-0.85 → Universe quality drives results (Scenario A)
- If Sharpe ≈ +0.05-0.20 → Small geographic effect exists (Scenario B)
- If Sharpe ≈ -0.30 to +0.05 → ML non-determinism or failed (Scenario C)

**Execution command:**
```bash
cd /d/Code/Hybrid-Pairs-Trading-Ensemble/Implementation/experimental-ablation
source ../../.venv/Scripts/activate
python scripts/run_multi_market_wfv.py \
    --market nse_nifty50 \
    --signal zscore \
    --n_folds 4 \
    --lookback 126
```

**Status:** Pending execution  
**Started:** [TO BE FILLED]  
**Completed:** [TO BE FILLED]  
**Result file:** `results/nse_nifty50/wfv_4folds_zscore_YYYYMMDD_HHMMSS.json`  
**Net Sharpe:** [TO BE FILLED]  
**Total Trades:** [TO BE FILLED]  

---

### **RUN 002: NSE Nifty 50 + Rolling + OU** (June 1, 2026)

**Purpose:** Control experiment with alternative signal model

**Configuration:**
- Market: NSE (National Stock Exchange of India)
- Universe: Nifty 50 (same 35 tickers as "India multi-market")
- Methodology: Rolling 12-month windows
- Signal: OUThreshold (lookback=126, entry_k=1.5, exit_k=0.2)
- Folds: 4 (test years 2021-2024)
- Transaction costs: 16.355 bps

**Hypothesis:**
- Validates consistency across signal models
- India OU was +0.200 (multi-market) → expect NSE Nifty 50 OU ≈ similar or lower

**Execution command:**
```bash
python scripts/run_multi_market_wfv.py \
    --market nse_nifty50 \
    --signal ou \
    --n_folds 4 \
    --lookback 126
```

**Status:** Pending execution  
**Started:** [TO BE FILLED]  
**Completed:** [TO BE FILLED]  
**Result file:** `results/nse_nifty50/wfv_4folds_ou_YYYYMMDD_HHMMSS.json`  
**Net Sharpe:** [TO BE FILLED]  
**Total Trades:** [TO BE FILLED]  

---

## COMPARISON MATRIX (TO BE FILLED POST-EXPERIMENTS)

### **2×2 Universe × Methodology Matrix**

| Universe | Methodology | Signal | Net Sharpe | Trades | Source | Status |
|----------|-------------|--------|------------|--------|--------|--------|
| Nifty 100 | Expanding | ZScore | -0.409 | 1,096 | Chapter 3 baseline | ✅ Complete |
| Nifty 100 | Rolling | ZScore | +0.052 | 293 | Section 3.6 | ✅ Complete |
| Nifty 50 | Expanding | ZScore | ??? | ??? | Future work | ❌ Not started |
| Nifty 50 | Rolling | ZScore | **[RUN 001]** | **[RUN 001]** | Control experiment | 🔄 Running |
| Nifty 100 | Rolling | OU | ??? | ??? | Not tested | ❌ Missing |
| Nifty 50 | Rolling | OU | **[RUN 002]** | **[RUN 002]** | Control experiment | 🔄 Running |

### **Multi-Market Comparison (Nifty 50 universe only)**

| Market | Exchange | Signal | Net Sharpe | Trades | Source | Notes |
|--------|----------|--------|------------|--------|--------|-------|
| India Multi-Market | NSE + others? | ZScore | +0.840 | 123 | Chapter 4, run 104009 | **Headline result** |
| NSE Single-Exchange | NSE | ZScore | **[RUN 001]** | **[RUN 001]** | Control experiment | **Critical comparison** |
| India Multi-Market | NSE + others? | OU | +0.200 | 26 | Chapter 4, run 104015 | Second signal |
| NSE Single-Exchange | NSE | OU | **[RUN 002]** | **[RUN 002]** | Control experiment | Validation |

**⚠️ KEY QUESTION:** What does "India multi-market" actually mean?
- Is it NSE data with different selection methodology?
- Is it truly multi-exchange (NSE + BSE)?
- **MUST CLARIFY** this in thesis to avoid confusion

---

## TRANSPARENCY REPORT (TO BE COMPLETED)

### **Multiple Runs Analysis**

For each (market, signal) pair, we will document:

| Market | Signal | Run Count | Run 1 Sharpe | Run 2 Sharpe | Run 3 Sharpe | Reported | Selection Criteria |
|--------|--------|-----------|--------------|--------------|--------------|----------|-------------------|
| India | ZScore | 3 | +0.398 (289t) | -0.386 (279t) | +0.840 (123t) | +0.840 | ??? |
| India | OU | 2 | 0.000 (0t) | [CHECK] | — | +0.200 | ??? |
| Brazil | OU | 3 | [CHECK] | [CHECK] | [CHECK] | +0.321 | ??? |
| Brazil | ZScore | 2 | [CHECK] | [CHECK] | — | -0.225 | ??? |
| UK | ZScore | 2 | [CHECK] | [CHECK] | — | -0.245 | ??? |
| UK | OU | 3 | [CHECK] | [CHECK] | [CHECK] | -0.405 | ??? |
| US | OU | 3 | [CHECK] | [CHECK] | [CHECK] | -0.254 | ??? |

**Action items:**
1. Load ALL JSON files
2. Extract Sharpe + trade count from each
3. Determine which run was reported in thesis
4. Document selection criteria (chronological? best? median?)
5. Calculate mean ± std across runs
6. Report variance explicitly in thesis

---

## DATA PROVENANCE

### **India Multi-Market (Nifty 50)**
- **Config file:** `experimental-ablation/configs/india.yaml`
- **Data source:** yfinance, NSE .NS tickers
- **Date range:** 2020-01-01 to 2025-05-01
- **Tickers (35):** RELIANCE.NS, TCS.NS, HDFCBANK.NS, INFY.NS, ICICIBANK.NS, ... (see config)
- **Cache location:** `experimental-ablation/data/india/prices_2020-01-01_2025-05-01.parquet`
- **File size:** 347,610 bytes

### **NSE Baseline (Nifty 100)**
- **Config file:** `experiments/config.py` (NSE_UNIVERSE constant)
- **Data source:** yfinance, NSE .NS tickers
- **Date range:** 2016-01-01 to 2026-03-31 (longer history for expanding windows)
- **Tickers (35):** HDFCBANK.NS, ICICIBANK.NS, SBIN.NS, ... (different list!)
- **Cache location:** Not explicitly cached in multi-market folder

### **NSE Nifty 50 (Control Experiment)**
- **Config file:** `experimental-ablation/configs/nse_nifty50.yaml` (NEW, created 2026-06-01)
- **Data source:** yfinance, NSE .NS tickers
- **Date range:** 2020-01-01 to 2025-05-01 (matches India multi-market)
- **Tickers (35):** Same as India multi-market (RELIANCE.NS, TCS.NS, etc.)
- **Cache location:** `experimental-ablation/data/nse_nifty50/prices_2020-01-01_2025-05-01.parquet`

**Universe overlap analysis:**
- Nifty 50 ∩ Nifty 100: 25 tickers (71% overlap)
- Only in Nifty 50: LT.NS, BHARTIARTL.NS, ASIANPAINT.NS, TITAN.NS, BAJFINANCE.NS, NTPC.NS, POWERGRID.NS, TATAMOTORS.NS, ADANIENT.NS, GRASIM.NS (10 tickers)
- Only in Nifty 100: ACC.NS, BAJAJ-AUTO.NS, BRITANNIA.NS, CIPLA.NS, DIVISLAB.NS, DRREDDY.NS, EICHERMOT.NS, HEROMOTOCO.NS, INDUSINDBK.NS, SHREECEM.NS (10 tickers)

---

## REPRODUCIBILITY NOTES

### **Random Seeds**
- Global: `RANDOM_SEED = 42` in `experiments/config.py`
- TensorFlow/Keras: Set via `tf.random.set_seed(42)` in selectors_ml.py
- NumPy: Set via `np.random.seed(42)` in experiment scripts

**⚠️ KNOWN ISSUE:** Despite seed=42, ML selectors (LSTM, Transformer, GNN) exhibit non-deterministic behavior on GPU.

**Evidence:**
- India ZScore run variance: +0.398 → -0.386 → +0.840 (1.2 Sharpe range!)

**Recommended fix:**
- Force CPU-only mode: `export CUDA_VISIBLE_DEVICES=-1` (slower but deterministic)
- OR: Exclude ML selectors, use 4 statistical selectors only

### **Software Versions**
[TO BE FILLED AFTER EXPERIMENT RUN]
- Python: ???
- TensorFlow: ???
- NumPy: ???
- Pandas: ???
- yfinance: ???

### **Hardware**
- Machine: Windows 11 (MSYS bash)
- CPU: ???
- GPU: ??? (if used)
- RAM: ???

---

## CHANGE LOG

### **2026-06-01: Pre-Salvage Archive**
- **Action:** Backed up all existing results to `results_archive/2026-06-01_pre-salvage/`
- **Reason:** Preserve original thesis state before running control experiments
- **Files archived:** All JSON files from `results/brazil/`, `results/india/`, `results/uk/`, `results/us/`
- **Archive size:** [TO BE FILLED]

### **2026-06-01: NSE Nifty 50 Config Created**
- **File:** `configs/nse_nifty50.yaml`
- **Purpose:** Control experiment to isolate universe quality from geographic effects
- **Key parameters:** Same as india.yaml (Nifty 50 tickers, rolling windows, 4 folds)
- **Difference:** Explicitly labeled as NSE (not "multi-market")

### **2026-06-01: Control Experiments Initiated**
- **RUN 001:** NSE Nifty 50 + Rolling + ZScore
- **RUN 002:** NSE Nifty 50 + Rolling + OU
- **Expected runtime:** 1-2 hours total

---

## POST-EXPERIMENT ACTIONS (CHECKLIST)

After RUN 001 and RUN 002 complete:

- [ ] Fill in result files, Sharpe ratios, trade counts in this log
- [ ] Load ALL JSON files from results_archive and document multiple runs
- [ ] Create transparency report table (mean ± std for each market×signal)
- [ ] Compare NSE Nifty 50 vs India multi-market (isolate geographic effect)
- [ ] Update comparison matrices above
- [ ] Choose scenario (A: universe quality, B: small geographic, C: ML issues)
- [ ] Draft new abstract + chapter structure
- [ ] Update thesis_drafts/ with chosen narrative
- [ ] Add confidence intervals to all figures (use variance across runs)
- [ ] Commit all results to git with clear message

---

## AUDIT TRAIL

**Who:** TARS (Hermes Agent)  
**When:** June 1, 2026, 09:43 AM IST  
**Why:** Critical confound identified in thesis (universe quality vs geography)  
**What:** Created control experiment to isolate effects before thesis submission  
**Approval:** Yash Sarang (user)  

**Next review:** After RUN 001 and RUN 002 complete (expected: same day, ~11:00 AM IST)

---

**END OF LOG (ACTIVE DOCUMENT - WILL BE UPDATED AS EXPERIMENTS RUN)**
