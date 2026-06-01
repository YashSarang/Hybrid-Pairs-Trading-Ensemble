# Control Experiment Status — 2026-06-01

## Current Status: **RUNNING**

**Job ID:** 8455  
**Node:** anandi (cn3_anandi partition)  
**Started:** Monday June 1, 2026 20:13:50 IST  
**Expected Completion:** 21:30-22:00 IST (~1-1.5 hours)  
**Status:** Running NSE Nifty 50 + Rolling + ZScore and OU experiments

---

## Experiment Overview

### Purpose
Isolate **universe quality effect** from **geographic effect** by running the same experiment on three universes:

1. **NSE Nifty 100** (Chapter 3 baseline): 35 tickers, different composition → Sharpe +0.052
2. **NSE Nifty 50** (Control, THIS RUN): 35 tickers, same as India multi-market → Sharpe **TBD**
3. **India Multi-Market** (Chapter 4): 35 tickers (Nifty 50) → Sharpe +0.840 (±0.631 variance)

### Key Question
**Does the high Sharpe (+0.840) in India multi-market come from:**
- **A) Universe quality** (Nifty 50 > Nifty 100 for pairs trading)?
- **B) Geographic diversification** (multi-market effect)?
- **C) ML non-determinism** (random variance, not a real effect)?

### Scenarios

| Scenario | NSE Nifty 50 Sharpe | Interpretation | Action |
|----------|---------------------|----------------|--------|
| **A** | +0.70 to +0.85 | Universe quality drives 90% of effect | "Stock selection matters" → Journal of Financial Markets |
| **B** | +0.05 to +0.20 | Small geographic effect | "Multi-market diversification" → Quantitative Finance |
| **C** | -0.30 to +0.05 | ML non-determinism, no real effect | "Reproducibility challenges" → Workshop only |

---

## Technical Details

### Configuration
- **Universe:** 35 tickers (same as India multi-market)
- **Data source:** Cached from `data/india/prices_2020-01-01_2025-05-01.parquet`
- **Signals:** ZScore (lookback=126), OU (lookback=126)
- **Window:** Rolling (4-fold walk-forward validation)
- **Selectors:** Ensemble of 8 (Correlation, Distance, Cointegration, Combined, ML, LSTM, Transformer, GNN)
- **Top pairs per fold:** 30
- **Transaction costs:** IndianCosts with 2.0 bps slippage per leg
- **Min hold:** 30 days

### Resources
- **CPUs:** 4
- **Memory:** 32GB
- **Time limit:** 4 hours
- **Partition:** cn3_anandi
- **QoS:** anandi
- **Account:** cminds_anandi

---

## Previous Attempts (Failed)

| Job ID | Status | Failure Reason | Fix |
|--------|--------|----------------|-----|
| 8438 | FAILED | yfinance API errors (all tickers failed) | Used cached data |
| 8450 | FAILED | `python: command not found` | Added conda activation |
| 8452 | FAILED | Wrong signal parameters (entry_threshold vs entry_z) | Fixed to entry_z/exit_z |
| 8453 | FAILED | Wrong selector parameters (top_n doesn't exist) | Removed parameters, use defaults |
| 8454 | FAILED | Wrong ensemble API (prices parameter) + IndianCosts API | Fixed to use selector.score_pairs + proper dataclass |
| **8455** | **RUNNING** | ✓ All APIs fixed | Expecting results in ~1 hour |

---

## Monitoring

### Check Status
```bash
ssh yash.sarang@kalpana.minds.iitb.ac.in 'squeue -u yash.sarang'
```

### View Log (live)
```bash
ssh yash.sarang@kalpana.minds.iitb.ac.in 'tail -f ~/Hybrid-Pairs-Trading-Ensemble/Implementation/experimental-ablation/logs/control_experiment_cached_8455.out'
```

### Check Results
```bash
ssh yash.sarang@kalpana.minds.iitb.ac.in 'ls -lh ~/Hybrid-Pairs-Trading-Ensemble/Implementation/experimental-ablation/results/nse_nifty50/*.json'
```

### Download Results (when complete)
```bash
cd /d/Code/Hybrid-Pairs-Trading-Ensemble/Implementation/experimental-ablation
mkdir -p results/nse_nifty50
scp yash.sarang@kalpana.minds.iitb.ac.in:~/Hybrid-Pairs-Trading-Ensemble/Implementation/experimental-ablation/results/nse_nifty50/*.json results/nse_nifty50/
```

---

## Next Steps (When Complete)

1. **Download results** from server to local machine
2. **Analyze Sharpe ratios:**
   - NSE Nifty 100 (baseline): +0.052
   - NSE Nifty 50 (control): **[PENDING]**
   - India Multi-Market: +0.840 (but high variance: -0.386 to +0.840)
3. **Determine scenario** (A/B/C) based on control results
4. **Create transparency report** documenting:
   - All 3 India ZScore runs: +0.398, -0.386, +0.840 → mean 0.284 ± 0.631
   - All 2 India OU runs: 0.000, +0.200
   - All US/UK/Brazil runs with variance
5. **Reframe thesis narrative:**
   - Update abstract with chosen finding
   - Rewrite Chapter 4 introduction and conclusions
   - Add confidence intervals to all figures
   - Expand discussion of ML reproducibility issues
6. **Update submission timeline:**
   - If Scenario A: Target JFM (July 15 or Aug-Sept)
   - If Scenario B: Target Quantitative Finance (rolling)
   - If Scenario C: Target NeurIPS ML Finance Workshop (Oct 20)

---

## Critical Context

### Pre-Salvage Experimental Runs (Reproducibility Crisis)

**India ZScore** (3 runs, May 29, 2026):
- Run 1 (070512): Sharpe +0.398, 289 trades
- Run 2 (083100): Sharpe -0.386, 279 trades
- Run 3 (104009): Sharpe +0.840, 123 trades
- **Mean:** 0.284, **Std:** 0.631
- **Thesis reports:** +0.840 (best of 3, no disclosure of variance)

**India OU** (2 runs):
- Run 1: Sharpe 0.000, 0 trades
- Run 2: Sharpe +0.200, 26 trades

**US Unknown Signal** (3 runs):
- Sharpe: 0.000, +0.116, +0.774

This variance indicates **severe ML non-determinism** that must be addressed before publication.

### Universe Comparison

**NSE Nifty 100** (experiments/config.py):
- 35 tickers: INDUSINDBK.NS, BAJAJ-AUTO.NS, HEROMOTOCO.NS, EICHERMOT.NS, BRITANNIA.NS, DRREDDY.NS, CIPLA.NS, DIVISLAB.NS, ACC.NS, SHREECEM.NS, etc.
- **Used in:** Chapter 3 baseline experiments
- **Result:** Sharpe +0.052 with Rolling Windows

**NSE Nifty 50** / India Multi-Market (configs/nse_nifty50.yaml):
- 35 tickers: RELIANCE.NS, TCS.NS, LT.NS, BHARTIARTL.NS, ASIANPAINT.NS, TITAN.NS, BAJFINANCE.NS, NTPC.NS, POWERGRID.NS, TATAMOTORS.NS, ADANIENT.NS, GRASIM.NS, etc.
- **Overlap with Nifty 100:** 25 tickers (71%)
- **Difference:** Nifty 50 has more mega-caps (LT, Bharti, Adani), Nifty 100 has better pharma/auto diversity
- **Used in:** Chapter 4 multi-market experiments + THIS CONTROL

### Files Modified/Created

**Local:**
- `run_control_experiment_cached.py` (10KB) - Fixed experiment script using cached data
- `slurm_control_experiment_cached.sh` (1.7KB) - SLURM submission script with conda activation
- `upload_and_submit.sh` (1.7KB) - Helper script for uploads
- `monitor_job.sh` (1.2KB) - Monitoring script
- `CONTROL_EXPERIMENT_STATUS.md` (this file)

**Server:**
- `/users/student/pg/pg24/yash.sarang/Hybrid-Pairs-Trading-Ensemble/Implementation/experimental-ablation/`
  - `run_control_experiment_cached.py` (uploaded)
  - `slurm_control_experiment_cached.sh` (uploaded)
  - `data/nse_nifty50/prices_2020-01-01_2025-05-01.parquet` (copied from India cache)
  - `logs/control_experiment_cached_8455.out` (actively writing)
  - `logs/control_experiment_cached_8455.err` (actively writing)
  - `results/nse_nifty50/` (will contain 2 JSON files when complete)

---

## Passwordless SSH Setup ✓

Successfully configured passwordless SSH to kalpana.minds.iitb.ac.in using key-based authentication.

**Test:**
```bash
ssh yash.sarang@kalpana.minds.iitb.ac.in 'echo "Success!"'
```

**Config:** `~/.ssh/config` contains:
```
Host kalpana
    HostName kalpana.minds.iitb.ac.in
    User yash.sarang
    IdentityFile ~/.ssh/id_rsa
```

---

**Last Updated:** 2026-06-01 20:15 IST  
**Next Check:** 2026-06-01 21:30 IST (when job should be complete)
