# CONTROL EXPERIMENT STATUS

**Job ID:** 8438  
**Node:** anandi  
**Started:** Mon Jun 1, 2026 18:00:50 IST  
**Estimated Runtime:** 1-2 hours  
**Status:** 🔄 RUNNING

---

## Live Monitoring

To check progress, run:

```bash
# Check job status
ssh yash.sarang@kalpana.minds.iitb.ac.in 'squeue -u yash.sarang'

# View live log
ssh yash.sarang@kalpana.minds.iitb.ac.in 'tail -f ~/Hybrid-Pairs-Trading-Ensemble/Implementation/experimental-ablation/logs/control_experiment_8438.out'

# Check for results
ssh yash.sarang@kalpana.minds.iitb.ac.in 'ls -lh ~/Hybrid-Pairs-Trading-Ensemble/Implementation/experimental-ablation/results/nse_nifty50/'
```

---

## What's Running

**Experiment:** NSE Nifty 50 + Rolling Windows (ZScore + OU signals)

**Purpose:** Control experiment to isolate universe quality from geographic effects

**Steps:**
1. ✓ Fetch NSE Nifty 50 price data (2020-2025)
2. 🔄 Run 4-fold walk-forward validation with ZScore signal (~45 min)
3. ⏳ Run 4-fold walk-forward validation with OU signal (~45 min)
4. ⏳ Save results to `results/nse_nifty50/wfv_4folds_*.json`

---

## Expected Output

Two result files:
- `results/nse_nifty50/wfv_4folds_zscore_<timestamp>.json`
- `results/nse_nifty50/wfv_4folds_ou_<timestamp>.json`

Each will contain:
- `avg_net_sharpe`: Mean Sharpe ratio across 4 folds
- `std_net_sharpe`: Standard deviation
- Fold-by-fold details
- Total trade count

---

## Next Steps After Completion

1. **Download results:**
   ```bash
   scp yash.sarang@kalpana.minds.iitb.ac.in:~/Hybrid-Pairs-Trading-Ensemble/Implementation/experimental-ablation/results/nse_nifty50/*.json /d/Code/Hybrid-Pairs-Trading-Ensemble/Implementation/experimental-ablation/results/nse_nifty50/
   ```

2. **Analyze results:**
   Compare NSE Nifty 50 vs India Multi-Market vs NSE Nifty 100

3. **Choose scenario:**
   - **Scenario A** (NSE Nifty 50 ≈ +0.75): Universe quality dominates
   - **Scenario B** (NSE Nifty 50 ≈ +0.10): Small geographic effect
   - **Scenario C** (NSE Nifty 50 ≈ -0.30): ML non-determinism issues

4. **Reframe thesis** based on chosen scenario

---

## Estimated Completion Time

- **Best case:** 18:45 IST (45 min from start)
- **Expected:** 19:00-19:30 IST (1-1.5 hours)
- **Worst case:** 20:00 IST (2 hours if data fetch is slow)

---

**Current time:** 18:01 IST  
**Check back at:** 19:00 IST for results
