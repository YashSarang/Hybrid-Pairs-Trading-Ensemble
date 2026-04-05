# Execution Plan: Hybrid Pairs Trading Ensemble

This document outlines the exact experiments to run on the CMInDS Kalpana cluster.
All commands should be executed via SLURM job scripts (see `jobs/` directory).

## Setup (One-time)

Run this once on the cluster to prepare your environment:

```bash
cd /users/student/pg/pg24/yash.sarang/Hybrid-Pairs-Trading-Ensemble
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Note:** Store the project in your home directory. Virtual environment will be created locally.

---

## Experiments Overview

### E4 — Walk-Forward Validation (CRITICAL - Run First)
**Purpose:** Out-of-sample validation with rolling windows
**Estimated Runtime:** 8-12 hours
**Priority:** Highest (headline result)

Two variants:
- `ou_only`: Uses best signal from statistical analysis (faster, ~8h)
- `all`: Full 4-model ensemble (slower, ~12h)

**Recommended:** Run `ou_only` first for quick results, then `all` for comparison.

### E3 — Ablation Study
**Purpose:** Measure contribution of each selector component
**Estimated Runtime:** 6-8 hours per stage
**Priority:** High

Two stages:
- `stat_ml`: Classical 4 selectors + XGBoost (baseline)
- `full`: All 8 selectors including deep learning (LSTM, Transformer, GNN)

### E1 — Frequency Comparison
**Purpose:** Compare daily vs hourly trading frequency
**Estimated Runtime:** 4-6 hours
**Priority:** Medium

Compares performance across different rebalancing frequencies.

---

## Execution Strategy

### Recommended Parallelism
Since cluster allows 2 concurrent jobs per user:

**Session 1 (Most Critical):** E4 with `ou_only` signal
**Session 2:** E3 ablation studies (stat_ml → full sequentially)
**Session 3:** E1 frequency comparison

All three can run concurrently as they write to separate timestamped result files.

### Resource Allocation
- **E4:** 1 GPU, 4 CPUs, 16GB RAM, 12h walltime
- **E3:** 1 GPU, 4 CPUs, 16GB RAM, 10h walltime  
- **E1:** 1 GPU, 4 CPUs, 16GB RAM, 8h walltime

---

## Job Submission

Use the provided SLURM scripts in the `jobs/` directory:

```bash
# Submit all three experiments
sbatch jobs/e4_walk_forward.sh
sbatch jobs/e3_ablation.sh
sbatch jobs/e1_frequency.sh

# Monitor progress
squeue -u <your_username>

# View logs
tail -f logs/e4_walk_forward_*.out
tail -f logs/e3_ablation_*.out
tail -f logs/e1_frequency_*.out
```

---

## Results

All results are saved to `experiments/results/` with timestamps:
- `walk_forward_YYYYMMDD_HHMMSS.json`
- `ablation_YYYYMMDD_HHMMSS.json`
- `freq_comparison_YYYYMMDD_HHMMSS.json`

---

## Troubleshooting

**Job fails immediately?**
- Check logs: `cat logs/e4_walk_forward_<job_id>.err`
- Verify environment: Run CUDA test job (see ServerGuidelines.md)
- Ensure requirements.txt is installed

**Job times out?**
- Increase `--time` in the SLURM script
- Check if data loading is slow (first run may be slower)

**GPU not being used?**
- Verify PyTorch CUDA installation (see ServerGuidelines.md FAQ)
- Check logs for device information
