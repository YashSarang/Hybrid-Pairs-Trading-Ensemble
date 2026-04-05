# SLURM Job Scripts for Hybrid Pairs Trading Ensemble

This directory contains SLURM job submission scripts for running experiments on the CMInDS Kalpana cluster.

## Quick Start

### One-time Setup
```bash
cd /users/student/pg/pg24/yash.sarang/Hybrid-Pairs-Trading-Ensemble
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Submit All Experiments
```bash
cd jobs/
chmod +x *.sh
./submit_all.sh
```

This submits all three experiments (E4, E3, E1) which can run concurrently.

---

## Individual Job Scripts

### E4: Walk-Forward Validation (CRITICAL)
**File:** `e4_walk_forward.sh`
**Runtime:** ~8-10 hours
**Priority:** Highest (headline result)

Runs out-of-sample validation with rolling windows using the best statistical signal (ou_only).

```bash
sbatch e4_walk_forward.sh
```

### E3: Ablation Study
**File:** `e3_ablation.sh`
**Runtime:** ~6-8 hours
**Priority:** High

Runs two stages sequentially:
1. `stat_ml`: Classical 4 selectors + XGBoost
2. `full`: All 8 selectors (includes LSTM, Transformer, GNN)

```bash
sbatch e3_ablation.sh
```

### E1: Frequency Comparison
**File:** `e1_frequency.sh`
**Runtime:** ~4-6 hours
**Priority:** Medium

Compares trading performance at daily (1D) vs hourly (1H) rebalancing frequencies.

```bash
sbatch e1_frequency.sh
```

---

## Resource Allocation

All scripts use:
- **GPU:** 1× L40S (48GB VRAM)
- **CPU:** 4 cores (within 8-per-GPU limit)
- **Memory:** 16GB RAM
- **Account:** cminds_anandi
- **Partition:** cn3_l40s

---

## Monitoring Jobs

### Check job status
```bash
squeue -u $(whoami)
```

### View live output
```bash
tail -f logs/e4_walk_forward_<job_id>.out
tail -f logs/e3_ablation_<job_id>.out
tail -f logs/e1_frequency_<job_id>.out
```

### Check for errors
```bash
cat logs/e4_walk_forward_<job_id>.err
cat logs/e3_ablation_<job_id>.err
cat logs/e1_frequency_<job_id>.err
```

### Cancel a job
```bash
scancel <job_id>
```

---

## Results

All results are saved to `experiments/results/` with timestamps:
- `walk_forward_YYYYMMDD_HHMMSS.json`
- `ablation_YYYYMMDD_HHMMSS.json`
- `freq_comparison_YYYYMMDD_HHMMSS.json`

---

## Troubleshooting

### Job fails immediately
1. Check error log: `cat logs/e4_walk_forward_<job_id>.err`
2. Verify environment is set up correctly
3. Test CUDA: Run the CUDA test job from ServerGuidelines.md

### Job times out
- Increase `--time` in the script (e.g., `--time=14:00:00`)
- First run may be slower due to data loading

### GPU not being used
- Verify PyTorch CUDA installation (see ServerGuidelines.md FAQ)
- Check logs for device information

### "Account is mandatory" error
- Ensure `#SBATCH --account=cminds_anandi` is in the script
- This is required by cluster policy

### "Too many CPUs for requested GPUs" error
- Maximum 8 CPUs per GPU allowed
- Current scripts use 4 CPUs per 1 GPU (safe)

---

## Recommended Execution Strategy

Since the cluster allows 2 concurrent jobs per user:

1. **Session 1:** Submit E4 (most critical, longest runtime)
2. **Session 2:** Submit E3 (ablation study)
3. **Session 3:** Submit E1 (frequency comparison)

All three can run concurrently as they write to separate timestamped result files.

---

## Cluster Policies

For full details, see `ServerGuidelines.md`:
- Max 2 running jobs per user
- Max 4 total jobs per user (2 running + 2 pending)
- Max 24-hour runtime per job
- Max 2 GPUs per job
- Must specify account, time, memory, and CPU count
- Only sbatch supported (no interactive jobs)

---

## Support

For cluster issues, contact CMInDS Helpdesk or refer to `ServerGuidelines.md`.
