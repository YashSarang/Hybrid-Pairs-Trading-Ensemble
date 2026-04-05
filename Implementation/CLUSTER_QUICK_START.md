# Quick Start: Running Experiments on Kalpana Cluster

## Prerequisites
- SSH access to `kalpana.minds.iitb.ac.in`
- LDAP credentials
- VPN if accessing from outside IITB network

---

## Step 1: Initial Setup (One-time)

```bash
# SSH into cluster
ssh <ldapusername>@kalpana.minds.iitb.ac.in

# Navigate to project
cd /users/student/pg/pg24/yash.sarang/Hybrid-Pairs-Trading-Ensemble

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Verify CUDA (optional but recommended)
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

---

## Step 2: Submit Experiments

### Option A: Submit All at Once (Recommended)
```bash
cd jobs/
./submit_all.sh
```

This submits E4, E3, and E1 which run concurrently.

### Option B: Submit Individually
```bash
# E4: Walk-Forward Validation (most important)
sbatch jobs/e4_walk_forward.sh

# E3: Ablation Study
sbatch jobs/e3_ablation.sh

# E1: Frequency Comparison
sbatch jobs/e1_frequency.sh
```

---

## Step 3: Monitor Progress

```bash
# Check job status
squeue -u $(whoami)

# View live output (replace <job_id> with actual ID)
tail -f logs/e4_walk_forward_<job_id>.out

# Check for errors
cat logs/e4_walk_forward_<job_id>.err
```

---

## Step 4: Retrieve Results

Results are saved to `experiments/results/` with timestamps:

```bash
# List all results
ls -lh experiments/results/

# View latest result
cat experiments/results/walk_forward_*.json | tail -1
```

---

## Key Information

| Experiment | Runtime | Priority | Command |
|-----------|---------|----------|---------|
| E4 (Walk-Forward) | 8-10h | Highest | `sbatch jobs/e4_walk_forward.sh` |
| E3 (Ablation) | 6-8h | High | `sbatch jobs/e3_ablation.sh` |
| E1 (Frequency) | 4-6h | Medium | `sbatch jobs/e1_frequency.sh` |

---

## Cluster Limits

- **Max 2 running jobs** per user
- **Max 24-hour runtime** per job
- **Max 2 GPUs** per job
- **200GB home directory quota**

---

## Troubleshooting

**Job fails immediately?**
```bash
cat logs/e4_walk_forward_<job_id>.err
```

**GPU not detected?**
- Verify PyTorch CUDA: `python -c "import torch; print(torch.cuda.is_available())"`
- See ServerGuidelines.md FAQ section

**Job times out?**
- Increase `--time` in the script (e.g., `--time=14:00:00`)
- Resubmit with longer walltime

**Out of disk space?**
- Check quota: `quota -u $(whoami)`
- Clean up old results: `rm experiments/results/old_*.json`

---

## Full Documentation

- **Cluster Policies:** See `ServerGuidelines.md`
- **Experiment Details:** See `Currently_Doing.md`
- **Job Scripts:** See `jobs/README.md`

---

## Support

For cluster issues: CMInDS Helpdesk
For code issues: Check logs and error messages
