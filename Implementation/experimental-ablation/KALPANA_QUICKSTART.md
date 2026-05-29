# CMInDS Kalpana Cluster - Quick Reference Guide (Updated May 2026)

## 🔑 Access
```bash
ssh <username>@kalpana.minds.iitb.ac.in
```
- **Home:** `/users/student/pg/pg24/yash.sarang`
- **Conda:** `source ~/miniconda3/bin/activate`

---

## 🖥️ Current Hardware (RTX 3060 Configuration)

| Partition | Node | GPU | Time Limit | Status |
|-----------|------|-----|------------|--------|
| `cn3_anandi` | anandi | 2× RTX 3060 (12GB) | 24h | ✅ Available |
| `cn3_anandi_interactive` | anandi | 2× RTX 3060 (12GB) | 1h | ❌ Disabled |

**Note:** Documentation mentions future L40S upgrade (`cn3_l40s` partition with 8× L40S). Currently **NOT available** — use `cn3_anandi` instead.

---

## ⚠️ **MANDATORY** Job Requirements

### Must Include (job will be rejected otherwise):
```bash
#SBATCH --account=cminds_anandi       # Project account (REQUIRED)
#SBATCH --partition=cn3_anandi        # Use anandi partition
#SBATCH --qos=anandi                  # Quality of Service
#SBATCH --gres=gpu:1                  # GPU request
#SBATCH --mem=16G                     # Memory (REQUIRED)
#SBATCH --time=12:00:00               # Max 24h (REQUIRED)
```

### Auto-allocate (don't specify):
- `--cpus-per-task` — Let SLURM decide based on GPU allocation
- `--ntasks` — Defaults to 1 (correct for single-node jobs)

---

## 📊 Resource Limits (QoS Policy)

| Limit | Value | Notes |
|-------|-------|-------|
| **Max running jobs per user** | 2 | Additional jobs queue automatically |
| **Max total jobs per user** | 4 | 2 running + 2 queued |
| **Max GPUs per job** | 2 | Use `--gres=gpu:1` or `gpu:2` |
| **Max runtime** | 24 hours | Jobs auto-killed after 24h |
| **Max memory per job** | 64G | Per-node limit |

---

## 📝 Standard Job Template

Save as `job.sbatch`:

```bash
#!/bin/bash
#SBATCH --job-name=my_experiment
#SBATCH --account=cminds_anandi
#SBATCH --partition=cn3_anandi
#SBATCH --qos=anandi
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Activate environment
source ~/miniconda3/bin/activate

# Navigate to project
cd ~/my_project

# Run script
python3 train.py
```

Submit:
```bash
mkdir -p logs  # Create log directory first!
sbatch job.sbatch
```

---

## 🔍 Monitoring Commands

```bash
# Check job status
squeue -u $USER

# Check detailed job info
scontrol show job <JOB_ID>

# Tail live output
tail -f logs/my_experiment_<JOB_ID>.out

# Cancel job
scancel <JOB_ID>

# Cancel all your jobs
scancel -u $USER

# Check node status
sinfo -N -l
```

---

## 🚨 Common Errors & Fixes

| Error | Cause | Fix |
|-------|-------|-----|
| `--account is mandatory` | Missing account flag | Add `#SBATCH --account=cminds_anandi` |
| `Invalid partition` | Wrong partition name | Use `cn3_anandi` (not `cn3_l40s`) |
| `Invalid qos` | Missing QoS | Add `#SBATCH --qos=anandi` |
| `Interactive jobs disabled` | Used `srun`/`salloc` | Use `sbatch` only |
| `QOSMaxJobsPerUserLimit` | >2 running jobs | Wait for jobs to complete |
| `Job violates QOS` | >2 GPUs requested | Use `--gres=gpu:1` or `gpu:2` max |

---

## 💾 Storage Policy

### Home Directory
- **Quota:** 200 GB per user
- **Path:** `~/` (backed up, persistent)
- **Use for:** Code, environments, small datasets

### Janaki NAS
- **Shared:** `/janaki/common` (read-only for most users)
- **Private:** `/janaki/backup/users/student/pg/pg24/yash.sarang` (500 GB quota)
- **Use for:** Large datasets, model checkpoints, backups

**⚠️ Check existing datasets before downloading:**
```bash
ls /janaki/common/Datasets
```

---

## 🐍 Environment Best Practices

### Option 1: Conda (recommended for ML/DL)
```bash
# Create environment
conda create -n myenv python=3.12
conda activate myenv

# Install PyTorch with CUDA
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Verify GPU access
python -c "import torch; print(torch.cuda.is_available())"
```

### Option 2: venv (lightweight)
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Always activate environment in SLURM script!**

---

## ✅ Pre-Flight Checklist

Before submitting jobs:

- [ ] Created `logs/` directory
- [ ] Specified `--account=cminds_anandi`
- [ ] Used `--partition=cn3_anandi` (NOT `cn3_l40s`)
- [ ] Added `--qos=anandi`
- [ ] Requested memory (`--mem=16G`)
- [ ] Set time limit (`--time=HH:MM:SS`, max 24h)
- [ ] Conda environment activates in script
- [ ] Tested code locally or in small test job first

---

## 🆘 Support

- **CMInDS Helpdesk:** [Submit ticket](https://cminds.iitb.ac.in/helpdesk)
- **Documentation (outdated):** `/home/yash.sarang/ServerGuidelines.md`
- **This guide:** Reflects **actual** cluster state as of May 2026

---

## 📌 Quick Reference Card

```bash
# Login
ssh yash.sarang@kalpana.minds.iitb.ac.in

# Activate conda
source ~/miniconda3/bin/activate

# Check queue
squeue -u $USER

# Submit job
sbatch job.sbatch

# Cancel job
scancel <JOB_ID>

# Watch logs
tail -f logs/*.out
```

---

**Last updated:** May 29, 2026  
**Current config:** 2× RTX 3060, anandi partition only  
**Future upgrade:** 8× L40S (documented but NOT yet deployed)
