#!/bin/bash
#SBATCH --job-name=robustness_35t
#SBATCH --output=/users/student/pg/pg24/yash.sarang/Hybrid-Pairs-Trading-Ensemble/Implementation/experiments/slurm_logs/robustness_35t_%j.log
#SBATCH --error=/users/student/pg/pg24/yash.sarang/Hybrid-Pairs-Trading-Ensemble/Implementation/experiments/slurm_logs/robustness_35t_%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --partition=cn3_anandi
#SBATCH --account=cminds_anandi
#SBATCH --qos=anandi

# ── Robustness Suite: 35-Ticker Nifty50 Matched Universe ─────────────────────
# Runs the IDENTICAL E4 pipeline on the Paper 2 35-ticker universe.
# Purpose: confirm Paper 1 results are not universe-size artefacts.
# Runtime estimate: ~1-2 hrs on 4 CPUs (32T x 496 pairs per fold, 6 folds).
# ─────────────────────────────────────────────────────────────────────────────

source /users/student/pg/pg24/yash.sarang/miniconda3/etc/profile.d/conda.sh
conda activate base
cd /users/student/pg/pg24/yash.sarang/Hybrid-Pairs-Trading-Ensemble

# Parquet already exists from previous runs — skip re-fetch for speed.
# Uncomment if you need to refresh the data cache:
# python Implementation/experiments/fetch_paper1_data.py

# ── CPU-only deterministic ML ─────────────────────────────────────────────────
export PAPER1_DATA_PATH="/users/student/pg/pg24/yash.sarang/Hybrid-Pairs-Trading-Ensemble/Implementation/experiments/data/nse_nifty100/prices_2015-01-01_2024-12-31.parquet"
export CUDA_VISIBLE_DEVICES=""
export TF_DETERMINISTIC_OPS=1
export TF_CPP_MIN_LOG_LEVEL=2
export PYTHONHASHSEED=42

echo "[Robustness-35T] stat_only + ou_only (primary robustness run)"
python Implementation/experiments/walk_forward_35t.py \
    --mode stat_only --s2 ou_only --top-k 10

echo "[Robustness-35T] stat_only + no_ml (ZScore+OU+Kalman ensemble)"
python Implementation/experiments/walk_forward_35t.py \
    --mode stat_only --s2 no_ml --top-k 10

echo "[Robustness-35T] stat_ml + ou_only (include MLSelector in S1)"
python Implementation/experiments/walk_forward_35t.py \
    --mode stat_ml --s2 ou_only --top-k 10

echo "[Robustness-35T] DONE — check results/walk_forward_35t_*.json"
