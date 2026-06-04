#!/bin/bash
#SBATCH --job-name=e4c_full
#SBATCH --output=/users/student/pg/pg24/yash.sarang/Hybrid-Pairs-Trading-Ensemble/Implementation/experiments/slurm_logs/e4c_full_%j.log
#SBATCH --error=/users/student/pg/pg24/yash.sarang/Hybrid-Pairs-Trading-Ensemble/Implementation/experiments/slurm_logs/e4c_full_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=cn3_anandi
#SBATCH --account=cminds_anandi
#SBATCH --qos=anandi

source /users/student/pg/pg24/yash.sarang/miniconda3/etc/profile.d/conda.sh
conda activate base
cd /users/student/pg/pg24/yash.sarang/Hybrid-Pairs-Trading-Ensemble

# Ensure cached data exists before running experiments
python Implementation/experiments/fetch_paper1_data.py


# ── CPU-only deterministic ML ──────────────────────────────────────────────
export PAPER1_DATA_PATH="/users/student/pg/pg24/yash.sarang/Hybrid-Pairs-Trading-Ensemble/Implementation/experiments/data/nse_nifty100/prices_2015-01-01_2024-12-31.parquet"
export CUDA_VISIBLE_DEVICES=""
export TF_DETERMINISTIC_OPS=1
export TF_CPP_MIN_LOG_LEVEL=2
export PYTHONHASHSEED=42

echo "[E4c] WFV full hybrid (CPU DL) + ou_only — run 1/3"
python Implementation/experiments/walk_forward.py \
    --mode full --s2 ou_only --top-k 10
echo "[E4c] WFV full hybrid (CPU DL) + ou_only — run 2/3"
python Implementation/experiments/walk_forward.py \
    --mode full --s2 ou_only --top-k 10
echo "[E4c] WFV full hybrid (CPU DL) + ou_only — run 3/3"
python Implementation/experiments/walk_forward.py \
    --mode full --s2 ou_only --top-k 10
echo "[E4c] DONE"
