#!/bin/bash
#SBATCH --job-name=e4a_stat
#SBATCH --output=/users/student/pg/pg24/yash.sarang/Hybrid-Pairs-Trading-Ensemble/Implementation/experiments/slurm_logs/e4a_stat_%j.log
#SBATCH --error=/users/student/pg/pg24/yash.sarang/Hybrid-Pairs-Trading-Ensemble/Implementation/experiments/slurm_logs/e4a_stat_%j.err
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
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

echo "[E4a] WFV stat_only + ou_only"
python Implementation/experiments/walk_forward.py \
    --mode stat_only --s2 ou_only --top-k 10
echo "[E4a] WFV stat_only + all signals"
python Implementation/experiments/walk_forward.py \
    --mode stat_only --s2 all --top-k 10
echo "[E4a] DONE"
