#!/bin/bash
#SBATCH --job-name=e4b_statml
#SBATCH --output=/users/student/pg/pg24/yash.sarang/Hybrid-Pairs-Trading-Ensemble/Implementation/experiments/slurm_logs/e4b_statml_%j.log
#SBATCH --error=/users/student/pg/pg24/yash.sarang/Hybrid-Pairs-Trading-Ensemble/Implementation/experiments/slurm_logs/e4b_statml_%j.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=6
#SBATCH --mem=12G
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

echo "[E4b] WFV stat_ml + ou_only — run 1/3"
python Implementation/experiments/walk_forward.py \
    --mode stat_ml --s2 ou_only --top-k 10
echo "[E4b] WFV stat_ml + ou_only — run 2/3"
python Implementation/experiments/walk_forward.py \
    --mode stat_ml --s2 ou_only --top-k 10
echo "[E4b] WFV stat_ml + ou_only — run 3/3"
python Implementation/experiments/walk_forward.py \
    --mode stat_ml --s2 ou_only --top-k 10
echo "[E4b] DONE"
