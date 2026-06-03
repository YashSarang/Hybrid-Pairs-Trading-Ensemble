#!/bin/bash
#SBATCH --job-name=e3_ablation
#SBATCH --output=/users/student/pg/pg24/yash.sarang/Hybrid-Pairs-Trading-Ensemble/Implementation/experiments/slurm_logs/e3_ablation_%j.log
#SBATCH --error=/users/student/pg/pg24/yash.sarang/Hybrid-Pairs-Trading-Ensemble/Implementation/experiments/slurm_logs/e3_ablation_%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --partition=cn3_anandi
#SBATCH --account=cminds_anandi
#SBATCH --qos=anandi

source /users/student/pg/pg24/yash.sarang/miniconda3/etc/profile.d/conda.sh
conda activate base
cd /users/student/pg/pg24/yash.sarang/Hybrid-Pairs-Trading-Ensemble

export CUDA_VISIBLE_DEVICES=""
export TF_DETERMINISTIC_OPS=1
export TF_CPP_MIN_LOG_LEVEL=2
export PYTHONHASHSEED=42

echo "[E3] Ablation stat_only (single selectors + stat ensemble)"
python Implementation/experiments/ablation.py \
    --mode stat_only --s2 ou_only

echo "[E3] Ablation stat_ml (adds XGBoost ML selector, CPU)"
python Implementation/experiments/ablation.py \
    --mode stat_ml --s2 ou_only

echo "[E3] Ablation full hybrid (CPU DL: LSTM+Transformer+GNN)"
python Implementation/experiments/ablation.py \
    --mode full --s2 ou_only

echo "[E3] DONE"
