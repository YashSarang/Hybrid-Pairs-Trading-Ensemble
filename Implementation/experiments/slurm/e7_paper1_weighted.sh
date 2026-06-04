#!/bin/bash
#SBATCH --job-name=e7_weighted
#SBATCH --output=/users/student/pg/pg24/yash.sarang/Hybrid-Pairs-Trading-Ensemble/Implementation/experiments/slurm_logs/e7_weighted_%j.log
#SBATCH --error=/users/student/pg/pg24/yash.sarang/Hybrid-Pairs-Trading-Ensemble/Implementation/experiments/slurm_logs/e7_weighted_%j.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=cn3_anandi
#SBATCH --account=cminds_anandi
#SBATCH --qos=anandi

source /users/student/pg/pg24/yash.sarang/miniconda3/etc/profile.d/conda.sh
conda activate base
cd /users/student/pg/pg24/yash.sarang/Hybrid-Pairs-Trading-Ensemble

python Implementation/experiments/fetch_paper1_data.py

export PAPER1_DATA_PATH="/users/student/pg/pg24/yash.sarang/Hybrid-Pairs-Trading-Ensemble/Implementation/experiments/data/nse_nifty100/prices_2015-01-01_2024-12-31.parquet"
export CUDA_VISIBLE_DEVICES=""
export TF_DETERMINISTIC_OPS=1
export TF_CPP_MIN_LOG_LEVEL=2
export PYTHONHASHSEED=42

echo "[E7-A] Weighted S1 (Corr=2.0, others=1.0) + OU-only S2 -- stat_ml"
python Implementation/experiments/walk_forward.py \
    --s1-weights '{"Correlation":2.0,"Distance":1.0,"Cointegration":1.0,"Combined":1.0,"ML":1.0}' \
    --s2 ou_only

echo "[E7-B] Weighted S1 (LSTM=3.0, Corr=2.0, others=1.0) + OU-only S2 -- full"
python Implementation/experiments/walk_forward.py \
    --s1-weights '{"Correlation":2.0,"Distance":1.0,"Cointegration":1.0,"Combined":1.0,"ML":1.0,"LSTM":3.0,"Transformer":1.0,"GNN":1.0}' \
    --s2 ou_only

echo "[E7-C] stat_only + OU-only S2 (replication check)"
python Implementation/experiments/walk_forward.py \
    --mode stat_only \
    --s2 ou_only

echo "[E7] DONE"
