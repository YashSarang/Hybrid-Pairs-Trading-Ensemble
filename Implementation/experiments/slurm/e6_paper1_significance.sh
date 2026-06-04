#!/bin/bash
#SBATCH --job-name=e6_sig
#SBATCH --output=/users/student/pg/pg24/yash.sarang/Hybrid-Pairs-Trading-Ensemble/Implementation/experiments/slurm_logs/e6_sig_%j.log
#SBATCH --error=/users/student/pg/pg24/yash.sarang/Hybrid-Pairs-Trading-Ensemble/Implementation/experiments/slurm_logs/e6_sig_%j.err
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
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

echo "[E6-1] stat_only + ou_only -- 10k bootstrap"
python Implementation/experiments/significance_tests.py \
    --mode stat_only --s2 ou_only --n_boot 10000

echo "[E6-2] stat_ml + ou_only -- 10k bootstrap"
python Implementation/experiments/significance_tests.py \
    --mode stat_ml --s2 ou_only --n_boot 10000

echo "[E6-3] full + ou_only -- 10k bootstrap"
python Implementation/experiments/significance_tests.py \
    --mode full --s2 ou_only --n_boot 10000

echo "[E6] DONE"
