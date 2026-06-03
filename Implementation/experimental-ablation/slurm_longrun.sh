#!/bin/bash
#SBATCH --job-name=nse_longrun_16fold
#SBATCH --output=/users/student/pg/pg24/yash.sarang/Hybrid-Pairs-Trading-Ensemble/Implementation/experimental-ablation/logs/longrun_%j.log
#SBATCH --error=/users/student/pg/pg24/yash.sarang/Hybrid-Pairs-Trading-Ensemble/Implementation/experimental-ablation/logs/longrun_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=cn3_anandi
#SBATCH --account=cminds_anandi
#SBATCH --qos=anandi

source /users/student/pg/pg24/yash.sarang/miniconda3/etc/profile.d/conda.sh
conda activate base

cd /users/student/pg/pg24/yash.sarang/Hybrid-Pairs-Trading-Ensemble
python Implementation/experimental-ablation/scripts/run_multi_market_wfv.py     --market nse_nifty50_longrun     --signal_model zscore     --n_folds 16     --selectors correlation distance cointegration combined
