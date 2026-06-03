#!/bin/bash
#SBATCH --job-name=nse50_8fold
#SBATCH --partition=cn3_anandi
#SBATCH --qos=anandi
#SBATCH --account=cminds_anandi
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/users/student/pg/pg24/yash.sarang/Hybrid-Pairs-Trading-Ensemble/Implementation/experimental-ablation/logs/nse50_8fold_%j.out
#SBATCH --error=/users/student/pg/pg24/yash.sarang/Hybrid-Pairs-Trading-Ensemble/Implementation/experimental-ablation/logs/nse50_8fold_%j.err

source ~/miniconda3/etc/profile.d/conda.sh
conda activate base

cd /users/student/pg/pg24/yash.sarang/Hybrid-Pairs-Trading-Ensemble/Implementation/experimental-ablation

echo '================================================================'
echo 'NSE NIFTY 50 8-FOLD EXTENSION EXPERIMENT'
echo 'Job ID: '
echo 'Node: '
echo 'Started: 'Wed Jun  3 11:54:31 IST 2026
echo '================================================================'

python scripts/run_multi_market_wfv.py   --market nse_nifty50_8fold   --selectors correlation distance cointegration combined   --signal_model zscore   --n_folds 8

echo '================================================================'
echo 'DONE: 'Wed Jun  3 11:54:31 IST 2026
echo '================================================================'
