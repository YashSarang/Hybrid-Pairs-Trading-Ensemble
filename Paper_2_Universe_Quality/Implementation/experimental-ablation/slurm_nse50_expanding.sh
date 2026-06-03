#!/bin/bash
#SBATCH --job-name=nse50_expand
#SBATCH --output=logs/nse50_expand_%j.out
#SBATCH --error=logs/nse50_expand_%j.err
#SBATCH --partition=cn3_anandi
#SBATCH --qos=anandi
#SBATCH --account=cminds_anandi
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00

echo "================================================================"
echo "NSE NIFTY 50 EXPERIMENT 2: EXPANDING WINDOWS"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Started: $(date)"
echo "================================================================"
echo ""

cd /users/student/pg/pg24/yash.sarang/Hybrid-Pairs-Trading-Ensemble/Implementation/experimental-ablation

source ~/miniconda3/etc/profile.d/conda.sh
conda activate base
echo "Python: $(which python)"

export CUDA_VISIBLE_DEVICES=""
export TF_CPP_MIN_LOG_LEVEL=3
export TF_ENABLE_ONEDNN_OPTS=0
export PYTHONUNBUFFERED=1

echo ""
echo "Running ZScore signal (expanding windows)..."
python scripts/run_multi_market_wfv.py \
    --market nse_nifty50_expanding \
    --signal_model zscore \
    --n_folds 4 \
    --selectors correlation distance cointegration combined

echo ""
echo "Running OU signal (expanding windows)..."
python scripts/run_multi_market_wfv.py \
    --market nse_nifty50_expanding \
    --signal_model ou \
    --n_folds 4 \
    --selectors correlation distance cointegration combined

echo ""
echo "================================================================"
echo "COMPLETE — Finished: $(date)"
echo "================================================================"
ls -lh results/nse_nifty50_expanding/*.json 2>/dev/null || echo "Check results dir"
