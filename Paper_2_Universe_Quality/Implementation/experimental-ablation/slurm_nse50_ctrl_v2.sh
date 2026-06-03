#!/bin/bash
#SBATCH --job-name=nse50_ctrl_v2
#SBATCH --output=logs/nse50_ctrl_v2_%j.out
#SBATCH --error=logs/nse50_ctrl_v2_%j.err
#SBATCH --partition=cn3_anandi
#SBATCH --qos=anandi
#SBATCH --account=cminds_anandi
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00

echo "================================================================"
echo "NSE NIFTY 50 CONTROL EXPERIMENT v2 (Production Script)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Started: $(date)"
echo "================================================================"
echo ""

cd /users/student/pg/pg24/yash.sarang/Hybrid-Pairs-Trading-Ensemble/Implementation/experimental-ablation

# Activate conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate base
echo "Python: $(which python)"
echo ""

# Disable GPU/CUDA to prevent TF hangs
export CUDA_VISIBLE_DEVICES=""
export TF_CPP_MIN_LOG_LEVEL=3
export TF_ENABLE_ONEDNN_OPTS=0
export PYTHONUNBUFFERED=1

echo "Running ZScore signal..."
python scripts/run_multi_market_wfv.py \
    --market nse_nifty50 \
    --signal_model zscore \
    --n_folds 4 \
    --selectors correlation distance cointegration combined
echo ""

echo "Running OU signal..."
python scripts/run_multi_market_wfv.py \
    --market nse_nifty50 \
    --signal_model ou \
    --n_folds 4 \
    --selectors correlation distance cointegration combined
echo ""

echo "================================================================"
echo "EXPERIMENTS COMPLETE"
echo "Finished: $(date)"
echo "================================================================"
echo ""
echo "Results:"
ls -lh results/nse_nifty50/*.json 2>/dev/null | tail -10 || echo "No results found"
