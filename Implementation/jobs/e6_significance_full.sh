#!/bin/bash
#SBATCH --job-name=e6_significance_full
#SBATCH --account=cminds_anandi
#SBATCH --partition=cn3_l40s
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

echo "=========================================="
echo "E6: Statistical Significance -- full-mode + E7 best result"
echo "Run AFTER e7_weighted_ensemble completes"
echo "Start time: $(date)"
echo "=========================================="

eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda activate pairs_trading

cd /users/student/pg/pg24/$(whoami)/Hybrid-Pairs-Trading-Ensemble/Implementation

# Run significance tests on full-mode WFV result (E4 full-mode)
echo "--- Significance on full-mode WFV result (E4 full) ---"
python experiments/significance_tests.py \
    --wfv walk_forward_20260406_011541.json \
    --mode full --s2 ou_only

# Run significance on E7 Config C (LSTM+Correlation only — best overall result, Net SR 0.451)
echo "--- Significance on E7 Config C: LSTM+Corr only ---"
python experiments/significance_tests.py \
    --wfv walk_forward_20260406_045231.json \
    --mode custom --s2 ou_only

echo "=========================================="
echo "End time: $(date)"
echo "=========================================="
