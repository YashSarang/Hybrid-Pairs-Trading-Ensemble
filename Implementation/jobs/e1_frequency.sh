#!/bin/bash
#SBATCH --job-name=e1_frequency
#SBATCH --account=cminds_anandi
#SBATCH --partition=cn3_l40s
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

echo "=========================================="
echo "E1: Frequency Comparison"
echo "Start time: $(date)"
echo "=========================================="

# Activate conda environment
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda activate pairs_trading

echo "Python: $(which python)"
echo "Conda env: $CONDA_DEFAULT_ENV"

# Run experiment
cd /users/student/pg/pg24/$(whoami)/Hybrid-Pairs-Trading-Ensemble/Implementation
python experiments/freq_comparison.py --mode full --freqs 1D 1H

echo "=========================================="
echo "End time: $(date)"
echo "=========================================="
