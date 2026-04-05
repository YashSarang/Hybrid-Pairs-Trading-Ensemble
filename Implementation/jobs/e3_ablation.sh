#!/bin/bash
#SBATCH --job-name=e3_ablation
#SBATCH --account=cminds_anandi
#SBATCH --partition=cn3_l40s
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=10:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

echo "=========================================="
echo "E3: Ablation Study"
echo "Start time: $(date)"
echo "=========================================="

# Activate conda environment
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda activate pairs_trading

echo "Python: $(which python)"
echo "Conda env: $CONDA_DEFAULT_ENV"

# Run experiments
cd /users/student/pg/pg24/$(whoami)/Hybrid-Pairs-Trading-Ensemble/Implementation

echo "Stage 1: stat_ml"
python experiments/ablation.py --mode stat_ml

echo ""
echo "Stage 2: full"
python experiments/ablation.py --mode full

echo "=========================================="
echo "End time: $(date)"
echo "=========================================="
