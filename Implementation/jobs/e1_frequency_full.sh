#!/bin/bash
#SBATCH --job-name=e1_frequency_full
#SBATCH --account=cminds_anandi
#SBATCH --partition=cn3_l40s
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

echo "=========================================="
echo "E1: Frequency Comparison -- full mode (all 8 selectors)"
echo "Purpose: Paper completeness; stat_only run already done (2026-04-02)"
echo "Start time: $(date)"
echo "=========================================="

eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda activate pairs_trading

cd /users/student/pg/pg24/$(whoami)/Hybrid-Pairs-Trading-Ensemble/Implementation

python experiments/freq_comparison.py --mode full

echo "=========================================="
echo "End time: $(date)"
echo "=========================================="
