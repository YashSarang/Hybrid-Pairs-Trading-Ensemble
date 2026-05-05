#!/bin/bash
#SBATCH --job-name=e3_transformer_rerun
#SBATCH --account=cminds_anandi
#SBATCH --partition=cn3_anandi
#SBATCH --qos=anandi
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

echo "=========================================="
echo "E3: Transformer_only ablation re-run"
echo "Fix: Lambda+GPU crash replaced with _PositionalEncodingLayer"
echo "Start time: $(date)"
echo "=========================================="

eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda activate pairs_trading

cd /users/student/pg/pg24/$(whoami)/Hybrid-Pairs-Trading-Ensemble/Implementation

# Run Stage 1 ablation, full mode, Stage 1 only
# This re-runs ALL Stage 1 configs (including Transformer_only which was missing)
# Stage 2 is skipped since those results are already valid
python experiments/ablation.py --mode full --stage 1

echo "=========================================="
echo "End time: $(date)"
echo "=========================================="
