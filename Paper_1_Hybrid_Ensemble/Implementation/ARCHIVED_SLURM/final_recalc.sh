#!/bin/bash
#SBATCH --job-name=final_recalc
#SBATCH --partition=cn3_anandi
#SBATCH --qos=anandi
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/final_recalc_%j.out
#SBATCH --error=logs/final_recalc_%j.err

source ~/miniconda3/etc/profile.d/conda.sh
conda activate pairs_trading

echo "=========================================="
echo "Running Baseline (Stat-only)"
echo "=========================================="
python experiments/walk_forward.py --mode stat_only --s2 ou_only

echo "=========================================="
echo "Running Full-Equal Ensemble"
echo "=========================================="
python experiments/walk_forward.py --mode full --s2 ou_only

echo "=========================================="
echo "Running Config C (Weighted Ensemble)"
echo "=========================================="
python experiments/walk_forward.py --s1-weights '{"LSTM":1,"Correlation":1,"Distance":0,"Cointegration":0,"Combined":0,"ML":0,"Transformer":0,"GNN":0}' --s2 ou_only

echo "=========================================="
echo "Running RL (E8)"
echo "=========================================="
python experiments/walk_forward.py --s1-weights '{"LSTM":1,"Correlation":1,"Distance":0,"Cointegration":0,"Combined":0,"ML":0,"Transformer":0,"GNN":0}' --s2 rl_only --top-k 10

echo "Done all recalculations!"
