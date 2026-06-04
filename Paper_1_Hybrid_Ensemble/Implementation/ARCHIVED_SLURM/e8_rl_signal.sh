#!/bin/bash
#SBATCH --job-name=e8_rl_signal
#SBATCH --account=cminds_anandi
#SBATCH --partition=cn3_anandi
#SBATCH --qos=anandi
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

echo "=========================================="
echo "E8: RL Signal Model (PPO)"
echo "Start time: $(date)"
echo "Config: LSTM+Correlation S1 + RL S2"
echo "=========================================="

eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda activate pairs_trading

cd /users/student/pg/pg24/$(whoami)/Hybrid-Pairs-Trading-Ensemble/Implementation

# Using the breakthrough S1 weights from E7 Config C
python experiments/walk_forward.py \
    --s1-weights '{"LSTM":1,"Correlation":1,"Distance":0,"Cointegration":0,"Combined":0,"ML":0,"Transformer":0,"GNN":0}' \
    --s2 rl_only \
    --top-k 10

echo "=========================================="
echo "E8 complete. End time: $(date)"
echo "=========================================="
