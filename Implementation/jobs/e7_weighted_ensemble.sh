#!/bin/bash
#SBATCH --job-name=e7_weighted_ensemble
#SBATCH --account=cminds_anandi
#SBATCH --partition=cn3_l40s
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

echo "=========================================="
echo "E7: Weighted Ensemble Walk-Forward"
echo "Start time: $(date)"
echo "Hypothesis: LSTM-heavy S1 + OU-only S2 beats equal-weight ensemble"
echo "=========================================="

eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda activate pairs_trading

cd /users/student/pg/pg24/$(whoami)/Hybrid-Pairs-Trading-Ensemble/Implementation

# ---------------------------------------------------------
# Config A: LSTM-heavy (3x), Correlation (2x), stat selectors (1x)
#           ML=0 (proven harmful), GNN=0 (poor OOS), Combined=0 (= Cointegration)
# ---------------------------------------------------------
echo ""
echo "--- Config A: LSTM=3, Correlation=2, Distance=1, Cointegration=1, Transformer=1 ---"
python experiments/walk_forward.py \
    --s1-weights '{"LSTM":3,"Correlation":2,"Distance":1,"Cointegration":1,"Combined":0,"ML":0,"Transformer":1,"GNN":0}' \
    --s2 ou_only

# ---------------------------------------------------------
# Config B: Stat + LSTM only (exclude GNN, ML, Combined)
#           Tightest, highest-quality selector set
# ---------------------------------------------------------
echo ""
echo "--- Config B: LSTM=1, Correlation=1, Distance=1, Cointegration=1 (stat+LSTM, no bad selectors) ---"
python experiments/walk_forward.py \
    --s1-weights '{"LSTM":1,"Correlation":1,"Distance":1,"Cointegration":1,"Combined":0,"ML":0,"Transformer":0,"GNN":0}' \
    --s2 ou_only

# ---------------------------------------------------------
# Config C: LSTM + Correlation only (most selective)
# ---------------------------------------------------------
echo ""
echo "--- Config C: LSTM=1, Correlation=1 only ---"
python experiments/walk_forward.py \
    --s1-weights '{"LSTM":1,"Correlation":1,"Distance":0,"Cointegration":0,"Combined":0,"ML":0,"Transformer":0,"GNN":0}' \
    --s2 ou_only

# ---------------------------------------------------------
# Config D: Full 8 selectors, but quality-weighted
#           (down-weight bad performers: ML=0, GNN=0.25, Combined=0.25)
# ---------------------------------------------------------
echo ""
echo "--- Config D: Full weighted (LSTM=3, Corr=2, Dist=1, Coint=1, Transformer=1, GNN=0.25, Combined=0.25, ML=0) ---"
python experiments/walk_forward.py \
    --s1-weights '{"LSTM":3,"Correlation":2,"Distance":1,"Cointegration":1,"Combined":0.25,"ML":0,"Transformer":1,"GNN":0.25}' \
    --s2 ou_only

echo "=========================================="
echo "E7 complete. End time: $(date)"
echo "=========================================="
