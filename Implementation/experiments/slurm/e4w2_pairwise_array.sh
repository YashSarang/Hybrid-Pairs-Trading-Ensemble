#!/bin/bash
# =============================================================================
# E4.W2 — Pairwise Ensemble Array Job
# 28 jobs, one per C(8,2) pairwise equal-weight combination
#
# Submit: sbatch slurm/e4w2_pairwise_array.sh
# Monitor: squeue -u $USER
#
# NOTE: Each task writes its own walk_forward_*.json to experiments/results/.
# Aggregate with: python experiments/ensemble_grid_search.py --no-triples --resume
# =============================================================================
#SBATCH --job-name=e4w2_pairs
#SBATCH --output=/users/student/pg/pg24/yash.sarang/Hybrid-Pairs-Trading-Ensemble/Implementation/experiments/slurm_logs/e4w2_%A_%a.log
#SBATCH --error=/users/student/pg/pg24/yash.sarang/Hybrid-Pairs-Trading-Ensemble/Implementation/experiments/slurm_logs/e4w2_%A_%a.err
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=cn3_anandi
#SBATCH --account=cminds_anandi
#SBATCH --qos=anandi
#SBATCH --array=1-28

# ---------------------------------------------------------------------------
# All C(8,2)=28 pairwise equal-weight combinations
# Format: JSON weight dicts. Each has exactly two selectors set to 0.5.
# ---------------------------------------------------------------------------
declare -A PAIR_WEIGHTS
PAIR_WEIGHTS[1]='{"Correlation":0.5,"Distance":0.5,"Cointegration":0.0,"Combined":0.0,"ML":0.0,"LSTM":0.0,"Transformer":0.0,"GNN":0.0}'
PAIR_WEIGHTS[2]='{"Correlation":0.5,"Distance":0.0,"Cointegration":0.5,"Combined":0.0,"ML":0.0,"LSTM":0.0,"Transformer":0.0,"GNN":0.0}'
PAIR_WEIGHTS[3]='{"Correlation":0.5,"Distance":0.0,"Cointegration":0.0,"Combined":0.5,"ML":0.0,"LSTM":0.0,"Transformer":0.0,"GNN":0.0}'
PAIR_WEIGHTS[4]='{"Correlation":0.5,"Distance":0.0,"Cointegration":0.0,"Combined":0.0,"ML":0.5,"LSTM":0.0,"Transformer":0.0,"GNN":0.0}'
PAIR_WEIGHTS[5]='{"Correlation":0.5,"Distance":0.0,"Cointegration":0.0,"Combined":0.0,"ML":0.0,"LSTM":0.5,"Transformer":0.0,"GNN":0.0}'
PAIR_WEIGHTS[6]='{"Correlation":0.5,"Distance":0.0,"Cointegration":0.0,"Combined":0.0,"ML":0.0,"LSTM":0.0,"Transformer":0.5,"GNN":0.0}'
PAIR_WEIGHTS[7]='{"Correlation":0.5,"Distance":0.0,"Cointegration":0.0,"Combined":0.0,"ML":0.0,"LSTM":0.0,"Transformer":0.0,"GNN":0.5}'
PAIR_WEIGHTS[8]='{"Correlation":0.0,"Distance":0.5,"Cointegration":0.5,"Combined":0.0,"ML":0.0,"LSTM":0.0,"Transformer":0.0,"GNN":0.0}'
PAIR_WEIGHTS[9]='{"Correlation":0.0,"Distance":0.5,"Cointegration":0.0,"Combined":0.5,"ML":0.0,"LSTM":0.0,"Transformer":0.0,"GNN":0.0}'
PAIR_WEIGHTS[10]='{"Correlation":0.0,"Distance":0.5,"Cointegration":0.0,"Combined":0.0,"ML":0.5,"LSTM":0.0,"Transformer":0.0,"GNN":0.0}'
PAIR_WEIGHTS[11]='{"Correlation":0.0,"Distance":0.5,"Cointegration":0.0,"Combined":0.0,"ML":0.0,"LSTM":0.5,"Transformer":0.0,"GNN":0.0}'
PAIR_WEIGHTS[12]='{"Correlation":0.0,"Distance":0.5,"Cointegration":0.0,"Combined":0.0,"ML":0.0,"LSTM":0.0,"Transformer":0.5,"GNN":0.0}'
PAIR_WEIGHTS[13]='{"Correlation":0.0,"Distance":0.5,"Cointegration":0.0,"Combined":0.0,"ML":0.0,"LSTM":0.0,"Transformer":0.0,"GNN":0.5}'
PAIR_WEIGHTS[14]='{"Correlation":0.0,"Distance":0.0,"Cointegration":0.5,"Combined":0.5,"ML":0.0,"LSTM":0.0,"Transformer":0.0,"GNN":0.0}'
PAIR_WEIGHTS[15]='{"Correlation":0.0,"Distance":0.0,"Cointegration":0.5,"Combined":0.0,"ML":0.5,"LSTM":0.0,"Transformer":0.0,"GNN":0.0}'
PAIR_WEIGHTS[16]='{"Correlation":0.0,"Distance":0.0,"Cointegration":0.5,"Combined":0.0,"ML":0.0,"LSTM":0.5,"Transformer":0.0,"GNN":0.0}'
PAIR_WEIGHTS[17]='{"Correlation":0.0,"Distance":0.0,"Cointegration":0.5,"Combined":0.0,"ML":0.0,"LSTM":0.0,"Transformer":0.5,"GNN":0.0}'
PAIR_WEIGHTS[18]='{"Correlation":0.0,"Distance":0.0,"Cointegration":0.5,"Combined":0.0,"ML":0.0,"LSTM":0.0,"Transformer":0.0,"GNN":0.5}'
PAIR_WEIGHTS[19]='{"Correlation":0.0,"Distance":0.0,"Cointegration":0.0,"Combined":0.5,"ML":0.5,"LSTM":0.0,"Transformer":0.0,"GNN":0.0}'
PAIR_WEIGHTS[20]='{"Correlation":0.0,"Distance":0.0,"Cointegration":0.0,"Combined":0.5,"ML":0.0,"LSTM":0.5,"Transformer":0.0,"GNN":0.0}'
PAIR_WEIGHTS[21]='{"Correlation":0.0,"Distance":0.0,"Cointegration":0.0,"Combined":0.5,"ML":0.0,"LSTM":0.0,"Transformer":0.5,"GNN":0.0}'
PAIR_WEIGHTS[22]='{"Correlation":0.0,"Distance":0.0,"Cointegration":0.0,"Combined":0.5,"ML":0.0,"LSTM":0.0,"Transformer":0.0,"GNN":0.5}'
PAIR_WEIGHTS[23]='{"Correlation":0.0,"Distance":0.0,"Cointegration":0.0,"Combined":0.0,"ML":0.5,"LSTM":0.5,"Transformer":0.0,"GNN":0.0}'
PAIR_WEIGHTS[24]='{"Correlation":0.0,"Distance":0.0,"Cointegration":0.0,"Combined":0.0,"ML":0.5,"LSTM":0.0,"Transformer":0.5,"GNN":0.0}'
PAIR_WEIGHTS[25]='{"Correlation":0.0,"Distance":0.0,"Cointegration":0.0,"Combined":0.0,"ML":0.5,"LSTM":0.0,"Transformer":0.0,"GNN":0.5}'
PAIR_WEIGHTS[26]='{"Correlation":0.0,"Distance":0.0,"Cointegration":0.0,"Combined":0.0,"ML":0.0,"LSTM":0.5,"Transformer":0.5,"GNN":0.0}'
PAIR_WEIGHTS[27]='{"Correlation":0.0,"Distance":0.0,"Cointegration":0.0,"Combined":0.0,"ML":0.0,"LSTM":0.5,"Transformer":0.0,"GNN":0.5}'
PAIR_WEIGHTS[28]='{"Correlation":0.0,"Distance":0.0,"Cointegration":0.0,"Combined":0.0,"ML":0.0,"LSTM":0.0,"Transformer":0.5,"GNN":0.5}'

WEIGHTS="${PAIR_WEIGHTS[$SLURM_ARRAY_TASK_ID]}"
echo "=== E4.W2 Array Task ${SLURM_ARRAY_TASK_ID}/28 — Weights: ${WEIGHTS} ==="

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
source /users/student/pg/pg24/yash.sarang/miniconda3/etc/profile.d/conda.sh
conda activate base
cd /users/student/pg/pg24/yash.sarang/Hybrid-Pairs-Trading-Ensemble

export PAPER1_DATA_PATH="/users/student/pg/pg24/yash.sarang/Hybrid-Pairs-Trading-Ensemble/Implementation/experiments/data/nse_nifty100/prices_2015-01-01_2024-12-31.parquet"
export CUDA_VISIBLE_DEVICES=""
export TF_DETERMINISTIC_OPS=1
export TF_CPP_MIN_LOG_LEVEL=2
export PYTHONHASHSEED=42

# Fetch data if not cached
python Implementation/experiments/fetch_paper1_data.py

# ---------------------------------------------------------------------------
# Run WFV
# ---------------------------------------------------------------------------
echo "[$(date)] Starting WFV for task ${SLURM_ARRAY_TASK_ID}"

python Implementation/experiments/walk_forward.py \
    --s1-weights "${WEIGHTS}" \
    --s2 ou_only \
    --top-k 10

EXIT_CODE=$?
echo "[$(date)] Finished task ${SLURM_ARRAY_TASK_ID} with exit code ${EXIT_CODE}"
exit ${EXIT_CODE}
