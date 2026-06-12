#!/bin/bash
# =============================================================================
# E4.S — Standalone Benchmark Array Job
# 8 jobs, one per selector (Corr, Dist, Coint, Comb, ML, LSTM, Trans, GNN)
#
# Submit: sbatch slurm/e4s_standalone_array.sh
# Monitor: squeue -u $USER
# =============================================================================
#SBATCH --job-name=e4s_standalone
#SBATCH --output=/users/student/pg/pg24/yash.sarang/Hybrid-Pairs-Trading-Ensemble/Implementation/experiments/slurm_logs/e4s_%A_%a.log
#SBATCH --error=/users/student/pg/pg24/yash.sarang/Hybrid-Pairs-Trading-Ensemble/Implementation/experiments/slurm_logs/e4s_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=cn3_anandi
#SBATCH --account=cminds_anandi
#SBATCH --qos=anandi
#SBATCH --array=1-8

# ---------------------------------------------------------------------------
# Array index → selector preset mapping
# ---------------------------------------------------------------------------
PRESETS=(
    ""              # placeholder so array is 1-indexed
    "corr_only"     # 1 — E4.S1
    "dist_only"     # 2 — E4.S2
    "coint_only"    # 3 — E4.S3
    "comb_only"     # 4 — E4.S4
    "ml_only"       # 5 — E4.S5
    "lstm_only"     # 6 — E4.S6
    "trans_only"    # 7 — E4.S7
    "gnn_only"      # 8 — E4.S8
)

PRESET="${PRESETS[$SLURM_ARRAY_TASK_ID]}"
echo "=== E4.S Array Task ${SLURM_ARRAY_TASK_ID} — Preset: ${PRESET} ==="

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
# Run WFV for this selector preset
# ---------------------------------------------------------------------------
echo "[$(date)] Starting WFV for preset=${PRESET}"

python Implementation/experiments/walk_forward.py \
    --s1-preset "${PRESET}" \
    --s2 ou_only \
    --top-k 10

EXIT_CODE=$?
echo "[$(date)] Finished preset=${PRESET} with exit code ${EXIT_CODE}"
exit ${EXIT_CODE}
