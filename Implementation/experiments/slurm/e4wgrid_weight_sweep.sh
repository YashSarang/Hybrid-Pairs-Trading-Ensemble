#!/bin/bash
# =============================================================================
# E4.W-Grid — Continuous Weight Grid Search for Top-2 Selectors
# 66 jobs — grid over simplex of 3 selectors with step=0.1 (no step=0.2 etc.)
# OR 11 jobs for 2-selector sweep with step=0.1
#
# Default (2-selector): sweeps alpha=Correlation weight, LSTM = 1-alpha,
# for alpha in {0.0, 0.1, 0.2, ..., 1.0} → 11 jobs (--array=1-11)
#
# 3-selector version: uncomment #SBATCH --array=1-66 below and the 3-sel block.
#
# Submit (2-selector sweep):
#   sbatch slurm/e4wgrid_weight_sweep.sh
#
# The two selectors and step size can be overridden via env vars:
#   SEL_A=Correlation SEL_B=LSTM sbatch slurm/e4wgrid_weight_sweep.sh
# =============================================================================
#SBATCH --job-name=e4wgrid
#SBATCH --output=/users/student/pg/pg24/yash.sarang/Hybrid-Pairs-Trading-Ensemble/Implementation/experiments/slurm_logs/e4wgrid_%A_%a.log
#SBATCH --error=/users/student/pg/pg24/yash.sarang/Hybrid-Pairs-Trading-Ensemble/Implementation/experiments/slurm_logs/e4wgrid_%A_%a.err
#SBATCH --time=05:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=10G
#SBATCH --partition=cn3_anandi
#SBATCH --account=cminds_anandi
#SBATCH --qos=anandi
#SBATCH --array=1-11

# Selectors to sweep (override via env or edit here after E4.S results)
SEL_A="${SEL_A:-Correlation}"
SEL_B="${SEL_B:-LSTM}"

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

# ---------------------------------------------------------------------------
# Resolve alpha from array task ID (task 1 = alpha=0.0, task 11 = alpha=1.0)
# ---------------------------------------------------------------------------
WEIGHTS=$(python - <<PYEOF
import json, sys

task_id = int("$SLURM_ARRAY_TASK_ID")
n_steps = 11
alpha = (task_id - 1) / (n_steps - 1)   # 0.0 to 1.0 inclusive

sel_a = "$SEL_A"
sel_b = "$SEL_B"

ALL = ["Correlation", "Distance", "Cointegration", "Combined",
       "ML", "LSTM", "Transformer", "GNN"]

weights = {k: 0.0 for k in ALL}
weights[sel_a] = round(alpha, 4)
weights[sel_b] = round(1.0 - alpha, 4)

print(json.dumps(weights))
print(f"# alpha({sel_a})={alpha:.2f}  alpha({sel_b})={1-alpha:.2f}", file=sys.stderr)
PYEOF
)

echo "=== E4.W-Grid Task ${SLURM_ARRAY_TASK_ID}/11 | ${SEL_A} vs ${SEL_B} ==="
echo "    Weights: ${WEIGHTS}"

# ---------------------------------------------------------------------------
# Fetch data if not cached
# ---------------------------------------------------------------------------
python Implementation/experiments/fetch_paper1_data.py

# ---------------------------------------------------------------------------
# Run WFV
# ---------------------------------------------------------------------------
echo "[$(date)] Starting"

python Implementation/experiments/walk_forward.py \
    --s1-weights "${WEIGHTS}" \
    --s2 ou_only \
    --top-k 10

EXIT_CODE=$?
echo "[$(date)] Finished with exit code ${EXIT_CODE}"
exit ${EXIT_CODE}
