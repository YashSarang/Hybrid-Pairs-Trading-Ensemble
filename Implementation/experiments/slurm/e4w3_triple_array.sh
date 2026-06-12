#!/bin/bash
# =============================================================================
# E4.W3 — Triple Ensemble Array Job
# 56 jobs, one per C(8,3) triple equal-weight combination
#
# Submit: sbatch slurm/e4w3_triple_array.sh
# Submit only positive-standalone subset (after reviewing E4.S results):
#   sbatch --array=<id1,id2,...> slurm/e4w3_triple_array.sh
#
# Mapping: Triple index → weight JSON is generated at runtime by Python
# (avoids hardcoding 56 entries). The mapping is deterministic based on
# itertools.combinations order over sorted(ALL_SELECTORS).
# =============================================================================
#SBATCH --job-name=e4w3_triples
#SBATCH --output=/users/student/pg/pg24/yash.sarang/Hybrid-Pairs-Trading-Ensemble/Implementation/experiments/slurm_logs/e4w3_%A_%a.log
#SBATCH --error=/users/student/pg/pg24/yash.sarang/Hybrid-Pairs-Trading-Ensemble/Implementation/experiments/slurm_logs/e4w3_%A_%a.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --partition=cn3_anandi
#SBATCH --account=cminds_anandi
#SBATCH --qos=anandi
#SBATCH --array=1-56

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
# Resolve weights for this array task via Python (deterministic ordering)
# ---------------------------------------------------------------------------
WEIGHTS=$(python - <<'PYEOF'
import sys, json
from itertools import combinations

ALL_SELECTORS = [
    "Correlation", "Distance", "Cointegration", "Combined",
    "ML", "LSTM", "Transformer", "GNN"
]
task_id = int("$SLURM_ARRAY_TASK_ID")

combos = list(combinations(ALL_SELECTORS, 3))   # 56 combos
if task_id < 1 or task_id > len(combos):
    print("{}", file=sys.stderr)
    sys.exit(1)

sel = combos[task_id - 1]   # 1-indexed
w = round(1.0 / len(sel), 8)
weights = {k: 0.0 for k in ALL_SELECTORS}
for s in sel:
    weights[s] = w
print(json.dumps(weights))
PYEOF
)

if [ -z "$WEIGHTS" ] || [ "$WEIGHTS" = "{}" ]; then
    echo "ERROR: Could not resolve weights for task ${SLURM_ARRAY_TASK_ID}" >&2
    exit 1
fi

echo "=== E4.W3 Array Task ${SLURM_ARRAY_TASK_ID}/56 ==="
echo "    Weights: ${WEIGHTS}"

# ---------------------------------------------------------------------------
# Fetch data if not cached
# ---------------------------------------------------------------------------
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
