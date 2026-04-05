#!/bin/bash
#SBATCH --job-name=run_full_experiments
#SBATCH --account=cminds_anandi
#SBATCH --partition=cn3_l40s
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=30:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# NOTE: Total wall-time estimate is ~30h (E4: 12h, E3: 10h, E1: 8h).
# If the cluster enforces a 24h limit, use submit_chained.sh instead,
# which submits the three experiments as dependent jobs in sequence.

echo "=========================================="
echo "Full Experiment Suite: E4 -> E3 -> E1"
echo "Start time: $(date)"
echo "Node: $SLURMD_NODENAME"
echo "Job ID: $SLURM_JOB_ID"
echo "=========================================="

# Activate conda environment
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda activate pairs_trading

echo "Python: $(which python)"
echo "Conda env: $CONDA_DEFAULT_ENV"
echo ""

WORK_DIR="/users/student/pg/pg24/$(whoami)/Hybrid-Pairs-Trading-Ensemble/Implementation"
cd "$WORK_DIR"

# Track overall status
E4_STATUS="NOT_RUN"
E3_STATUS="NOT_RUN"
E1_STATUS="NOT_RUN"

# ---------------------------------------------------------------------------
# E4 — Walk-Forward Validation (full mode, ou_only)
# This is the primary result: does the full ensemble beat stat-only?
# ---------------------------------------------------------------------------
echo "=========================================="
echo "E4: Walk-Forward Validation (full, ou_only)"
echo "Start: $(date)"
echo "=========================================="

python experiments/walk_forward.py --mode full --s2 ou_only
E4_EXIT=$?

if [[ $E4_EXIT -eq 0 ]]; then
    E4_STATUS="OK"
    echo "E4 PASSED (exit 0)"
else
    E4_STATUS="FAILED (exit $E4_EXIT)"
    echo "E4 FAILED with exit code $E4_EXIT — continuing to E3"
fi

echo "E4 end: $(date)"
echo ""

# ---------------------------------------------------------------------------
# E3 — Ablation Study (full mode, both stages)
# Proves ensemble > individual. Stage 2 uses pairs from full S1 ensemble.
# ---------------------------------------------------------------------------
echo "=========================================="
echo "E3: Ablation Study (full, stages 1+2)"
echo "Start: $(date)"
echo "=========================================="

python experiments/ablation.py --mode full --stage 0
E3_EXIT=$?

if [[ $E3_EXIT -eq 0 ]]; then
    E3_STATUS="OK"
    echo "E3 PASSED (exit 0)"
else
    E3_STATUS="FAILED (exit $E3_EXIT)"
    echo "E3 FAILED with exit code $E3_EXIT — continuing to E1"
fi

echo "E3 end: $(date)"
echo ""

# ---------------------------------------------------------------------------
# E1 — Frequency Comparison (full mode)
# Confirms daily > hourly with all 8 selectors active (not just stat-only).
# ---------------------------------------------------------------------------
echo "=========================================="
echo "E1: Frequency Comparison (full, 1D vs 1H)"
echo "Start: $(date)"
echo "=========================================="

python experiments/freq_comparison.py --mode full --freqs 1D 1H
E1_EXIT=$?

if [[ $E1_EXIT -eq 0 ]]; then
    E1_STATUS="OK"
    echo "E1 PASSED (exit 0)"
else
    E1_STATUS="FAILED (exit $E1_EXIT)"
    echo "E1 FAILED with exit code $E1_EXIT"
fi

echo "E1 end: $(date)"
echo ""

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo "=========================================="
echo "SUMMARY"
echo "  E4 Walk-Forward (full): $E4_STATUS"
echo "  E3 Ablation (full):     $E3_STATUS"
echo "  E1 Freq Comparison:     $E1_STATUS"
echo "End time: $(date)"
echo "=========================================="

# Exit non-zero if any experiment failed
if [[ "$E4_STATUS" != "OK" || "$E3_STATUS" != "OK" || "$E1_STATUS" != "OK" ]]; then
    exit 1
fi
exit 0
