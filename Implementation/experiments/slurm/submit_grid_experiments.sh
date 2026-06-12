#!/bin/bash
# =============================================================================
# submit_grid_experiments.sh
# Master submission script for E4.S + E4.W2 + E4.W3 + E4.W-Grid
#
# Usage:
#   # Phase 1 — standalone + pairwise (fastest, submit first):
#   bash slurm/submit_grid_experiments.sh phase1
#
#   # Phase 2 — triples + weight grid (submit after reviewing Phase 1 results):
#   bash slurm/submit_grid_experiments.sh phase2
#
#   # All at once:
#   bash slurm/submit_grid_experiments.sh all
# =============================================================================

set -euo pipefail

PHASE="${1:-phase1}"
SLURM_DIR="$(dirname "$0")"

echo "========================================================"
echo "  Ensemble Grid Search — SLURM Job Submission"
echo "  Phase: ${PHASE}"
echo "  Time: $(date)"
echo "========================================================"

submit_phase1() {
    echo ""
    echo "[Phase 1] Submitting E4.S (8 standalone jobs) ..."
    JID_S=$(sbatch --parsable "${SLURM_DIR}/e4s_standalone_array.sh")
    echo "  E4.S job ID: ${JID_S}"

    echo ""
    echo "[Phase 1] Submitting E4.W2 (28 pairwise jobs) ..."
    JID_W2=$(sbatch --parsable "${SLURM_DIR}/e4w2_pairwise_array.sh")
    echo "  E4.W2 job ID: ${JID_W2}"

    echo ""
    echo "Phase 1 submitted:"
    echo "  E4.S  (standalone):   Job ${JID_S}  (8 tasks)"
    echo "  E4.W2 (pairwise):     Job ${JID_W2} (28 tasks)"
    echo ""
    echo "Monitor with: squeue -u \$USER"
    echo "Once complete, run: python Implementation/experiments/aggregate_grid_results.py"
    echo "Then submit Phase 2 based on which standalone selectors have positive Net SR."
}

submit_phase2() {
    echo ""
    echo "[Phase 2] Submitting E4.W3 (56 triple jobs) ..."
    JID_W3=$(sbatch --parsable "${SLURM_DIR}/e4w3_triple_array.sh")
    echo "  E4.W3 job ID: ${JID_W3}"

    echo ""
    echo "[Phase 2] Submitting E4.W-Grid (11 Corr-vs-LSTM weight sweep jobs) ..."
    JID_GRID=$(sbatch --parsable "${SLURM_DIR}/e4wgrid_weight_sweep.sh")
    echo "  E4.W-Grid job ID: ${JID_GRID}"

    echo ""
    echo "Phase 2 submitted:"
    echo "  E4.W3     (triples):      Job ${JID_W3}   (56 tasks)"
    echo "  E4.W-Grid (weight sweep): Job ${JID_GRID} (11 tasks)"
}

case "${PHASE}" in
    phase1) submit_phase1 ;;
    phase2) submit_phase2 ;;
    all)
        submit_phase1
        submit_phase2
        ;;
    *)
        echo "Unknown phase '${PHASE}'. Use: phase1 | phase2 | all"
        exit 1
        ;;
esac

echo ""
echo "========================================================"
echo "  SLURM job submission complete."
echo "  Results will appear in:"
echo "    Implementation/experiments/results/walk_forward_*.json"
echo "  Aggregate with:"
echo "    python Implementation/experiments/aggregate_grid_results.py"
echo "========================================================"
