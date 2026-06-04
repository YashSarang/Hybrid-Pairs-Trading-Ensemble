#!/bin/bash
# run_full_experiments.sh
# -----------------------
# Replication script: submits all experiments in dependency order.
# Respects the cluster's 2-running + 2-pending job limit via SLURM --dependency chains.
#
# Usage (from Implementation/ directory):
#   bash jobs/run_full_experiments.sh
#
# The submission order is:
#   Round 1 (parallel): E4 + E3
#   Round 2 (after Round 1): E7 + E3-transformer
#   Round 3 (after Round 2): E1 + E6
#
# E2 (hold sweep) and E5 (benchmarks) are fast local runs — see scripts.md.

set -euo pipefail

echo "============================================================"
echo "Submitting full experiment replication suite"
echo "All experiments: E1, E3, E4, E6, E7"
echo "============================================================"
echo ""

# --- Round 1: independent, submit together ---
E4_JOB=$(sbatch --parsable jobs/e4_walk_forward.sh)
echo "  [Round 1] E4 Walk-Forward   (~12h) -> job $E4_JOB"

E3_JOB=$(sbatch --parsable jobs/e3_ablation.sh)
echo "  [Round 1] E3 Ablation       (~10h) -> job $E3_JOB"

# --- Round 2: after Round 1 completes ---
E7_JOB=$(sbatch --parsable --dependency=afterok:${E4_JOB}:${E3_JOB} jobs/e7_weighted_ensemble.sh)
echo "  [Round 2] E7 Weighted       (~4h, after E4+E3) -> job $E7_JOB"

E3T_JOB=$(sbatch --parsable --dependency=afterok:${E3_JOB} jobs/e3_transformer_rerun.sh)
echo "  [Round 2] E3 Transformer    (~3h, after E3)    -> job $E3T_JOB"

# --- Round 3: after Round 2 completes ---
E1_JOB=$(sbatch --parsable --dependency=afterok:${E3T_JOB} jobs/e1_frequency.sh)
echo "  [Round 3] E1 Frequency      (~2h, after E3-T)  -> job $E1_JOB"

E6_JOB=$(sbatch --parsable --dependency=afterok:${E7_JOB} jobs/e6_significance_full.sh)
echo "  [Round 3] E6 Significance   (~1h, after E7)    -> job $E6_JOB"

echo ""
echo "All jobs queued with dependency chains."
echo ""
echo "Monitor : squeue -u \$(whoami)"
echo "Logs    : tail -f logs/e4_walk_forward_${E4_JOB}.out"
echo "          tail -f logs/e3_ablation_${E3_JOB}.out"
echo "          tail -f logs/e7_weighted_ensemble_${E7_JOB}.out"
echo "          tail -f logs/e6_significance_full_${E6_JOB}.out"
echo ""
echo "Cancel all: scancel $E4_JOB $E3_JOB $E7_JOB $E3T_JOB $E1_JOB $E6_JOB"
