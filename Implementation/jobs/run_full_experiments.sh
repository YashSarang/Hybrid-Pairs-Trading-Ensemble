#!/bin/bash
# run_full_experiments.sh
# -----------------------
# Submits E4, E3, and E1 as three independent parallel SLURM jobs.
# Each gets its own L40S GPU (48 GB VRAM) and runs concurrently.
#
# Usage (from Implementation/ directory):
#   bash jobs/run_full_experiments.sh
#
# Monitor:
#   squeue -u $(whoami)
#
# Cancel all three:
#   scancel <E4_JOB_ID> <E3_JOB_ID> <E1_JOB_ID>

set -euo pipefail

echo "Submitting full-mode experiment suite ..."

E4_JOB=$(sbatch --parsable jobs/e4_walk_forward.sh)
echo "  E4 Walk-Forward  (full, ou_only, ~12h) -> job $E4_JOB"

E3_JOB=$(sbatch --parsable jobs/e3_ablation.sh)
echo "  E3 Ablation      (full, stages 1+2, ~10h) -> job $E3_JOB"

E1_JOB=$(sbatch --parsable jobs/e1_frequency.sh)
echo "  E1 Freq Compare  (full, 1D vs 1H, ~1h)  -> job $E1_JOB"

echo ""
echo "All three jobs submitted in parallel."
echo ""
echo "Monitor : squeue -u \$(whoami)"
echo "Logs    : tail -f logs/e4_walk_forward_${E4_JOB}.out"
echo "          tail -f logs/e3_ablation_${E3_JOB}.out"
echo "          tail -f logs/e1_frequency_${E1_JOB}.out"
echo "Cancel  : scancel $E4_JOB $E3_JOB $E1_JOB"
