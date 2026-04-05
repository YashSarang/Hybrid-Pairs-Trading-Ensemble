#!/bin/bash
# submit_chained.sh
# -----------------
# Submits E4, E3, E1 as dependent SLURM jobs so each runs in its own
# time slot (12h / 10h / 8h) in sequence.
#
# Use this instead of run_full_experiments.sh if the cluster enforces
# a 24h wall-time limit per job.
#
# Usage (from Implementation/ directory):
#   bash jobs/submit_chained.sh
#
# To cancel the whole chain:
#   scancel <E4_JOB_ID> <E3_JOB_ID> <E1_JOB_ID>

set -euo pipefail

echo "Submitting chained experiment jobs ..."

# Submit E4 (no dependency — runs immediately)
E4_JOB=$(sbatch --parsable jobs/e4_walk_forward.sh)
echo "  E4 submitted: job $E4_JOB"

# Submit E3 after E4 succeeds
E3_JOB=$(sbatch --parsable --dependency=afterok:$E4_JOB jobs/e3_ablation.sh)
echo "  E3 submitted: job $E3_JOB  (waits for E4 $E4_JOB)"

# Submit E1 after E3 succeeds
E1_JOB=$(sbatch --parsable --dependency=afterok:$E3_JOB jobs/e1_frequency.sh)
echo "  E1 submitted: job $E1_JOB  (waits for E3 $E3_JOB)"

echo ""
echo "Chain: E4 ($E4_JOB) -> E3 ($E3_JOB) -> E1 ($E1_JOB)"
echo ""
echo "Monitor:  squeue -u \$(whoami)"
echo "Cancel:   scancel $E4_JOB $E3_JOB $E1_JOB"
echo "Logs:     tail -f logs/e4_walk_forward_${E4_JOB}.out"
