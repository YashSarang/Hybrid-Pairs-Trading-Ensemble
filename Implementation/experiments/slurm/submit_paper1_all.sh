#!/bin/bash
# Paper 1 — master SLURM submit script
# Run on kalpana after: git pull origin main
# Usage: bash Implementation/experiments/slurm/submit_paper1_all.sh

set -e
mkdir -p /users/student/pg/pg24/yash.sarang/Hybrid-Pairs-Trading-Ensemble/Implementation/experiments/slurm_logs

echo "=== Submitting Paper 1 E1-E6 experiments ==="

J1=$(sbatch --parsable Implementation/experiments/slurm/e1_paper1_freq.sh)
echo "E1  submitted: $J1"

J3=$(sbatch --parsable Implementation/experiments/slurm/e3_paper1_ablation.sh)
echo "E3  submitted: $J3"

J4A=$(sbatch --parsable Implementation/experiments/slurm/e4a_paper1_stat.sh)
echo "E4a submitted: $J4A"

J4B=$(sbatch --parsable Implementation/experiments/slurm/e4b_paper1_statml.sh)
echo "E4b submitted: $J4B"

J4C=$(sbatch --parsable Implementation/experiments/slurm/e4c_paper1_full.sh)
echo "E4c submitted: $J4C  (slow — 24h wall)"

J5=$(sbatch --parsable --dependency=afterok:${J4A} \
    Implementation/experiments/slurm/e5_paper1_benchmark.sh)
echo "E5  submitted: $J5  (depends on E4a)"

J6=$(sbatch --parsable --dependency=afterok:${J4A}:${J4B}:${J4C} \
    Implementation/experiments/slurm/e6_paper1_significance.sh)
echo "E6  submitted: $J6  (depends on E4a+E4b+E4c)"

echo ""
echo "Monitor: squeue -u yash.sarang"
echo "Logs:    /users/student/pg/pg24/yash.sarang/Hybrid-Pairs-Trading-Ensemble/Implementation/experiments/slurm_logs/"
