#!/bin/bash
# Submit all three experiments

sbatch jobs/e4_walk_forward.sh
sbatch jobs/e3_ablation.sh
sbatch jobs/e1_frequency.sh

echo "Jobs submitted. Check status with: squeue -u \$(whoami)"
