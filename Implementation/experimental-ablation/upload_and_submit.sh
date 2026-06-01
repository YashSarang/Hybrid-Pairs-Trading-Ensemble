#!/bin/bash
# upload_and_submit.sh
# Manual upload and submission script (requires password entry)

set -e

HOST="yash.sarang@kalpana.minds.iitb.ac.in"
LOCAL_DIR="/d/Code/Hybrid-Pairs-Trading-Ensemble/Implementation/experimental-ablation"
REMOTE_DIR="/users/student/pg/pg24/yash.sarang/Hybrid-Pairs-Trading-Ensemble/Implementation/experimental-ablation"

echo "================================================================"
echo "UPLOADING FILES AND SUBMITTING SLURM JOB"
echo "================================================================"
echo ""
echo "Password: yash.sarang"
echo ""

# Upload Python script
echo "=== Step 1: Uploading run_control_experiment_cached.py ==="
scp "$LOCAL_DIR/run_control_experiment_cached.py" "$HOST:$REMOTE_DIR/"
echo "✓ Python script uploaded"
echo ""

# Upload SLURM script  
echo "=== Step 2: Uploading slurm_control_experiment_cached.sh ==="
scp "$LOCAL_DIR/slurm_control_experiment_cached.sh" "$HOST:$REMOTE_DIR/"
echo "✓ SLURM script uploaded"
echo ""

# Submit job
echo "=== Step 3: Submitting SLURM job ==="
ssh "$HOST" << 'ENDSSH'
cd ~/Hybrid-Pairs-Trading-Ensemble/Implementation/experimental-ablation

echo "Submitting job..."
sbatch slurm_control_experiment_cached.sh

echo ""
echo "Job queue:"
squeue -u yash.sarang -o "%.10i %.12j %.8T %.10M %.10L %.6D %.20R"
ENDSSH

echo ""
echo "================================================================"
echo "DONE"
echo "================================================================"
echo ""
echo "Monitor with:"
echo "  ssh $HOST 'tail -f ~/Hybrid-Pairs-Trading-Ensemble/Implementation/experimental-ablation/logs/control_experiment_cached_*.out'"
