#!/bin/bash
#SBATCH --job-name=nse_n50_simple
#SBATCH --output=logs/control_experiment_simple_%j.out
#SBATCH --error=logs/control_experiment_simple_%j.err
#SBATCH --partition=cn3_anandi
#SBATCH --qos=anandi
#SBATCH --account=cminds_anandi
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00

echo "================================================================"
echo "NSE NIFTY 50 CONTROL EXPERIMENT (Statistical Selectors Only)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Started: $(date)"
echo "================================================================"
echo ""

cd /users/student/pg/pg24/yash.sarang/Hybrid-Pairs-Trading-Ensemble/Implementation/experimental-ablation

echo "Working directory: $(pwd)"

# Activate conda environment
echo "Activating conda environment..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate base
echo "Python: $(which python)"
echo ""

echo "Running NSE Nifty 50 control experiment (statistical selectors only)..."
python run_control_experiment_simple.py

echo ""
echo "================================================================"
echo "EXPERIMENT COMPLETE"
echo "Finished: $(date)"
echo "================================================================"
echo ""
echo "Results:"
find results/nse_nifty50 -name "*_statistical_*.json" -type f -exec ls -lh {} \; 2>/dev/null || echo "No results found"
