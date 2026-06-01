#!/bin/bash
#SBATCH --job-name=nse_nifty50_ctrl
#SBATCH --output=logs/control_experiment_cached_%j.out
#SBATCH --error=logs/control_experiment_cached_%j.err
#SBATCH --partition=cn3_anandi
#SBATCH --qos=anandi
#SBATCH --account=cminds_anandi
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00

echo "================================================================"
echo "NSE NIFTY 50 CONTROL EXPERIMENT (CACHED DATA)"
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
echo "Python version: $(python --version)"
echo ""

# Copy India data to NSE Nifty 50 cache location
echo "Setting up cached data..."
mkdir -p data/nse_nifty50
cp data/india/prices_2020-01-01_2025-05-01.parquet data/nse_nifty50/prices_2020-01-01_2025-05-01.parquet
echo "✓ Cache ready: data/nse_nifty50/prices_2020-01-01_2025-05-01.parquet"
echo ""

echo "Running NSE Nifty 50 control experiment (using cached data)..."
python run_control_experiment_cached.py

echo ""
echo "================================================================"
echo "EXPERIMENT COMPLETE"
echo "Finished: $(date)"
echo "================================================================"
echo ""
echo "Results saved to:"
find results/nse_nifty50 -name "*.json" -type f -exec ls -lh {} \; 2>/dev/null || echo "No results found"
