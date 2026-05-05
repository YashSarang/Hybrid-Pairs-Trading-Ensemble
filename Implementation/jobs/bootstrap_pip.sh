#!/bin/bash
#SBATCH --job-name=bootstrap_pip
#SBATCH --account=cminds_anandi
#SBATCH --partition=cn3_anandi
#SBATCH --qos=anandi
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

echo "=========================================="
echo "Bootstrapping pip installation"
echo "=========================================="

# Download get-pip.py
echo "Downloading get-pip.py..."
curl -s https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py

# Install pip using get-pip.py
echo "Installing pip..."
python3 /tmp/get-pip.py --user

# Verify pip installation
echo ""
echo "Verifying pip installation..."
python3 -m pip --version

# Install requirements
PROJ_DIR="/users/student/pg/pg24/$(whoami)/Hybrid-Pairs-Trading-Ensemble"
cd $PROJ_DIR

echo ""
echo "Installing requirements..."
python3 -m pip install --user -r requirements.txt

echo ""
echo "=========================================="
echo "Bootstrap complete"
echo "=========================================="
