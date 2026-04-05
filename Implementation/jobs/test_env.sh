#!/bin/bash
#SBATCH --job-name=test_env
#SBATCH --account=cminds_anandi
#SBATCH --partition=cn3_l40s
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

echo "=========================================="
echo "Testing environment setup"
echo "=========================================="

echo ""
echo "Python version:"
python3 --version

echo ""
echo "Checking for pip:"
python3 -m pip --version 2>&1 || echo "pip not found via -m pip"

echo ""
echo "Checking for get-pip.py:"
which get-pip.py 2>&1 || echo "get-pip.py not found"

echo ""
echo "Checking available Python packages:"
python3 -c "import sys; print('Python path:', sys.path)"

echo ""
echo "Checking for apt:"
which apt && apt --version || echo "apt not available"

echo ""
echo "Checking for conda:"
which conda && conda --version || echo "conda not available"

echo ""
echo "Checking for mamba:"
which mamba && mamba --version || echo "mamba not available"

echo ""
echo "Checking for uv:"
which uv && uv --version || echo "uv not available"

echo ""
echo "Checking installed Python packages:"
python3 -c "import pkgutil; mods = [name for _, name, _ in pkgutil.iter_modules()]; print('\\n'.join(sorted(mods)[:20]))"

echo ""
echo "=========================================="
