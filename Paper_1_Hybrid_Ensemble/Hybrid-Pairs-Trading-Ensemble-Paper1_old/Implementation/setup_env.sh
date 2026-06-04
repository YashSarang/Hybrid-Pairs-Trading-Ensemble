#!/bin/bash
# One-time setup script - installs Miniconda and creates environment

echo "=========================================="
echo "Setting up Conda environment"
echo "=========================================="

# 1. Download and install Miniconda
if [ ! -d "$HOME/miniconda3" ]; then
    echo "Installing Miniconda..."
    curl -s https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p $HOME/miniconda3
    rm /tmp/miniconda.sh
else
    echo "Miniconda already installed"
fi

# 2. Initialize conda
source $HOME/miniconda3/etc/profile.d/conda.sh

# 3. Create environment
echo ""
echo "Creating conda environment 'pairs_trading'..."
conda create -n pairs_trading python=3.10 -y

# 4. Activate environment
conda activate pairs_trading

# 5. Install requirements
echo ""
echo "Installing requirements..."
cd ~/Hybrid-Pairs-Trading-Ensemble/Implementation
pip install -r requirements.txt

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "Environment: pairs_trading"
echo "Location: $HOME/miniconda3/envs/pairs_trading"
echo ""
echo "You can now submit jobs with:"
echo "  sbatch jobs/e4_walk_forward.sh"
echo "  sbatch jobs/e3_ablation.sh"
echo "  sbatch jobs/e1_frequency.sh"
