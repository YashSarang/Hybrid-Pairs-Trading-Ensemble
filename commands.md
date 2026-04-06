conda activate pairs_trading
squeue -u $(whoami)
nvidia-smi


git pull
git add *
git commit -m "Whatever"
git push -u origin main