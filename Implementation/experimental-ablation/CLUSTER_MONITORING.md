# Cluster Monitoring Cheat Sheet

## Check job status
```bash
ssh yash.sarang@kalpana.minds.iitb.ac.in "squeue -u yash.sarang"
```

## Check logs (live tail)
```bash
# India ZScore (Job 8159 - currently running)
ssh yash.sarang@kalpana.minds.iitb.ac.in "tail -f ~/Hybrid-Pairs-Trading-Ensemble/Implementation/experimental-ablation/logs/india_zscore_8159.out"

# Brazil ZScore (Job 8161 - currently running)
ssh yash.sarang@kalpana.minds.iitb.ac.in "tail -f ~/Hybrid-Pairs-Trading-Ensemble/Implementation/experimental-ablation/logs/brazil_zscore_8161.out"
```

## Check all log files
```bash
ssh yash.sarang@kalpana.minds.iitb.ac.in "ls -lht ~/Hybrid-Pairs-Trading-Ensemble/Implementation/experimental-ablation/logs/*.out | head -10"
```

## Download results (when complete)
```bash
# Download all result JSONs
scp 'yash.sarang@kalpana.minds.iitb.ac.in:~/Hybrid-Pairs-Trading-Ensemble/Implementation/experimental-ablation/results/*/wfv_*.json' \
    /c/Code/Hybrid-Pairs-Trading-Ensemble/Implementation/experimental-ablation/results/
```

## Cancel a job (if needed)
```bash
ssh yash.sarang@kalpana.minds.iitb.ac.in "scancel 8159"  # Replace with job ID
```

## Cancel all jobs (if needed)
```bash
ssh yash.sarang@kalpana.minds.iitb.ac.in "scancel -u yash.sarang"
```
