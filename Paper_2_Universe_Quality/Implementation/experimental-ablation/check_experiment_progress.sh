#!/bin/bash
# check_experiment_progress.sh
# Quick status check for Job 8455

HOST="yash.sarang@kalpana.minds.iitb.ac.in"

echo "=========================================="
echo "NSE NIFTY 50 CONTROL EXPERIMENT - Job 8455"
echo "=========================================="
echo ""

# Check if job is still running
JOB_STATUS=$(ssh $HOST 'squeue -u yash.sarang -o "%.10i %.8T %.10M" | grep 8455')

if [ -n "$JOB_STATUS" ]; then
    echo "Status: RUNNING"
    echo "$JOB_STATUS"
    echo ""
    
    # Show last 30 lines of log
    echo "Recent progress:"
    ssh $HOST 'tail -30 ~/Hybrid-Pairs-Trading-Ensemble/Implementation/experimental-ablation/logs/control_experiment_cached_8455.out'
else
    echo "Status: COMPLETED or NOT FOUND"
    echo ""
    
    # Check for results
    echo "Results:"
    ssh $HOST 'ls -lh ~/Hybrid-Pairs-Trading-Ensemble/Implementation/experimental-ablation/results/nse_nifty50/*.json 2>/dev/null'
    
    echo ""
    echo "Full log:"
    ssh $HOST 'tail -100 ~/Hybrid-Pairs-Trading-Ensemble/Implementation/experimental-ablation/logs/control_experiment_cached_8455.out'
fi
