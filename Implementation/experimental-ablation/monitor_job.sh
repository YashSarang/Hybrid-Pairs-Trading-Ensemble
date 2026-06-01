#!/bin/bash
# monitor_job.sh
# Monitor the running SLURM job

HOST="yash.sarang@kalpana.minds.iitb.ac.in"

echo "================================================================"
echo "MONITORING NSE NIFTY 50 CONTROL EXPERIMENT"
echo "================================================================"
echo ""
echo "Password: yash.sarang"
echo ""

ssh "$HOST" << 'ENDSSH'
cd ~/Hybrid-Pairs-Trading-Ensemble/Implementation/experimental-ablation

echo "=== Job Queue ==="
squeue -u yash.sarang -o "%.10i %.12j %.8T %.10M %.10L %.6D %.20R"

echo ""
echo "=== Latest Job Log ==="
LOG_FILE=$(ls -t logs/control_experiment_cached_*.out 2>/dev/null | head -1)
if [ -n "$LOG_FILE" ]; then
    echo "Log file: $LOG_FILE"
    echo ""
    tail -100 "$LOG_FILE"
else
    echo "No log file found yet"
fi

echo ""
echo "=== Results Generated ==="
ls -lht results/nse_nifty50/*.json 2>/dev/null | head -5 || echo "No results yet"

echo ""
echo "=== Latest Error Log ==="
ERR_FILE=$(ls -t logs/control_experiment_cached_*.err 2>/dev/null | head -1)
if [ -n "$ERR_FILE" ]; then
    echo "Error file: $ERR_FILE"
    echo ""
    tail -20 "$ERR_FILE"
else
    echo "No error file found yet"
fi
ENDSSH
