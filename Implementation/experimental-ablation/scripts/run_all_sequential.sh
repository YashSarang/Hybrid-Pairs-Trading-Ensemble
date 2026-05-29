#!/usr/bin/env bash
# Sequential multi-market WFV runner
# Runs experiments one at a time to avoid GPU memory conflicts

set -euo pipefail

PYTHON="/c/Python313/python.exe"
SCRIPT="run_multi_market_wfv.py"
LOG_DIR="../logs"

mkdir -p "$LOG_DIR"

echo "🚀 Starting sequential WFV experiments..."
echo "================================================"
echo "Total: 7 experiments × ~6-8h each = ~48h total"
echo "================================================"
echo ""

experiments=(
  "us:ou"
  "india:zscore"
  "india:ou"
  "brazil:zscore"
  "brazil:ou"
  "uk:zscore"
  "uk:ou"
)

for i in "${!experiments[@]}"; do
  IFS=':' read -r market signal <<< "${experiments[$i]}"
  num=$((i + 1))
  
  echo "[$num/7] Running $market + ${signal}Threshold..."
  echo "  Started: $(date '+%Y-%m-%d %H:%M:%S')"
  
  log_file="$LOG_DIR/${market}_${signal}.log"
  
  $PYTHON $SCRIPT --market "$market" --signal_model "$signal" --n_folds 4 \
    > "$log_file" 2>&1
  
  exit_code=$?
  
  if [ $exit_code -eq 0 ]; then
    echo "  ✅ Completed: $(date '+%Y-%m-%d %H:%M:%S')"
  else
    echo "  ❌ Failed with exit code $exit_code"
    echo "  See log: $log_file"
  fi
  
  echo ""
done

echo "================================================"
echo "✅ All experiments complete!"
echo "================================================"
