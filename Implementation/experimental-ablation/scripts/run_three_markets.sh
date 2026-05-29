#!/usr/bin/env bash
# Sequential runner for 3 remaining markets
# Waits for India to finish, then runs Brazil, then UK

set -euo pipefail

PYTHON="/c/Python313/python.exe"
SCRIPT="run_multi_market_wfv.py"

echo "🚀 Sequential 3-Market WFV Runner"
echo "================================================"
echo ""

# Check if India is already running
if ps aux | grep -q "[r]un_multi_market_wfv.py --market india"; then
  echo "✅ India already running (PID: $(ps aux | grep '[r]un_multi_market_wfv.py --market india' | awk '{print $2}'))"
  echo "   Waiting for completion..."
  
  # Wait for India to finish
  while ps aux | grep -q "[r]un_multi_market_wfv.py --market india"; do
    sleep 60
  done
  
  echo "✅ India completed!"
  echo ""
else
  echo "[1/3] Starting India + ZScore..."
  echo "  Started: $(date '+%Y-%m-%d %H:%M:%S')"
  
  $PYTHON $SCRIPT --market india --signal_model zscore --n_folds 4
  
  echo "  ✅ Completed: $(date '+%Y-%m-%d %H:%M:%S')"
  echo ""
fi

echo "[2/3] Starting Brazil + ZScore..."
echo "  Started: $(date '+%Y-%m-%d %H:%M:%S')"

$PYTHON $SCRIPT --market brazil --signal_model zscore --n_folds 4

echo "  ✅ Completed: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

echo "[3/3] Starting UK + ZScore..."
echo "  Started: $(date '+%Y-%m-%d %H:%M:%S')"

$PYTHON $SCRIPT --market uk --signal_model zscore --n_folds 4

echo "  ✅ Completed: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

echo "================================================"
echo "✅ All 3 markets complete!"
echo "================================================"
echo ""
echo "Results saved to:"
echo "  - experimental-ablation/results/india/wfv_4folds_zscore_*.json"
echo "  - experimental-ablation/results/brazil/wfv_4folds_zscore_*.json"
echo "  - experimental-ablation/results/uk/wfv_4folds_zscore_*.json"
echo ""
echo "Next steps:"
echo "  python compare_markets.py --markets us india brazil uk"
echo "  python visualize_cross_market.py --markets us india brazil uk"
