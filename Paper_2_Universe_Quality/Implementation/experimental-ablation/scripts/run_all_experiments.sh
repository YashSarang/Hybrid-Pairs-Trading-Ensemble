#!/usr/bin/env bash
# Multi-market WFV launcher — runs all 7 experiments in parallel
# Each experiment: 4 markets × 2 signal models - 1 (US+ZScore already done)

set -euo pipefail

PYTHON="/c/Python313/python.exe"
SCRIPT="run_multi_market_wfv.py"
LOG_DIR="../logs"

mkdir -p "$LOG_DIR"

echo "🚀 Launching 7 parallel WFV experiments..."
echo "================================================"

# US + OUThreshold
echo "[1/7] US + OUThreshold..."
nohup $PYTHON $SCRIPT --market us --signal_model ou --n_folds 4 \
  > "$LOG_DIR/us_ou.log" 2>&1 &
echo "  PID: $!"

# India + ZScore
echo "[2/7] India + ZScoreThreshold..."
nohup $PYTHON $SCRIPT --market india --signal_model zscore --n_folds 4 \
  > "$LOG_DIR/india_zscore.log" 2>&1 &
echo "  PID: $!"

# India + OUThreshold
echo "[3/7] India + OUThreshold..."
nohup $PYTHON $SCRIPT --market india --signal_model ou --n_folds 4 \
  > "$LOG_DIR/india_ou.log" 2>&1 &
echo "  PID: $!"

# Brazil + ZScore
echo "[4/7] Brazil + ZScoreThreshold..."
nohup $PYTHON $SCRIPT --market brazil --signal_model zscore --n_folds 4 \
  > "$LOG_DIR/brazil_zscore.log" 2>&1 &
echo "  PID: $!"

# Brazil + OUThreshold
echo "[5/7] Brazil + OUThreshold..."
nohup $PYTHON $SCRIPT --market brazil --signal_model ou --n_folds 4 \
  > "$LOG_DIR/brazil_ou.log" 2>&1 &
echo "  PID: $!"

# UK + ZScore
echo "[6/7] UK + ZScoreThreshold..."
nohup $PYTHON $SCRIPT --market uk --signal_model zscore --n_folds 4 \
  > "$LOG_DIR/uk_zscore.log" 2>&1 &
echo "  PID: $!"

# UK + OUThreshold
echo "[7/7] UK + OUThreshold..."
nohup $PYTHON $SCRIPT --market uk --signal_model ou --n_folds 4 \
  > "$LOG_DIR/uk_ou.log" 2>&1 &
echo "  PID: $!"

echo "================================================"
echo "✅ All 7 experiments launched!"
echo ""
echo "Monitor progress:"
echo "  tail -f $LOG_DIR/*.log"
echo ""
echo "Check running processes:"
echo "  ps aux | grep run_multi_market_wfv"
echo ""
echo "Estimated completion: 6-8 hours"
echo "================================================"
