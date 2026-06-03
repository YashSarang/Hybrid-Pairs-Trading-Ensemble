#!/usr/bin/env bash
# run_control_experiment.sh
# Execute the critical NSE Nifty 50 control experiment

set -e

cd /d/Code/Hybrid-Pairs-Trading-Ensemble/Implementation/experimental-ablation

echo "==================================="
echo "NSE NIFTY 50 CONTROL EXPERIMENT"
echo "==================================="
echo ""
echo "Purpose: Test if Nifty 50 outperforms Nifty 100 on SAME exchange (NSE)"
echo "This isolates universe quality from geographic effects"
echo ""

# Activate venv
source ../../.venv/Scripts/activate

echo "Step 1: Fetch NSE Nifty 50 data..."
python scripts/fetch_market_data.py --market nse_nifty50

echo ""
echo "Step 2: Run NSE Nifty 50 + Rolling + ZScore..."
python scripts/run_multi_market_wfv.py \
    --market nse_nifty50 \
    --signal zscore \
    --n_folds 4 \
    --lookback 126

echo ""
echo "Step 3: Run NSE Nifty 50 + Rolling + OU..."
python scripts/run_multi_market_wfv.py \
    --market nse_nifty50 \
    --signal ou \
    --n_folds 4 \
    --lookback 126

echo ""
echo "==================================="
echo "✅ CONTROL EXPERIMENT COMPLETE"
echo "==================================="
echo ""
echo "Check results in:"
echo "  results/nse_nifty50/wfv_4folds_zscore_*.json"
echo "  results/nse_nifty50/wfv_4folds_ou_*.json"
echo ""
echo "Next: Analyze results to determine narrative (Scenario A, B, or C)"
