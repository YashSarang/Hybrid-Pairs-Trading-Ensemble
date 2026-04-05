#!/usr/bin/env bash
# =============================================================================
# run_full_experiments.sh
# =============================================================================
# Runs the three remaining full-mode experiments in priority order:
#
#   E4 — Walk-Forward Validation  (full mode, ou_only)     ~8-12h
#   E3 — Ablation Study           (full mode, both stages) ~6-10h
#   E1 — Frequency Comparison     (full mode)              ~30-60m
#
# Each experiment logs to experiments/logs/<experiment>_<timestamp>.log
# and saves JSON results to experiments/results/ as usual.
#
# Usage (from repo root, with venv active):
#   bash experiments/run_full_experiments.sh
#
#   # Or run a single experiment:
#   bash experiments/run_full_experiments.sh e4
#   bash experiments/run_full_experiments.sh e3
#   bash experiments/run_full_experiments.sh e1
# =============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOGS_DIR="$REPO_ROOT/experiments/logs"
mkdir -p "$LOGS_DIR"

TS="$(date +%Y%m%d_%H%M%S)"

# ---------------------------------------------------------------------------
# Helper: run an experiment, tee output to log, print elapsed time
# ---------------------------------------------------------------------------
run_experiment() {
    local label="$1"
    local script="$2"
    shift 2
    local args=("$@")

    local log_file="$LOGS_DIR/${label}_${TS}.log"

    echo ""
    echo "============================================================"
    echo "  STARTING: $label"
    echo "  Command : python $script ${args[*]}"
    echo "  Log     : $log_file"
    echo "  Started : $(date '+%Y-%m-%d %H:%M:%S')"
    echo "============================================================"

    local t0=$SECONDS
    python "$REPO_ROOT/$script" "${args[@]}" 2>&1 | tee "$log_file"
    local exit_code=${PIPESTATUS[0]}
    local elapsed=$(( SECONDS - t0 ))

    echo ""
    if [[ $exit_code -eq 0 ]]; then
        echo "  DONE: $label  ($(( elapsed / 3600 ))h $(( (elapsed % 3600) / 60 ))m $(( elapsed % 60 ))s)"
    else
        echo "  FAILED: $label  (exit code $exit_code)  — check $log_file"
    fi
    echo "============================================================"
    return $exit_code
}

# ---------------------------------------------------------------------------
# Which experiments to run (default: all three in order)
# ---------------------------------------------------------------------------
RUN_TARGET="${1:-all}"

case "$RUN_TARGET" in

    e4|all)
        run_experiment \
            "e4_wfv_full" \
            "experiments/walk_forward.py" \
            --mode full \
            --s2 ou_only
        ;;&   # fall-through only when target is "all"

    e3|all)
        run_experiment \
            "e3_ablation_full" \
            "experiments/ablation.py" \
            --mode full \
            --stage 0    # 0 = both Stage 1 and Stage 2
        ;;&

    e1|all)
        run_experiment \
            "e1_freq_full" \
            "experiments/freq_comparison.py" \
            --mode full \
            --freqs 1D 1H
        ;;

    *)
        echo "Unknown target '$RUN_TARGET'. Use: all | e4 | e3 | e1"
        exit 1
        ;;
esac

echo ""
echo "All requested experiments complete. Results in experiments/results/"
