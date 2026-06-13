"""
experiments/results_sync.py
===========================
Helper script to sync grid search walk_forward JSON files from Kalpana
to local directory, and run the local aggregator.

Usage:
  python Implementation/experiments/results_sync.py --loop --interval 60
  python Implementation/experiments/results_sync.py --once
"""
import argparse
import subprocess
import time
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent.parent
LOCAL_RESULTS_DIR = ROOT / "Implementation" / "experiments" / "results"
REMOTE_PATH = "kalpana:~/Hybrid-Pairs-Trading-Ensemble/Implementation/experiments/results/walk_forward_*.json"
AGGREGATOR_SCRIPT = ROOT / "Implementation" / "experiments" / "aggregate_grid_results.py"

def sync_results():
    log.info("Checking Kalpana queue status...")
    try:
        q_out = subprocess.check_output(
            ["ssh", "kalpana", "squeue -u $USER"], 
            text=True, 
            stderr=subprocess.DEVNULL
        )
        print(q_out)
    except Exception as exc:
        log.warning(f"Could not fetch queue status: {exc}")

    log.info("Syncing new walk_forward JSON files from Kalpana...")
    # scp command to copy walk_forward_*.json files
    try:
        subprocess.run(
            ["scp", REMOTE_PATH, str(LOCAL_RESULTS_DIR)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        log.info("Sync complete.")
    except Exception as exc:
        log.warning(f"Sync command finished with some issues (usually means no new files or file list too long): {exc}")

    log.info("Running local results aggregator...")
    try:
        subprocess.run(
            ["python", str(AGGREGATOR_SCRIPT)],
            check=True
        )
    except Exception as exc:
        log.error(f"Aggregator execution failed: {exc}")

def main():
    parser = argparse.ArgumentParser(description="Sync Kalpana WFV results and aggregate locally.")
    parser.add_argument("--once", action="store_true", help="Sync once and exit.")
    parser.add_argument("--loop", action="store_true", help="Sync in a loop.")
    parser.add_argument("--interval", type=int, default=120, help="Loop interval in seconds.")
    args = parser.parse_args()

    if args.once or not args.loop:
        sync_results()
    else:
        log.info(f"Starting sync daemon. Interval: {args.interval} seconds. Press Ctrl+C to stop.")
        try:
            while True:
                sync_results()
                time.sleep(args.interval)
        except KeyboardInterrupt:
            log.info("Sync daemon stopped.")

if __name__ == "__main__":
    main()
