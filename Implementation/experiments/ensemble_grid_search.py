"""
experiments/ensemble_grid_search.py
=====================================
Exhaustive Stage-1 Selector Ensemble Grid Search
-------------------------------------------------
Runs Walk-Forward Validation (E4) for:

  E4.S  — 8 single-selector standalone benchmarks
  E4.W2 — All C(8,2)=28 pairwise equal-weight ensembles
  E4.W3 — All C(8,3)=56 triple equal-weight ensembles   [optional]

Each combination is evaluated using the SAME pipeline as walk_forward.py
(6-fold expanding WFV, OU-only S2, top-k=10, 16.28 bps costs) via subprocess.

Results are saved to:
  experiments/results/grid_search/ensemble_grid_results.json

Usage
-----
# Run all groups (standalone + pairwise + triples):
    python experiments/ensemble_grid_search.py

# Only E4.S + E4.W2 (faster — recommended first pass):
    python experiments/ensemble_grid_search.py --no-triples

# Dry-run (print combinations without executing):
    python experiments/ensemble_grid_search.py --dry-run

# Resume from a previous partial run (skip completed combos):
    python experiments/ensemble_grid_search.py --resume

# Parallelism (concurrent subprocess workers):
    python experiments/ensemble_grid_search.py --workers 4
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Paths & logging
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent          # Implementation/
RESULTS_DIR = Path(__file__).parent / "results" / "grid_search"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

AGGREGATE_FILE = RESULTS_DIR / "ensemble_grid_results.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Selector registry — must match config.py keys exactly
# ---------------------------------------------------------------------------

ALL_SELECTORS: List[str] = [
    "Correlation",
    "Distance",
    "Cointegration",
    "Combined",
    "ML",
    "LSTM",
    "Transformer",
    "GNN",
]

# Human-readable short labels for printing
SHORT = {
    "Correlation":   "Corr",
    "Distance":      "Dist",
    "Cointegration": "Coint",
    "Combined":      "Comb",
    "ML":            "ML",
    "LSTM":          "LSTM",
    "Transformer":   "Trans",
    "GNN":           "GNN",
}

# ---------------------------------------------------------------------------
# Combination generators
# ---------------------------------------------------------------------------


def _make_weights(selectors: Tuple[str, ...]) -> Dict[str, float]:
    """Equal-weight dict for an arbitrary subset of selectors."""
    zero = {k: 0.0 for k in ALL_SELECTORS}
    w = 1.0 / len(selectors)
    for s in selectors:
        zero[s] = w
    return zero


def _combo_label(selectors: Tuple[str, ...]) -> str:
    return "+".join(SHORT[s] for s in selectors)


def generate_combinations(
    include_triples: bool = True,
) -> List[Tuple[str, Tuple[str, ...]]]:
    """Return (label, selector_tuple) for all planned experiments."""
    combos: List[Tuple[str, Tuple[str, ...]]] = []

    # E4.S — standalone single selectors
    for s in ALL_SELECTORS:
        combos.append((SHORT[s] + "_only", (s,)))

    # E4.W2 — all pairwise
    for pair in combinations(ALL_SELECTORS, 2):
        combos.append((_combo_label(pair), pair))

    # E4.W3 — all triples
    if include_triples:
        for triple in combinations(ALL_SELECTORS, 3):
            combos.append((_combo_label(triple), triple))

    return combos


# ---------------------------------------------------------------------------
# Single run executor
# ---------------------------------------------------------------------------

PYTHON = sys.executable   # reuse the same interpreter that launched this script


def _run_single(
    label: str,
    selectors: Tuple[str, ...],
    s2: str = "ou_only",
    top_k: int = 10,
    dry_run: bool = False,
) -> Dict:
    """Execute walk_forward.py for one combination and return parsed result."""
    weights = _make_weights(selectors)
    weights_json = json.dumps({k: v for k, v in weights.items()})

    cmd = [
        PYTHON,
        str(ROOT / "experiments" / "walk_forward.py"),
        "--s1-weights", weights_json,
        "--s2", s2,
        "--top-k", str(top_k),
    ]

    if dry_run:
        log.info(f"[DRY-RUN] {label:40s}  cmd={' '.join(cmd[:4])} ...")
        return {"label": label, "selectors": list(selectors), "status": "dry_run"}

    t0 = time.time()
    log.info(f"[START] {label:40s}  ({len(selectors)} selectors)")
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(ROOT),
            timeout=3600,   # 1h per run — generous for LSTM-heavy combos
        )
        elapsed = time.time() - t0

        if proc.returncode != 0:
            log.error(f"[FAIL ] {label}  rc={proc.returncode}  elapsed={elapsed:.0f}s")
            log.error(f"  stderr: {proc.stderr[-500:]}")
            return {
                "label": label,
                "selectors": list(selectors),
                "status": "error",
                "returncode": proc.returncode,
                "stderr_tail": proc.stderr[-500:],
                "elapsed_s": round(elapsed, 1),
            }

        # Parse the results JSON written by walk_forward.py
        # walk_forward.py prints the path at the end: "Results saved -> <path>"
        result_path: Optional[Path] = None
        for line in proc.stdout.splitlines() + proc.stderr.splitlines():
            if "Results saved" in line and "walk_forward_" in line:
                # Extract the path from the log line
                parts = line.split("->")
                if len(parts) >= 2:
                    candidate = parts[-1].strip()
                    p = Path(candidate)
                    if p.exists():
                        result_path = p
                        break

        if result_path is None:
            # Fall back: find the most recently created walk_forward_*.json
            all_json = sorted(
                (ROOT / "experiments" / "results").glob("walk_forward_*.json"),
                key=lambda f: f.stat().st_mtime,
                reverse=True,
            )
            if all_json:
                result_path = all_json[0]

        summary: Dict = {
            "label": label,
            "selectors": list(selectors),
            "weights": {k: round(v, 6) for k, v in weights.items() if v > 0},
            "status": "ok",
            "elapsed_s": round(elapsed, 1),
        }

        if result_path and result_path.exists():
            with open(result_path) as fh:
                data = json.load(fh)
            agg = data.get("aggregate", {})
            summary["result_file"] = str(result_path)
            summary["mean_net_sharpe"]   = agg.get("mean_net_sharpe")
            summary["mean_gross_sharpe"] = agg.get("mean_gross_sharpe")
            summary["mean_net_cagr"]     = agg.get("mean_net_ann_ret_pct")
            summary["mean_maxdd_pct"]    = agg.get("mean_net_maxdd_pct")
            summary["total_trades"]      = agg.get("total_trades")
            summary["pct_folds_positive"]= agg.get("pct_folds_net_positive")
        else:
            summary["status"] = "result_not_found"
            log.warning(f"  Could not locate result JSON for {label}")

        log.info(
            f"[DONE ] {label:40s}  "
            f"NetSR={summary.get('mean_net_sharpe', 'N/A')}  "
            f"elapsed={elapsed:.0f}s"
        )
        return summary

    except subprocess.TimeoutExpired:
        elapsed = time.time() - t0
        log.error(f"[TIMEOUT] {label}  elapsed={elapsed:.0f}s")
        return {
            "label": label,
            "selectors": list(selectors),
            "status": "timeout",
            "elapsed_s": round(elapsed, 1),
        }
    except Exception as exc:
        elapsed = time.time() - t0
        log.error(f"[ERROR ] {label}  {exc}")
        return {
            "label": label,
            "selectors": list(selectors),
            "status": "exception",
            "error": str(exc),
            "elapsed_s": round(elapsed, 1),
        }


# ---------------------------------------------------------------------------
# Aggregate & display
# ---------------------------------------------------------------------------


def _print_ranked_table(results: List[Dict]) -> None:
    """Print results sorted by mean_net_sharpe descending."""
    ok = [r for r in results if r.get("status") == "ok" and r.get("mean_net_sharpe") is not None]
    ok.sort(key=lambda r: r["mean_net_sharpe"], reverse=True)

    header = f"{'Rank':>4}  {'Label':<40}  {'Selectors':<5}  {'NetSR':>7}  {'GrossSR':>7}  {'NetCAGR':>8}  {'MaxDD':>6}  {'Trades':>6}  {'Folds+':>6}"
    sep = "-" * len(header)
    print("\n" + sep)
    print("  ENSEMBLE GRID SEARCH — RANKED RESULTS (by Mean OOS Net Sharpe)")
    print(sep)
    print(header)
    print(sep)
    for rank, r in enumerate(ok, 1):
        n_sel = len(r.get("selectors", []))
        print(
            f"{rank:>4}  {r['label']:<40}  {n_sel:>5}  "
            f"{r['mean_net_sharpe']:>7.4f}  "
            f"{(r.get('mean_gross_sharpe') or 0):>7.4f}  "
            f"{(r.get('mean_net_cagr') or 0):>8.2f}  "
            f"{(r.get('mean_maxdd_pct') or 0):>6.2f}  "
            f"{(r.get('total_trades') or 0):>6}  "
            f"{(r.get('pct_folds_positive') or 0):>6.0f}%"
        )
    print(sep + "\n")

    # Also print failed runs
    failed = [r for r in results if r.get("status") != "ok"]
    if failed:
        print(f"  {len(failed)} run(s) failed / timed out:")
        for r in failed:
            print(f"    {r['label']:40s}  status={r['status']}")
        print()


def _save_aggregate(results: List[Dict], started_at: str) -> None:
    """Save all results and a summary to AGGREGATE_FILE."""
    ok = [r for r in results if r.get("status") == "ok" and r.get("mean_net_sharpe") is not None]
    ok.sort(key=lambda r: r["mean_net_sharpe"], reverse=True)

    payload = {
        "experiment": "E4_ensemble_grid_search",
        "started_at": started_at,
        "finished_at": datetime.now().isoformat(),
        "total_runs": len(results),
        "successful_runs": len(ok),
        "best_config": ok[0] if ok else None,
        "top_10": ok[:10],
        "all_results": results,   # full detail for downstream analysis
    }
    with open(AGGREGATE_FILE, "w") as fh:
        json.dump(payload, fh, indent=2, default=str)
    log.info(f"Aggregate results saved -> {AGGREGATE_FILE}")


# ---------------------------------------------------------------------------
# Resume support
# ---------------------------------------------------------------------------


def _load_completed_labels() -> set:
    """Load already-completed labels from a previous partial run."""
    if not AGGREGATE_FILE.exists():
        return set()
    try:
        with open(AGGREGATE_FILE) as fh:
            data = json.load(fh)
        return {r["label"] for r in data.get("all_results", []) if r.get("status") == "ok"}
    except Exception:
        return set()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="E4 Ensemble Grid Search — exhaustive Stage-1 weight space exploration",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--no-triples", action="store_true",
                        help="Skip E4.W3 triple combinations (only standalone + pairwise).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print planned runs without executing any.")
    parser.add_argument("--resume", action="store_true",
                        help="Skip combinations already present in AGGREGATE_FILE.")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of parallel subprocess workers. "
                             "Use 1 for sequential (safest), >1 for parallelism on multi-core machines. "
                             "On Kalpana, prefer SLURM array jobs instead.")
    parser.add_argument("--s2", type=str, default="ou_only",
                        choices=["all", "no_ml", "ou_only"],
                        help="Stage 2 signal config for all runs.")
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    started_at = datetime.now().isoformat()
    combos = generate_combinations(include_triples=not args.no_triples)

    completed = _load_completed_labels() if args.resume else set()
    if completed:
        log.info(f"[Resume] Skipping {len(completed)} already-completed runs.")

    pending = [(lbl, sel) for lbl, sel in combos if lbl not in completed]
    log.info(
        f"Ensemble grid search: {len(combos)} total combos, "
        f"{len(pending)} to run "
        f"({'triples included' if not args.no_triples else 'triples skipped'})"
    )

    # Load any pre-existing results if resuming
    existing_results: List[Dict] = []
    if args.resume and AGGREGATE_FILE.exists():
        try:
            with open(AGGREGATE_FILE) as fh:
                old = json.load(fh)
            existing_results = old.get("all_results", [])
        except Exception:
            pass

    results: List[Dict] = list(existing_results)

    if args.dry_run:
        log.info("=== DRY RUN — planned combinations ===")
        for i, (lbl, sel) in enumerate(pending, 1):
            log.info(f"  {i:3d}. {lbl:40s}  selectors={sel}")
        log.info(f"Total: {len(pending)} runs")
        return

    t_total = time.time()

    if args.workers == 1:
        # Sequential execution — simplest, easiest to debug
        for i, (lbl, sel) in enumerate(pending, 1):
            log.info(f"\n[{i}/{len(pending)}] Starting: {lbl}")
            r = _run_single(lbl, sel, s2=args.s2, top_k=args.top_k, dry_run=False)
            results.append(r)
            # Checkpoint after each run so resume works on interruption
            _save_aggregate(results, started_at)
    else:
        # Parallel execution
        log.info(f"Running {len(pending)} combinations with {args.workers} parallel workers")
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(_run_single, lbl, sel, args.s2, args.top_k, False): lbl
                for lbl, sel in pending
            }
            done_count = 0
            for fut in as_completed(futures):
                lbl = futures[fut]
                try:
                    r = fut.result()
                except Exception as exc:
                    r = {"label": lbl, "status": "exception", "error": str(exc)}
                results.append(r)
                done_count += 1
                log.info(f"  Progress: {done_count}/{len(pending)} completed")
                _save_aggregate(results, started_at)

    total_elapsed = time.time() - t_total
    log.info(f"\nAll runs completed in {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")

    _print_ranked_table(results)
    _save_aggregate(results, started_at)


if __name__ == "__main__":
    main()
