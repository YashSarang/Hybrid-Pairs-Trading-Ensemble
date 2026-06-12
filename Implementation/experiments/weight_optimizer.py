"""
experiments/weight_optimizer.py
================================
E4.O — Scipy-Optimised Stage-1 Ensemble Weights
-------------------------------------------------
Searches for the weight vector w* on the Stage-1 selector simplex that
maximises OOS Net Sharpe, evaluated on a HELD-OUT training-period subset
to avoid lookahead into the true OOS test folds.

Protocol (purged cross-validation):
    Weight search  →  Folds 1–4  (optimiser "training" folds)
    Weight validation  →  Fold 5  (held-out, never used during search)
    Final OOS test  →  Fold 6  (untouched; reported as primary result)

This is the academically rigorous separation required when learning
parameters from a held-out fold.

Optimisation method:
    scipy.optimize.minimize with SLSQP (sequential quadratic programming).
    Constraints: sum(w)=1, w_k >= 0  (probability simplex).
    Multiple random restarts (N=200 default) to escape local minima.

Usage
-----
    # Optimise over all 8 selectors:
    python experiments/weight_optimizer.py

    # Optimise over specific selectors only (recommended: top performers from E4.S):
    python experiments/weight_optimizer.py --selectors Correlation LSTM Transformer

    # Custom number of restarts:
    python experiments/weight_optimizer.py --restarts 50

    # S2 signal config:
    python experiments/weight_optimizer.py --s2 ou_only

Outputs
-------
    experiments/results/grid_search/weight_optimizer_results.json
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = Path(__file__).parent / "results" / "grid_search"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = RESULTS_DIR / "weight_optimizer_results.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

PYTHON = sys.executable

ALL_SELECTORS = ["Correlation", "Distance", "Cointegration", "Combined",
                 "ML", "LSTM", "Transformer", "GNN"]

# Folds to use for weight optimisation (held-out: Fold 5 = index 4, Fold 6 = index 5)
OPT_FOLD_INDICES = [0, 1, 2, 3]   # Folds 1–4
VAL_FOLD_INDEX   = 4               # Fold 5
TEST_FOLD_INDEX  = 5               # Fold 6

# ---------------------------------------------------------------------------
# Walk-forward runner (single call)
# ---------------------------------------------------------------------------


def _run_wfv(weights: Dict[str, float], s2: str, top_k: int) -> Optional[Dict]:
    """Run walk_forward.py and return the parsed JSON aggregate, or None on failure."""
    weights_json = json.dumps(weights)
    cmd = [
        PYTHON,
        str(ROOT / "experiments" / "walk_forward.py"),
        "--s1-weights", weights_json,
        "--s2", s2,
        "--top-k", str(top_k),
    ]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, cwd=str(ROOT), timeout=3600
        )
        if proc.returncode != 0:
            return None

        # Find the freshest result JSON
        result_files = sorted(
            (ROOT / "experiments" / "results").glob("walk_forward_*.json"),
            key=lambda f: f.stat().st_mtime, reverse=True
        )
        if not result_files:
            return None

        with open(result_files[0]) as fh:
            return json.load(fh)

    except Exception as exc:
        log.warning(f"  WFV run failed: {exc}")
        return None


def _sharpe_from_folds(data: Dict, fold_indices: List[int]) -> float:
    """Extract mean Net Sharpe from specific fold indices."""
    folds = data.get("folds", [])
    sharpes = []
    for i in fold_indices:
        if i < len(folds):
            ns = folds[i].get("net_sharpe")
            if ns is not None and np.isfinite(ns):
                sharpes.append(ns)
    return float(np.mean(sharpes)) if sharpes else -99.0


# ---------------------------------------------------------------------------
# Scipy optimiser
# ---------------------------------------------------------------------------


def _optimise(
    selectors: List[str],
    s2: str,
    top_k: int,
    n_restarts: int,
    opt_fold_indices: List[int],
    rng: np.random.Generator,
) -> Tuple[np.ndarray, float]:
    """Run SLSQP with n_restarts random initialisations.

    Returns (best_weights_array, best_objective) where objective = mean Net Sharpe
    over opt_fold_indices.
    """
    try:
        from scipy.optimize import minimize
    except ImportError:
        log.error("scipy not available. Install scipy to use weight_optimizer.py")
        sys.exit(1)

    n = len(selectors)

    def _objective(w: np.ndarray) -> float:
        """Negative mean Net Sharpe (SLSQP minimises)."""
        w_clipped = np.clip(w, 0, 1)
        w_norm = w_clipped / (w_clipped.sum() + 1e-9)
        weights_dict = {k: 0.0 for k in ALL_SELECTORS}
        for k, v in zip(selectors, w_norm):
            weights_dict[k] = float(v)

        data = _run_wfv(weights_dict, s2, top_k)
        if data is None:
            return 99.0   # large penalty
        sr = _sharpe_from_folds(data, opt_fold_indices)
        log.debug(f"  w={[round(float(x),3) for x in w_norm]}  SR={sr:.4f}")
        return -sr   # minimise negative SR

    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    bounds = [(0.0, 1.0)] * n

    best_w = np.ones(n) / n     # default: equal weight
    best_obj = 99.0

    for restart in range(n_restarts):
        # Random Dirichlet initialisation (uniform on simplex)
        w0 = rng.dirichlet(np.ones(n))
        log.info(f"  Restart {restart+1}/{n_restarts}: w0={[round(float(x),3) for x in w0]}")

        try:
            res = minimize(
                _objective,
                w0,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"maxiter": 50, "ftol": 1e-4},
            )
            if res.fun < best_obj:
                best_obj = res.fun
                best_w = res.x
                log.info(f"    New best OBJ={-best_obj:.4f}  w={[round(float(x),3) for x in best_w]}")
        except Exception as exc:
            log.warning(f"  Restart {restart+1} failed: {exc}")

    # Normalise best weights
    best_w = np.clip(best_w, 0, 1)
    best_w = best_w / (best_w.sum() + 1e-9)
    return best_w, -best_obj


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="E4.O — Scipy-optimised Stage-1 ensemble weights",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--selectors", nargs="+", default=ALL_SELECTORS,
        choices=ALL_SELECTORS,
        help="Selectors to optimise over. Default: all 8. Recommend passing "
             "only selectors with positive standalone Net SR from E4.S results."
    )
    parser.add_argument("--restarts", type=int, default=200,
                        help="Number of random restarts for SLSQP.")
    parser.add_argument("--s2", type=str, default="ou_only",
                        choices=["all", "no_ml", "ou_only"])
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    started_at = datetime.now().isoformat()

    log.info("=" * 64)
    log.info("E4.O — Scipy Weight Optimiser")
    log.info(f"  Selectors:  {args.selectors}")
    log.info(f"  Opt folds:  {[f'Fold{i+1}' for i in OPT_FOLD_INDICES]}")
    log.info(f"  Val fold:   Fold{VAL_FOLD_INDEX+1}")
    log.info(f"  Test fold:  Fold{TEST_FOLD_INDEX+1}")
    log.info(f"  Restarts:   {args.restarts}")
    log.info("=" * 64)

    # ---- Step 1: Optimise on Folds 1-4 ----
    log.info("\n[Step 1] Optimising weights on Folds 1–4 ...")
    best_w, best_sr = _optimise(
        selectors=args.selectors,
        s2=args.s2,
        top_k=args.top_k,
        n_restarts=args.restarts,
        opt_fold_indices=OPT_FOLD_INDICES,
        rng=rng,
    )

    opt_weights = {s: round(float(w), 6) for s, w in zip(args.selectors, best_w)}
    # Fill non-selected selectors with 0
    full_opt_weights = {k: 0.0 for k in ALL_SELECTORS}
    full_opt_weights.update(opt_weights)

    log.info(f"\nOptimised weights (Folds 1-4 SR={best_sr:.4f}):")
    for s, w in sorted(opt_weights.items(), key=lambda x: -x[1]):
        if w > 0.005:
            log.info(f"  {s:20s}: {w:.4f}")

    # ---- Step 2: Validate on Fold 5 (held-out) ----
    log.info("\n[Step 2] Validating optimised weights on Fold 5 (held-out) ...")
    val_data = _run_wfv(full_opt_weights, args.s2, args.top_k)
    val_sr = _sharpe_from_folds(val_data, [VAL_FOLD_INDEX]) if val_data else None

    # ---- Step 3: Equal-weight baseline for comparison ----
    log.info("\n[Step 3] Running equal-weight baseline for comparison ...")
    n = len(args.selectors)
    eq_weights = {k: 0.0 for k in ALL_SELECTORS}
    for s in args.selectors:
        eq_weights[s] = round(1.0 / n, 6)
    eq_data = _run_wfv(eq_weights, args.s2, args.top_k)
    eq_opt_sr   = _sharpe_from_folds(eq_data, OPT_FOLD_INDICES) if eq_data else None
    eq_val_sr   = _sharpe_from_folds(eq_data, [VAL_FOLD_INDEX]) if eq_data else None
    eq_test_sr  = _sharpe_from_folds(eq_data, [TEST_FOLD_INDEX]) if eq_data else None

    # ---- Step 4: True OOS test on Fold 6 (optimised) ----
    log.info("\n[Step 4] OOS test on Fold 6 (never touched during optimisation) ...")
    test_data = _run_wfv(full_opt_weights, args.s2, args.top_k)
    test_sr = _sharpe_from_folds(test_data, [TEST_FOLD_INDEX]) if test_data else None

    # ---- Summary ----
    print("\n" + "=" * 64)
    print("  WEIGHT OPTIMISER RESULTS")
    print("=" * 64)
    print(f"  Selectors optimised:       {args.selectors}")
    print(f"\n  {'Metric':<35}  {'Optimised':>10}  {'Equal-Weight':>12}")
    print(f"  {'-'*60}")
    print(f"  {'Folds 1-4 train SR':<35}  {best_sr:>10.4f}  {(eq_opt_sr or 0):>12.4f}")
    print(f"  {'Fold 5 val SR (held-out)':<35}  {(val_sr or 0):>10.4f}  {(eq_val_sr or 0):>12.4f}")
    print(f"  {'Fold 6 OOS test SR':<35}  {(test_sr or 0):>10.4f}  {(eq_test_sr or 0):>12.4f}")
    print(f"\n  Optimised weight vector:")
    for s, w in sorted(opt_weights.items(), key=lambda x: -x[1]):
        if w > 0.001:
            print(f"    {s:20s}: {w:.4f}  {'█' * int(w * 40)}")
    print("=" * 64 + "\n")

    # ---- Save results ----
    payload = {
        "experiment": "E4.O_weight_optimizer",
        "started_at": started_at,
        "finished_at": datetime.now().isoformat(),
        "selectors": args.selectors,
        "n_restarts": args.restarts,
        "s2": args.s2,
        "top_k": args.top_k,
        "seed": args.seed,
        "protocol": {
            "opt_folds": [f"Fold{i+1}" for i in OPT_FOLD_INDICES],
            "val_fold": f"Fold{VAL_FOLD_INDEX+1}",
            "test_fold": f"Fold{TEST_FOLD_INDEX+1}",
        },
        "optimised_weights": opt_weights,
        "full_weights": full_opt_weights,
        "equal_weights": {k: round(v, 6) for k, v in eq_weights.items() if v > 0},
        "performance": {
            "optimised": {
                "opt_folds_sr":  round(best_sr, 4),
                "val_fold5_sr":  round(val_sr, 4) if val_sr else None,
                "test_fold6_sr": round(test_sr, 4) if test_sr else None,
            },
            "equal_weight": {
                "opt_folds_sr":  round(eq_opt_sr, 4) if eq_opt_sr else None,
                "val_fold5_sr":  round(eq_val_sr, 4) if eq_val_sr else None,
                "test_fold6_sr": round(eq_test_sr, 4) if eq_test_sr else None,
            },
        },
        "interpretation": (
            "If optimised Fold 6 OOS SR ≈ equal-weight Fold 6 SR, "
            "weight optimisation provides no significant benefit beyond heuristic equal-weighting. "
            "This supports the parsimony principle. "
            "If optimised significantly outperforms, refine claim in §5.2.4."
        ),
    }
    with open(OUT_FILE, "w") as fh:
        json.dump(payload, fh, indent=2, default=str)
    log.info(f"Results saved -> {OUT_FILE}")


if __name__ == "__main__":
    main()
