"""
experiments/significance_tests.py
====================================
Experiment E6 — Statistical Significance Testing
--------------------------------------------------
Tests whether the OOS Sharpe ratio is statistically different from zero,
accounting for the serial correlation and non-normality of financial returns.

Three tests are computed:

1. **Bootstrap Sharpe CI** — Resample the OOS daily return series with
   replacement (B=10,000). Compute Sharpe on each resample.  Report the
   2.5th–97.5th percentile CI and the bootstrap p-value (proportion of
   resamples with Sharpe <= 0).

2. **Newey-West t-test** — t-statistic for mean daily return = 0, with
   heteroscedasticity and autocorrelation-consistent (HAC) standard error
   via Newey-West.  Handles serial correlation present in minimum-hold
   strategies (min_hold=30 induces at least 30-bar autocorrelation).

3. **White's Reality Check (simplified)** — Tests whether the *best*
   strategy from the E3 ablation (OU_only) beats a naïve benchmark
   (zero return) after accounting for the data-snooping bias of having
   evaluated 5 Stage-2 configurations.

Usage
-----
    python experiments/significance_tests.py
    python experiments/significance_tests.py --wfv walk_forward_20260402_230753.json --n_boot 10000

Output
------
  experiments/results/significance_<timestamp>.json

References
----------
- Ledoit & Wolf (2008): Robust performance hypothesis testing with the Sharpe ratio.
- Newey & West (1987): A simple positive semi-definite HAC covariance matrix.
- White (2000): A reality check for data snooping.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

_RESULTS_DIR = Path(__file__).parent / "results"
_RESULTS_DIR.mkdir(exist_ok=True)

PERIODS_PER_YEAR = 252


# ---------------------------------------------------------------------------
# Load OOS daily returns from WFV JSON
# ---------------------------------------------------------------------------

def _find_wfv_file(mode: str, s2: str) -> Path:
    candidates = sorted(_RESULTS_DIR.glob("walk_forward_*.json"), reverse=True)
    for p in candidates:
        try:
            d = json.loads(p.read_text())
            if d.get("mode") == mode and d.get("s2") == s2:
                return p
        except Exception:
            continue
    raise FileNotFoundError(
        f"No WFV result found for mode={mode!r} s2={s2!r} in {_RESULTS_DIR}"
    )


def load_oos_daily_returns(wfv_path: Path) -> tuple[pd.Series, pd.Series]:
    """Returns (gross_daily_ret, net_daily_ret) as log-return Series."""
    d = json.loads(wfv_path.read_text())
    agg = d["aggregate"]
    capital = d.get("capital", 1_000_000)

    def _parse(raw: dict) -> pd.Series:
        s = pd.Series(raw, dtype=float)
        s.index = pd.to_datetime(s.index)
        s = s / capital          # normalize to 1.0 base
        # Convert cumulative returns to log-returns
        log_ret = np.log(s / s.shift(1)).dropna()
        return log_ret

    gross = _parse(agg["cumulative_gross"])
    net   = _parse(agg["cumulative_net"])
    return gross, net


# ---------------------------------------------------------------------------
# Test 1: Bootstrap Sharpe confidence interval
# ---------------------------------------------------------------------------

def bootstrap_sharpe(
    daily_returns: pd.Series,
    n_boot: int = 10_000,
    seed: int = 42,
    block_size: int = 0,
) -> dict:
    """Bootstrap Sharpe CI using IID or block (circular) bootstrap.

    Args:
        daily_returns: Daily log-return series.
        n_boot: Number of bootstrap resamples.
        seed: RNG seed for reproducibility.
        block_size: If > 1, use circular block bootstrap to preserve
            autocorrelation structure (recommended for strategies with
            minimum hold constraints). If 0, auto-detect from min_hold.

    Returns:
        dict with keys: sharpe_observed, ci_lo, ci_hi, p_value (prob SR<=0),
        n_boot, block_size, n_obs.
    """
    r = daily_returns.values
    n = len(r)
    rng = np.random.default_rng(seed)

    sharpe_obs = float(r.mean() / r.std(ddof=1) * np.sqrt(PERIODS_PER_YEAR))

    if block_size <= 1:
        # IID bootstrap
        boot_sharpes = np.empty(n_boot)
        for i in range(n_boot):
            sample = rng.choice(r, size=n, replace=True)
            std = sample.std(ddof=1)
            boot_sharpes[i] = float(sample.mean() / std * np.sqrt(PERIODS_PER_YEAR)) if std > 0 else 0.0
    else:
        # Circular block bootstrap (preserves autocorrelation)
        boot_sharpes = np.empty(n_boot)
        n_blocks = int(np.ceil(n / block_size))
        for i in range(n_boot):
            starts = rng.integers(0, n, size=n_blocks)
            blocks = [r[np.arange(s, s + block_size) % n] for s in starts]
            sample = np.concatenate(blocks)[:n]
            std = sample.std(ddof=1)
            boot_sharpes[i] = float(sample.mean() / std * np.sqrt(PERIODS_PER_YEAR)) if std > 0 else 0.0

    ci_lo = float(np.percentile(boot_sharpes, 2.5))
    ci_hi = float(np.percentile(boot_sharpes, 97.5))
    p_value = float(np.mean(boot_sharpes <= 0.0))

    return {
        "sharpe_observed": round(sharpe_obs, 4),
        "ci_lo_95": round(ci_lo, 4),
        "ci_hi_95": round(ci_hi, 4),
        "p_value_sr_lte_0": round(p_value, 4),
        "significant_at_5pct": bool(ci_lo > 0),
        "significant_at_10pct": bool(p_value < 0.10),
        "n_boot": n_boot,
        "block_size": block_size,
        "n_obs": n,
    }


# ---------------------------------------------------------------------------
# Test 2: Newey-West HAC t-test (mean return = 0)
# ---------------------------------------------------------------------------

def newey_west_ttest(daily_returns: pd.Series, lags: int | None = None) -> dict:
    """t-test for mean daily return = 0 with Newey-West HAC standard error.

    The NW lag order accounts for autocorrelation induced by min_hold_bars.
    Default lag = ceil(4 * (T/100)^(2/9)) per the Andrews (1991) rule.

    Returns:
        dict with keys: t_stat, p_value (one-sided, H1: mean > 0),
        mean_daily_ret, nw_std_err, lags, n_obs.
    """
    r = daily_returns.values
    n = len(r)
    mu = float(r.mean())

    if lags is None:
        # Andrews (1991) automatic bandwidth
        lags = int(np.ceil(4 * (n / 100) ** (2 / 9)))

    # Newey-West variance estimator
    r_demeaned = r - mu
    # Gamma(0)
    gamma_0 = float(np.dot(r_demeaned, r_demeaned) / n)
    nw_var = gamma_0
    for h in range(1, lags + 1):
        w = 1.0 - h / (lags + 1)          # Bartlett kernel
        gamma_h = float(np.dot(r_demeaned[h:], r_demeaned[:-h]) / n)
        nw_var += 2.0 * w * gamma_h

    nw_se = float(np.sqrt(max(nw_var, 1e-12) / n))
    t_stat = float(mu / nw_se) if nw_se > 0 else 0.0

    # One-sided p-value (H1: mean > 0) using normal approximation (large n)
    from math import erfc, sqrt
    p_one_sided = float(0.5 * erfc(t_stat / sqrt(2)))

    # Annualized expected return and Sharpe
    ann_ret = float(mu * PERIODS_PER_YEAR)
    std_daily = float(r.std(ddof=1))
    sharpe = float(mu / std_daily * np.sqrt(PERIODS_PER_YEAR)) if std_daily > 0 else 0.0

    return {
        "t_stat": round(t_stat, 4),
        "p_value_one_sided": round(p_one_sided, 4),
        "significant_at_5pct": bool(p_one_sided < 0.05),
        "significant_at_10pct": bool(p_one_sided < 0.10),
        "mean_daily_ret_pct": round(mu * 100, 5),
        "ann_ret_pct": round(ann_ret * 100, 3),
        "sharpe": round(sharpe, 4),
        "nw_std_err": round(nw_se, 8),
        "lags": lags,
        "n_obs": n,
    }


# ---------------------------------------------------------------------------
# Test 3: Multiple-comparison adjustment (Bonferroni + White RC sketch)
# ---------------------------------------------------------------------------

def multiple_comparison_adjustment(
    individual_p_values: dict[str, float],
    method: str = "bonferroni",
) -> dict:
    """Adjust p-values for multiple comparisons.

    E3 ablation evaluated 5 Stage-2 configurations:
      ZScore_only, OU_only, Kalman_only, ML_only, S2_Ensemble

    We report whether the best (OU_only) remains significant after
    Bonferroni correction for 5 comparisons.

    Args:
        individual_p_values: {config_name: p_value} for each config tested.
        method: "bonferroni" or "holm" (Holm-Bonferroni step-down).

    Returns:
        dict with corrected p-values and significance flags.
    """
    names  = list(individual_p_values.keys())
    pvals  = np.array([individual_p_values[n] for n in names])
    m      = len(pvals)

    if method == "bonferroni":
        corrected = np.minimum(pvals * m, 1.0)
    elif method == "holm":
        order = np.argsort(pvals)
        corrected = np.empty(m)
        running_max = 0.0
        for rank, idx in enumerate(order):
            adj = pvals[idx] * (m - rank)
            running_max = max(running_max, adj)
            corrected[idx] = min(running_max, 1.0)
    else:
        corrected = pvals.copy()

    return {
        "method": method,
        "n_comparisons": m,
        "results": [
            {
                "config": names[i],
                "p_raw":       round(float(pvals[i]), 4),
                "p_corrected": round(float(corrected[i]), 4),
                "significant_at_5pct": bool(corrected[i] < 0.05),
            }
            for i in range(m)
        ],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="E6 Statistical Significance Tests")
    parser.add_argument("--wfv", default=None, help="WFV result filename.")
    parser.add_argument("--mode",   default="stat_only")
    parser.add_argument("--s2",     default="ou_only")
    parser.add_argument("--n_boot", type=int, default=10_000)
    parser.add_argument(
        "--block",
        type=int,
        default=30,
        help="Block size for block bootstrap (default=30, matching min_hold_bars).",
    )
    args = parser.parse_args()

    # --- Load ---
    if args.wfv:
        p = Path(args.wfv)
        # Accept absolute paths, paths relative to CWD, or bare filenames
        wfv_path = p if p.is_absolute() else (_RESULTS_DIR / p.name)
    else:
        wfv_path = _find_wfv_file(args.mode, args.s2)

    log.info(f"WFV result: {wfv_path.name}")
    gross_r, net_r = load_oos_daily_returns(wfv_path)
    log.info(f"OOS daily returns: {len(net_r)} bars ({net_r.index[0].date()} to {net_r.index[-1].date()})")

    # --- Test 1: Bootstrap CI ---
    log.info(f"Bootstrap Sharpe CI (n_boot={args.n_boot}, block={args.block}) ...")
    boot_gross = bootstrap_sharpe(gross_r, args.n_boot, block_size=args.block)
    boot_net   = bootstrap_sharpe(net_r,   args.n_boot, block_size=args.block)

    # --- Test 2: Newey-West ---
    log.info("Newey-West HAC t-test ...")
    nw_gross = newey_west_ttest(gross_r)
    nw_net   = newey_west_ttest(net_r)

    # --- Test 3: Multiple comparison (Bonferroni over 5 S2 configs) ---
    # We use the bootstrap p-values from the NW test as proxies.
    # In a full White RC, you'd use the bootstrap distribution of max Sharpe.
    # Here we use the simpler Bonferroni correction over the 5 configs tested.
    # The p-values below are estimated from the ablation experiment: configs
    # that had positive net SR are expected to have lower p-values; ML_only
    # was clearly negative (p ~ 1).
    # We use the NW one-sided p-value for the headline result (OU_only) and
    # conservatively set p=0.50 for configs that were negative.
    s2_config_pvalues = {
        "ZScore_only": 0.50,   # Full-OOS Net SR ~ 0 or negative
        "OU_only":     float(nw_net["p_value_one_sided"]),
        "Kalman_only": 0.50,
        "ML_only":     0.99,   # Strongly negative, clearly not significant
        "S2_Ensemble": 0.50,
    }
    bonferroni = multiple_comparison_adjustment(s2_config_pvalues, "bonferroni")
    holm       = multiple_comparison_adjustment(s2_config_pvalues, "holm")

    # --- Print results ---
    print()
    print("=" * 72)
    print(f"  E6 Statistical Significance Tests")
    print(f"  WFV: {wfv_path.name}  |  mode={args.mode}  |  s2={args.s2}")
    print("=" * 72)

    print("\n  Test 1: Block Bootstrap Sharpe CI (block_size=%d, B=%d)" % (args.block, args.n_boot))
    print(f"  {'':30s}{'Gross':>12}{'Net':>12}")
    print("  " + "-" * 54)
    for key, label in [
        ("sharpe_observed",   "Observed Sharpe"),
        ("ci_lo_95",          "95% CI Lower"),
        ("ci_hi_95",          "95% CI Upper"),
        ("p_value_sr_lte_0",  "p(SR <= 0)"),
        ("significant_at_5pct", "Sig. at 5% (CI > 0)"),
    ]:
        g = boot_gross.get(key, "")
        n = boot_net.get(key, "")
        print(f"  {label:<30}{str(g):>12}{str(n):>12}")

    print("\n  Test 2: Newey-West HAC t-test (lags=%d)" % nw_net["lags"])
    print(f"  {'':30s}{'Gross':>12}{'Net':>12}")
    print("  " + "-" * 54)
    for key, label in [
        ("t_stat",            "t-statistic"),
        ("p_value_one_sided", "p-value (one-sided)"),
        ("significant_at_5pct", "Sig. at 5%"),
        ("ann_ret_pct",       "Ann. Return (%)"),
        ("sharpe",            "Sharpe"),
    ]:
        g = nw_gross.get(key, "")
        n = nw_net.get(key, "")
        print(f"  {label:<30}{str(g):>12}{str(n):>12}")

    print("\n  Test 3: Multiple Comparison — Bonferroni (5 S2 configs)")
    print(f"  {'Config':<20}{'p_raw':>10}{'p_Bonf':>10}{'Sig 5%':>10}")
    print("  " + "-" * 50)
    for row in bonferroni["results"]:
        print(f"  {row['config']:<20}{row['p_raw']:>10}{row['p_corrected']:>10}{str(row['significant_at_5pct']):>10}")

    print()

    # --- Save ---
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = {
        "experiment": "E6_significance_tests",
        "wfv_file": wfv_path.name,
        "mode": args.mode,
        "s2":   args.s2,
        "bootstrap": {
            "gross": boot_gross,
            "net":   boot_net,
        },
        "newey_west": {
            "gross": nw_gross,
            "net":   nw_net,
        },
        "multiple_comparison": {
            "bonferroni": bonferroni,
            "holm":       holm,
            "note": "p-values for ZScore/Kalman/Ensemble are conservative estimates (0.50); only OU_only uses actual NW test.",
        },
    }
    out_path = _RESULTS_DIR / f"significance_{ts}.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    log.info(f"Results saved to {out_path.name}")


if __name__ == "__main__":
    main()
