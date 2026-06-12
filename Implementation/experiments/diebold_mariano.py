# experiments/diebold_mariano.py
"""
Diebold-Mariano Pairwise Tests for Paper 1
==========================================
Compares the predictive accuracy and return streams of the full hybrid ensemble
(wfv_file: walk_forward_20260604_093332.json) against individual selectors:
- stat_only (walk_forward_20260604_102006.json)
- stat_ml (walk_forward_20260604_071048.json)

Computes the Diebold-Mariano test statistic for a given forecast horizon (default: 30,
matching the min_hold constraint) to evaluate whether the difference in net OOS return
MSE or absolute loss is statistically significant.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

def diebold_mariano_test(
    realized: pd.Series,
    forecast_1: pd.Series,
    forecast_2: pd.Series,
    h: int = 30,
    loss_type: str = "mse"
) -> dict:
    """Diebold-Mariano test for predictive accuracy comparison.
    
    H0: both models have same accuracy (expected loss difference = 0).
    H1: Model 2 is more accurate than Model 1 (or different).
    
    forecast_1: Daily return series of Model 1 (e.g. baseline stat_only)
    forecast_2: Daily return series of Model 2 (e.g. full hybrid ensemble)
    realized: Target series (e.g. realized returns or simply treating it as 0 baseline
              or comparing returns differences directly).
              If comparing daily returns, we can define loss of forecasting a benchmark:
              e.g. squared tracking error vs benchmark, or just direct PnL variance.
              Alternatively, we can compare returns streams using mean squared errors or 
              the standard DM setup for return difference significance.
    """
    # Align indexes
    df = pd.concat([forecast_1, forecast_2, realized], axis=1, join="inner").dropna()
    f1 = df.iloc[:, 0].values
    f2 = df.iloc[:, 1].values
    y = df.iloc[:, 2].values
    T = len(y)
    
    if loss_type == "mse":
        d = (y - f1) ** 2 - (y - f2) ** 2
    elif loss_type == "mae":
        d = np.abs(y - f1) - np.abs(y - f2)
    elif loss_type == "return":
        # Compare return streams directly (Model 2 return - Model 1 return)
        d = f2 - f1
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
        
    d_mean = np.mean(d)
    
    # Autocovariance up to lag h-1 (accounting for min_hold overlap)
    gamma_0 = np.var(d, ddof=0)
    var_d = gamma_0
    for lag in range(1, h):
        if T - lag > 0:
            cov = np.cov(d[lag:], d[:-lag], ddof=0)[0, 1]
            var_d += 2.0 * cov
            
    # DM statistic
    dm_stat = float(d_mean / np.sqrt(max(var_d / T, 1e-12)))
    
    # One-sided and two-sided p-values
    from scipy.stats import norm
    p_two_sided = float(2 * (1 - norm.cdf(np.abs(dm_stat))))
    # One-sided (H1: Model 2 is better than Model 1, i.e. d_mean > 0 for return type, or d_mean > 0 for loss reduction)
    if loss_type == "return":
        p_one_sided = float(1 - norm.cdf(dm_stat))
    else:
        p_one_sided = float(1 - norm.cdf(dm_stat)) if d_mean > 0 else float(norm.cdf(dm_stat))
        
    return {
        "dm_statistic": round(dm_stat, 4),
        "p_value_two_sided": round(p_two_sided, 4),
        "p_value_one_sided": round(p_one_sided, 4),
        "mean_difference": round(d_mean, 6),
        "n_obs": T
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h", type=int, default=30, help="Forecast horizon (min_hold=30)")
    args = parser.parse_args()
    
    results_dir = Path("Implementation/experiments/results")
    
    # Load daily returns for the three core runs
    def get_returns(file_name: str) -> pd.Series:
        path = results_dir / file_name
        with open(path, "r") as f:
            data = json.load(f)
        capital = data.get("capital", 1000000.0)
        cum_net = pd.Series(data['aggregate']['cumulative_net'])
        cum_net.index = pd.to_datetime(cum_net.index)
        daily_net = cum_net.diff().fillna(cum_net - capital)
        return daily_net / capital  # return fraction
        
    print("Loading daily return streams...")
    ret_stat = get_returns("walk_forward_20260604_102006.json")
    ret_ml = get_returns("walk_forward_20260604_071048.json")
    ret_full = get_returns("walk_forward_20260604_093332.json")
    
    # Realized target for tracking error / loss.
    # In standard finance literature comparing returns streams directly, we set realized=0
    # to evaluate whether returns differ significantly (H1: full hybrid return > stat_only).
    realized_zero = pd.Series(0.0, index=ret_full.index)
    
    # 1. Full vs Stat_only
    print(f"\n--- Diebold-Mariano pairwise test: full vs stat_only (h={args.h}) ---")
    dm_full_vs_stat = diebold_mariano_test(realized_zero, ret_stat, ret_full, h=args.h, loss_type="return")
    print(json.dumps(dm_full_vs_stat, indent=2))
    
    # 2. Full vs Stat_ml
    print(f"\n--- Diebold-Mariano pairwise test: full vs stat_ml (h={args.h}) ---")
    dm_full_vs_ml = diebold_mariano_test(realized_zero, ret_ml, ret_full, h=args.h, loss_type="return")
    print(json.dumps(dm_full_vs_ml, indent=2))
    
    # 3. Stat_only vs Stat_ml
    print(f"\n--- Diebold-Mariano pairwise test: stat_only vs stat_ml (h={args.h}) ---")
    dm_stat_vs_ml = diebold_mariano_test(realized_zero, ret_ml, ret_stat, h=args.h, loss_type="return")
    print(json.dumps(dm_stat_vs_ml, indent=2))
    
    # Save results
    output_path = results_dir / "diebold_mariano_results.json"
    with open(output_path, "w") as f:
        json.dump({
            "full_vs_stat_only": dm_full_vs_stat,
            "full_vs_stat_ml": dm_full_vs_ml,
            "stat_only_vs_stat_ml": dm_stat_vs_ml,
            "horizon": args.h
        }, f, indent=2)
    print(f"\nResults saved to {output_path.name}")

if __name__ == "__main__":
    main()
