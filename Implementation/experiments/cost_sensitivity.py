# experiments/cost_sensitivity.py
"""
Cost Sensitivity Analysis for Paper 1
=====================================
Analyzes the impact of transaction costs (±5 bps, ±10 bps, etc.) on strategy
performance metrics for the three primary configurations:
- stat_only + ou_only
- stat_ml + ou_only
- full + ou_only

Reads the canonical OOS returns and trade logs from the walk_forward JSON files
and applies various cost fractions to compute Net Sharpe, Net Return, and Max Drawdown.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

def _metrics_from_pnl(pnl: pd.Series, capital: float, periods_per_year: int = 252):
    if pnl.empty:
        return {"Return": 0.0, "Sharpe": 0.0, "Volatility": 0.0, "MaxDrawdown": 0.0}
    eq = capital + pnl.cumsum()
    ret = pnl / max(capital, 1.0)
    max_eq = eq.cummax().replace(0, np.nan)
    dd = (max_eq - eq) / max_eq
    r = ret.replace([np.inf, -np.inf], np.nan).dropna()
    vol = float(r.std(ddof=0) * math.sqrt(periods_per_year)) if len(r) else 0.0
    sharpe = (
        0.0 if len(r) < 2 or r.std(ddof=0) == 0
        else float((r.mean() / r.std(ddof=0)) * math.sqrt(periods_per_year))
    )
    # Annualized return
    ann_ret = float(r.mean() * periods_per_year) if len(r) else 0.0
    max_dd = float(dd.max()) if not dd.isna().all() else 0.0
    return {
        "Return": ann_ret * 100,
        "Sharpe": sharpe,
        "Volatility": vol * 100,
        "MaxDrawdown": max_dd * 100,
    }

def analyze_cost_sensitivity(wfv_path: Path, config_name: str):
    with open(wfv_path, "r") as f:
        data = json.load(f)
        
    capital = data.get("capital", 1000000.0)
    
    # Reconstruct daily gross pnl
    cum_gross = pd.Series(data['aggregate']['cumulative_gross'])
    cum_gross.index = pd.to_datetime(cum_gross.index)
    daily_gross = cum_gross.diff().fillna(cum_gross - capital)
    
    # We also need the daily turnover.
    # If the JSON doesn't contain a direct daily turnover time series, we can estimate it
    # from the trade list or backtest files. Let's look at the walk_forward JSON structure.
    # If daily net and daily gross are available, turnover is proportional to (daily_gross - daily_net).
    # Specifically, old_cost_bps = 16.28 bps.
    # old_cost_frac = 16.28 / 10000 = 0.001628
    # cost_incurred = daily_gross - daily_net
    # Therefore, notional_turnover = cost_incurred / old_cost_frac
    
    cum_net = pd.Series(data['aggregate']['cumulative_net'])
    cum_net.index = pd.to_datetime(cum_net.index)
    daily_net = cum_net.diff().fillna(cum_net - capital)
    
    cost_incurred = daily_gross - daily_net
    
    # Baseline cost fraction
    baseline_bps = 16.28
    baseline_frac = baseline_bps / 10000.0
    
    # Avoid division by zero if no trades occurred
    if baseline_frac == 0:
        notional_turnover = pd.Series(0.0, index=daily_gross.index)
    else:
        notional_turnover = cost_incurred / baseline_frac
        
    cost_scenarios = [0.0, 5.0, 11.28, 16.28, 21.28, 30.0, 50.0]
    results = []
    
    for bps in cost_scenarios:
        cost_frac = bps / 10000.0
        scenario_costs = notional_turnover * cost_frac
        scenario_net_pnl = daily_gross - scenario_costs
        
        metrics = _metrics_from_pnl(scenario_net_pnl, capital, 252)
        results.append({
            "Bps": bps,
            "Net Sharpe": round(metrics["Sharpe"], 3),
            "Net CAGR %": round(metrics["Return"], 2),
            "MaxDD %": round(metrics["MaxDrawdown"], 2),
        })
        
    return results

if __name__ == "__main__":
    results_dir = Path("Implementation/experiments/results")
    
    # Target files
    configs = {
        "stat_only + ou_only": results_dir / "walk_forward_20260604_102006.json",
        "stat_ml + ou_only": results_dir / "walk_forward_20260604_071048.json",
        "full + ou_only": results_dir / "walk_forward_20260604_093332.json"
    }
    
    final_output = {}
    for name, path in configs.items():
        if path.exists():
            print(f"Analyzing {name}...")
            final_output[name] = analyze_cost_sensitivity(path, name)
        else:
            print(f"Warning: {path} not found.")
            
    # Save the output to sensitivity_analysis.json
    output_path = results_dir / "cost_sensitivity_analysis.json"
    with open(output_path, "w") as f:
        json.dump(final_output, f, indent=2)
        
    # Print Markdown table format
    for name, res in final_output.items():
        print(f"\n### Cost Sensitivity: {name}")
        print("| Round-Trip Cost (bps) | Net Sharpe | Net CAGR % | MaxDD % |")
        print("|---|---|---|---|")
        for r in res:
            print(f"| {r['Bps']:.2f} bps | {r['Net Sharpe']:.3f} | {r['Net CAGR %']:.2f}% | {r['MaxDD %']:.2f}% |")
