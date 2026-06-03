#!/usr/bin/env python3
"""
compare_markets.py

Aggregate WFV results across markets and generate comparative tables.

Usage:
    python compare_markets.py --markets india us brazil uk
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict
import pandas as pd
import numpy as np

def load_latest_results(market: str) -> Dict:
    """Load most recent WFV result JSON for a market."""
    results_dir = Path(__file__).parent.parent / "results" / market
    
    if not results_dir.exists():
        raise FileNotFoundError(f"No results directory for {market}: {results_dir}")
    
    # Find most recent JSON
    json_files = list(results_dir.glob("wfv_*.json"))
    
    if not json_files:
        raise FileNotFoundError(f"No WFV results found in {results_dir}")
    
    latest = max(json_files, key=lambda p: p.stat().st_mtime)
    
    with open(latest, 'r') as f:
        return json.load(f)

def build_selector_ranking_table(results: Dict[str, Dict]) -> pd.DataFrame:
    """
    Build cross-market selector ranking table.
    
    Returns DataFrame:
        Rows: Selectors
        Columns: Markets (Net Sharpe values)
    """
    selector_names = [
        'correlation', 'distance', 'cointegration', 'combined',
        'ml', 'lstm', 'transformer', 'gnn'
    ]
    
    data = {}
    
    for market, res in results.items():
        # Extract per-selector performance from fold results
        # This requires selector-level isolation runs or parsing selector_scores
        # For now, use placeholder (requires ablation runs)
        data[market.upper()] = [0.0] * len(selector_names)  # Placeholder
    
    df = pd.DataFrame(data, index=selector_names)
    df.index.name = 'Selector'
    
    return df

def build_cost_sensitivity_table(results: Dict[str, Dict]) -> pd.DataFrame:
    """
    Build transaction cost sensitivity table.
    
    Columns: Market, Cost (bps), Gross Sharpe, Net Sharpe, Degradation
    """
    rows = []
    
    for market, res in results.items():
        rows.append({
            'Market': res['market'],
            'Cost_bps': res['transaction_cost_bps'],
            'Gross_Sharpe': res['avg_gross_sharpe'],
            'Net_Sharpe': res['avg_net_sharpe'],
            'Degradation': res['avg_gross_sharpe'] - res['avg_net_sharpe'],
            'Degradation_pct': (
                (res['avg_gross_sharpe'] - res['avg_net_sharpe']) / res['avg_gross_sharpe'] * 100
                if res['avg_gross_sharpe'] != 0 else 0
            )
        })
    
    df = pd.DataFrame(rows)
    df = df.sort_values('Cost_bps')
    
    return df

def build_regime_performance_table(results: Dict[str, Dict]) -> pd.DataFrame:
    """
    Build regime (fold-level) performance table.
    
    Rows: Folds
    Columns: Markets (Net Sharpe per fold)
    """
    # Determine max folds
    max_folds = max(len(res['folds']) for res in results.values())
    
    data = {}
    
    for market, res in results.items():
        sharpes = []
        for fold in res['folds']:
            if 'metrics' in fold:
                sharpes.append(fold['metrics']['net_sharpe'])
            else:
                sharpes.append(0.0)
        
        # Pad to max_folds
        while len(sharpes) < max_folds:
            sharpes.append(np.nan)
        
        data[market.upper()] = sharpes
    
    df = pd.DataFrame(data, index=[f"Fold {i+1}" for i in range(max_folds)])
    df.index.name = 'Fold'
    
    return df

def build_parsimony_table(results: Dict[str, Dict]) -> pd.DataFrame:
    """
    Build parsimony principle test table.
    
    Placeholder: requires 2-selector vs 8-selector ablation runs.
    """
    rows = []
    
    for market, res in results.items():
        rows.append({
            'Market': res['market'],
            '8-Selector_Net_Sharpe': res['avg_net_sharpe'],  # Full ensemble
            '2-Selector_Net_Sharpe': np.nan,  # Requires lstm+corr ablation
            'Best_Single_Net_Sharpe': np.nan,  # Requires single-selector runs
            'Parsimony_Wins': np.nan
        })
    
    df = pd.DataFrame(rows)
    
    return df

def generate_summary_stats(results: Dict[str, Dict]) -> pd.DataFrame:
    """Overall summary statistics per market."""
    rows = []
    
    for market, res in results.items():
        rows.append({
            'Market': res['market'],
            'Code': res['market_code'],
            'Folds': res['n_folds'],
            'Avg_Net_Sharpe': f"{res['avg_net_sharpe']:.3f} ± {res['std_net_sharpe']:.3f}",
            'Avg_Gross_Sharpe': f"{res['avg_gross_sharpe']:.3f}",
            'Cost_bps': f"{res['transaction_cost_bps']:.1f}",
            'Selectors': ', '.join(res['selectors'])
        })
    
    df = pd.DataFrame(rows)
    
    return df

def main():
    parser = argparse.ArgumentParser(description="Compare multi-market WFV results")
    parser.add_argument(
        '--markets',
        nargs='+',
        required=True,
        choices=['india', 'us', 'brazil', 'uk'],
        help='Markets to compare'
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Multi-Market WFV Results Comparison")
    print("=" * 70)
    print()
    
    # Load results
    results = {}
    
    for market in args.markets:
        try:
            print(f"Loading {market}...", end=" ")
            results[market] = load_latest_results(market)
            print(f"✅ {results[market]['n_folds']} folds")
        except FileNotFoundError as e:
            print(f"❌ {e}")
    
    if not results:
        print("\n❌ No results loaded. Exiting.\n")
        return
    
    print()
    
    # Generate tables
    output_dir = Path(__file__).parent.parent / "results" / "cross_market_summary"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Summary stats
    print("=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)
    summary = generate_summary_stats(results)
    print(summary.to_string(index=False))
    print()
    summary.to_csv(output_dir / "summary_stats.csv", index=False)
    print(f"💾 Saved: {output_dir / 'summary_stats.csv'}\n")
    
    # 2. Cost sensitivity
    print("=" * 70)
    print("TRANSACTION COST SENSITIVITY")
    print("=" * 70)
    cost_table = build_cost_sensitivity_table(results)
    print(cost_table.to_string(index=False))
    print()
    cost_table.to_csv(output_dir / "cost_sensitivity.csv", index=False)
    print(f"💾 Saved: {output_dir / 'cost_sensitivity.csv'}\n")
    
    # 3. Regime performance
    print("=" * 70)
    print("REGIME PERFORMANCE (Net Sharpe per Fold)")
    print("=" * 70)
    regime_table = build_regime_performance_table(results)
    print(regime_table.to_string())
    print()
    regime_table.to_csv(output_dir / "regime_performance.csv")
    print(f"💾 Saved: {output_dir / 'regime_performance.csv'}\n")
    
    # 4. Parsimony test (placeholder)
    print("=" * 70)
    print("PARSIMONY PRINCIPLE TEST")
    print("=" * 70)
    print("⚠️  Requires ablation runs:")
    print("   - 8-selector ensemble (baseline)")
    print("   - 2-selector (lstm+correlation)")
    print("   - Single-selector runs (8 runs per market)")
    print()
    parsimony_table = build_parsimony_table(results)
    print(parsimony_table.to_string(index=False))
    print()
    parsimony_table.to_csv(output_dir / "parsimony_test.csv", index=False)
    print(f"💾 Saved: {output_dir / 'parsimony_test.csv'}\n")
    
    # 5. Selector ranking (placeholder)
    print("=" * 70)
    print("SELECTOR RANKING (Net Sharpe)")
    print("=" * 70)
    print("⚠️  Requires per-selector ablation runs")
    print()
    
    print("=" * 70)
    print("DONE")
    print("=" * 70)
    print(f"All tables saved to: {output_dir}")
    print()

if __name__ == "__main__":
    main()
