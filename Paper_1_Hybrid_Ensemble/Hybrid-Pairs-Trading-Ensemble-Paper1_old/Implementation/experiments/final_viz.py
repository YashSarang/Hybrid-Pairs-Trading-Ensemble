"""Final Visualization script for Thesis Figures.

Generates comparative equity curves and drawdown plots for the three core
configurations: Stat-only, Equal-Weight Ensemble, and Weighted Ensemble (Config C).
"""
import json
import os
import matplotlib.pyplot as plt
import pandas as pd
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Result files
RESULTS = {
    "Stat-Only Baseline": "experiments/results/walk_forward_20260402_230753.json",
    "Equal-Weight Ensemble": "experiments/results/walk_forward_20260406_011541.json",
    "Weighted Ensemble (Config C)": "experiments/results/walk_forward_20260506_022235.json"
}

OUTPUT_DIR = "reports/thesis_figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_equity_curve(path):
    print(f"Loading cumulative_net from {path}...")
    with open(path, 'r') as f:
        data = json.load(f)
    
    cumulative_net = data.get('aggregate', {}).get('cumulative_net', {})
    if not cumulative_net:
        print(f"  Warning: No cumulative_net found in {path}")
        return pd.Series()
        
    # Convert dictionary back to pandas Series
    # Keys are string dates 'YYYY-MM-DD', values are floats
    idx = pd.to_datetime(list(cumulative_net.keys()))
    vals = list(cumulative_net.values())
    
    equity = pd.Series(vals, index=idx)
    equity = equity.sort_index()
    
    # Adding initial capital back if it's purely PnL, but the cumulative_net in walk_forward 
    # uses pnl_net.cumsum() which is pure PnL, not capital+PnL.
    # Actually wait: walk_forward.py says:
    # `equity_net = (cfg.capital + pnl_net.cumsum())` 
    # Let me check walk_forward.py: aggregate_folds uses `pnl_net_oos`.
    # `full_pnl = pd.concat(all_pnl).sort_index()`
    # `cumulative_net = full_pnl.cumsum() + capital`
    # Let's check the first value. If it's around 1M, we don't need to add it.
    if len(equity) > 0 and equity.iloc[0] < 500000:
        # It's pure PnL
        equity = equity + 1000000.0
    elif len(equity) > 0 and equity.iloc[0] > 900000:
        # Already has capital
        pass
    
    return equity

def plot_comparison():
    plt.figure(figsize=(12, 7))
    
    for label, path in RESULTS.items():
        equity = load_equity_curve(path)
        if equity.empty: continue
        # Normalize to 100 for comparison
        normalized = (equity / equity.iloc[0]) * 100
        plt.plot(normalized, label=f"{label} (Final: {normalized.iloc[-1]:.1f})")

    plt.title("Comparative Net Equity Curves (2020-2025 Out-of-Sample)", fontsize=14, fontweight='bold')
    plt.xlabel("Trading Days (Stitched OOS Folds)", fontsize=12)
    plt.ylabel("Normalized Equity (Base=100)", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper left')
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/equity_comparison.png", dpi=300)
    print(f"Saved equity_comparison.png to {OUTPUT_DIR}")

def plot_drawdowns():
    plt.figure(figsize=(12, 5))
    
    for label, path in RESULTS.items():
        equity = load_equity_curve(path)
        if equity.empty: continue
        rolling_max = equity.cummax()
        drawdown = (equity - rolling_max) / rolling_max
        plt.fill_between(range(len(drawdown)), drawdown * 100, 0, alpha=0.3, label=label)

    plt.title("Portfolio Drawdowns (%)", fontsize=14, fontweight='bold')
    plt.xlabel("Trading Days", fontsize=12)
    plt.ylabel("Drawdown %", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='lower left')
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/drawdown_comparison.png", dpi=300)
    print(f"Saved drawdown_comparison.png to {OUTPUT_DIR}")

if __name__ == "__main__":
    print("Generating final thesis visualizations directly from JSON aggregates...")
    plot_comparison()
    plot_drawdowns()
    print("Done!")
