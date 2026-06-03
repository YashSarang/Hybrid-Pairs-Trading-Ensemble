#!/usr/bin/env python3
"""
visualize_cross_market.py

Generate comparative visualizations across markets.

Usage:
    python visualize_cross_market.py --markets india us brazil uk
"""

import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
sns.set_palette("husl")

def load_summary_tables(summary_dir: Path) -> dict:
    """Load all CSV tables from cross_market_summary/."""
    tables = {}
    
    for csv_file in summary_dir.glob("*.csv"):
        df = pd.read_csv(csv_file)
        tables[csv_file.stem] = df
    
    return tables

def plot_cost_sensitivity(df: pd.DataFrame, output_path: Path):
    """
    Plot transaction cost sensitivity.
    X-axis: Cost (bps)
    Y-axis: Sharpe Ratio
    Two lines: Gross vs Net
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    markets = df['Market'].tolist()
    costs = df['Cost_bps'].tolist()
    gross = df['Gross_Sharpe'].tolist()
    net = df['Net_Sharpe'].tolist()
    
    ax.plot(costs, gross, 'o-', label='Gross Sharpe', linewidth=2, markersize=8)
    ax.plot(costs, net, 's-', label='Net Sharpe', linewidth=2, markersize=8)
    
    # Annotate markets
    for i, market in enumerate(markets):
        ax.annotate(
            market,
            (costs[i], net[i]),
            textcoords="offset points",
            xytext=(0, -15),
            ha='center',
            fontsize=9
        )
    
    ax.set_xlabel('Transaction Cost (bps, round-trip)', fontsize=12)
    ax.set_ylabel('Sharpe Ratio', fontsize=12)
    ax.set_title('Transaction Cost Sensitivity Across Markets', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"✅ Saved: {output_path}")
    plt.close()

def plot_regime_heatmap(df: pd.DataFrame, output_path: Path):
    """
    Plot fold-level performance heatmap.
    Rows: Folds
    Columns: Markets
    Color: Net Sharpe
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Remove index name for cleaner plot
    df_plot = df.copy()
    df_plot.index.name = None
    
    sns.heatmap(
        df_plot,
        annot=True,
        fmt='.2f',
        cmap='RdYlGn',
        center=0,
        cbar_kws={'label': 'Net Sharpe Ratio'},
        linewidths=0.5,
        ax=ax
    )
    
    ax.set_title('Regime Performance (Net Sharpe per Fold)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Fold (6-month test period)', fontsize=11)
    ax.set_xlabel('Market', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"✅ Saved: {output_path}")
    plt.close()

def plot_degradation_bar(df: pd.DataFrame, output_path: Path):
    """
    Plot gross-to-net degradation as grouped bar chart.
    X-axis: Markets
    Y-axis: Sharpe Ratio
    Two bars per market: Gross vs Net
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    markets = df['Market'].tolist()
    gross = df['Gross_Sharpe'].tolist()
    net = df['Net_Sharpe'].tolist()
    
    x = range(len(markets))
    width = 0.35
    
    bars1 = ax.bar([i - width/2 for i in x], gross, width, label='Gross Sharpe', alpha=0.8)
    bars2 = ax.bar([i + width/2 for i in x], net, width, label='Net Sharpe', alpha=0.8)
    
    ax.set_xlabel('Market', fontsize=12)
    ax.set_ylabel('Sharpe Ratio', fontsize=12)
    ax.set_title('Gross vs Net Sharpe Ratio by Market', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(markets, fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(True, axis='y', alpha=0.3)
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
    
    # Annotate degradation %
    for i, (g, n) in enumerate(zip(gross, net)):
        if g != 0:
            pct = (g - n) / g * 100
            ax.text(
                i, max(g, n) + 0.05,
                f"-{pct:.0f}%",
                ha='center',
                fontsize=9,
                color='red'
            )
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"✅ Saved: {output_path}")
    plt.close()

def plot_summary_table(df: pd.DataFrame, output_path: Path):
    """
    Render summary stats table as image (for inclusion in reports).
    """
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc='center',
        loc='center',
        colWidths=[0.12] * len(df.columns)
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Header styling
    for i in range(len(df.columns)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.title('Multi-Market Summary Statistics', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Visualize cross-market results")
    parser.add_argument(
        '--markets',
        nargs='+',
        default=['india', 'us', 'brazil', 'uk'],
        help='Markets to include'
    )
    
    args = parser.parse_args()
    
    # Paths
    summary_dir = Path(__file__).parent.parent / "results" / "cross_market_summary"
    
    if not summary_dir.exists():
        print(f"❌ Summary directory not found: {summary_dir}")
        print("   Run: python compare_markets.py --markets {' '.join(args.markets)}")
        return
    
    print("=" * 60)
    print("Multi-Market Visualization")
    print("=" * 60)
    print()
    
    # Load tables
    print("Loading summary tables...", end=" ")
    tables = load_summary_tables(summary_dir)
    print(f"✅ {len(tables)} tables loaded\n")
    
    # Generate plots
    output_dir = summary_dir / "plots"
    output_dir.mkdir(exist_ok=True)
    
    print("Generating visualizations:\n")
    
    # 1. Cost sensitivity
    if 'cost_sensitivity' in tables:
        plot_cost_sensitivity(
            tables['cost_sensitivity'],
            output_dir / "cost_sensitivity.png"
        )
    
    # 2. Regime heatmap
    if 'regime_performance' in tables:
        plot_regime_heatmap(
            tables['regime_performance'],
            output_dir / "regime_heatmap.png"
        )
    
    # 3. Degradation bar chart
    if 'cost_sensitivity' in tables:
        plot_degradation_bar(
            tables['cost_sensitivity'],
            output_dir / "degradation_bars.png"
        )
    
    # 4. Summary table image
    if 'summary_stats' in tables:
        plot_summary_table(
            tables['summary_stats'],
            output_dir / "summary_table.png"
        )
    
    print()
    print("=" * 60)
    print(f"All plots saved to: {output_dir}")
    print("=" * 60)

if __name__ == "__main__":
    main()
