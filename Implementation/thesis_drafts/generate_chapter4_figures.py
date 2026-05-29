#!/usr/bin/env python3
"""
Generate Chapter 4 Figures: Multi-Market Validation

Creates 6 publication-ready figures:
1. Multi-market Sharpe comparison (bar chart)
2. Cost vs Performance scatter plot
3. India vs NSE fold-by-fold comparison
4. Trade efficiency (Sharpe per trade)
5. Geographic diversification heatmap
6. Signal model comparison (ZScore vs OU)
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Set publication quality
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'sans-serif'
sns.set_palette("husl")

# ============================================================================
# DATA
# ============================================================================

# Multi-market results (from MULTI_MARKET_RESULTS.md)
results = pd.DataFrame([
    {"Market": "India", "Signal": "ZScore", "Sharpe": 0.840, "Trades": 123, "TxCost_bps": 16.4, "Gross": 1.042},
    {"Market": "Brazil", "Signal": "OU", "Sharpe": 0.321, "Trades": 32, "TxCost_bps": 8.4, "Gross": 0.390},
    {"Market": "India", "Signal": "OU", "Sharpe": 0.200, "Trades": 26, "TxCost_bps": 16.4, "Gross": 0.243},
    {"Market": "NSE Rolling", "Signal": "ZScore", "Sharpe": 0.052, "Trades": 293, "TxCost_bps": 16.4, "Gross": 0.109},
    {"Market": "Brazil", "Signal": "ZScore", "Sharpe": -0.225, "Trades": 115, "TxCost_bps": 8.4, "Gross": -0.074},
    {"Market": "UK", "Signal": "ZScore", "Sharpe": -0.245, "Trades": 111, "TxCost_bps": 8.0, "Gross": -0.156},
    {"Market": "US", "Signal": "OU", "Sharpe": -0.254, "Trades": 39, "TxCost_bps": 2.7, "Gross": -0.233},
    {"Market": "UK", "Signal": "OU", "Sharpe": -0.405, "Trades": 42, "TxCost_bps": 8.0, "Gross": -0.371},
    {"Market": "NSE Expanding", "Signal": "ZScore", "Sharpe": -0.409, "Trades": 1096, "TxCost_bps": 16.4, "Gross": 0.117},
])

# Calculate metrics
results['Sharpe_per_Trade'] = results['Sharpe'] / results['Trades']
results['Cost_Drag'] = results['Gross'] - results['Sharpe']
results['Multiplier_vs_NSE'] = results['Sharpe'] / 0.052  # vs rolling NSE

# India fold-by-fold (approximate from text, will refine)
india_folds = pd.DataFrame([
    {"Fold": "F1 (2021)", "India": 0.50, "NSE_Rolling": 0.572},  # Estimated
    {"Fold": "F2 (2022)", "India": 1.30, "NSE_Rolling": 0.847},  # Estimated
    {"Fold": "F3 (2023)", "India": 0.90, "NSE_Rolling": -0.485},  # Estimated
    {"Fold": "F4 (2024)", "India": 0.68, "NSE_Rolling": -1.270},  # Estimated
])

# ============================================================================
# FIGURE 4.1: Multi-Market Sharpe Comparison (Bar Chart)
# ============================================================================

fig, ax = plt.subplots(figsize=(12, 6))

# Sort by Sharpe descending
sorted_results = results.sort_values('Sharpe', ascending=False)

# Create labels
labels = [f"{row['Market']}\n{row['Signal']}" for _, row in sorted_results.iterrows()]
sharpes = sorted_results['Sharpe'].values
colors = ['#2ecc71' if s > 0 else '#e74c3c' for s in sharpes]

# Highlight top performer and baselines
bar_colors = []
for i, row in enumerate(sorted_results.itertuples()):
    if row.Market == "India" and row.Signal == "ZScore":
        bar_colors.append('#f39c12')  # Gold for winner
    elif "NSE" in row.Market:
        bar_colors.append('#95a5a6')  # Grey for baselines
    else:
        bar_colors.append(colors[i])

bars = ax.barh(labels, sharpes, color=bar_colors, edgecolor='black', linewidth=1.2)

# Add value labels
for i, (bar, val) in enumerate(zip(bars, sharpes)):
    x = val + 0.05 if val > 0 else val - 0.05
    ha = 'left' if val > 0 else 'right'
    ax.text(x, i, f'{val:.3f}', va='center', ha=ha, fontsize=9, fontweight='bold')

# Add reference lines
ax.axvline(0, color='black', linewidth=1.5, linestyle='-')
ax.axvline(0.052, color='blue', linewidth=1, linestyle='--', label='Rolling NSE Baseline (+0.052)', alpha=0.7)

ax.set_xlabel('Net Sharpe Ratio', fontsize=12, fontweight='bold')
ax.set_title('Figure 4.1: Multi-Market Performance Comparison\n(4 Markets × 2 Signal Models = 7 Experiments)', 
             fontsize=14, fontweight='bold', pad=20)
ax.legend(loc='lower right', fontsize=10)
ax.grid(axis='x', alpha=0.3)
ax.set_xlim(-0.6, 1.0)

plt.tight_layout()
plt.savefig('thesis_drafts/figures/figure_4.1_multimarket_sharpe.png', dpi=300, bbox_inches='tight')
plt.savefig('thesis_drafts/figures/figure_4.1_multimarket_sharpe.pdf', bbox_inches='tight')
plt.close()

print("✓ Figure 4.1 saved")

# ============================================================================
# FIGURE 4.2: Cost vs Performance Scatter Plot
# ============================================================================

fig, ax = plt.subplots(figsize=(10, 8))

# Scatter plot
for _, row in results.iterrows():
    if "NSE" in row['Market']:
        marker = 's'  # Square for baselines
        size = 200
        alpha = 0.6
    else:
        marker = 'o'
        size = 150
        alpha = 0.9
    
    color = '#2ecc71' if row['Sharpe'] > 0 else '#e74c3c'
    ax.scatter(row['TxCost_bps'], row['Sharpe'], s=size, marker=marker, 
               color=color, alpha=alpha, edgecolor='black', linewidth=1.5)
    
    # Label
    offset_x = 0.3
    offset_y = 0.03
    if row['Market'] == "India" and row['Signal'] == "ZScore":
        offset_y = 0.06  # Move label up for winner
    ax.text(row['TxCost_bps'] + offset_x, row['Sharpe'] + offset_y, 
            f"{row['Market'][:3]}-{row['Signal'][:2]}", 
            fontsize=9, ha='left')

# Reference line
ax.axhline(0, color='black', linewidth=1.5, linestyle='-')
ax.axhline(0.052, color='blue', linewidth=1, linestyle='--', label='Rolling NSE (+0.052)', alpha=0.7)

ax.set_xlabel('Transaction Cost (basis points)', fontsize=12, fontweight='bold')
ax.set_ylabel('Net Sharpe Ratio', fontsize=12, fontweight='bold')
ax.set_title('Figure 4.2: Transaction Cost vs Performance\n(High Costs Do NOT Prevent India Profitability)', 
             fontsize=14, fontweight='bold', pad=20)
ax.legend(loc='upper right', fontsize=10)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('thesis_drafts/figures/figure_4.2_cost_vs_performance.png', dpi=300, bbox_inches='tight')
plt.savefig('thesis_drafts/figures/figure_4.2_cost_vs_performance.pdf', bbox_inches='tight')
plt.close()

print("✓ Figure 4.2 saved")

# ============================================================================
# FIGURE 4.3: India vs NSE Fold-by-Fold Comparison
# ============================================================================

fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(len(india_folds))
width = 0.35

india_bars = ax.bar(x - width/2, india_folds['India'], width, label='India Multi-Market (Nifty 50)', 
                     color='#f39c12', edgecolor='black', linewidth=1.2)
nse_bars = ax.bar(x + width/2, india_folds['NSE_Rolling'], width, label='NSE Rolling (Nifty 100)', 
                   color='#3498db', edgecolor='black', linewidth=1.2)

# Add value labels
for bars in [india_bars, nse_bars]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.05 if height > 0 else height - 0.12,
                f'{height:.2f}', ha='center', va='bottom' if height > 0 else 'top', 
                fontsize=9, fontweight='bold')

ax.axhline(0, color='black', linewidth=1.5, linestyle='-')
ax.set_xlabel('Fold (Test Year)', fontsize=12, fontweight='bold')
ax.set_ylabel('Net Sharpe Ratio', fontsize=12, fontweight='bold')
ax.set_title('Figure 4.3: India Multi-Market vs NSE Rolling (Fold-by-Fold)\n(Same Years, Different Universes)', 
             fontsize=14, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(india_folds['Fold'])
ax.legend(loc='upper right', fontsize=10)
ax.grid(axis='y', alpha=0.3)

# Add note
note = "Note: India aggregate +0.840 (4 folds) vs NSE aggregate +0.052 (6 folds)\nData: Estimated fold distribution (exact JSON extraction pending)"
ax.text(0.5, -0.25, note, transform=ax.transAxes, fontsize=8, ha='center', 
        style='italic', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.tight_layout()
plt.savefig('thesis_drafts/figures/figure_4.3_india_vs_nse_folds.png', dpi=300, bbox_inches='tight')
plt.savefig('thesis_drafts/figures/figure_4.3_india_vs_nse_folds.pdf', bbox_inches='tight')
plt.close()

print("✓ Figure 4.3 saved (with estimated data)")

# ============================================================================
# FIGURE 4.4: Trade Efficiency (Sharpe per Trade)
# ============================================================================

fig, ax = plt.subplots(figsize=(12, 6))

# Sort by Sharpe per trade
sorted_eff = results.sort_values('Sharpe_per_Trade', ascending=False)

labels = [f"{row['Market']}\n{row['Signal']}" for _, row in sorted_eff.iterrows()]
efficiency = sorted_eff['Sharpe_per_Trade'].values * 1000  # Scale to per-1000-trades for readability

# Color coding
bar_colors = []
for _, row in sorted_eff.iterrows():
    if row['Market'] == "India" and row['Signal'] == "ZScore":
        bar_colors.append('#f39c12')  # Gold
    elif "NSE" in row['Market']:
        bar_colors.append('#95a5a6')  # Grey
    elif efficiency[sorted_eff.index.get_loc(row.name)] > 0:
        bar_colors.append('#2ecc71')
    else:
        bar_colors.append('#e74c3c')

bars = ax.barh(labels, efficiency, color=bar_colors, edgecolor='black', linewidth=1.2)

# Add value labels
for i, (bar, val, orig) in enumerate(zip(bars, efficiency, sorted_eff['Sharpe_per_Trade'])):
    x = val + 0.5 if val > 0 else val - 0.5
    ha = 'left' if val > 0 else 'right'
    ax.text(x, i, f'{val:.2f}', va='center', ha=ha, fontsize=9, fontweight='bold')

ax.axvline(0, color='black', linewidth=1.5, linestyle='-')
ax.set_xlabel('Sharpe per 1000 Trades (×1000 scaling)', fontsize=12, fontweight='bold')
ax.set_title('Figure 4.4: Trade Efficiency Comparison\n(India: 34x More Efficient per Trade than NSE Rolling)', 
             fontsize=14, fontweight='bold', pad=20)
ax.grid(axis='x', alpha=0.3)

# Add note
note_text = "India ZScore: 0.840 Sharpe / 123 trades = 6.8 Sharpe per 1000 trades\nNSE Rolling: 0.052 Sharpe / 293 trades = 0.2 Sharpe per 1000 trades (34x less efficient)"
ax.text(0.98, 0.02, note_text, transform=ax.transAxes, fontsize=9, ha='right', va='bottom',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('thesis_drafts/figures/figure_4.4_trade_efficiency.png', dpi=300, bbox_inches='tight')
plt.savefig('thesis_drafts/figures/figure_4.4_trade_efficiency.pdf', bbox_inches='tight')
plt.close()

print("✓ Figure 4.4 saved")

# ============================================================================
# FIGURE 4.5: Geographic Diversification Heatmap
# ============================================================================

# Create pivot table for heatmap
heatmap_data = results[~results['Market'].str.contains('NSE')].pivot_table(
    index='Market', columns='Signal', values='Sharpe', fill_value=np.nan
)

# Reorder for clarity
market_order = ['India', 'Brazil', 'US', 'UK']
signal_order = ['ZScore', 'OU']
heatmap_data = heatmap_data.reindex(index=market_order, columns=signal_order)

fig, ax = plt.subplots(figsize=(8, 6))

# Create heatmap
sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='RdYlGn', center=0, 
            vmin=-0.5, vmax=1.0, cbar_kws={'label': 'Net Sharpe Ratio'},
            linewidths=2, linecolor='black', ax=ax, annot_kws={'fontsize': 12, 'fontweight': 'bold'})

ax.set_xlabel('Signal Model', fontsize=12, fontweight='bold')
ax.set_ylabel('Market', fontsize=12, fontweight='bold')
ax.set_title('Figure 4.5: Geographic Diversification Matrix\n(Performance by Market × Signal)', 
             fontsize=14, fontweight='bold', pad=20)

# Add note
note = "Green = Positive Sharpe (Profitable)\nRed = Negative Sharpe (Unprofitable)\nWhite = No Data"
ax.text(0.5, -0.12, note, transform=ax.transAxes, fontsize=9, ha='center', 
        style='italic', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

plt.tight_layout()
plt.savefig('thesis_drafts/figures/figure_4.5_geographic_heatmap.png', dpi=300, bbox_inches='tight')
plt.savefig('thesis_drafts/figures/figure_4.5_geographic_heatmap.pdf', bbox_inches='tight')
plt.close()

print("✓ Figure 4.5 saved")

# ============================================================================
# FIGURE 4.6: Signal Model Comparison (ZScore vs OU)
# ============================================================================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Group by signal
zscore_results = results[results['Signal'] == 'ZScore']
ou_results = results[results['Signal'] == 'OU']

# Panel 1: ZScore markets
markets_z = [row['Market'] for _, row in zscore_results.iterrows()]
sharpes_z = [row['Sharpe'] for _, row in zscore_results.iterrows()]
colors_z = ['#f39c12' if 'India' in m and 'Expanding' not in m else '#95a5a6' if 'NSE' in m else '#2ecc71' if s > 0 else '#e74c3c' 
            for m, s in zip(markets_z, sharpes_z)]

bars1 = ax1.barh(markets_z, sharpes_z, color=colors_z, edgecolor='black', linewidth=1.2)

for i, (bar, val) in enumerate(zip(bars1, sharpes_z)):
    x = val + 0.04 if val > 0 else val - 0.04
    ha = 'left' if val > 0 else 'right'
    ax1.text(x, i, f'{val:.3f}', va='center', ha=ha, fontsize=9, fontweight='bold')

ax1.axvline(0, color='black', linewidth=1.5, linestyle='-')
ax1.axvline(0.052, color='blue', linewidth=1, linestyle='--', alpha=0.5)
ax1.set_xlabel('Net Sharpe Ratio', fontsize=11, fontweight='bold')
ax1.set_title('ZScore Signal Model\n(4 markets + 2 NSE baselines)', fontsize=12, fontweight='bold')
ax1.grid(axis='x', alpha=0.3)

# Panel 2: OU markets
markets_ou = [row['Market'] for _, row in ou_results.iterrows()]
sharpes_ou = [row['Sharpe'] for _, row in ou_results.iterrows()]
colors_ou = ['#f39c12' if 'India' in m else '#2ecc71' if s > 0 else '#e74c3c' 
             for m, s in zip(markets_ou, sharpes_ou)]

bars2 = ax2.barh(markets_ou, sharpes_ou, color=colors_ou, edgecolor='black', linewidth=1.2)

for i, (bar, val) in enumerate(zip(bars2, sharpes_ou)):
    x = val + 0.04 if val > 0 else val - 0.04
    ha = 'left' if val > 0 else 'right'
    ax2.text(x, i, f'{val:.3f}', va='center', ha=ha, fontsize=9, fontweight='bold')

ax2.axvline(0, color='black', linewidth=1.5, linestyle='-')
ax2.axvline(0.052, color='blue', linewidth=1, linestyle='--', alpha=0.5, label='Rolling NSE ZScore')
ax2.set_xlabel('Net Sharpe Ratio', fontsize=11, fontweight='bold')
ax2.set_title('OU Signal Model\n(4 markets tested)', fontsize=12, fontweight='bold')
ax2.legend(loc='lower right', fontsize=9)
ax2.grid(axis='x', alpha=0.3)

fig.suptitle('Figure 4.6: Signal Model Comparison (ZScore vs OU Across Markets)', 
             fontsize=14, fontweight='bold', y=1.02)

plt.tight_layout()
plt.savefig('thesis_drafts/figures/figure_4.6_signal_comparison.png', dpi=300, bbox_inches='tight')
plt.savefig('thesis_drafts/figures/figure_4.6_signal_comparison.pdf', bbox_inches='tight')
plt.close()

print("✓ Figure 4.6 saved")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*70)
print("✅ ALL 6 CHAPTER 4 FIGURES GENERATED")
print("="*70)
print("\nFigures saved to: thesis_drafts/figures/")
print("\n1. figure_4.1_multimarket_sharpe.{png,pdf}")
print("   → Bar chart: Multi-market performance comparison")
print("\n2. figure_4.2_cost_vs_performance.{png,pdf}")
print("   → Scatter: Transaction cost vs Sharpe (India paradox)")
print("\n3. figure_4.3_india_vs_nse_folds.{png,pdf}")
print("   → Bar chart: India vs NSE fold-by-fold (estimated data)")
print("\n4. figure_4.4_trade_efficiency.{png,pdf}")
print("   → Bar chart: Sharpe per trade (India 34x better)")
print("\n5. figure_4.5_geographic_heatmap.{png,pdf}")
print("   → Heatmap: Market × Signal performance matrix")
print("\n6. figure_4.6_signal_comparison.{png,pdf}")
print("   → Dual panel: ZScore vs OU signal models")
print("\n" + "="*70)
print("NOTE: Figure 4.3 uses ESTIMATED India fold-by-fold data.")
print("      Extract exact data from JSON for publication version.")
print("="*70)
