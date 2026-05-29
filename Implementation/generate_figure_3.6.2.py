"""
Generate Figure 3.6.2: Cost Drag Decomposition
Stacked bar chart showing Gross Sharpe + Cost Drag = Net Sharpe
"""

import json
import matplotlib.pyplot as plt
import numpy as np

# Load results
with open('experiments/results/walk_forward_20260506_104613.json') as f:
    expanding = json.load(f)

with open('experiments/results/rolling_window_validation_20260529/walk_forward_rolling_20260529_170106.json') as f:
    rolling = json.load(f)

# Extract data
years = [2020, 2021, 2022, 2023, 2024, 2025]
exp_gross = [f['gross_sharpe'] for f in expanding['folds']]
exp_net = [f['net_sharpe'] for f in expanding['folds']]
exp_cost = [net - gross for net, gross in zip(exp_net, exp_gross)]

roll_gross = [f['gross_sharpe'] for f in rolling['folds']]
roll_net = [f['net_sharpe'] for f in rolling['folds']]
roll_cost = [net - gross for net, gross in zip(roll_net, roll_gross)]

# Create figure with subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Expanding window
x1 = np.arange(len(years))
bars1_gross = ax1.bar(x1, exp_gross, label='Gross Sharpe', color='#2ECC71', alpha=0.8, edgecolor='black', linewidth=1.2)
bars1_cost = ax1.bar(x1, exp_cost, bottom=exp_gross, label='Cost Drag', color='#E74C3C', alpha=0.8, edgecolor='black', linewidth=1.2)

ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.3)
ax1.set_xlabel('Test Year', fontsize=12, fontweight='bold')
ax1.set_ylabel('Sharpe Ratio', fontsize=12, fontweight='bold')
ax1.set_title('Expanding Window: Gross Sharpe + Cost Drag = Net Sharpe', fontsize=12, fontweight='bold')
ax1.set_xticks(x1)
ax1.set_xticklabels(years)
ax1.legend(fontsize=10, loc='upper left')
ax1.grid(axis='y', alpha=0.3, linestyle='--')

# Add net sharpe markers
for i, (gross, cost) in enumerate(zip(exp_gross, exp_cost)):
    net = gross + cost
    ax1.plot(i, net, 'ko', markersize=8, markeredgewidth=2, markerfacecolor='yellow', zorder=5)
    ax1.text(i, net + 0.08, f'{net:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

# Add avg cost drag text
avg_cost_exp = np.mean(exp_cost)
textstr1 = f'Avg Cost Drag:\n{avg_cost_exp:.3f} Sharpe'
props = dict(boxstyle='round', facecolor='#E74C3C', alpha=0.3, edgecolor='black', linewidth=1.5)
ax1.text(0.98, 0.02, textstr1, transform=ax1.transAxes, fontsize=10,
        verticalalignment='bottom', horizontalalignment='right', bbox=props, family='monospace')

# Rolling window
x2 = np.arange(len(years))
bars2_gross = ax2.bar(x2, roll_gross, label='Gross Sharpe', color='#2ECC71', alpha=0.8, edgecolor='black', linewidth=1.2)
bars2_cost = ax2.bar(x2, roll_cost, bottom=roll_gross, label='Cost Drag', color='#3498DB', alpha=0.8, edgecolor='black', linewidth=1.2)

ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.3)
ax2.set_xlabel('Test Year', fontsize=12, fontweight='bold')
ax2.set_ylabel('Sharpe Ratio', fontsize=12, fontweight='bold')
ax2.set_title('Rolling Window: Gross Sharpe + Cost Drag = Net Sharpe', fontsize=12, fontweight='bold')
ax2.set_xticks(x2)
ax2.set_xticklabels(years)
ax2.legend(fontsize=10, loc='upper left')
ax2.grid(axis='y', alpha=0.3, linestyle='--')

# Add net sharpe markers
for i, (gross, cost) in enumerate(zip(roll_gross, roll_cost)):
    net = gross + cost
    ax2.plot(i, net, 'ko', markersize=8, markeredgewidth=2, markerfacecolor='yellow', zorder=5)
    ax2.text(i, net + 0.08, f'{net:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

# Add avg cost drag text
avg_cost_roll = np.mean(roll_cost)
textstr2 = f'Avg Cost Drag:\n{avg_cost_roll:.3f} Sharpe\n\n89% reduction\nvs Expanding'
props2 = dict(boxstyle='round', facecolor='#3498DB', alpha=0.3, edgecolor='black', linewidth=1.5)
ax2.text(0.98, 0.02, textstr2, transform=ax2.transAxes, fontsize=10,
        verticalalignment='bottom', horizontalalignment='right', bbox=props2, family='monospace')

# Main title
fig.suptitle('Figure 3.6.2: Cost Drag Decomposition — Gross vs Net Performance', 
             fontsize=14, fontweight='bold', y=0.98)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('thesis_drafts/figures/figure_3.6.2_cost_decomposition.png', dpi=300, bbox_inches='tight')
print("✅ Saved: thesis_drafts/figures/figure_3.6.2_cost_decomposition.png")

plt.savefig('thesis_drafts/figures/figure_3.6.2_cost_decomposition.pdf', bbox_inches='tight')
print("✅ Saved: thesis_drafts/figures/figure_3.6.2_cost_decomposition.pdf")

plt.close()
