"""
Generate Figure 3.6.1: Fold-by-Fold Net Sharpe Comparison
Bar chart with expanding vs rolling side-by-side
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
exp_sharpes = [f['net_sharpe'] for f in expanding['folds']]
roll_sharpes = [f['net_sharpe'] for f in rolling['folds']]

# Create figure
fig, ax = plt.subplots(figsize=(12, 6))

x = np.arange(len(years))
width = 0.35

# Plot bars
bars1 = ax.bar(x - width/2, exp_sharpes, width, label='Expanding Window', 
               color='#E74C3C', alpha=0.8, edgecolor='black', linewidth=1.2)
bars2 = ax.bar(x + width/2, roll_sharpes, width, label='Rolling Window (12-month)', 
               color='#3498DB', alpha=0.8, edgecolor='black', linewidth=1.2)

# Add horizontal line at zero
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.3)

# Customize
ax.set_xlabel('Test Year', fontsize=12, fontweight='bold')
ax.set_ylabel('Net Sharpe Ratio', fontsize=12, fontweight='bold')
ax.set_title('Figure 3.6.1: Walk-Forward Validation — Net Sharpe by Test Year', 
             fontsize=14, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(years)
ax.legend(fontsize=11, loc='upper left')
ax.grid(axis='y', alpha=0.3, linestyle='--')

# Add value labels on bars
def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        label_y = height + 0.05 if height >= 0 else height - 0.12
        ax.text(bar.get_x() + bar.get_width()/2., label_y,
                f'{height:.3f}',
                ha='center', va='bottom' if height >= 0 else 'top', 
                fontsize=9, fontweight='bold')

add_labels(bars1)
add_labels(bars2)

# Add aggregate means as text box
exp_mean = np.mean(exp_sharpes)
roll_mean = np.mean(roll_sharpes)
textstr = f'Mean Net Sharpe:\nExpanding: {exp_mean:.3f}\nRolling: {roll_mean:.3f}\nΔ = {roll_mean - exp_mean:+.3f} (+{(roll_mean - exp_mean)/abs(exp_mean)*100:.1f}%)'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8, edgecolor='black', linewidth=1.5)
ax.text(0.98, 0.02, textstr, transform=ax.transAxes, fontsize=10,
        verticalalignment='bottom', horizontalalignment='right', bbox=props, family='monospace')

plt.tight_layout()
plt.savefig('thesis_drafts/figures/figure_3.6.1_fold_comparison.png', dpi=300, bbox_inches='tight')
print("✅ Saved: thesis_drafts/figures/figure_3.6.1_fold_comparison.png")

# Also save as PDF for LaTeX
plt.savefig('thesis_drafts/figures/figure_3.6.1_fold_comparison.pdf', bbox_inches='tight')
print("✅ Saved: thesis_drafts/figures/figure_3.6.1_fold_comparison.pdf")

plt.close()
