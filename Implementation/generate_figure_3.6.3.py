"""
Generate Figure 3.6.3: Trade Frequency Consistency
Line chart showing trade counts by fold with consistency bands
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
exp_trades = [f['total_trades'] for f in expanding['folds']]
roll_trades = [f['total_trades'] for f in rolling['folds']]

# Create figure
fig, ax = plt.subplots(figsize=(12, 7))

x = np.arange(len(years))

# Plot lines
line1 = ax.plot(x, exp_trades, marker='o', markersize=10, linewidth=2.5, 
                label='Expanding Window', color='#E74C3C', markeredgecolor='black', 
                markeredgewidth=1.5)
line2 = ax.plot(x, roll_trades, marker='s', markersize=10, linewidth=2.5, 
                label='Rolling Window (12-month)', color='#3498DB', markeredgecolor='black',
                markeredgewidth=1.5)

# Add consistency bands (mean ± std)
exp_mean = np.mean(exp_trades)
exp_std = np.std(exp_trades, ddof=1)
roll_mean = np.mean(roll_trades)
roll_std = np.std(roll_trades, ddof=1)

ax.axhline(y=exp_mean, color='#E74C3C', linestyle='--', linewidth=1.5, alpha=0.6, label=f'Expanding Mean: {exp_mean:.1f}')
ax.fill_between(x, exp_mean - exp_std, exp_mean + exp_std, color='#E74C3C', alpha=0.15)

ax.axhline(y=roll_mean, color='#3498DB', linestyle='--', linewidth=1.5, alpha=0.6, label=f'Rolling Mean: {roll_mean:.1f}')
ax.fill_between(x, roll_mean - roll_std, roll_mean + roll_std, color='#3498DB', alpha=0.15)

# Customize
ax.set_xlabel('Test Year', fontsize=12, fontweight='bold')
ax.set_ylabel('Total Trades', fontsize=12, fontweight='bold')
ax.set_title('Figure 3.6.3: Trade Frequency by Fold — Consistency Comparison', 
             fontsize=14, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(years)
ax.legend(fontsize=11, loc='upper right')
ax.grid(axis='both', alpha=0.3, linestyle='--')

# Add value labels
for i, (exp_t, roll_t) in enumerate(zip(exp_trades, roll_trades)):
    ax.text(i, exp_t + 8, f'{exp_t}', ha='center', va='bottom', fontsize=9, fontweight='bold', color='#E74C3C')
    ax.text(i, roll_t - 8, f'{roll_t}', ha='center', va='top', fontsize=9, fontweight='bold', color='#3498DB')

# Add statistics box
cv_exp = (exp_std / exp_mean) * 100
cv_roll = (roll_std / roll_mean) * 100
total_reduction = (sum(exp_trades) - sum(roll_trades)) / sum(exp_trades) * 100

textstr = (f'CONSISTENCY METRICS\n'
           f'─────────────────────────\n'
           f'Expanding:\n'
           f'  Range: {min(exp_trades)}-{max(exp_trades)} trades\n'
           f'  Std Dev: {exp_std:.1f}\n'
           f'  CV: {cv_exp:.1f}%\n'
           f'\n'
           f'Rolling:\n'
           f'  Range: {min(roll_trades)}-{max(roll_trades)} trades\n'
           f'  Std Dev: {roll_std:.1f}\n'
           f'  CV: {cv_roll:.1f}%\n'
           f'\n'
           f'Total Reduction: {total_reduction:.1f}%\n'
           f'Rolling is {(cv_exp/cv_roll - 1)*100:.0f}% more consistent')

props = dict(boxstyle='round', facecolor='wheat', alpha=0.9, edgecolor='black', linewidth=1.5)
ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=9,
        verticalalignment='top', horizontalalignment='left', bbox=props, family='monospace')

plt.tight_layout()
plt.savefig('thesis_drafts/figures/figure_3.6.3_trade_consistency.png', dpi=300, bbox_inches='tight')
print("✅ Saved: thesis_drafts/figures/figure_3.6.3_trade_consistency.png")

plt.savefig('thesis_drafts/figures/figure_3.6.3_trade_consistency.pdf', bbox_inches='tight')
print("✅ Saved: thesis_drafts/figures/figure_3.6.3_trade_consistency.pdf")

plt.close()
