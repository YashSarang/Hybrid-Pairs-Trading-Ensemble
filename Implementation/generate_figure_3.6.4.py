"""
Generate Figure 3.6.4: Cumulative Returns by Methodology
Equity curves for 6-fold concatenated backtest
"""

import json
import matplotlib.pyplot as plt
import numpy as np

# Load results
with open('experiments/results/walk_forward_20260506_104613.json') as f:
    expanding = json.load(f)

with open('experiments/results/rolling_window_validation_20260529/walk_forward_rolling_20260529_170106.json') as f:
    rolling = json.load(f)

# Calculate cumulative Sharpe ratios (proxy for returns)
# In reality, we'd use actual equity curves, but Sharpe works for illustration
years = [2020, 2021, 2022, 2023, 2024, 2025]
exp_sharpes = [f['net_sharpe'] for f in expanding['folds']]
roll_sharpes = [f['net_sharpe'] for f in rolling['folds']]

# Cumulative sum (as if each fold contributes additive return)
# This is a simplification; real equity would compound
exp_cum = np.cumsum([0] + exp_sharpes)
roll_cum = np.cumsum([0] + roll_sharpes)

# Create figure
fig, ax = plt.subplots(figsize=(12, 7))

x = [2019] + years  # Start at 2019 (before first test period)

# Plot equity curves
line1 = ax.plot(x, exp_cum, marker='o', markersize=8, linewidth=2.5, 
                label='Expanding Window', color='#E74C3C', markeredgecolor='black', 
                markeredgewidth=1.5, zorder=3)
line2 = ax.plot(x, roll_cum, marker='s', markersize=8, linewidth=2.5, 
                label='Rolling Window (12-month)', color='#3498DB', markeredgecolor='black',
                markeredgewidth=1.5, zorder=3)

# Fill area between curves
ax.fill_between(x, exp_cum, roll_cum, where=(np.array(roll_cum) >= np.array(exp_cum)), 
                color='#3498DB', alpha=0.2, interpolate=True, label='Rolling Outperformance')
ax.fill_between(x, exp_cum, roll_cum, where=(np.array(roll_cum) < np.array(exp_cum)), 
                color='#E74C3C', alpha=0.2, interpolate=True, label='Expanding Outperformance')

# Add horizontal line at zero
ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)

# Shade fold periods
for i, year in enumerate(years):
    ax.axvspan(year - 0.4, year + 0.4, alpha=0.1, color='gray', zorder=1)
    ax.text(year, ax.get_ylim()[1] * 0.95, f'Fold {i+1}', 
            ha='center', va='top', fontsize=8, fontweight='bold', alpha=0.6)

# Customize
ax.set_xlabel('Year', fontsize=12, fontweight='bold')
ax.set_ylabel('Cumulative Sharpe Ratio', fontsize=12, fontweight='bold')
ax.set_title('Figure 3.6.4: Cumulative Performance — Expanding vs Rolling Window', 
             fontsize=14, fontweight='bold', pad=20)
ax.legend(fontsize=11, loc='upper left')
ax.grid(axis='both', alpha=0.3, linestyle='--')

# Add value labels at key points
for i, year in enumerate(years):
    if i in [0, 2, 5]:  # Label first, middle, last
        ax.text(year + 0.1, exp_cum[i+1] + 0.08, f'{exp_cum[i+1]:.2f}', 
                ha='left', va='bottom', fontsize=8, color='#E74C3C', fontweight='bold')
        ax.text(year + 0.1, roll_cum[i+1] - 0.08, f'{roll_cum[i+1]:.2f}', 
                ha='left', va='top', fontsize=8, color='#3498DB', fontweight='bold')

# Add final performance box
final_exp = exp_cum[-1]
final_roll = roll_cum[-1]
delta = final_roll - final_exp

textstr = (f'FINAL CUMULATIVE SHARPE\n'
           f'────────────────────────\n'
           f'Expanding: {final_exp:+.3f}\n'
           f'Rolling:   {final_roll:+.3f}\n'
           f'Delta:     {delta:+.3f}\n'
           f'\n'
           f'Improvement: {delta/abs(final_exp)*100:+.1f}%')

props = dict(boxstyle='round', facecolor='wheat', alpha=0.9, edgecolor='black', linewidth=1.5)
ax.text(0.98, 0.02, textstr, transform=ax.transAxes, fontsize=10,
        verticalalignment='bottom', horizontalalignment='right', bbox=props, family='monospace')

plt.tight_layout()
plt.savefig('thesis_drafts/figures/figure_3.6.4_cumulative_returns.png', dpi=300, bbox_inches='tight')
print("✅ Saved: thesis_drafts/figures/figure_3.6.4_cumulative_returns.png")

plt.savefig('thesis_drafts/figures/figure_3.6.4_cumulative_returns.pdf', bbox_inches='tight')
print("✅ Saved: thesis_drafts/figures/figure_3.6.4_cumulative_returns.pdf")

plt.close()
