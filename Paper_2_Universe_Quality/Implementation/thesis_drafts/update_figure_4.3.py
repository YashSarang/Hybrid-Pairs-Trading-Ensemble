#!/usr/bin/env python3
"""
Update Figure 4.3 with EXACT India vs NSE fold-by-fold data
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

# EXACT DATA from JSON
india_folds = pd.DataFrame([
    {"Fold": "F1 (2021)", "India": 0.604, "NSE_Rolling": 0.572},
    {"Fold": "F2 (2022)", "India": -0.080, "NSE_Rolling": 0.847},
    {"Fold": "F3 (2023)", "India": 1.996, "NSE_Rolling": -0.485},
    {"Fold": "F4 (2024)", "India": 0.840, "NSE_Rolling": -1.270},
])

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
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.08 if height > 0 else height - 0.12,
                f'{height:.2f}', ha='center', va='bottom' if height > 0 else 'top', 
                fontsize=9, fontweight='bold')

ax.axhline(0, color='black', linewidth=1.5, linestyle='-')
ax.set_xlabel('Fold (Test Year)', fontsize=12, fontweight='bold')
ax.set_ylabel('Net Sharpe Ratio', fontsize=12, fontweight='bold')
ax.set_title('Figure 4.3: India Multi-Market vs NSE Rolling (Fold-by-Fold)\n(Same Years, Different Universes: Nifty 50 vs Nifty 100)', 
             fontsize=14, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(india_folds['Fold'])
ax.legend(loc='upper right', fontsize=10)
ax.grid(axis='y', alpha=0.3)
ax.set_ylim(-1.5, 2.2)

# Add note with EXACT aggregate data
note = "India aggregate: +0.840 (4 folds, 123 trades)\nNSE Rolling aggregate: +0.052 (6 folds, 293 trades)\nIndia wins 3/4 folds (75%)"
ax.text(0.5, -0.25, note, transform=ax.transAxes, fontsize=9, ha='center', 
        style='italic', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('thesis_drafts/figures/figure_4.3_india_vs_nse_folds.png', dpi=300, bbox_inches='tight')
plt.savefig('thesis_drafts/figures/figure_4.3_india_vs_nse_folds.pdf', bbox_inches='tight')
plt.close()

print("✅ Figure 4.3 UPDATED with exact fold-by-fold data from JSON")
print("\nIndia vs NSE Fold-by-Fold Comparison:")
print(india_folds.to_string(index=False))
print(f"\nIndia wins: 3/4 folds (75%)")
print(f"India aggregate: +0.840 Sharpe")
print(f"NSE aggregate: +0.052 Sharpe")
print(f"India multiplier: {0.840/0.052:.1f}x better")
