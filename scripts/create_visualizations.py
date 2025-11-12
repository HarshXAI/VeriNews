"""
Generate visual summary of our journey
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Create output directory
output_dir = Path("experiments/visualizations")
output_dir.mkdir(parents=True, exist_ok=True)

# Data
stages = ['Original\n"Baseline"', 'True\nBaseline', 'After\nHPO', 'After\nEnsemble', 'Target']
f1_scores = [91.76, 86.61, 88.56, 87.72, 90.00]
colors = ['lightcoral', 'orange', 'lightgreen', 'yellow', 'lightblue']
labels = ['Lucky split\n(not reproducible)', 'Seed 314\n(stratified)', '+1.95 pts\n(GATv2 HPO)', '-0.84 pts\n(failed)', 'Realistic\ngoal']

# Create figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Chart 1: Progress bars
bars = ax1.barh(stages, f1_scores, color=colors, edgecolor='black', linewidth=1.5)
ax1.axvline(x=90, color='green', linestyle='--', linewidth=2, label='90% Target')
ax1.axvline(x=95, color='red', linestyle='--', linewidth=2, label='95% Target')
ax1.set_xlabel('F1 Score (%)', fontsize=12, fontweight='bold')
ax1.set_title('Journey to Improved F1 Score', fontsize=14, fontweight='bold')
ax1.set_xlim(85, 96)
ax1.legend(fontsize=10)
ax1.grid(axis='x', alpha=0.3)

# Add values on bars
for i, (bar, score, label) in enumerate(zip(bars, f1_scores, labels)):
    ax1.text(score + 0.3, i, f'{score:.2f}%', va='center', fontsize=11, fontweight='bold')
    ax1.text(84.5, i, label, va='center', ha='right', fontsize=8, style='italic')

# Chart 2: Timeline
attempts = ['Baseline\n(Lucky)', 'Baseline\n(Real)', 'Threshold\nOpt', 'GATv2\nLarge', 'Ensemble\nv1', 'HPO\n(seed 42)', 'HPO\n(seed 314)', 'Ensemble\nv2', 'Best']
results = [91.76, 86.61, 86.09, 84.18, 86.13, 84.96, 88.56, 87.72, 88.56]
colors_timeline = ['red', 'orange', 'red', 'red', 'red', 'red', 'green', 'yellow', 'green']

ax2.plot(range(len(attempts)), results, 'o-', linewidth=2, markersize=8, color='gray', alpha=0.5)
scatter = ax2.scatter(range(len(attempts)), results, c=colors_timeline, s=200, edgecolors='black', linewidth=2, zorder=3)
ax2.axhline(y=90, color='green', linestyle='--', linewidth=2, alpha=0.5, label='90% Target')
ax2.axhline(y=95, color='red', linestyle='--', linewidth=2, alpha=0.5, label='95% Target')
ax2.set_xticks(range(len(attempts)))
ax2.set_xticklabels(attempts, rotation=45, ha='right', fontsize=9)
ax2.set_ylabel('F1 Score (%)', fontsize=12, fontweight='bold')
ax2.set_xlabel('Attempt', fontsize=12, fontweight='bold')
ax2.set_title('All Attempts Timeline', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=10)
ax2.set_ylim(82, 96)

# Add annotations for key points
ax2.annotate('Discovery:\nLucky split!', xy=(1, 86.61), xytext=(1.5, 84), 
            arrowprops=dict(arrowstyle='->', color='red', lw=2),
            fontsize=9, fontweight='bold', color='red')
ax2.annotate('Best Result:\n88.56%', xy=(6, 88.56), xytext=(5, 91), 
            arrowprops=dict(arrowstyle='->', color='green', lw=2),
            fontsize=9, fontweight='bold', color='green')

plt.tight_layout()
plt.savefig(output_dir / 'journey_summary.png', dpi=300, bbox_inches='tight')
print(f"✅ Saved visualization to {output_dir / 'journey_summary.png'}")

# Chart 3: Component analysis
fig, ax = plt.subplots(figsize=(10, 6))

components = ['Baseline\n(seed 314)', 'HPO\nGain', 'Ensemble\nGain\n(failed)', 'Gap to\n90%', 'Gap to\n95%']
values = [86.61, 1.95, -0.84, 1.44, 6.44]
colors_comp = ['blue', 'green', 'red', 'orange', 'darkred']

cumulative = [86.61]
for i in range(1, len(values)):
    if i <= 2:  # Add gains
        cumulative.append(cumulative[-1] + values[i])
    else:  # Gaps are not cumulative
        cumulative.append(cumulative[2])

# Create stacked bar chart
bars = []
bottom = 0
for i, (val, color) in enumerate(zip(values[:3], colors_comp[:3])):
    if val > 0:
        bar = ax.bar(0, val, bottom=bottom, color=color, edgecolor='black', linewidth=2, width=0.5)
        ax.text(0, bottom + val/2, f'+{val:.2f}', ha='center', va='center', fontsize=12, fontweight='bold', color='white')
        bottom += val
    else:
        bar = ax.bar(0.6, abs(val), color=color, edgecolor='black', linewidth=2, width=0.5)
        ax.text(0.6, abs(val)/2, f'{val:.2f}', ha='center', va='center', fontsize=12, fontweight='bold', color='white')

# Add targets
ax.axhline(y=90, color='green', linestyle='--', linewidth=2, label='90% Target')
ax.axhline(y=95, color='red', linestyle='--', linewidth=2, label='95% Target')

# Add current position
ax.scatter([0], [88.56], s=500, c='gold', marker='*', edgecolors='black', linewidth=2, zorder=10, label='Current: 88.56%')

ax.set_xlim(-0.5, 1.5)
ax.set_ylim(85, 96)
ax.set_xticks([0, 0.6])
ax.set_xticklabels(['Gains', 'Losses'], fontsize=12, fontweight='bold')
ax.set_ylabel('F1 Score (%)', fontsize=12, fontweight='bold')
ax.set_title('Performance Components Breakdown', fontsize=14, fontweight='bold')
ax.legend(fontsize=10, loc='upper left')
ax.grid(axis='y', alpha=0.3)

# Add text annotations
ax.text(-0.25, 86.61, 'Baseline\n86.61%', fontsize=10, ha='center', fontweight='bold')
ax.text(-0.25, 88.3, 'After HPO\n88.56%', fontsize=10, ha='center', fontweight='bold', color='green')
ax.text(1.1, 90, '90% target\n(+1.44 needed)', fontsize=9, ha='left', va='center')
ax.text(1.1, 95, '95% target\n(+6.44 needed)', fontsize=9, ha='left', va='center')

plt.tight_layout()
plt.savefig(output_dir / 'component_analysis.png', dpi=300, bbox_inches='tight')
print(f"✅ Saved visualization to {output_dir / 'component_analysis.png'}")

print("\n" + "="*80)
print("VISUALIZATION COMPLETE")
print("="*80)
print(f"\nGenerated 2 visualizations:")
print(f"  1. journey_summary.png - Shows our complete journey")
print(f"  2. component_analysis.png - Breaks down performance components")
print(f"\nLocation: {output_dir}/")
