"""
Threshold Sensitivity Sweep

Tests different p_keep thresholds to find optimal trade-off between:
- Trade frequency
- Expectancy
- Maximum drawdown
- Winrate
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

print("="*80)
print("PHASE 5: THRESHOLD SENSITIVITY SWEEP")
print("="*80)

# Paths
ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "output/phase4_quality"
OUTPUT_DIR = ROOT / "output/phase5_quality"
PLOTS_DIR = OUTPUT_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Threshold grid
THRESHOLDS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

# Load predictions
print(f"\n[1/4] Loading validation predictions...")
predictions_df = pd.read_csv(DATA_DIR / "predictions.csv")
print(f"  Loaded {len(predictions_df):,} predictions")

y_true = predictions_df['actual'].values
p_keep = predictions_df['p_keep'].values

# Load meta for trading metrics
print(f"\n[2/4] Computing metrics for each threshold...")

results = []

for threshold in THRESHOLDS:
    print(f"\n  Threshold: {threshold:.1f}")
    
    # Predict based on threshold
    y_pred = (p_keep >= threshold).astype(int)
    
    # Classification metrics
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # Trading metrics (filter = only trade if predicted KEEP)
    kept_mask = y_pred == 1
    
    if kept_mask.sum() == 0:
        # No trades kept
        num_trades = 0
        winrate = 0.0
        avg_r = 0.0
        total_r = 0.0
        maxdd_r = 0.0
    else:
        kept_events = predictions_df[kept_mask]
        
        num_trades = len(kept_events)
        winners = (kept_events['hit'] == 'tp').sum()
        winrate = winners / num_trades if num_trades > 0 else 0.0
        
        # Compute R metrics
        total_r = kept_events['outcome_rr'].sum()
        avg_r = total_r / num_trades if num_trades > 0 else 0.0
        
        # Max drawdown (simple cumulative)
        cumulative_r = kept_events['outcome_rr'].cumsum().values
        peak_r = np.maximum.accumulate(cumulative_r)
        drawdown = peak_r - cumulative_r
        maxdd_r = drawdown.max() if len(drawdown) > 0 else 0.0
    
    print(f"    Trades: {num_trades:,} | WR: {winrate:.1%} | Exp: {avg_r:+.4f}R | DD: {maxdd_r:.2f}R | F1: {f1:.4f}")
    
    results.append({
        'threshold': threshold,
        'num_trades': num_trades,
        'winrate': winrate,
        'avg_r': avg_r,
        'total_r': total_r,
        'maxdd_r': maxdd_r,
        'accuracy': acc,
        'precision_keep': precision,
        'recall_keep': recall,
        'f1_keep': f1
    })

# Convert to DataFrame
results_df = pd.DataFrame(results)

# Save results
csv_path = OUTPUT_DIR / "threshold_sweep_results.csv"
results_df.to_csv(csv_path, index=False)
print(f"\n[OK] Saved results: {csv_path}")

# Generate plots
print(f"\n[3/4] Generating plots...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Threshold vs Expectancy
ax1 = axes[0, 0]
ax1.plot(results_df['threshold'], results_df['avg_r'], marker='o', linewidth=2, markersize=8)
ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax1.set_xlabel('Threshold', fontsize=11)
ax1.set_ylabel('Expectancy (avg R)', fontsize=11)
ax1.set_title('Threshold vs Expectancy', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)

# Highlight max
max_exp_idx = results_df['avg_r'].idxmax()
max_exp = results_df.loc[max_exp_idx]
ax1.scatter([max_exp['threshold']], [max_exp['avg_r']], color='red', s=200, zorder=5, marker='*')
ax1.text(max_exp['threshold'], max_exp['avg_r']*1.1, 
         f"Max: {max_exp['threshold']:.1f} ({max_exp['avg_r']:.4f}R)", 
         ha='center', fontsize=9, weight='bold')

# Plot 2: Threshold vs Max DD
ax2 = axes[0, 1]
ax2.plot(results_df['threshold'], results_df['maxdd_r'], marker='o', linewidth=2, markersize=8, color='orange')
ax2.set_xlabel('Threshold', fontsize=11)
ax2.set_ylabel('Max Drawdown (R)', fontsize=11)
ax2.set_title('Threshold vs Max Drawdown', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)

# Highlight min
min_dd_idx = results_df['maxdd_r'].idxmin()
min_dd = results_df.loc[min_dd_idx]
ax2.scatter([min_dd['threshold']], [min_dd['maxdd_r']], color='green', s=200, zorder=5, marker='*')
ax2.text(min_dd['threshold'], min_dd['maxdd_r']*1.1, 
         f"Min: {min_dd['threshold']:.1f} ({min_dd['maxdd_r']:.2f}R)", 
         ha='center', fontsize=9, weight='bold')

# Plot 3: Threshold vs Number of Trades
ax3 = axes[1, 0]
ax3.plot(results_df['threshold'], results_df['num_trades'], marker='o', linewidth=2, markersize=8, color='green')
ax3.set_xlabel('Threshold', fontsize=11)
ax3.set_ylabel('Number of Trades', fontsize=11)
ax3.set_title('Threshold vs Trade Frequency', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3)

# Plot 4: Threshold vs F1 (KEEP)
ax4 = axes[1, 1]
ax4.plot(results_df['threshold'], results_df['f1_keep'], marker='o', linewidth=2, markersize=8, color='purple')
ax4.set_xlabel('Threshold', fontsize=11)
ax4.set_ylabel('F1 Score (KEEP class)', fontsize=11)
ax4.set_title('Threshold vs F1 Score', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plots_path = PLOTS_DIR / "threshold_sensitivity.png"
plt.savefig(plots_path, dpi=150, bbox_inches='tight')
print(f"  [OK] Saved: {plots_path}")

# Pareto frontier plot (Expectancy vs Max DD)
fig2, ax = plt.subplots(figsize=(10, 7))
scatter = ax.scatter(results_df['maxdd_r'], results_df['avg_r'], 
                     c=results_df['threshold'], s=200, cmap='viridis', 
                     edgecolors='black', linewidth=1.5)
ax.set_xlabel('Max Drawdown (R)', fontsize=12)
ax.set_ylabel('Expectancy (avg R)', fontsize=12)
ax.set_title('Pareto Frontier: Expectancy vs Max DD', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)

# Annotate each point
for _, row in results_df.iterrows():
    ax.annotate(f"{row['threshold']:.1f}", 
                (row['maxdd_r'], row['avg_r']),
                textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)

cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Threshold', fontsize=11)
plt.tight_layout()

pareto_path = PLOTS_DIR / "threshold_pareto.png"
plt.savefig(pareto_path, dpi=150, bbox_inches='tight')
print(f"  [OK] Saved: {pareto_path}")

# Analysis and recommendations
print(f"\n[4/4] Analysis and recommendations...")

# Best expectancy
best_exp = results_df.loc[results_df['avg_r'].idxmax()]
print(f"\n  Best Expectancy:")
print(f"    Threshold: {best_exp['threshold']:.1f}")
print(f"    Expectancy: {best_exp['avg_r']:+.4f}R")
print(f"    Max DD: {best_exp['maxdd_r']:.2f}R")
print(f"    Trades: {best_exp['num_trades']} | WR: {best_exp['winrate']:.1%}")

# Best DD (min drawdown)
best_dd = results_df.loc[results_df['maxdd_r'].idxmin()]
print(f"\n  Lowest Drawdown:")
print(f"    Threshold: {best_dd['threshold']:.1f}")
print(f"    Max DD: {best_dd['maxdd_r']:.2f}R")
print(f"    Expectancy: {best_dd['avg_r']:+.4f}R")
print(f"    Trades: {best_dd['num_trades']} | WR: {best_dd['winrate']:.1%}")

# Balanced (default 0.5)
balanced = results_df[results_df['threshold'] == 0.5].iloc[0]
print(f"\n  Balanced (default t=0.5):")
print(f"    Expectancy: {balanced['avg_r']:+.4f}R")
print(f"    Max DD: {balanced['maxdd_r']:.2f}R")
print(f"    Trades: {balanced['num_trades']} | WR: {balanced['winrate']:.1%}")

# Recommendations
recommendations = f"""
THRESHOLD RECOMMENDATIONS

Based on sweep of thresholds from {min(THRESHOLDS):.1f} to {max(THRESHOLDS):.1f}:

1. **Conservative** (Maximize Expectancy):
   - Threshold: {best_exp['threshold']:.1f}
   - Expectancy: {best_exp['avg_r']:+.4f}R
   - Max DD: {best_exp['maxdd_r']:.2f}R
   - Trades: {best_exp['num_trades']}
   - Use when: Prioritizing quality over quantity

2. **Defensive** (Minimize Drawdown):
   - Threshold: {best_dd['threshold']:.1f}
   - Expectancy: {best_dd['avg_r']:+.4f}R
   - Max DD: {best_dd['maxdd_r']:.2f}R
   - Trades: {best_dd['num_trades']}
   - Use when: Risk aversion is priority

3. **Balanced** (Default):
   - Threshold: 0.5
   - Expectancy: {balanced['avg_r']:+.4f}R
   - Max DD: {balanced['maxdd_r']:.2f}R
   - Trades: {balanced['num_trades']}
   - Use when: Standard operating mode

TRADE-OFFS:
- Lower threshold (0.2-0.4): More trades, potentially lower expectancy
- Higher threshold (0.6-0.8): Fewer trades, potentially higher expectancy
- Sweet spot appears to be around {best_exp['threshold']:.1f} for this validation set

See plots in: {PLOTS_DIR}/
"""

print(f"\n{recommendations}")

# Save recommendations
rec_path = OUTPUT_DIR / "threshold_recommendations.txt"
with open(rec_path, 'w') as f:
    f.write("THRESHOLD SENSITIVITY ANALYSIS\n")
    f.write("="*80 + "\n\n")
    f.write(recommendations)

print(f"\nSaved recommendations: {rec_path}")
print(f"\n{'='*80}")
print("THRESHOLD SWEEP COMPLETE!")
print("="*80)
print(f"\nOutput files:")
print(f"  - {csv_path}")
print(f"  - {plots_path}")
print(f"  - {pareto_path}")
print(f"  - {rec_path}")
