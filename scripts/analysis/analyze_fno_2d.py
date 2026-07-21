"""
Comprehensive FNO 2D analysis: training evolution, cross-field metrics, spectral breakdown.
Generates plots matching the PINN ensemble analysis style.
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Load data
results_dir = Path(__file__).parent / "results_fno_2d"
history_file = results_dir / "fno2d_history.json"
metrics_file = results_dir / "fno2d_metrics.json"

with open(history_file) as f:
    history = json.load(f)

with open(metrics_file) as f:
    metrics = json.load(f)

# Extract training data
epochs = np.array([h["epoch"] for h in history])
train_loss = np.array([h["train_total"] for h in history])
val_loss = np.array([h["val_total"] for h in history])
elapsed_mins = np.array([h["elapsed"] / 60 for h in history])

# Per-channel validation losses
val_mse = np.array([h["val"]["mse"] for h in history])
val_spec_xy = np.array([h["val"]["spec_xy"] for h in history])
val_spec_t = np.array([h["val"]["spec_t"] for h in history])

fig = plt.figure(figsize=(16, 12))

# ── 1. Training evolution (total loss) ──
ax1 = plt.subplot(3, 3, 1)
ax1.semilogy(epochs, train_loss, 'o-', label='Train', alpha=0.7, markersize=4)
ax1.semilogy(epochs, val_loss, 's-', label='Val', alpha=0.7, markersize=4)
best_epoch = metrics["epoch_loaded"]
ax1.axvline(best_epoch, color='red', linestyle='--', alpha=0.5, label=f'Best (ep {best_epoch})')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('FNO 2D: Training Evolution')
ax1.legend()
ax1.grid(True, alpha=0.3)

# ── 2. Validation loss components ──
ax2 = plt.subplot(3, 3, 2)
ax2.semilogy(epochs, val_mse, 'o-', label='MSE', markersize=4)
ax2.semilogy(epochs, val_spec_xy, 's-', label='Spatial PSD', markersize=4)
ax2.semilogy(epochs, val_spec_t, '^-', label='Temporal PSD', markersize=4)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss component')
ax2.set_title('FNO 2D: Validation Loss Components')
ax2.legend()
ax2.grid(True, alpha=0.3)

# ── 3. Training time ──
ax3 = plt.subplot(3, 3, 3)
ax3.plot(epochs, elapsed_mins, 'o-', color='purple', markersize=4)
ax3.set_xlabel('Epoch')
ax3.set_ylabel('Elapsed time (minutes)')
ax3.set_title('FNO 2D: Training Duration')
ax3.grid(True, alpha=0.3)

# ── 4-6. Per-channel metrics (bar charts) ──
channels = ['z', 'y', 'p']
colors = {'z': '#1f77b4', 'y': '#ff7f0e', 'p': '#2ca02c'}
labels_long = {'z': 'Correlation (z)', 'y': 'Volatility (y)', 'p': 'Density (p)'}

for idx, ch in enumerate(channels, start=4):
    ax = plt.subplot(3, 3, idx)
    m = metrics[ch]
    x_pos = np.arange(3)
    vals = [m['sp_err'], m['tp_err'], m['var_ratio']]
    labels = ['sp_err', 'tp_err', 'var_ratio']
    bars = ax.bar(x_pos, vals, color=colors[ch], alpha=0.7, edgecolor='black', linewidth=1.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel('Value')
    ax.set_title(f'FNO 2D: {labels_long[ch]}')
    ax.grid(True, alpha=0.3, axis='y')
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2e}' if height < 0.01 else f'{height:.3f}',
                ha='center', va='bottom', fontsize=9)

# ── 7. Cross-field comparison: sp_err ──
ax7 = plt.subplot(3, 3, 7)
sp_errs = [metrics[ch]['sp_err'] for ch in channels]
bars = ax7.bar(channels, sp_errs, color=[colors[ch] for ch in channels], alpha=0.7, edgecolor='black', linewidth=1.5)
ax7.set_ylabel('Spatial Error (sp_err)')
ax7.set_title('FNO 2D: Spatial PSD Error (Cross-field)')
ax7.grid(True, alpha=0.3, axis='y')
for bar in bars:
    height = bar.get_height()
    ax7.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.2e}', ha='center', va='bottom', fontsize=10)

# ── 8. Cross-field comparison: tp_err ──
ax8 = plt.subplot(3, 3, 8)
tp_errs = [metrics[ch]['tp_err'] for ch in channels]
bars = ax8.bar(channels, tp_errs, color=[colors[ch] for ch in channels], alpha=0.7, edgecolor='black', linewidth=1.5)
ax8.set_ylabel('Temporal Error (tp_err)')
ax8.set_title('FNO 2D: Temporal PSD Error (Cross-field)')
ax8.grid(True, alpha=0.3, axis='y')
for bar in bars:
    height = bar.get_height()
    ax8.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.2e}', ha='center', va='bottom', fontsize=10)

# ── 9. Cross-field comparison: var_ratio ──
ax9 = plt.subplot(3, 3, 9)
var_ratios = [metrics[ch]['var_ratio'] for ch in channels]
bars = ax9.bar(channels, var_ratios, color=[colors[ch] for ch in channels], alpha=0.7, edgecolor='black', linewidth=1.5)
ax9.axhline(1.0, color='red', linestyle='--', alpha=0.5, linewidth=2, label='Target (1.0)')
ax9.set_ylabel('Variance Ratio')
ax9.set_title('FNO 2D: Variance Ratio (Cross-field)')
ax9.legend()
ax9.grid(True, alpha=0.3, axis='y')
for bar in bars:
    height = bar.get_height()
    ax9.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.3f}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig(results_dir / "fno2d_comprehensive_analysis.png", dpi=150, bbox_inches='tight')
print(f"[OK] Saved comprehensive FNO analysis -> {results_dir / 'fno2d_comprehensive_analysis.png'}")

# ── Summary table ──
print("\n" + "="*80)
print("FNO 2D COMPREHENSIVE METRICS")
print("="*80)
print(f"\nTraining Summary:")
print(f"  Epochs trained: {len(history)}")
print(f"  Best epoch: {best_epoch} (val_loss={metrics['val_total_loaded']:.6f})")
print(f"  Total time: {elapsed_mins[-1]:.1f} minutes")
print(f"  Best val loss: {metrics['val_total_loaded']:.6e}")

print(f"\nPer-Channel Metrics:")
print(f"{'Channel':<12} {'sp_err':<15} {'tp_err':<15} {'var_ratio':<15}")
print("-" * 60)
for ch in channels:
    m = metrics[ch]
    print(f"{ch:<12} {m['sp_err']:<15.3e} {m['tp_err']:<15.3e} {m['var_ratio']:<15.3f}")

print(f"\nVariance Details (Pred vs True):")
print(f"{'Channel':<12} {'var_pred':<20} {'var_true':<20}")
print("-" * 55)
for ch in channels:
    m = metrics[ch]
    print(f"{ch:<12} {m['var_pred']:<20.6e} {m['var_true']:<20.6e}")

print("\n" + "="*80)
