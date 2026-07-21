"""
Comprehensive Spectral-PINN 2D analysis: ensemble metrics across seeds.
Matches the FNO analysis style.
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Load data
results_dir = Path(__file__).parent / "results_pinn_2d_spectral"
eval_file = results_dir / "pinn_spectral_eval.json"

with open(eval_file) as f:
    data = json.load(f)

channels = ['z', 'y', 'p']
colors = {'z': '#1f77b4', 'y': '#ff7f0e', 'p': '#2ca02c'}
labels_long = {'z': 'Correlation (z)', 'y': 'Volatility (y)', 'p': 'Density (p)'}

fig = plt.figure(figsize=(16, 12))

# ── 1-3. Per-channel metrics (bar charts) ──
for idx, ch in enumerate(channels, start=1):
    ax = plt.subplot(3, 3, idx)
    ensemble_stats = data['ensemble'][ch]

    x_pos = np.arange(3)
    vals = [
        ensemble_stats['sp_err']['mean'],
        ensemble_stats['tp_err']['mean'],
        ensemble_stats['var_ratio']['mean'],
    ]
    errs = [
        ensemble_stats['sp_err']['std'],
        ensemble_stats['tp_err']['std'],
        ensemble_stats['var_ratio']['std'],
    ]
    labels = ['sp_err', 'tp_err', 'var_ratio']

    bars = ax.bar(x_pos, vals, yerr=errs, capsize=5, color=colors[ch], alpha=0.7,
                   edgecolor='black', linewidth=1.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel('Value')
    ax.set_title(f'Spectral-PINN 2D: {labels_long[ch]} (8 seeds)')
    ax.grid(True, alpha=0.3, axis='y')
    # Add value labels on bars
    for bar, val, err in zip(bars, vals, errs):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + err + 0.05,
                f'{height:.2e}' if height < 0.01 else f'{height:.3f}',
                ha='center', va='bottom', fontsize=9)

# ── 4. Cross-field sp_err ──
ax = plt.subplot(3, 3, 4)
sp_errs = [data['ensemble'][ch]['sp_err']['mean'] for ch in channels]
sp_stds = [data['ensemble'][ch]['sp_err']['std'] for ch in channels]
bars = ax.bar(channels, sp_errs, yerr=sp_stds, capsize=5,
              color=[colors[ch] for ch in channels], alpha=0.7, edgecolor='black', linewidth=1.5)
ax.set_ylabel('Spatial Error (sp_err)')
ax.set_title('Spectral-PINN 2D: Spatial PSD Error (Cross-field)')
ax.grid(True, alpha=0.3, axis='y')
for bar, val, err in zip(bars, sp_errs, sp_stds):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + err + 0.01,
            f'{height:.2e}', ha='center', va='bottom', fontsize=10)

# ── 5. Cross-field tp_err ──
ax = plt.subplot(3, 3, 5)
tp_errs = [data['ensemble'][ch]['tp_err']['mean'] for ch in channels]
tp_stds = [data['ensemble'][ch]['tp_err']['std'] for ch in channels]
bars = ax.bar(channels, tp_errs, yerr=tp_stds, capsize=5,
              color=[colors[ch] for ch in channels], alpha=0.7, edgecolor='black', linewidth=1.5)
ax.set_ylabel('Temporal Error (tp_err)')
ax.set_title('Spectral-PINN 2D: Temporal PSD Error (Cross-field)')
ax.grid(True, alpha=0.3, axis='y')
for bar, val, err in zip(bars, tp_errs, tp_stds):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + err + 0.01,
            f'{height:.2e}', ha='center', va='bottom', fontsize=10)

# ── 6. Cross-field var_ratio ──
ax = plt.subplot(3, 3, 6)
var_ratios = [data['ensemble'][ch]['var_ratio']['mean'] for ch in channels]
var_stds = [data['ensemble'][ch]['var_ratio']['std'] for ch in channels]
bars = ax.bar(channels, var_ratios, yerr=var_stds, capsize=5,
              color=[colors[ch] for ch in channels], alpha=0.7, edgecolor='black', linewidth=1.5)
ax.axhline(1.0, color='red', linestyle='--', alpha=0.5, linewidth=2, label='Target (1.0)')
ax.set_ylabel('Variance Ratio')
ax.set_title('Spectral-PINN 2D: Variance Ratio (Cross-field)')
ax.set_ylim([0, 1.5])
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
for bar, val, err in zip(bars, var_ratios, var_stds):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + err + 0.02,
            f'{height:.3f}', ha='center', va='bottom', fontsize=10)

# ── 7. Seed variance: sp_err ──
ax = plt.subplot(3, 3, 7)
for ch_idx, ch in enumerate(channels):
    seed_sp_errs = []
    for seed in range(1, 9):
        # Compute mean across test seeds for this PINN seed
        test_sp_errs = []
        for ts in data['test_seeds']:
            if str(seed) in data['per_seed'] and ts in data['per_seed'][str(seed)][ch]:
                test_sp_errs.append(data['per_seed'][str(seed)][ts][ch]['sp_err'])
        if test_sp_errs:
            seed_sp_errs.append(np.mean(test_sp_errs))
    seeds = np.arange(1, len(seed_sp_errs) + 1)
    ax.plot(seeds, seed_sp_errs, 'o-', label=ch, color=colors[ch], markersize=6)

ax.set_xlabel('PINN Seed')
ax.set_ylabel('Spatial PSD Error (sp_err)')
ax.set_title('Spectral-PINN 2D: Per-Seed Variability (sp_err)')
ax.legend()
ax.grid(True, alpha=0.3)

# ── 8. Seed variance: tp_err ──
ax = plt.subplot(3, 3, 8)
for ch_idx, ch in enumerate(channels):
    seed_tp_errs = []
    for seed in range(1, 9):
        test_tp_errs = []
        for ts in data['test_seeds']:
            if str(seed) in data['per_seed'] and ts in data['per_seed'][str(seed)][ch]:
                test_tp_errs.append(data['per_seed'][str(seed)][ts][ch]['tp_err'])
        if test_tp_errs:
            seed_tp_errs.append(np.mean(test_tp_errs))
    seeds = np.arange(1, len(seed_tp_errs) + 1)
    ax.plot(seeds, seed_tp_errs, 's-', label=ch, color=colors[ch], markersize=6)

ax.set_xlabel('PINN Seed')
ax.set_ylabel('Temporal PSD Error (tp_err)')
ax.set_title('Spectral-PINN 2D: Per-Seed Variability (tp_err)')
ax.legend()
ax.grid(True, alpha=0.3)

# ── 9. Seed variance: var_ratio ──
ax = plt.subplot(3, 3, 9)
for ch_idx, ch in enumerate(channels):
    seed_var_ratios = []
    for seed in range(1, 9):
        test_var_ratios = []
        for ts in data['test_seeds']:
            if str(seed) in data['per_seed'] and ts in data['per_seed'][str(seed)][ch]:
                test_var_ratios.append(data['per_seed'][str(seed)][ts][ch]['var_ratio'])
        if test_var_ratios:
            seed_var_ratios.append(np.mean(test_var_ratios))
    seeds = np.arange(1, len(seed_var_ratios) + 1)
    ax.plot(seeds, seed_var_ratios, '^-', label=ch, color=colors[ch], markersize=6)

ax.axhline(1.0, color='red', linestyle='--', alpha=0.5, linewidth=1.5)
ax.set_xlabel('PINN Seed')
ax.set_ylabel('Variance Ratio')
ax.set_title('Spectral-PINN 2D: Per-Seed Variability (var_ratio)')
ax.set_ylim([0, 2])
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(results_dir / "spectral_pinn_2d_comprehensive_analysis.png", dpi=150, bbox_inches='tight')
print(f"[OK] Saved comprehensive Spectral-PINN analysis -> {results_dir / 'spectral_pinn_2d_comprehensive_analysis.png'}")

# ── Summary table ──
print("\n" + "="*80)
print("SPECTRAL-PINN 2D COMPREHENSIVE METRICS (8 seeds)")
print("="*80)

print(f"\nPer-Channel Metrics:")
print(f"{'Channel':<12} {'sp_err':<20} {'tp_err':<20} {'var_ratio':<15}")
print("-" * 70)
for ch in channels:
    m = data['ensemble'][ch]
    print(f"{ch:<12} {m['sp_err']['mean']:.3e} ± {m['sp_err']['std']:.3e}  "
          f"{m['tp_err']['mean']:.3e} ± {m['tp_err']['std']:.3e}  "
          f"{m['var_ratio']['mean']:.3f} ± {m['var_ratio']['std']:.3f}")

print(f"\nVariance Details:")
print(f"{'Channel':<12} {'var_pred':<20} {'var_true':<20}")
print("-" * 55)
for ch in channels:
    # Aggregate variance stats
    var_preds = []
    var_trues = []
    for seed in range(1, 9):
        for ts in data['test_seeds']:
            if str(seed) in data['per_seed'] and ts in data['per_seed'][str(seed)][ch]:
                m = data['per_seed'][str(seed)][ts][ch]
                var_preds.append(m['var_pred'])
                var_trues.append(m['var_true'])
    if var_preds:
        print(f"{ch:<12} {np.mean(var_preds):<20.6e} {np.mean(var_trues):<20.6e}")

print("\n" + "="*80)
