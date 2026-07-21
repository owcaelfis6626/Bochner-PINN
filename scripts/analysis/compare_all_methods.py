"""
Comprehensive 3-way comparison: FNO vs vanilla PINN vs Spectral-PINN.
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Load all results
print("[compare] Loading evaluation results...")

with open(Path(__file__).parent / "results_fno_2d" / "fno2d_metrics.json") as f:
    fno_metrics = json.load(f)

with open(Path(__file__).parent / "results_pinn_2d" / "pinn_ensemble_eval.json") as f:
    vanilla_pinn_eval = json.load(f)

with open(Path(__file__).parent / "results_pinn_2d_spectral" / "pinn_spectral_eval.json") as f:
    spectral_pinn_eval = json.load(f)

channels = ['z', 'y', 'p']
labels_long = {'z': 'Correlation (z)', 'y': 'Volatility (y)', 'p': 'Density (p)'}
colors = {'z': '#1f77b4', 'y': '#ff7f0e', 'p': '#2ca02c'}

# Extract metrics
fno_data = {ch: {} for ch in channels}
for ch in channels:
    fno_data[ch] = {
        'sp_err': fno_metrics[ch]['sp_err'],
        'tp_err': fno_metrics[ch]['tp_err'],
        'var_ratio': fno_metrics[ch]['var_ratio'],
    }

vanilla_pinn_data = {ch: {} for ch in channels}
for ch in channels:
    if ch in vanilla_pinn_eval['pinn_ensemble']:
        vanilla_pinn_data[ch] = {
            'sp_err': vanilla_pinn_eval['pinn_ensemble'][ch]['sp_err_mean'],
            'tp_err': vanilla_pinn_eval['pinn_ensemble'][ch]['tp_err_mean'],
            'var_ratio': vanilla_pinn_eval['pinn_ensemble'][ch]['var_ratio_mean'],
            'sp_err_std': vanilla_pinn_eval['pinn_ensemble'][ch]['sp_err_std'],
            'tp_err_std': vanilla_pinn_eval['pinn_ensemble'][ch]['tp_err_std'],
            'var_ratio_std': vanilla_pinn_eval['pinn_ensemble'][ch]['var_ratio_std'],
        }

spectral_pinn_data = {ch: {} for ch in channels}
for ch in channels:
    if ch in spectral_pinn_eval['ensemble']:
        spectral_pinn_data[ch] = {
            'sp_err': spectral_pinn_eval['ensemble'][ch]['sp_err']['mean'],
            'tp_err': spectral_pinn_eval['ensemble'][ch]['tp_err']['mean'],
            'var_ratio': spectral_pinn_eval['ensemble'][ch]['var_ratio']['mean'],
            'sp_err_std': spectral_pinn_eval['ensemble'][ch]['sp_err']['std'],
            'tp_err_std': spectral_pinn_eval['ensemble'][ch]['tp_err']['std'],
            'var_ratio_std': spectral_pinn_eval['ensemble'][ch]['var_ratio']['std'],
        }

# Create comparison plots
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Row 1: sp_err comparison
for idx, ch in enumerate(channels):
    ax = axes[0, idx]
    methods = ['FNO', 'Vanilla\nPINN', 'Spectral-\nPINN']
    sp_errs = [
        fno_data[ch]['sp_err'],
        vanilla_pinn_data[ch]['sp_err'],
        spectral_pinn_data[ch]['sp_err'],
    ]
    errs = [
        0,
        vanilla_pinn_data[ch]['sp_err_std'],
        spectral_pinn_data[ch]['sp_err_std'],
    ]

    bars = ax.bar(methods, sp_errs, yerr=errs, capsize=5, color=['#d62728', '#1f77b4', '#2ca02c'],
                   alpha=0.7, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Spatial PSD Error (log-MSE)')
    ax.set_title(f'{labels_long[ch]} - Spatial Error', fontweight='bold')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')
    for i, (bar, val) in enumerate(zip(bars, sp_errs)):
        ax.text(bar.get_x() + bar.get_width()/2., val * 1.5,
                f'{val:.2e}', ha='center', va='bottom', fontsize=9, fontweight='bold')

# Row 2: tp_err comparison
for idx, ch in enumerate(channels):
    ax = axes[1, idx]
    methods = ['FNO', 'Vanilla\nPINN', 'Spectral-\nPINN']
    tp_errs = [
        fno_data[ch]['tp_err'],
        vanilla_pinn_data[ch]['tp_err'],
        spectral_pinn_data[ch]['tp_err'],
    ]
    errs = [
        0,
        vanilla_pinn_data[ch]['tp_err_std'],
        spectral_pinn_data[ch]['tp_err_std'],
    ]

    bars = ax.bar(methods, tp_errs, yerr=errs, capsize=5, color=['#d62728', '#1f77b4', '#2ca02c'],
                   alpha=0.7, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Temporal PSD Error (log-MSE)')
    ax.set_title(f'{labels_long[ch]} - Temporal Error', fontweight='bold')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')
    for i, (bar, val) in enumerate(zip(bars, tp_errs)):
        ax.text(bar.get_x() + bar.get_width()/2., val * 1.5,
                f'{val:.2e}', ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(Path(__file__).parent / "comparison_sp_tp_errors.png", dpi=150, bbox_inches='tight')
print("[OK] Saved comparison_sp_tp_errors.png")

# Variance ratio comparison
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for idx, ch in enumerate(channels):
    ax = axes[idx]
    methods = ['FNO', 'Vanilla\nPINN', 'Spectral-\nPINN']
    var_ratios = [
        fno_data[ch]['var_ratio'],
        vanilla_pinn_data[ch]['var_ratio'],
        spectral_pinn_data[ch]['var_ratio'],
    ]
    errs = [
        0,
        vanilla_pinn_data[ch]['var_ratio_std'],
        spectral_pinn_data[ch]['var_ratio_std'],
    ]

    bars = ax.bar(methods, var_ratios, yerr=errs, capsize=5, color=['#d62728', '#1f77b4', '#2ca02c'],
                   alpha=0.7, edgecolor='black', linewidth=1.5)
    ax.axhline(1.0, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Target (1.0)')
    ax.set_ylabel('Variance Ratio (Pred / True)')
    ax.set_title(f'{labels_long[ch]} - Variance Matching', fontweight='bold')
    ax.set_ylim([0, 2.5])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    for i, (bar, val) in enumerate(zip(bars, var_ratios)):
        ax.text(bar.get_x() + bar.get_width()/2., val + 0.05,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(Path(__file__).parent / "comparison_var_ratios.png", dpi=150, bbox_inches='tight')
print("[OK] Saved comparison_var_ratios.png")

# Print comprehensive table
print("\n" + "="*100)
print("3-WAY COMPARISON: FNO vs Vanilla PINN vs Spectral-PINN")
print("="*100)
print("\nSPATIAL PSD ERROR (sp_err, lower is better):")
print("-" * 100)
print(f"{'Channel':<15} {'FNO':<20} {'Vanilla PINN':<25} {'Spectral-PINN':<25} {'Improvement*':<15}")
print("-" * 100)
for ch in channels:
    fno_sp = fno_data[ch]['sp_err']
    vanilla_sp = vanilla_pinn_data[ch]['sp_err']
    spectral_sp = spectral_pinn_data[ch]['sp_err']
    vanilla_std = vanilla_pinn_data[ch]['sp_err_std']
    spectral_std = spectral_pinn_data[ch]['sp_err_std']

    # Compute improvement ratio (vanilla vs spectral)
    if vanilla_sp > 0:
        improve_ratio = vanilla_sp / spectral_sp
    else:
        improve_ratio = 0

    print(f"{ch:<15} {fno_sp:.3e}          {vanilla_sp:.3e} ± {vanilla_std:.3e}  {spectral_sp:.3e} ± {spectral_std:.3e}  {improve_ratio:.1f}x")

print("\nTEMPORAL PSD ERROR (tp_err, lower is better):")
print("-" * 100)
print(f"{'Channel':<15} {'FNO':<20} {'Vanilla PINN':<25} {'Spectral-PINN':<25} {'Improvement*':<15}")
print("-" * 100)
for ch in channels:
    fno_tp = fno_data[ch]['tp_err']
    vanilla_tp = vanilla_pinn_data[ch]['tp_err']
    spectral_tp = spectral_pinn_data[ch]['tp_err']
    vanilla_std = vanilla_pinn_data[ch]['tp_err_std']
    spectral_std = spectral_pinn_data[ch]['tp_err_std']

    if vanilla_tp > 0:
        improve_ratio = vanilla_tp / spectral_tp
    else:
        improve_ratio = 0

    print(f"{ch:<15} {fno_tp:.3e}          {vanilla_tp:.3e} ± {vanilla_std:.3e}  {spectral_tp:.3e} ± {spectral_std:.3e}  {improve_ratio:.1f}x")

print("\nVARIANCE RATIO (target = 1.0, higher variance underestimation marked with *):")
print("-" * 100)
print(f"{'Channel':<15} {'FNO':<20} {'Vanilla PINN':<25} {'Spectral-PINN':<25} {'Notes':<15}")
print("-" * 100)
for ch in channels:
    fno_var = fno_data[ch]['var_ratio']
    vanilla_var = vanilla_pinn_data[ch]['var_ratio']
    spectral_var = spectral_pinn_data[ch]['var_ratio']
    vanilla_std = vanilla_pinn_data[ch]['var_ratio_std']
    spectral_std = spectral_pinn_data[ch]['var_ratio_std']

    notes = ""
    if vanilla_var < 0.3:
        notes = "Vanilla: severe underest."
    if spectral_var < 0.5 or spectral_var > 2.0:
        notes += " Spectral: outliers"

    print(f"{ch:<15} {fno_var:.3f}            {vanilla_var:.3f} ± {vanilla_std:.3f}        {spectral_var:.3f} ± {spectral_std:.3f}        {notes:<15}")

print("\n" + "="*100)
print("* Improvement = Vanilla PINN error / Spectral-PINN error (higher is better)")
print("="*100)
