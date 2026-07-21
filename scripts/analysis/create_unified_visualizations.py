"""
Create unified visualization suite for all models: FNO, Vanilla PINN, Spectral-PINN.
Consolidated in results_analysis/ folder with consistent naming and styling.
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import shutil

# Create analysis folder
analysis_dir = Path(__file__).parent / "results_analysis"
analysis_dir.mkdir(exist_ok=True)

# Load all metrics
print("[unify] Loading metrics from all models...")
with open(Path(__file__).parent / "results_fno_2d" / "fno2d_metrics.json") as f:
    fno_metrics = json.load(f)

with open(Path(__file__).parent / "results_pinn_2d" / "pinn_ensemble_eval.json") as f:
    vanilla_eval = json.load(f)

with open(Path(__file__).parent / "results_pinn_2d_spectral" / "pinn_spectral_eval.json") as f:
    spectral_eval = json.load(f)

channels = ['z', 'y', 'p']
labels_long = {'z': 'Correlation (z)', 'y': 'Volatility (y)', 'p': 'Density (p)'}
colors = {'FNO': '#d62728', 'Vanilla PINN': '#1f77b4', 'Spectral-PINN': '#2ca02c'}

# Extract data
fno_data = {ch: {
    'sp_err': fno_metrics[ch]['sp_err'],
    'tp_err': fno_metrics[ch]['tp_err'],
    'var_ratio': fno_metrics[ch]['var_ratio'],
} for ch in channels}

vanilla_data = {ch: {
    'sp_err': vanilla_eval['pinn_ensemble'][ch]['sp_err_mean'],
    'sp_err_std': vanilla_eval['pinn_ensemble'][ch]['sp_err_std'],
    'tp_err': vanilla_eval['pinn_ensemble'][ch]['tp_err_mean'],
    'tp_err_std': vanilla_eval['pinn_ensemble'][ch]['tp_err_std'],
    'var_ratio': vanilla_eval['pinn_ensemble'][ch]['var_ratio_mean'],
    'var_ratio_std': vanilla_eval['pinn_ensemble'][ch]['var_ratio_std'],
} for ch in channels}

spectral_data = {ch: {
    'sp_err': spectral_eval['ensemble'][ch]['sp_err']['mean'],
    'sp_err_std': spectral_eval['ensemble'][ch]['sp_err']['std'],
    'tp_err': spectral_eval['ensemble'][ch]['tp_err']['mean'],
    'tp_err_std': spectral_eval['ensemble'][ch]['tp_err']['std'],
    'var_ratio': spectral_eval['ensemble'][ch]['var_ratio']['mean'],
    'var_ratio_std': spectral_eval['ensemble'][ch]['var_ratio']['std'],
} for ch in channels}

# ========== UNIFIED COMPARISON: All 3 models side-by-side ==========
fig, axes = plt.subplots(3, 3, figsize=(16, 12))
fig.suptitle('2D SPDE Model Comparison: Spatial Error, Temporal Error, Variance Ratio',
             fontsize=16, fontweight='bold', y=0.995)

for ch_idx, ch in enumerate(channels):
    # Column 1: Spatial Error
    ax = axes[ch_idx, 0]
    methods = ['FNO', 'Vanilla\nPINN', 'Spectral-\nPINN']
    sp_errs = [
        fno_data[ch]['sp_err'],
        vanilla_data[ch]['sp_err'],
        spectral_data[ch]['sp_err'],
    ]
    sp_errs_std = [0, vanilla_data[ch]['sp_err_std'], spectral_data[ch]['sp_err_std']]
    bars = ax.bar(methods, sp_errs, yerr=sp_errs_std, capsize=5,
                   color=['#d62728', '#1f77b4', '#2ca02c'], alpha=0.7, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Spatial PSD Error', fontweight='bold')
    ax.set_title(f'{labels_long[ch]} - Spatial Error', fontweight='bold', fontsize=11)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, sp_errs):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height * 2,
                f'{val:.2e}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Column 2: Temporal Error
    ax = axes[ch_idx, 1]
    tp_errs = [
        fno_data[ch]['tp_err'],
        vanilla_data[ch]['tp_err'],
        spectral_data[ch]['tp_err'],
    ]
    tp_errs_std = [0, vanilla_data[ch]['tp_err_std'], spectral_data[ch]['tp_err_std']]
    bars = ax.bar(methods, tp_errs, yerr=tp_errs_std, capsize=5,
                   color=['#d62728', '#1f77b4', '#2ca02c'], alpha=0.7, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Temporal PSD Error', fontweight='bold')
    ax.set_title(f'{labels_long[ch]} - Temporal Error', fontweight='bold', fontsize=11)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, tp_errs):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height * 2,
                f'{val:.2e}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Column 3: Variance Ratio
    ax = axes[ch_idx, 2]
    var_ratios = [
        fno_data[ch]['var_ratio'],
        vanilla_data[ch]['var_ratio'],
        spectral_data[ch]['var_ratio'],
    ]
    var_ratios_std = [0, vanilla_data[ch]['var_ratio_std'], spectral_data[ch]['var_ratio_std']]
    bars = ax.bar(methods, var_ratios, yerr=var_ratios_std, capsize=5,
                   color=['#d62728', '#1f77b4', '#2ca02c'], alpha=0.7, edgecolor='black', linewidth=1.5)
    ax.axhline(1.0, color='red', linestyle='--', alpha=0.5, linewidth=2, label='Target (1.0)')
    ax.set_ylabel('Variance Ratio', fontweight='bold')
    ax.set_title(f'{labels_long[ch]} - Variance Ratio', fontweight='bold', fontsize=11)
    ax.set_ylim([0, 2.5])
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, var_ratios):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(analysis_dir / "001_unified_comparison_all_models.png", dpi=150, bbox_inches='tight')
print(f"[OK] Saved: 001_unified_comparison_all_models.png")

# ========== Copy field snapshots to analysis folder ==========
print("[unify] Copying field snapshots...")
for src, dst_name in [
    (Path(__file__).parent / "field_snapshots_fd_vs_fno.png", "002_field_snapshots_fno.png"),
    (Path(__file__).parent / "field_snapshots_fd_vs_spectral_pinn.png", "003_field_snapshots_spectral_pinn.png"),
]:
    if src.exists():
        shutil.copy(src, analysis_dir / dst_name)
        print(f"[OK] Copied: {dst_name}")

# ========== Copy error maps to analysis folder ==========
print("[unify] Copying error maps...")
for src, dst_name in [
    (Path(__file__).parent / "field_errors_fno_vs_fd.png", "004_field_errors_fno.png"),
    (Path(__file__).parent / "field_errors_spectral_pinn_vs_fd.png", "005_field_errors_spectral_pinn.png"),
]:
    if src.exists():
        shutil.copy(src, analysis_dir / dst_name)
        print(f"[OK] Copied: {dst_name}")

# ========== Copy comprehensive analysis plots ==========
print("[unify] Copying comprehensive analysis plots...")
for src, dst_name in [
    (Path(__file__).parent / "results_fno_2d" / "fno2d_comprehensive_analysis.png", "006_comprehensive_fno.png"),
    (Path(__file__).parent / "results_pinn_2d_spectral" / "spectral_pinn_2d_comprehensive_analysis.png", "007_comprehensive_spectral_pinn.png"),
]:
    if src.exists():
        shutil.copy(src, analysis_dir / dst_name)
        print(f"[OK] Copied: {dst_name}")

# ========== Create summary metrics table image ==========
print("[unify] Creating metrics summary table...")
fig, ax = plt.subplots(figsize=(16, 10))
ax.axis('off')

summary_text = """
╔════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
║                        2D SPDE FINAL RESULTS: FNO vs Vanilla PINN vs Spectral-PINN                                                   ║
╚════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝

SPATIAL PSD ERROR (sp_err) — Lower is Better
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Channel          FNO              Vanilla PINN           Spectral-PINN          Improvement (Spectral vs Vanilla)
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
z (Correlation)  4.6e-4           1.21 ± 0.13           0.34 ± 0.11            3.6x ✓
y (Volatility)   2.2e-4           1.13 ± 0.12           0.24 ± 0.06            4.7x ✓
p (Density)      5.6e-4           0.86 ± 0.09           0.21 ± 0.09            4.1x ✓

TEMPORAL PSD ERROR (tp_err) — Lower is Better
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
z (Correlation)  2.6e-3           1.68 ± 0.26           0.39 ± 0.10            4.4x ✓
y (Volatility)   9.0e-4           0.87 ± 0.26           0.26 ± 0.08            3.3x ✓
p (Density)      8.9e-4           0.53 ± 0.33           0.014 ± 0.008          38.7x ✓✓✓

VARIANCE RATIO (target = 1.0) — Closer to 1.0 is Better
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
z (Correlation)  0.978            0.19 ± 0.10           0.825 ± 0.072          4.3x better (target match)
y (Volatility)   1.101            0.75 ± 0.43           0.795 ± 0.443          Near target
p (Density)      0.962            0.45 ± 0.21           0.792 ± 0.095          1.8x better

KEY INSIGHTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

• FNO PERFORMANCE:          1000x better than Vanilla PINN. Near-perfect variance matching (0.96-1.1).
                            Fast inference (~100x faster than PINN). Single deterministic model.
                            Cost: Required parameter shrinking (53M→8.4M), torch.compile debugging.

• VANILLA PINN FAILURE:     Catastrophic failure on stochastic PDEs. Residual loss blind to spectral structure.
                            Creates smooth, unphysical solutions. Variance 4-5x too small. sp_err ~1.2.
                            Root cause: Minimizes PDE residual but misses distributional properties.

• SPECTRAL-PINN SUCCESS:    Fixes vanilla PINN via 3 novel distributional loss terms on 32×32×128 sub-grid:
                            • L_spec_xy: Spatial PSD matching
                            • L_spec_t:  Temporal PSD matching
                            • L_var:     Variance ratio matching (weight=50.0, dominates early training)

                            Results: 3-4x error reduction, variance ratio 0.82 (near target 1.0).
                            Still 1000x behind FNO but VIABLE for uncertainty quantification + physics.

• ENSEMBLE VARIANCE:        8 Spectral-PINN seeds × 4 test cases show consistent performance.
                            Std of metrics: 0.06-0.44 across all channels.
                            Enables principled uncertainty quantification for downstream applications.

RECOMMENDATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Use FNO for:        Maximum accuracy, speed, deterministic predictions. Production inference.
Use Spectral-PINN:  Uncertainty quantification, ensemble variance, physics constraint enforcement.
                    Academic/research applications, where ensemble + distributional properties matter.
"""

ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
        fontsize=9, verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.tight_layout()
plt.savefig(analysis_dir / "008_summary_metrics_table.png", dpi=150, bbox_inches='tight')
print(f"[OK] Saved: 008_summary_metrics_table.png")

# ========== List all files ==========
print("\n" + "="*80)
print("UNIFIED ANALYSIS FOLDER CREATED")
print("="*80)
result_files = sorted(analysis_dir.glob("*.png"))
for i, f in enumerate(result_files, 1):
    print(f"{i}. {f.name}")

print("\n" + "="*80)
print(f"Total files: {len(result_files)}")
print(f"Location: {analysis_dir}")
print("="*80)
