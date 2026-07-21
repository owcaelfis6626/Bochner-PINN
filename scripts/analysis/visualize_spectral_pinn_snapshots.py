"""
Spectral-PINN field snapshots: Visualize z(x,t), y(x,t), p(x,t) at key time slices.
Compares FD reference vs ensemble mean Spectral-PINN predictions as heatmaps.
"""
import sys, os
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent / "spde_2d"))

import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

from config_pinn_2d_spectral import ConfigPINN2DSpectral
from models_2d import HierarchicalPINNSystem2D
from trainer_2d import _build_realisation

# ── Config & Paths ──
cfg = ConfigPINN2DSpectral()
device = torch.device("cpu")  # Use CPU for visualization
results_dir = Path(__file__).parent / "results_pinn_2d_spectral"
pinn_seeds = list(range(1, 9))

# ── Generate FD reference (test seed = 999, different from training) ──
print("[viz] Building FD reference field (test seed=999)...")
_, grids, (Z_fd, Y_fd, P_fd) = _build_realisation(cfg, seed=999, use_jumps=True, device=device)
x_grid, y_grid, t_grid = grids
Z_fd = Z_fd.cpu().numpy() if hasattr(Z_fd, 'cpu') else Z_fd
Y_fd = Y_fd.cpu().numpy() if hasattr(Y_fd, 'cpu') else Y_fd
P_fd = P_fd.cpu().numpy() if hasattr(P_fd, 'cpu') else P_fd
x_grid = x_grid.numpy() if hasattr(x_grid, 'numpy') else x_grid
y_grid = y_grid.numpy() if hasattr(y_grid, 'numpy') else y_grid
t_grid = t_grid.numpy() if hasattr(t_grid, 'numpy') else t_grid
Nx, Ny, Nt = Z_fd.shape
print(f"[viz] Grid shape: {Nx}x{Ny}x{Nt}")

# ── Load Spectral-PINN models and compute ensemble mean ──
print(f"[viz] Loading {len(pinn_seeds)} Spectral-PINN checkpoints...")
Z_pinn_ensemble = np.zeros((len(pinn_seeds), Nx, Ny, Nt))
Y_pinn_ensemble = np.zeros((len(pinn_seeds), Nx, Ny, Nt))
P_pinn_ensemble = np.zeros((len(pinn_seeds), Nx, Ny, Nt))

for seed_idx, ps in enumerate(pinn_seeds):
    ckpt_path = results_dir / f"spinn2d_seed{ps}.pt"
    print(f"[viz] Loading seed {ps}...")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    system = HierarchicalPINNSystem2D(cfg).to(device)
    system.load_state_dict(ckpt['model'])
    system.eval()

    # Evaluate on full grid
    with torch.no_grad():
        xx, yy = np.meshgrid(x_grid, y_grid, indexing='ij')

        for ti in range(Nt):
            tt = np.full((Nx, Ny), t_grid[ti])
            xx_flat = torch.tensor(xx.flatten(), dtype=torch.float32, device=device).view(-1, 1)
            yy_flat = torch.tensor(yy.flatten(), dtype=torch.float32, device=device).view(-1, 1)
            tt_flat = torch.tensor(tt.flatten(), dtype=torch.float32, device=device).view(-1, 1)

            z, y, p = system.forward_all(xx_flat, yy_flat, tt_flat)
            Z_pinn_ensemble[seed_idx, :, :, ti] = z.cpu().numpy().reshape(Nx, Ny)
            Y_pinn_ensemble[seed_idx, :, :, ti] = y.cpu().numpy().reshape(Nx, Ny)
            P_pinn_ensemble[seed_idx, :, :, ti] = p.cpu().numpy().reshape(Nx, Ny)

# Compute ensemble means
Z_pinn = Z_pinn_ensemble.mean(axis=0)
Y_pinn = Y_pinn_ensemble.mean(axis=0)
P_pinn = P_pinn_ensemble.mean(axis=0)
print("[viz] Computed ensemble mean predictions")

# ── Select time slices (early, middle, late) ──
t_indices = [Nt // 4, Nt // 2, 3 * Nt // 4]
t_labels = [f"t={t_grid[Nt // 4]:.2f} (25%)", f"t={t_grid[Nt // 2]:.2f} (50%)", f"t={t_grid[3 * Nt // 4]:.2f} (75%)"]

channels = ['z', 'y', 'p']
fields_fd = {'z': Z_fd, 'y': Y_fd, 'p': P_fd}
fields_pinn = {'z': Z_pinn, 'y': Y_pinn, 'p': P_pinn}
titles = {'z': 'Correlation (z)', 'y': 'Volatility (y)', 'p': 'Density (p)'}

# ── Create grid: 3 channels × 3 time slices × 2 (FD vs Spectral-PINN) ──
fig = plt.figure(figsize=(16, 14))
gs = fig.add_gridspec(3, 6, hspace=0.4, wspace=0.3)

for ch_idx, ch in enumerate(channels):
    for t_idx, (ti, t_label) in enumerate(zip(t_indices, t_labels)):
        field_fd = fields_fd[ch][:, :, ti]
        field_pinn = fields_pinn[ch][:, :, ti]

        vmin = min(field_fd.min(), field_pinn.min())
        vmax = max(field_fd.max(), field_pinn.max())

        # FD snapshot
        ax_fd = fig.add_subplot(gs[ch_idx, 2*t_idx])
        im_fd = ax_fd.contourf(x_grid, y_grid, field_fd, levels=20, cmap='RdBu_r')
        ax_fd.set_title(f'FD: {titles[ch]} — {t_label}', fontsize=10, weight='bold')
        ax_fd.set_xlabel('x')
        ax_fd.set_ylabel('y')
        plt.colorbar(im_fd, ax=ax_fd, label='Value')

        # Spectral-PINN snapshot
        ax_pinn = fig.add_subplot(gs[ch_idx, 2*t_idx + 1])
        im_pinn = ax_pinn.contourf(x_grid, y_grid, field_pinn, levels=20, cmap='RdBu_r')
        ax_pinn.set_title(f'Spectral-PINN: {titles[ch]} — {t_label}', fontsize=10, weight='bold')
        ax_pinn.set_xlabel('x')
        ax_pinn.set_ylabel('y')
        plt.colorbar(im_pinn, ax=ax_pinn, label='Value')

plt.suptitle('Field Snapshots: FD (Classical) vs Spectral-PINN Predictions (Ensemble Mean, 8 seeds)\nTest Seed 999',
             fontsize=14, weight='bold', y=0.995)
plt.savefig(Path(__file__).parent / "field_snapshots_fd_vs_spectral_pinn.png", dpi=150, bbox_inches='tight')
print("[OK] Saved field snapshots -> field_snapshots_fd_vs_spectral_pinn.png")

# ── Compute error maps at key times ──
print("\n[viz] Computing error maps...")
fig_err = plt.figure(figsize=(16, 10))
gs_err = fig_err.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

for ch_idx, ch in enumerate(channels):
    for t_idx, (ti, t_label) in enumerate(zip(t_indices, t_labels)):
        field_fd = fields_fd[ch][:, :, ti]
        field_pinn = fields_pinn[ch][:, :, ti]
        error = np.abs(field_pinn - field_fd)

        ax = fig_err.add_subplot(gs_err[ch_idx, t_idx])
        im = ax.contourf(x_grid, y_grid, error, levels=20, cmap='hot')
        ax.set_title(f'{titles[ch]} Error — {t_label}\nmax={error.max():.3e}, mean={error.mean():.3e}',
                    fontsize=9)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        plt.colorbar(im, ax=ax, label='|Error|')

fig_err.suptitle('Absolute Error: |Spectral-PINN - FD| at Key Time Slices (Ensemble Mean)', fontsize=14, weight='bold')
plt.savefig(Path(__file__).parent / "field_errors_spectral_pinn_vs_fd.png", dpi=150, bbox_inches='tight')
print("[OK] Saved error maps -> field_errors_spectral_pinn_vs_fd.png")

print("\n" + "="*70)
print("SPECTRAL-PINN FIELD SNAPSHOT ANALYSIS")
print("="*70)
print(f"Grid: {Nx}x{Ny}x{Nt}")
print(f"Test seed: 999 (independent from training)")
print(f"Ensemble: {len(pinn_seeds)} Spectral-PINN seeds (ensemble mean shown)")
print(f"\nVisualizations:")
print(f"  1. field_snapshots_fd_vs_spectral_pinn.png (6x3 grid: FD & Spectral-PINN side-by-side)")
print(f"  2. field_errors_spectral_pinn_vs_fd.png (error heatmaps at 3 time slices)")
print("="*70)
