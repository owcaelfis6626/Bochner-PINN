"""
Field snapshots: Visualize z(x,t), y(x,t), p(x,t) at key time slices.
Compares FD reference vs FNO predictions as heatmaps.
"""
import sys, os
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent / "spde_2d"))

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
from pathlib import Path

from config_pinn_2d import ConfigPINN2D
from models_2d import HierarchicalPINNSystem2D
from trainer_2d import _build_realisation, _spatial_psd_xy, _temporal_psd

# ── Config & Paths ──
cfg = ConfigPINN2D()
device = torch.device("cpu")  # Use CPU for visualization
results_dir = Path(__file__).parent / "results_fno_2d"
fno_ckpt = results_dir / "fno_best.pt"

# ── Generate FD reference (test seed = 999, different from training) ──
print("[viz] Building FD reference field (test seed=999)...")
_, grids, (Z_fd, Y_fd, P_fd) = _build_realisation(cfg, seed=999, use_jumps=True, device=device)
x_grid, y_grid, t_grid = grids
# Convert to numpy for visualization
Z_fd = Z_fd.cpu().numpy() if hasattr(Z_fd, 'cpu') else Z_fd
Y_fd = Y_fd.cpu().numpy() if hasattr(Y_fd, 'cpu') else Y_fd
P_fd = P_fd.cpu().numpy() if hasattr(P_fd, 'cpu') else P_fd
x_grid = x_grid.numpy() if hasattr(x_grid, 'numpy') else x_grid
y_grid = y_grid.numpy() if hasattr(y_grid, 'numpy') else y_grid
t_grid = t_grid.numpy() if hasattr(t_grid, 'numpy') else t_grid
Nx, Ny, Nt = Z_fd.shape
print(f"[viz] Grid shape: {Nx}x{Ny}x{Nt}")

# ── Load FNO model ──
print(f"[viz] Loading FNO checkpoint: {fno_ckpt}")
from fno_model_2d import FNO3d
ckpt = torch.load(fno_ckpt, map_location=device, weights_only=False)
from config_fno_2d import FNOConfig2D
fno_cfg = FNOConfig2D()
fno_model = FNO3d(
    in_channels=fno_cfg.in_channels,
    out_channels=fno_cfg.out_channels,
    modes_x=fno_cfg.modes_x,
    modes_y=fno_cfg.modes_y,
    modes_t=fno_cfg.modes_t,
    width=fno_cfg.width,
    n_layers=fno_cfg.n_layers,
    activation=fno_cfg.activation,
).to(device)
fno_model.load_state_dict(ckpt["model"])
fno_model.eval()
print("[viz] FNO model loaded.")

# ── Prepare FNO input (8-channel: noise_z, noise_y, noise_p, jump_z, jump_y, IC_z, IC_y, IC_p) ──
print("[viz] Preparing FNO input...")
from noise_2d import generate_noise_field
from vg_jumps_2d import generate_vg_jump_field
rng = np.random.default_rng(999)
nz = generate_noise_field(cfg, "z", rng)
ny = generate_noise_field(cfg, "y", rng)
npf = generate_noise_field(cfg, "p", rng)
jz = generate_vg_jump_field(cfg, rng=rng)
jy = generate_vg_jump_field(cfg, rng=rng)
nz_net = nz + jz / cfg.dt
ny_net = ny + jy / cfg.dt

# Stack into (8, Nx, Ny, Nt) input
fno_input = np.stack([nz_net, ny_net, npf, jz, jy, nz, ny, npf], axis=0)
fno_input_t = torch.tensor(fno_input, dtype=torch.float32, device=device).unsqueeze(0)  # (1, 8, Nx, Ny, Nt)
print(f"[viz] FNO input shape: {fno_input_t.shape}")

# ── FNO forward pass ──
print("[viz] Running FNO forward pass...")
with torch.no_grad():
    out_fno = fno_model(fno_input_t)  # (1, 3, Nx, Ny, Nt)
    out_fno = out_fno.squeeze(0).numpy()  # (3, Nx, Ny, Nt)
    Z_fno = out_fno[0]
    Y_fno = out_fno[1]
    P_fno = out_fno[2]
print(f"[viz] FNO output shapes: Z={Z_fno.shape}, Y={Y_fno.shape}, P={P_fno.shape}")

# ── Select time slices (early, middle, late) ──
t_indices = [Nt // 4, Nt // 2, 3 * Nt // 4]
t_labels = [f"t={t_grid[ti]:.2f} (25%)" for ti in t_indices]
t_labels = [f"t={t_grid[Nt // 4]:.2f} (25%)", f"t={t_grid[Nt // 2]:.2f} (50%)", f"t={t_grid[3 * Nt // 4]:.2f} (75%)"]

channels = ['z', 'y', 'p']
fields_fd = {'z': Z_fd, 'y': Y_fd, 'p': P_fd}
fields_fno = {'z': Z_fno, 'y': Y_fno, 'p': P_fno}
titles = {'z': 'Correlation (z)', 'y': 'Volatility (y)', 'p': 'Density (p)'}

# ── Create grid: 3 channels × 3 time slices × 2 (FD vs FNO) ──
fig = plt.figure(figsize=(16, 14))
gs = fig.add_gridspec(3, 6, hspace=0.4, wspace=0.3)

for ch_idx, ch in enumerate(channels):
    for t_idx, (ti, t_label) in enumerate(zip(t_indices, t_labels)):
        field_fd = fields_fd[ch][:, :, ti]
        field_fno = fields_fno[ch][:, :, ti]

        vmin = min(field_fd.min(), field_fno.min())
        vmax = max(field_fd.max(), field_fno.max())

        # FD snapshot
        ax_fd = fig.add_subplot(gs[ch_idx, 2*t_idx])
        im_fd = ax_fd.contourf(x_grid, y_grid, field_fd, levels=20, cmap='RdBu_r')
        ax_fd.set_title(f'FD: {titles[ch]} — {t_label}', fontsize=10, weight='bold')
        ax_fd.set_xlabel('x')
        ax_fd.set_ylabel('y')
        plt.colorbar(im_fd, ax=ax_fd, label='Value')

        # FNO snapshot
        ax_fno = fig.add_subplot(gs[ch_idx, 2*t_idx + 1])
        im_fno = ax_fno.contourf(x_grid, y_grid, field_fno, levels=20, cmap='RdBu_r')
        ax_fno.set_title(f'FNO: {titles[ch]} — {t_label}', fontsize=10, weight='bold')
        ax_fno.set_xlabel('x')
        ax_fno.set_ylabel('y')
        plt.colorbar(im_fno, ax=ax_fno, label='Value')

plt.suptitle('Field Snapshots: FD (Classical) vs FNO Predictions\nTest Seed 999',
             fontsize=14, weight='bold', y=0.995)
plt.savefig(Path(__file__).parent / "field_snapshots_fd_vs_fno.png", dpi=150, bbox_inches='tight')
print("[OK] Saved field snapshots -> field_snapshots_fd_vs_fno.png")

# ── Compute error maps at key times ──
print("\n[viz] Computing error maps...")
fig_err = plt.figure(figsize=(16, 10))
gs_err = fig_err.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

for ch_idx, ch in enumerate(channels):
    for t_idx, (ti, t_label) in enumerate(zip(t_indices, t_labels)):
        field_fd = fields_fd[ch][:, :, ti]
        field_fno = fields_fno[ch][:, :, ti]
        error = np.abs(field_fno - field_fd)

        ax = fig_err.add_subplot(gs_err[ch_idx, t_idx])
        im = ax.contourf(x_grid, y_grid, error, levels=20, cmap='hot')
        ax.set_title(f'{titles[ch]} Error — {t_label}\nmax={error.max():.3e}, mean={error.mean():.3e}',
                    fontsize=9)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        plt.colorbar(im, ax=ax, label='|Error|')

fig_err.suptitle('Absolute Error: |FNO - FD| at Key Time Slices', fontsize=14, weight='bold')
plt.savefig(Path(__file__).parent / "field_errors_fno_vs_fd.png", dpi=150, bbox_inches='tight')
print("[OK] Saved error maps -> field_errors_fno_vs_fd.png")

print("\n" + "="*70)
print("FIELD SNAPSHOT ANALYSIS")
print("="*70)
print(f"Grid: {Nx}×{Ny}×{Nt}")
print(f"Test seed: 999 (independent from training)")
print(f"FNO model: {fno_ckpt.name}")
print(f"\nVisualizations:")
print(f"  1. field_snapshots_fd_vs_fno.png (6×3 grid: FD & FNO side-by-side)")
print(f"  2. field_errors_fno_vs_fd.png (error heatmaps at 3 time slices)")
print("="*70)
