"""
Evaluate Spectral-PINN 2D ensemble (8 seeds) on test set.
Computes sp_err, tp_err, var_ratio for each seed and ensemble statistics.
"""
import json
import numpy as np
import torch
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent / "spde_2d"))

from config_pinn_2d_spectral import ConfigPINN2DSpectral
from models_2d import HierarchicalPINNSystem2D
from trainer_2d import _build_realisation, _spatial_psd_xy, _temporal_psd

# ── Config ──
cfg = ConfigPINN2DSpectral()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
results_dir = Path(__file__).parent / "results_pinn_2d_spectral"
test_seeds = [2001, 2002, 2003, 2004]
pinn_seeds = list(range(1, 9))  # 8 seeds

print(f"[eval-spectral] device={device}")
print(f"[eval-spectral] test_seeds={test_seeds}")
print(f"[eval-spectral] pinn_seeds={pinn_seeds}")

# ── Load test realisations ──
test_data = {}
for ts in test_seeds:
    print(f"[eval-spectral] Building test realisation seed={ts}...")
    _, grids, fields = _build_realisation(cfg, seed=ts, use_jumps=True, device=device)
    Z_fd, Y_fd, P_fd = fields
    test_data[ts] = {
        'Z': Z_fd.cpu().numpy() if hasattr(Z_fd, 'cpu') else Z_fd,
        'Y': Y_fd.cpu().numpy() if hasattr(Y_fd, 'cpu') else Y_fd,
        'P': P_fd.cpu().numpy() if hasattr(P_fd, 'cpu') else P_fd,
        'grids': grids,
    }

# ── Evaluate each PINN seed ──
results = {}
for ps in pinn_seeds:
    ckpt_path = results_dir / f"spinn2d_seed{ps}.pt"
    print(f"\n[eval-spectral] === PINN seed {ps} ===")

    if not ckpt_path.exists():
        print(f"[eval-spectral] WARNING: {ckpt_path} not found!")
        continue

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    system = HierarchicalPINNSystem2D(cfg).to(device)
    system.load_state_dict(ckpt['model'])
    system.eval()

    seed_metrics = {'z': {}, 'y': {}, 'p': {}}

    for ts in test_seeds:
        print(f"  Test seed {ts}...")
        data = test_data[ts]
        x_grid, y_grid, t_grid = data['grids']
        Nx, Ny, Nt = len(x_grid), len(y_grid), len(t_grid)

        # Build meshgrid for PINN inference
        xx, yy = np.meshgrid(x_grid, y_grid, indexing='ij')

        # Evaluate PINN on dense grid
        with torch.no_grad():
            z_pred = np.zeros((Nx, Ny, Nt))
            y_pred = np.zeros((Nx, Ny, Nt))
            p_pred = np.zeros((Nx, Ny, Nt))

            # Process in batches to avoid memory issues
            batch_size = 512
            for ti in range(Nt):
                tt = np.full((Nx, Ny), t_grid[ti])
                xx_flat = torch.tensor(xx.flatten(), dtype=torch.float32, device=device).view(-1, 1)
                yy_flat = torch.tensor(yy.flatten(), dtype=torch.float32, device=device).view(-1, 1)
                tt_flat = torch.tensor(tt.flatten(), dtype=torch.float32, device=device).view(-1, 1)

                z, y, p = system.forward_all(xx_flat, yy_flat, tt_flat)
                z_pred[:, :, ti] = z.cpu().numpy().reshape(Nx, Ny)
                y_pred[:, :, ti] = y.cpu().numpy().reshape(Nx, Ny)
                p_pred[:, :, ti] = p.cpu().numpy().reshape(Nx, Ny)

        # Compute metrics
        for ch, pred, fd in [('z', z_pred, data['Z']),
                              ('y', y_pred, data['Y']),
                              ('p', p_pred, data['P'])]:
            sp_pred = _spatial_psd_xy(torch.tensor(pred, dtype=torch.float32, device=device).unsqueeze(0)).squeeze(0)
            sp_fd = _spatial_psd_xy(torch.tensor(fd, dtype=torch.float32, device=device).unsqueeze(0)).squeeze(0)
            sp_err = ((torch.log10(sp_pred.clamp(min=1e-12)) - torch.log10(sp_fd.clamp(min=1e-12))) ** 2).mean().item()

            tp_pred = _temporal_psd(torch.tensor(pred, dtype=torch.float32, device=device).unsqueeze(0)).squeeze(0)
            tp_fd = _temporal_psd(torch.tensor(fd, dtype=torch.float32, device=device).unsqueeze(0)).squeeze(0)
            tp_err = ((torch.log10(tp_pred.clamp(min=1e-12)) - torch.log10(tp_fd.clamp(min=1e-12))) ** 2).mean().item()

            var_pred = float(pred.var())
            var_fd = float(fd.var())
            var_ratio = var_pred / (var_fd + 1e-12)

            seed_metrics[ch][ts] = {
                'sp_err': float(sp_err),
                'tp_err': float(tp_err),
                'var_ratio': float(var_ratio),
                'var_pred': var_pred,
                'var_true': var_fd,
            }

    results[ps] = seed_metrics

# ── Compute ensemble statistics ──
ensemble_stats = {'z': {}, 'y': {}, 'p': {}}
for ch in ['z', 'y', 'p']:
    for metric in ['sp_err', 'tp_err', 'var_ratio']:
        all_values = []
        for ps in pinn_seeds:
            for ts in test_seeds:
                if ps in results and ts in results[ps][ch]:
                    all_values.append(results[ps][ch][ts][metric])

        if all_values:
            ensemble_stats[ch][metric] = {
                'mean': float(np.mean(all_values)),
                'std': float(np.std(all_values)),
                'min': float(np.min(all_values)),
                'max': float(np.max(all_values)),
            }

# ── Save results ──
output = {
    'n_pinn_seeds': len(pinn_seeds),
    'test_seeds': test_seeds,
    'per_seed': results,
    'ensemble': ensemble_stats,
}

out_file = results_dir / "pinn_spectral_eval.json"
with open(out_file, 'w') as f:
    json.dump(output, f, indent=2)

print(f"\n[OK] Saved Spectral-PINN eval -> {out_file}")

# ── Print summary ──
print("\n" + "="*70)
print("SPECTRAL-PINN 2D ENSEMBLE EVALUATION (8 seeds × 4 test seeds)")
print("="*70)
for ch in ['z', 'y', 'p']:
    print(f"\nChannel: {ch}")
    for metric in ['sp_err', 'tp_err', 'var_ratio']:
        stats = ensemble_stats[ch][metric]
        print(f"  {metric:12s}: {stats['mean']:.3e} ± {stats['std']:.3e} "
              f"(min={stats['min']:.3e}, max={stats['max']:.3e})")

print("\n" + "="*70)
