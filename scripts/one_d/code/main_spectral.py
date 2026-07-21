#!/usr/bin/env python3
"""
Spectral/moment-loss variant for the correlation field z.

Motivation: a PINN cannot match a specific realization of a stochastic
field pointwise (confirmed empirically), but it can match the field's
*distributional fingerprint* — mean, variance, spatial/temporal power
spectra. These are deterministic functions of the SPDE and its driving
noise statistics, independent of which particular noise path you drew.

Plan:
  - For z: compute empirical targets from an ensemble of FD realizations,
    replace pointwise data loss with a spectral/moment loss. PDE residual,
    IC, BC are retained.
  - For y, p: unchanged (near-deterministic, baseline results are fine).

The evaluation table is extended to report moment and spectrum error for
z, alongside pointwise L2 for y and p.

Output dir: results_spectral/
"""

import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import Config
from noise import generate_noise_field, NoiseInterpolator
from models import HierarchicalPINNSystem
from classical_solver import solve_classical
from trainer import (
    Trainer,
    _sample_interior, _sample_ic, _sample_bc, _sample_data,
)
from pde_residuals import (
    compute_all_residuals,
    ic_loss, bc_loss_neumann, bc_loss_dirichlet,
    normalisation_penalty,
)
from visualization import (
    evaluate_pinn_on_grid, plot_field_comparison, plot_snapshots,
    plot_training_loss, plot_cross_sections, plot_combined_evolution,
)
from main import parse_args, apply_overrides


DEFAULT_OUTPUT_DIR = "results_spectral"
N_ENSEMBLE = 32
NX_EVAL = 64           # subsampled grid for spectral eval each epoch
NT_EVAL = 128
W_SPECTRAL = 5.0       # weight on spectral loss (replaces w_data for z)


# ═══════════════════════════════════════════════════════════════════════
#  Ensemble target generation
# ═══════════════════════════════════════════════════════════════════════

def generate_ensemble_z(cfg: Config, n: int, base_seed: int = 123):
    """Run FD n times with different noise seeds, return stacked z fields."""
    Z_list = []
    rng_master = np.random.default_rng(base_seed)
    for i in range(n):
        seed = int(rng_master.integers(0, 10_000_000))
        rng = np.random.default_rng(seed)
        nz = generate_noise_field(cfg, "z", rng)
        ny = generate_noise_field(cfg, "y", rng)
        npp = generate_noise_field(cfg, "p", rng)
        _, _, Z, _, _ = solve_classical(cfg, nz, ny, npp)
        Z_list.append(Z.astype(np.float32))
    return np.stack(Z_list, axis=0)   # (n, Nx, Nt)


def subsample_indices(full_n: int, target_n: int) -> np.ndarray:
    """Evenly-spaced integer indices, inclusive endpoints."""
    return np.linspace(0, full_n - 1, target_n).astype(int)


def compute_spectral_targets(Z_ens: np.ndarray, cfg: Config,
                              Nx_eval: int, Nt_eval: int, device):
    """
    Subsample each ensemble member to the evaluation grid and compute:
      - mean / variance (scalars)
      - spatial PSD averaged over t and ensemble
      - temporal PSD averaged over x and ensemble
    Returns a dict of torch tensors on `device`.
    """
    ix = subsample_indices(cfg.Nx, Nx_eval)
    it = subsample_indices(cfg.Nt, Nt_eval)
    Z_sub = Z_ens[:, ix][:, :, it]                       # (n, Nx_eval, Nt_eval)

    mean_target = float(Z_sub.mean())
    var_target = float(Z_sub.var())

    # Spatial PSD: rfft along x, average magnitude over (ens, t)
    Fx = np.fft.rfft(Z_sub, axis=1)                       # (n, kx, t)
    spatial_psd = (np.abs(Fx) ** 2).mean(axis=(0, 2))     # (kx,)

    # Temporal PSD: rfft along t, average over (ens, x)
    Ft = np.fft.rfft(Z_sub, axis=2)                       # (n, x, w)
    temporal_psd = (np.abs(Ft) ** 2).mean(axis=(0, 1))    # (w,)

    return {
        "mean": torch.tensor(mean_target, device=device),
        "var":  torch.tensor(var_target, device=device),
        "spatial_psd":  torch.tensor(spatial_psd,  dtype=torch.float32, device=device),
        "temporal_psd": torch.tensor(temporal_psd, dtype=torch.float32, device=device),
        "x_idx": ix,
        "t_idx": it,
        "Z_ens": Z_ens,                                   # keep for post-hoc eval
    }


def build_eval_grid(cfg: Config, Nx_eval: int, Nt_eval: int, device):
    """Return flat (N,1) tensors x_eval, t_eval; shape used for spectral loss."""
    x_lin = np.linspace(cfg.x_min, cfg.x_max, Nx_eval, dtype=np.float32)
    t_lin = np.linspace(cfg.t_min, cfg.t_max, Nt_eval, dtype=np.float32)
    xx, tt = np.meshgrid(x_lin, t_lin, indexing="ij")
    x_eval = torch.tensor(xx.ravel(), device=device).unsqueeze(-1)
    t_eval = torch.tensor(tt.ravel(), device=device).unsqueeze(-1)
    return x_eval, t_eval


# ═══════════════════════════════════════════════════════════════════════
#  Spectral loss
# ═══════════════════════════════════════════════════════════════════════

def spectral_loss(Z_pred: torch.Tensor, targets: dict,
                   eps: float = 1e-10) -> tuple:
    """
    Z_pred: (Nx_eval, Nt_eval) torch tensor (with grad).
    Returns (total_loss, dict_of_components) for logging.
    """
    mean_p = Z_pred.mean()
    var_p = Z_pred.var()

    Fx = torch.fft.rfft(Z_pred, dim=0)
    spatial_psd = (Fx.abs() ** 2).mean(dim=1)

    Ft = torch.fft.rfft(Z_pred, dim=1)
    temporal_psd = (Ft.abs() ** 2).mean(dim=0)

    L_mean = (mean_p - targets["mean"]) ** 2
    L_var  = (var_p  - targets["var"])  ** 2

    # Log-space matching for spectra: wide dynamic range
    L_spec_x = F.mse_loss(torch.log(spatial_psd + eps),
                           torch.log(targets["spatial_psd"] + eps))
    L_spec_t = F.mse_loss(torch.log(temporal_psd + eps),
                           torch.log(targets["temporal_psd"] + eps))

    # Relative weighting: var is typically O(0.001), spectra log-values O(1-10)
    total = L_mean + 100.0 * L_var + L_spec_x + L_spec_t
    return total, {
        "mean": L_mean.item(),
        "var": L_var.item(),
        "spec_x": L_spec_x.item(),
        "spec_t": L_spec_t.item(),
    }


# ═══════════════════════════════════════════════════════════════════════
#  Spectral trainer — replaces z's pointwise data loss with spectral loss
# ═══════════════════════════════════════════════════════════════════════

class SpectralTrainer(Trainer):
    def __init__(self, *args, spectral_targets, x_eval, t_eval, **kwargs):
        super().__init__(*args, **kwargs)
        self.spec_targets = spectral_targets
        self.x_eval = x_eval
        self.t_eval = t_eval
        self.history.setdefault("spec_z", [])

    def _step(self, use_pde: bool, pde_ramp: float = 1.0):
        cfg = self.cfg
        dev = self.device

        # Sampling (same as parent)
        x_ic, t_ic, z0, y0, p0 = _sample_ic(cfg, cfg.batch_ic, dev)
        x_lo, x_hi, t_bc = _sample_bc(cfg, cfg.batch_bc, dev)
        x_d, t_d, z_d, y_d, p_d = _sample_data(
            self.x_grid, self.t_grid,
            self.Z_ref, self.Y_ref, self.P_ref,
            cfg.batch_data, dev,
        )

        # IC
        L_ic = (ic_loss(self.system.net_z, x_ic, t_ic, z0)
                + ic_loss(self.system.net_y, x_ic, t_ic, y0)
                + ic_loss(self.system.net_p, x_ic, t_ic, p0))

        # BC
        L_bc = (bc_loss_neumann(self.system.net_z, x_lo, t_bc)
                + bc_loss_neumann(self.system.net_z, x_hi, t_bc)
                + bc_loss_neumann(self.system.net_y, x_lo, t_bc)
                + bc_loss_neumann(self.system.net_y, x_hi, t_bc)
                + bc_loss_dirichlet(self.system.net_p, x_lo, t_bc)
                + bc_loss_dirichlet(self.system.net_p, x_hi, t_bc))

        # Data loss for y, p ONLY (z uses spectral loss instead)
        _, y_pred, p_pred = self.system.forward_all(x_d, t_d)
        L_data = (torch.mean((y_pred - y_d) ** 2)
                  + torch.mean((p_pred - p_d) ** 2))

        # Constraints
        L_pos_y = torch.mean(torch.relu(-y_pred) ** 2)
        t_norm = torch.rand(4, 1, device=dev) * (cfg.t_max - cfg.t_min) + cfg.t_min
        L_norm_p = normalisation_penalty(self.system.net_p, t_norm, cfg, n_quad=48)

        # Spectral loss for z on eval grid
        z_eval = self.system.net_z(self.x_eval, self.t_eval)
        Z_eval = z_eval.view(NX_EVAL, NT_EVAL)
        L_spec_z, spec_components = spectral_loss(Z_eval, self.spec_targets)

        # PDE residuals
        if use_pde:
            x_c, t_c = _sample_interior(cfg, cfg.batch_coll, dev)
            R_z, R_y, R_p = compute_all_residuals(
                self.system, x_c, t_c,
                self.noise_z, self.noise_y, self.noise_p, cfg,
            )
            L_pde_z = torch.mean(R_z ** 2)
            L_pde_y = torch.mean(R_y ** 2)
            L_pde_p = torch.mean(R_p ** 2)
        else:
            L_pde_z = L_pde_y = L_pde_p = torch.tensor(0.0, device=dev)
            pde_ramp = 0.0

        loss = (pde_ramp * cfg.w_pde_z * L_pde_z
                + pde_ramp * cfg.w_pde_y * L_pde_y
                + pde_ramp * cfg.w_pde_p * L_pde_p
                + cfg.w_ic * L_ic
                + cfg.w_bc * L_bc
                + cfg.w_data * L_data
                + cfg.w_pos_y * L_pos_y
                + cfg.w_norm_p * L_norm_p
                + W_SPECTRAL * L_spec_z)

        self.optimiser.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.system.parameters(),
                                       max_norm=cfg.grad_clip)
        self.optimiser.step()

        return {
            "total": loss.item(),
            "pde_z": L_pde_z.item(),
            "pde_y": L_pde_y.item(),
            "pde_p": L_pde_p.item(),
            "ic":    L_ic.item(),
            "bc":    L_bc.item(),
            "data":  L_data.item(),
            "pos_y": L_pos_y.item(),
            "norm_p": L_norm_p.item(),
            "spec_z": L_spec_z.item(),
        }


# ═══════════════════════════════════════════════════════════════════════
#  Post-hoc spectral evaluation + plots
# ═══════════════════════════════════════════════════════════════════════

def spectral_eval_report(Z_pinn_full: np.ndarray, Z_ens: np.ndarray,
                          out_path: str):
    """
    Compare statistical fingerprint of the PINN's z field against the FD
    ensemble. Prints a summary and saves a figure.
    """
    # Match shapes via full-grid Z (PINN) vs ensemble Z (many x Nx x Nt)
    pinn_mean = Z_pinn_full.mean()
    pinn_var = Z_pinn_full.var()
    ens_mean = Z_ens.mean()
    ens_var = Z_ens.var()

    # Spatial PSD (avg over t)
    Fx_pinn = np.abs(np.fft.rfft(Z_pinn_full, axis=0)) ** 2
    sp_pinn = Fx_pinn.mean(axis=1)
    Fx_ens = np.abs(np.fft.rfft(Z_ens, axis=1)) ** 2
    sp_ens = Fx_ens.mean(axis=(0, 2))

    # Temporal PSD (avg over x)
    Ft_pinn = np.abs(np.fft.rfft(Z_pinn_full, axis=1)) ** 2
    tp_pinn = Ft_pinn.mean(axis=0)
    Ft_ens = np.abs(np.fft.rfft(Z_ens, axis=2)) ** 2
    tp_ens = Ft_ens.mean(axis=(0, 1))

    print("\n" + "=" * 62)
    print("  z-field statistical fingerprint (PINN vs FD ensemble)")
    print("-" * 62)
    print(f"  mean      : PINN={pinn_mean: .4e}  Ens={ens_mean: .4e}  "
          f"|diff|={abs(pinn_mean-ens_mean): .4e}")
    print(f"  variance  : PINN={pinn_var: .4e}  Ens={ens_var: .4e}  "
          f"ratio={pinn_var/max(ens_var,1e-15): .3f}")
    eps = 1e-12
    sp_err = np.mean((np.log(sp_pinn + eps) - np.log(sp_ens + eps)) ** 2)
    tp_err = np.mean((np.log(tp_pinn + eps) - np.log(tp_ens + eps)) ** 2)
    print(f"  log spatial PSD MSE : {sp_err:.4e}")
    print(f"  log temporal PSD MSE: {tp_err:.4e}")
    print("=" * 62 + "\n")

    # Plot spectra overlay
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].loglog(sp_ens, "k-", lw=1.5, label="FD ensemble")
    axes[0].loglog(sp_pinn, "r--", lw=1.5, label="PINN")
    axes[0].set_title("Spatial PSD  (avg over t)")
    axes[0].set_xlabel("wavenumber k")
    axes[0].set_ylabel("|FFT|²")
    axes[0].legend(); axes[0].grid(True, alpha=0.3, which="both")

    axes[1].loglog(tp_ens, "k-", lw=1.5, label="FD ensemble")
    axes[1].loglog(tp_pinn, "r--", lw=1.5, label="PINN")
    axes[1].set_title("Temporal PSD  (avg over x)")
    axes[1].set_xlabel("frequency ω")
    axes[1].legend(); axes[1].grid(True, alpha=0.3, which="both")

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")

    return {"sp_err": sp_err, "tp_err": tp_err,
            "var_ratio": pinn_var / max(ens_var, 1e-15)}


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    cfg = Config()
    cfg.output_dir = DEFAULT_OUTPUT_DIR
    apply_overrides(cfg, args)

    os.makedirs(cfg.output_dir, exist_ok=True)

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Config: Nx={cfg.Nx}, Nt={cfg.Nt}, epochs={cfg.num_epochs}")
    print(f"[SPEC] ensemble={N_ENSEMBLE}, eval grid={NX_EVAL}x{NT_EVAL}, "
          f"w_spectral={W_SPECTRAL}")

    # ── Reference FD run (same seed for y/p data loss) ───────────────
    print("\n[1/5] Generating noise fields + reference FD solution ...")
    t0 = time.time()
    noise_z_grid = generate_noise_field(cfg, "z", rng)
    noise_y_grid = generate_noise_field(cfg, "y", rng)
    noise_p_grid = generate_noise_field(cfg, "p", rng)
    x_grid, t_grid, Z_ref, Y_ref, P_ref = solve_classical(
        cfg, noise_z_grid, noise_y_grid, noise_p_grid,
    )
    print(f"  Done in {time.time()-t0:.2f}s")

    # ── Ensemble for z statistics ────────────────────────────────────
    print(f"\n[2/5] Running ensemble of {N_ENSEMBLE} FD realizations ...")
    t0 = time.time()
    Z_ens = generate_ensemble_z(cfg, N_ENSEMBLE, base_seed=cfg.seed + 1000)
    print(f"  Done in {time.time()-t0:.2f}s  (shape: {Z_ens.shape})")

    # ── Build spectral targets ───────────────────────────────────────
    spectral_targets = compute_spectral_targets(
        Z_ens, cfg, NX_EVAL, NT_EVAL, device,
    )
    print(f"  target mean  = {spectral_targets['mean'].item(): .4e}")
    print(f"  target var   = {spectral_targets['var'].item(): .4e}")
    print(f"  target spatial PSD length: {spectral_targets['spatial_psd'].shape[0]}")
    print(f"  target temporal PSD length: {spectral_targets['temporal_psd'].shape[0]}")

    x_eval, t_eval = build_eval_grid(cfg, NX_EVAL, NT_EVAL, device)

    # ── Build PINN and train ─────────────────────────────────────────
    print("\n[3/5] Building PINN system ...")
    system = HierarchicalPINNSystem(cfg).to(device)
    n_params = sum(p.numel() for p in system.parameters())
    print(f"  Total parameters: {n_params:,}")

    noise_interp_z = NoiseInterpolator(noise_z_grid, x_grid, t_grid, device)
    noise_interp_y = NoiseInterpolator(noise_y_grid, x_grid, t_grid, device)
    noise_interp_p = NoiseInterpolator(noise_p_grid, x_grid, t_grid, device)

    trainer = SpectralTrainer(
        cfg, system,
        noise_interp_z, noise_interp_y, noise_interp_p,
        x_grid, t_grid, Z_ref, Y_ref, P_ref,
        device,
        spectral_targets=spectral_targets,
        x_eval=x_eval, t_eval=t_eval,
    )
    print("\n[4/5] Training ...")
    history = trainer.train()

    # ── Visualise + report ───────────────────────────────────────────
    print("\n[5/5] Generating visualisations ...")

    # Evaluate on full FD grid
    Z_pinn, Y_pinn, P_pinn = evaluate_pinn_on_grid(
        system, x_grid, t_grid, device,
    )

    # Standard plots
    out = cfg.output_dir
    plot_combined_evolution(x_grid, t_grid,
                             Z_ref, Y_ref, P_ref,
                             Z_pinn, Y_pinn, P_pinn,
                             os.path.join(out, "combined_evolution.png"))
    for ref, pinn, lbl, tag in zip(
        [Z_ref, Y_ref, P_ref], [Z_pinn, Y_pinn, P_pinn],
        ["z (correlation)", "y (volatility)", "p (density)"],
        ["z", "y", "p"],
    ):
        plot_field_comparison(x_grid, t_grid, ref, pinn, lbl,
                               os.path.join(out, f"field_{tag}.png"))
        plot_snapshots(x_grid, t_grid, ref, pinn, lbl,
                        os.path.join(out, f"snapshots_{tag}.png"))
    plot_training_loss(history, os.path.join(out, "training_loss.png"))
    plot_cross_sections(x_grid, t_grid,
                         [Z_ref, Y_ref, P_ref], [Z_pinn, Y_pinn, P_pinn],
                         ["z (correlation)", "y (volatility)", "p (density)"],
                         os.path.join(out, "cross_sections.png"),
                         x_vals=(-1.0, 0.0, 1.0))

    # Spectral report + plots
    spec_report = spectral_eval_report(
        Z_pinn, Z_ens,
        os.path.join(out, "spectral_fingerprint.png"),
    )

    # Pointwise metrics for y, p (still meaningful)
    print("=" * 62)
    print(f"  {'Field':<20} {'L2-rel':<14} {'Linf':<14} {'RMSE':<14}")
    print("-" * 62)
    dx, dt = cfg.dx, cfg.dt
    for ref, pinn, lbl in zip([Y_ref, P_ref], [Y_pinn, P_pinn],
                              ["y (volatility)", "p (density)"]):
        err = np.abs(ref - pinn)
        l2_ref = np.sqrt(np.sum(ref ** 2) * dx * dt)
        l2_err = np.sqrt(np.sum(err ** 2) * dx * dt)
        l2_rel = l2_err / max(l2_ref, 1e-15)
        linf = err.max()
        rmse = np.sqrt(np.mean(err ** 2))
        print(f"  {lbl:<20} {l2_rel:<14.4e} {linf:<14.4e} {rmse:<14.4e}")
    print("=" * 62)

    # Save model
    model_path = os.path.join(cfg.output_dir, "model.pt")
    torch.save(system.state_dict(), model_path)
    print(f"\nModel saved to {model_path}")
    print("Done.")


if __name__ == "__main__":
    main()
