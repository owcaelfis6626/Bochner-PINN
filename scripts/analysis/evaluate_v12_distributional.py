#!/usr/bin/env python3
"""
Distributional re-evaluation of v12 ensemble.

The v12 report compared PSD( mean over models )  vs  mean over FD realizations of PSDs.
That is the wrong comparison: averaging network outputs pointwise before taking PSD
collapses the realized variance and gives an artificially smooth "PSD of the mean".

This script recomputes the proper distributional metrics:
    PSD-per-realization → mean across realizations        (both sides)

Also reports the spread (std across the 6 PINN models) so we can see whether
the ensemble is tight or loose as a sample of the stochastic process.

Inputs:  results_spectral_v12/ensemble_fields.npz  (already contains Z_ens_pinn)
Outputs: results_spectral_v12/distributional_fingerprint.png
         results_spectral_v12/distributional_metrics.txt
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import Config
from main_spectral_v3 import generate_ensemble_zy


V12_DIR = "results_spectral_v12"
NPZ_IN  = os.path.join(V12_DIR, "ensemble_fields.npz")
FIG_OUT = os.path.join(V12_DIR, "distributional_fingerprint.png")
TXT_OUT = os.path.join(V12_DIR, "distributional_metrics.txt")

N_ENSEMBLE = 256  # same as v12


def compute_psds(Z_ens: np.ndarray) -> tuple:
    """
    Z_ens: (N_real, Nx, Nt) — stack of realizations
    Returns (spatial_psd_mean, temporal_psd_mean, spatial_psds_per, temporal_psds_per)
    where _mean is mean over realizations and _per is (N_real, K).
    """
    # Spatial PSD: rfft along x, squared magnitude, averaged over t
    Fx = np.fft.rfft(Z_ens, axis=1)                   # (N, kx, t)
    spatial_psds = (np.abs(Fx) ** 2).mean(axis=2)     # (N, kx)

    # Temporal PSD: rfft along t, squared magnitude, averaged over x
    Ft = np.fft.rfft(Z_ens, axis=2)                   # (N, x, w)
    temporal_psds = (np.abs(Ft) ** 2).mean(axis=1)    # (N, w)

    return (spatial_psds.mean(axis=0),
            temporal_psds.mean(axis=0),
            spatial_psds,
            temporal_psds)


def log_mse(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    return float(np.mean((np.log(a + eps) - np.log(b + eps)) ** 2))


def main():
    cfg = Config()

    print(f"[load] {NPZ_IN}")
    data = np.load(NPZ_IN)
    Z_ens_pinn = data["Z_ens_pinn"]           # (6, Nx, Nt)
    N_pinn, Nx, Nt = Z_ens_pinn.shape
    print(f"  PINN ensemble: {N_pinn} models, Nx={Nx}, Nt={Nt}")

    # Regenerate the FD ensemble with the same recipe v12 used
    print(f"\n[regen FD ensemble] {N_ENSEMBLE} realizations, seed={cfg.seed+1000}")
    Z_fd, Y_fd = generate_ensemble_zy(cfg, N_ENSEMBLE, base_seed=cfg.seed + 1000)
    print(f"  FD ensemble shape: {Z_fd.shape}")

    # ── Distributional PSDs ──────────────────────────────────────────────
    sp_fd_mean, tp_fd_mean, sp_fd_all, tp_fd_all = compute_psds(Z_fd)
    sp_pinn_mean, tp_pinn_mean, sp_pinn_all, tp_pinn_all = compute_psds(Z_ens_pinn)

    # ── Old (incorrect) comparison for reference ─────────────────────────
    Z_pinn_mean_field = Z_ens_pinn.mean(axis=0)                  # (Nx, Nt)
    Fx_mf = np.abs(np.fft.rfft(Z_pinn_mean_field, axis=0)) ** 2
    sp_pinn_mf = Fx_mf.mean(axis=1)
    Ft_mf = np.abs(np.fft.rfft(Z_pinn_mean_field, axis=1)) ** 2
    tp_pinn_mf = Ft_mf.mean(axis=0)

    # ── Metrics ──────────────────────────────────────────────────────────
    sp_err_dist = log_mse(sp_pinn_mean, sp_fd_mean)
    tp_err_dist = log_mse(tp_pinn_mean, tp_fd_mean)
    sp_err_old  = log_mse(sp_pinn_mf,   sp_fd_mean)
    tp_err_old  = log_mse(tp_pinn_mf,   tp_fd_mean)

    # Variance statistics
    var_fd_per   = Z_fd.var(axis=(1, 2))                  # (256,)
    var_pinn_per = Z_ens_pinn.var(axis=(1, 2))            # (6,)
    var_ratio_dist = var_pinn_per.mean() / var_fd_per.mean()
    var_ratio_old  = Z_pinn_mean_field.var() / Z_fd.mean(0).var() if Z_fd.mean(0).var() > 0 else np.nan

    # ── Report ───────────────────────────────────────────────────────────
    lines = []
    def p(s):
        print(s); lines.append(s)

    p("=" * 70)
    p("  DISTRIBUTIONAL RE-EVALUATION OF V12 ENSEMBLE")
    p("=" * 70)
    p(f"  PINN ensemble size: {N_pinn}")
    p(f"  FD   ensemble size: {N_ENSEMBLE}")
    p("")
    p("  Variance (per-realization, then averaged)")
    p(f"    FD   : mean={var_fd_per.mean():.4e}  std={var_fd_per.std():.4e}")
    p(f"    PINN : mean={var_pinn_per.mean():.4e}  std={var_pinn_per.std():.4e}")
    p(f"    var_ratio  (distributional)   = {var_ratio_dist:.3f}")
    p(f"    var_ratio  (old, mean-field)  = {var_ratio_old:.3f}")
    p("")
    p("  log-PSD MSE (PINN vs FD)")
    p(f"    spatial  (distributional)  = {sp_err_dist:.4e}")
    p(f"    spatial  (old, mean-field) = {sp_err_old:.4e}")
    p(f"    temporal (distributional)  = {tp_err_dist:.4e}")
    p(f"    temporal (old, mean-field) = {tp_err_old:.4e}")
    p("")
    p("  Interpretation:")
    p("    If distributional << old => v12's ensemble reporting was artificially")
    p("    degraded by the mean-field PSD mistake; individual models are fine.")
    p("    If distributional ~= old => the PSD mismatch is a real PINN limitation,")
    p("    not an averaging artifact.")
    p("=" * 70)

    with open(TXT_OUT, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nSaved metrics: {TXT_OUT}")

    # ── Plot ─────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    # Spatial
    ax = axes[0]
    for i in range(N_ENSEMBLE):
        ax.loglog(sp_fd_all[i], color="k", lw=0.3, alpha=0.08)
    ax.loglog(sp_fd_mean,   "k-",  lw=2.0, label=f"FD mean (N={N_ENSEMBLE})")
    for i in range(N_pinn):
        ax.loglog(sp_pinn_all[i], color="C3", lw=0.6, alpha=0.5)
    ax.loglog(sp_pinn_mean,   "r-",  lw=2.0, label=f"PINN mean (N={N_pinn})")
    ax.loglog(sp_pinn_mf,     "r--", lw=1.5, label="PINN mean-field (old, wrong)")
    ax.set_title("Spatial PSD")
    ax.set_xlabel("wavenumber k")
    ax.set_ylabel("|FFT|²")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which="both")

    # Temporal
    ax = axes[1]
    for i in range(N_ENSEMBLE):
        ax.loglog(tp_fd_all[i], color="k", lw=0.3, alpha=0.08)
    ax.loglog(tp_fd_mean,   "k-",  lw=2.0, label=f"FD mean (N={N_ENSEMBLE})")
    for i in range(N_pinn):
        ax.loglog(tp_pinn_all[i], color="C3", lw=0.6, alpha=0.5)
    ax.loglog(tp_pinn_mean,   "r-",  lw=2.0, label=f"PINN mean (N={N_pinn})")
    ax.loglog(tp_pinn_mf,     "r--", lw=1.5, label="PINN mean-field (old, wrong)")
    ax.set_title("Temporal PSD")
    ax.set_xlabel("frequency omega")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which="both")

    fig.suptitle("v12 — distributional vs mean-field PSD comparison", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG_OUT, bbox_inches="tight", dpi=130)
    plt.close(fig)
    print(f"Saved figure:  {FIG_OUT}")


if __name__ == "__main__":
    main()
