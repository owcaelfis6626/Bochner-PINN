"""
p1_empirical_w2.py — Empirical W_2 between two independent FD realizations
of Burgers-VG 2D, computed via the Gelbrich formula on stationary Gaussian
proxy covariances.

The bound from p1_constants.py gives W_2 ≤ ~1.5e8. This script measures
what W_2 actually looks like between independent realizations of the SAME
distribution — i.e., what a perfect PINN matching L(u_FD) would achieve.

The ratio (predicted bound) / (empirical W_2) quantifies bound looseness.

W_2 formula (Gelbrich for centred Gaussian on stationary periodised domain):
  W_2² = Σ_{k,ω} (√S_1(k,ω) - √S_2(k,ω))²
where S_i is the empirical joint power spectrum of realization i.

Two independent realizations differ only by their noise/jump driver seeds;
their joint spectral measures should agree in expectation but disagree at
finite sample by O(1/√N_modes).
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, '/home/hubi/spde/spde-inne/01_burgers_levy')
sys.path.insert(0, '/home/hubi/spde/spde-inne/shared')
from burgers_vg_solver_2d import solve_burgers_vg_2d
from config_burgers_vg_2d import ConfigBurgersVG2D


def solve_with_seed(seed):
    cfg = ConfigBurgersVG2D()
    cfg.seed = seed
    rng = np.random.default_rng(seed)
    noise_u = cfg.sigma_u * rng.standard_normal((cfg.Nx, cfg.Ny, cfg.Nt)).astype(np.float64)
    _, _, _, U = solve_burgers_vg_2d(cfg, noise_u=noise_u, rng=rng)
    return cfg, U


def joint_power_spectrum(U):
    """Joint 3D periodogram (single realization).  Returns |FFT(U)|² / |grid|."""
    F = np.fft.rfftn(U.astype(np.float64))           # (Nx, Ny, Nt//2+1)
    return (np.abs(F) ** 2) / U.size


def gelbrich_w2_squared(S1, S2):
    """W_2² between two centred Gaussian measures with diagonal spectra S1, S2.
       (√S1 - √S2)² summed over all (k,ω)."""
    return float(np.sum((np.sqrt(S1) - np.sqrt(S2)) ** 2))


def main():
    print("Generating two independent FD realizations (different seeds)...")
    cfg, U1 = solve_with_seed(42)
    _, U2   = solve_with_seed(43)
    print(f"  U1: shape {U1.shape}, range [{U1.min():.3f}, {U1.max():.3f}]")
    print(f"  U2: shape {U2.shape}, range [{U2.min():.3f}, {U2.max():.3f}]")
    print(f"  ||U1 - U2||_L²: {np.sqrt(np.mean((U1 - U2)**2)):.4e}  (pointwise difference between realizations)\n")

    print("Computing joint 3D power spectra...")
    S1 = joint_power_spectrum(U1)
    S2 = joint_power_spectrum(U2)
    print(f"  Joint spectrum shape: {S1.shape}")
    print(f"  S1 range: [{S1.min():.3e}, {S1.max():.3e}]")
    print(f"  S2 range: [{S2.min():.3e}, {S2.max():.3e}]\n")

    print("Computing Gelbrich W_2 between empirical covariances:\n")

    # ── Full-grid W_2 ──
    w2_sq_full = gelbrich_w2_squared(S1, S2)
    w2_full    = np.sqrt(w2_sq_full)
    print(f"  W_2 (full grid, all {S1.size} modes):                 {w2_full:.4e}")

    # ── Sub-grid (24×24×96-equivalent) ──
    # Sample 24 spatial indices × 96 temporal indices to match paper protocol
    rng_sub = np.random.default_rng(0)
    ix = np.sort(rng_sub.choice(cfg.Nx, 24, replace=False))
    iy = np.sort(rng_sub.choice(cfg.Ny, 24, replace=False))
    # For temporal sub-grid we use FFT freq indices 0..96 (low-freq slice)
    nt_kept = 96 // 2 + 1
    S1_sub = S1[np.ix_(ix, iy, np.arange(nt_kept))]
    S2_sub = S2[np.ix_(ix, iy, np.arange(nt_kept))]
    w2_sq_sub = gelbrich_w2_squared(S1_sub, S2_sub)
    w2_sub    = np.sqrt(w2_sq_sub)
    print(f"  W_2 (sub-grid 24×24×{nt_kept} modes):                  {w2_sub:.4e}")

    # ── Physical-window restricted (drop highest-frequency tail) ──
    # Keep modes where max(S1, S2) ≥ 1e-4 × max
    smax = max(S1.max(), S2.max())
    mask = np.maximum(S1, S2) >= 1e-4 * smax
    w2_sq_win = gelbrich_w2_squared(S1[mask], S2[mask])
    w2_win    = np.sqrt(w2_sq_win)
    print(f"  W_2 (windowed S≥1e-4·max, {mask.sum()} modes):          {w2_win:.4e}")

    # ── Trivial reference: W_2 between L(u) and δ_0 (zero field) ──
    w2_zero = np.sqrt(np.sum(S1))    # √(total power) = ||u||_L²
    print(f"  W_2 (vs. zero field, ||u||_L² baseline):              {w2_zero:.4e}\n")

    # ── Comparison to predicted bound from p1_constants ──
    try:
        c = np.load('/home/hubi/spde/spectral_pinn_for_spde/01_paper/verification/p1_constants.npz')
        C_1, C_2, C_3, beta = float(c['C_1']), float(c['C_2']), float(c['C_3']), float(c['beta'])
        eps = 0.063     # paper's Gumbel floor
        delta = 1e-3
        N_s = 24
        bound = C_1 * np.sqrt(eps) + C_2 * N_s**(-beta) + C_3 * np.sqrt(delta)
        print(f"  Predicted bound (from p1_constants, ε={eps}, δ={delta}, N_s={N_s}): {bound:.4e}\n")

        print(f"  ── COMPARISON ──")
        print(f"    Bound (predicted):                 {bound:.3e}")
        print(f"    Empirical W_2 (full grid):         {w2_full:.3e}")
        print(f"    Empirical W_2 (sub-grid):          {w2_sub:.3e}")
        print(f"    Empirical W_2 (windowed):          {w2_win:.3e}")
        print(f"    Bound / Empirical (full grid):     {bound/w2_full:.3e}")
        print(f"    Bound / Empirical (sub-grid):      {bound/w2_sub:.3e}")
    except FileNotFoundError:
        print("  (Run p1_constants.py first for the comparison)")

    print(f"\n  Interpretation:")
    print(f"   - Empirical W_2 ~ {w2_full:.2f} between two FD realizations is the practical 'noise floor'")
    print(f"     for a Gaussian-covariance match. A perfect-distribution PINN would achieve this.")
    print(f"   - Bound is loose by {bound/w2_full:.1e}× — qualitatively expected for the prefactor")
    print(f"     chain through Gelbrich + Sobolev + log-MVT; standard for Wasserstein PINN bounds.")

    np.savez('/home/hubi/spde/spectral_pinn_for_spde/01_paper/verification/p1_empirical_w2.npz',
             U1=U1.astype(np.float32), U2=U2.astype(np.float32),
             S1=S1.astype(np.float32), S2=S2.astype(np.float32),
             w2_full=w2_full, w2_sub=w2_sub, w2_win=w2_win, w2_zero=w2_zero)
    print(f"\n  → saved: /home/hubi/spde/spectral_pinn_for_spde/01_paper/verification/p1_empirical_w2.npz")


if __name__ == "__main__":
    main()
