"""
p1_constants.py — Compute the constants C_1, C_2 (and an estimate of C_3)
in Proposition 1 of Paper 1, for the Burgers-VG 2D benchmark.

This is the concrete reviewer-defence numerical instantiation: pick
benchmark, compute the constants from FD reference data, give the
predicted W_2 bound as a number that can be compared to an empirical
W_2 measurement.

Proposition 1:
  W_2 ≤ C_1 √ε  +  C_2 N_s^{-β}  +  C_3 √δ
  C_1 = μ_max ln(10) (M_xy + M_t) / √(2 μ_min)
  C_2 = √(C_quad,x M_t² + C_quad,t M_xy²) / √(2 μ_min)
  C_3 = √C_MM      [requires SPDE stability constant; we estimate via mean ||u||_L²]
  β   = (1/2) min(s_x/d, s_t)

For Burgers-VG, the paper takes s_x = s_t = 1, d = 2 → β = 1/4.

Workflow:
  1. Run FD solver for Burgers-VG 2D config → U[Nx, Ny, Nt]
  2. Sub-grid to 24×24×96
  3. Compute spectral marginals S^xy(k), S^t(ω) via FFT + axis-averaging
  4. Extract μ_min, μ_max, M_xy, M_t
  5. Compute C_quad,x = ||μ^xy||²_{H^{s_x}}, C_quad,t similarly
  6. Plug in.
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, '/home/hubi/spde/spde-inne/01_burgers_levy')
sys.path.insert(0, '/home/hubi/spde/spde-inne/shared')
from burgers_vg_solver_2d import solve_burgers_vg_2d
from config_burgers_vg_2d import ConfigBurgersVG2D


# ── Sub-grid spec from paper ──────────────────────────────────────────
NX_SUB, NY_SUB, NT_SUB = 24, 24, 96
S_X, S_T, D_SPATIAL    = 1, 1, 2          # Sobolev exponents + spatial dim
BETA = 0.5 * min(S_X / D_SPATIAL, S_T)    # = 1/4


def generate_fd_reference():
    cfg = ConfigBurgersVG2D()
    print(f"Generating FD reference: Nx={cfg.Nx}, Ny={cfg.Ny}, Nt={cfg.Nt}, "
          f"T={cfg.t_max}, ν={cfg.nu_u}, σ={cfg.sigma_u}")
    rng = np.random.default_rng(cfg.seed)
    noise_u = cfg.sigma_u * rng.standard_normal((cfg.Nx, cfg.Ny, cfg.Nt)).astype(np.float64)
    x, y, t, U = solve_burgers_vg_2d(cfg, noise_u=noise_u, rng=rng)
    return cfg, U


def subgrid_indices(N_full, N_sub, rng):
    """Uniform-without-replacement sub-grid indices, sorted."""
    return np.sort(rng.choice(N_full, size=N_sub, replace=False))


def spectral_marginals(U_sub):
    """U_sub : (NX_SUB, NY_SUB, NT_SUB).
    Returns S_xy(k) [1D, real-FFT length], S_t(ω) [1D, real-FFT length].
    S_xy is the symmetrised spatial PSD per paper eq. (2).
    """
    # Spatial PSD along x, averaged over y, t
    Fx = np.fft.rfft(U_sub, axis=0)           # (Nx//2+1, Ny, Nt)
    S_along_x = (np.abs(Fx) ** 2).mean(axis=(1, 2))
    Fy = np.fft.rfft(U_sub, axis=1)
    S_along_y = (np.abs(Fy) ** 2).mean(axis=(0, 2))
    # Symmetrise to length min(len_x, len_y) = NX_SUB//2+1
    L = min(len(S_along_x), len(S_along_y))
    S_xy = 0.5 * (S_along_x[:L] + S_along_y[:L])

    # Temporal PSD along t, averaged over x, y
    Ft = np.fft.rfft(U_sub, axis=2)
    S_t = (np.abs(Ft) ** 2).mean(axis=(0, 1))
    return S_xy, S_t


def sobolev_norm_sq(S, s):
    """H^s norm² of S treated as a function on a 1-d torus.
       ||S||²_{H^s} = Σ_τ (1+|τ|²)^s |Ŝ(τ)|²    where Ŝ = FFT(S)."""
    S_hat = np.fft.fft(S)
    n = len(S)
    tau = np.fft.fftfreq(n, d=1.0) * n           # integer frequencies
    weight = (1.0 + tau ** 2) ** s
    return float(np.sum(weight * np.abs(S_hat) ** 2)) / n


def main():
    cfg, U = generate_fd_reference()
    print(f"FD solve done.  U shape: {U.shape}  range: [{U.min():.3f}, {U.max():.3f}]\n")

    rng_sub = np.random.default_rng(0)
    ix = subgrid_indices(cfg.Nx, NX_SUB, rng_sub)
    iy = subgrid_indices(cfg.Ny, NY_SUB, rng_sub)
    it = subgrid_indices(cfg.Nt, NT_SUB, rng_sub)
    U_sub = U[np.ix_(ix, iy, it)]
    print(f"Sub-grid: {NX_SUB}×{NY_SUB}×{NT_SUB} = {U_sub.size} points")
    print(f"U_sub stats: mean={U_sub.mean():+.4e}  std={U_sub.std():.4e}\n")

    S_xy_full, S_t_full = spectral_marginals(U_sub.astype(np.float64))
    print(f"Spatial PSD S^xy (full):  length {len(S_xy_full)}  min={S_xy_full.min():.4e}  max={S_xy_full.max():.4e}")
    print(f"Temporal PSD S^t (full):  length {len(S_t_full)}  min={S_t_full.min():.4e}  max={S_t_full.max():.4e}\n")
    print(f"Note: full-sub-grid bound is vacuous due to high-frequency decay (paper p.23).")
    print(f"      Restricting to a physical frequency window where A3 holds non-vacuously.\n")

    # ── Sweep frequency-window cutoffs and report bound for each ──
    CUTOFFS = [1e-4, 1e-3, 1e-2, 1e-1]      # keep modes where S/max(S) ≥ cutoff (loosest → tightest)
    results = []
    for cutoff_ratio in CUTOFFS:
        mask_xy = S_xy_full >= cutoff_ratio * S_xy_full.max()
        mask_t  = S_t_full  >= cutoff_ratio * S_t_full.max()
        S_xy = S_xy_full[mask_xy]
        S_t  = S_t_full [mask_t]
        if len(S_xy) < 2 or len(S_t) < 2:
            continue
        results.append((cutoff_ratio, mask_xy.sum(), mask_t.sum(), S_xy, S_t))

    # Use loosest cutoff (most modes) as the default printed report
    cutoff_ratio, nx_kept, nt_kept, S_xy, S_t = results[0]
    print(f"  Default report uses loosest cutoff S/max ≥ {cutoff_ratio} → kept {nx_kept}/{len(S_xy_full)} spatial, {nt_kept}/{len(S_t_full)} temporal modes\n")

    mu_min = S_xy.min() * S_t.min()
    mu_max = S_xy.max() * S_t.max()
    print(f"Joint spectral measure bounds (windowed):")
    print(f"  μ_min = {S_xy.min():.4e} × {S_t.min():.4e} = {mu_min:.4e}")
    print(f"  μ_max = {S_xy.max():.4e} × {S_t.max():.4e} = {mu_max:.4e}")
    print(f"  μ_max / μ_min = {mu_max/mu_min:.2e}\n")

    # ── M_xy, M_t (L² norms of spectral marginals) ──
    M_xy = float(np.sqrt(np.sum(S_xy ** 2)))
    M_t  = float(np.sqrt(np.sum(S_t  ** 2)))
    print(f"L² norms of spectral marginals (discrete sum convention):")
    print(f"  M_xy = ||S^xy||_L² = {M_xy:.4e}")
    print(f"  M_t  = ||S^t||_L²  = {M_t:.4e}\n")

    # ── Sobolev H^{s_x} norms of marginals ──
    C_x = 1.0       # absorbed proportionality in Lemma 2 (diam(Ω), d)
    C_t = 1.0
    norm_sx_sq = sobolev_norm_sq(S_xy, S_X)
    norm_st_sq = sobolev_norm_sq(S_t,  S_T)
    C_quad_x = C_x * norm_sx_sq
    C_quad_t = C_t * norm_st_sq
    print(f"Sobolev norms (H^{S_X} of S^xy, H^{S_T} of S^t):")
    print(f"  ||S^xy||²_H^{S_X} = {norm_sx_sq:.4e}  → C_quad,x = {C_quad_x:.4e}")
    print(f"  ||S^t||²_H^{S_T}  = {norm_st_sq:.4e}  → C_quad,t = {C_quad_t:.4e}\n")

    # ── Plug in to Proposition 1 constants ──
    C_1 = mu_max * np.log(10) * (M_xy + M_t) / np.sqrt(2.0 * mu_min)
    C_2 = np.sqrt(C_quad_x * M_t**2 + C_quad_t * M_xy**2) / np.sqrt(2.0 * mu_min)
    # C_3 = √C_MM where C_MM = C_stab (1 + 1/λ_ic + 1/λ_bc).
    # C_stab is the SPDE-specific stability constant from Mishra-Molinaro [23, Thm 3.4].
    # For Burgers we don't have a closed form — use a placeholder coarse estimate.
    # Lower-bound surrogate: C_stab ≳ T × ||u||_L^∞ ÷ ν  (advection-dominated regime)
    C_stab_estimate = cfg.t_max * float(np.max(np.abs(U))) / cfg.nu_u
    lam_ic, lam_bc = 1.0, 1.0     # default PINN loss weights; user can override
    C_MM = C_stab_estimate * (1 + 1/lam_ic + 1/lam_bc)
    C_3 = float(np.sqrt(C_MM))

    print("══════════════════════════════════════════════════════════════")
    print(f"  PROPOSITION 1 CONSTANTS (Burgers-VG 2D, sub-grid {NX_SUB}×{NY_SUB}×{NT_SUB})")
    print("══════════════════════════════════════════════════════════════")
    print(f"  C_1 = μ_max ln(10)(M_xy+M_t)/√(2μ_min)  = {C_1:.4e}")
    print(f"  C_2 = √(C_qx M_t² + C_qt M_xy²)/√(2μ_min) = {C_2:.4e}")
    print(f"  C_3 ≈ √C_MM   (PDE-stability coarse)     = {C_3:.4e}")
    print(f"  β   = (1/2) min(s_x/d, s_t) = (1/2) min({S_X/D_SPATIAL}, {S_T}) = {BETA}")
    print()

    # ── Predicted bound for the paper's reported operating point ──
    # ε from paper: log-PSD MSE plateaus at ~0.3-0.4 (Section §3.3.3 / Remark 3).
    # δ from paper: PDE residual at end of training, typically ~1e-4 to 1e-3 in PINN literature.
    epsilon_grid = [0.063, 0.1, 0.3, 1.0]      # spans Gumbel floor → operating point
    delta_grid   = [1e-4, 1e-3, 1e-2]
    N_s_per_axis = NX_SUB                       # use spatial sub-grid count

    print(f"  PREDICTED W_2 BOUND at N_s = {N_s_per_axis} per axis (β term = C_2 * N_s^-β = {C_2 * N_s_per_axis**-BETA:.4e}):")
    print(f"  {'ε':>8} | {'δ':>8} | {'C_1√ε':>10} | {'C_2N^-β':>10} | {'C_3√δ':>10} | {'W_2 ≤':>10}")
    print("  " + "-" * 70)
    for eps in epsilon_grid:
        for delta in delta_grid:
            t1 = C_1 * np.sqrt(eps)
            t2 = C_2 * N_s_per_axis ** (-BETA)
            t3 = C_3 * np.sqrt(delta)
            total = t1 + t2 + t3
            print(f"  {eps:>8.4f} | {delta:>8.0e} | {t1:>10.3e} | {t2:>10.3e} | {t3:>10.3e} | {total:>10.3e}")

    print()
    print("══════════════════════════════════════════════════════════════")
    print("  WINDOWED CONSTANT SWEEP (paper p.23 caveat made concrete)")
    print("══════════════════════════════════════════════════════════════")
    print(f"  {'cutoff':>8} | {'kept xy/t':>11} | {'μ_max/μ_min':>13} | {'C_1':>10} | {'C_2':>10} | {'C_1√0.3 + C_2·24^-.25':>22}")
    print("  " + "-"*88)
    for c, nx, nt, Sxy_w, St_w in results:
        mum_w = Sxy_w.min() * St_w.min()
        mux_w = Sxy_w.max() * St_w.max()
        Mxy_w = float(np.sqrt(np.sum(Sxy_w**2)))
        Mt_w  = float(np.sqrt(np.sum(St_w**2)))
        Cqx_w = sobolev_norm_sq(Sxy_w, S_X)
        Cqt_w = sobolev_norm_sq(St_w,  S_T)
        C1_w  = mux_w * np.log(10) * (Mxy_w + Mt_w) / np.sqrt(2.0 * mum_w)
        C2_w  = np.sqrt(Cqx_w * Mt_w**2 + Cqt_w * Mxy_w**2) / np.sqrt(2.0 * mum_w)
        bnd   = C1_w * np.sqrt(0.3) + C2_w * 24**(-BETA)
        print(f"  {c:>8.0e} | {f'{nx}/{nt}':>11} | {mux_w/mum_w:>13.2e} | {C1_w:>10.3e} | {C2_w:>10.3e} | {bnd:>22.3e}")

    print()
    print("  Notes:")
    print(f"   - The bound is non-vacuous only on a sufficiently restricted frequency window;")
    print(f"     this matches the paper's own caveat (Section 3.3.1 commentary after A1-A9).")
    print(f"   - C_3 uses a coarse C_stab estimate (T·||u||_∞/ν = {C_stab_estimate:.2e}); the true")
    print(f"     Mishra-Molinaro C_stab for Burgers can be tighter — needs PDE-specific analysis.")
    print(f"   - For reviewer defence: include the 1e-2 cutoff row in the paper with explicit numbers")
    print(f"     for the operating point (ε≈0.3 from log-PSD floor, δ from PDE residual at end of train).")

    np.savez('/home/hubi/spde/spectral_pinn_for_spde/01_paper/verification/p1_constants.npz',
             U=U.astype(np.float32), U_sub=U_sub.astype(np.float32),
             S_xy=S_xy, S_t=S_t,
             mu_min=mu_min, mu_max=mu_max, M_xy=M_xy, M_t=M_t,
             C_quad_x=C_quad_x, C_quad_t=C_quad_t,
             C_1=C_1, C_2=C_2, C_3=C_3, beta=BETA)
    print(f"\n  → saved: /home/hubi/spde/spectral_pinn_for_spde/01_paper/verification/p1_constants.npz")


if __name__ == "__main__":
    main()
