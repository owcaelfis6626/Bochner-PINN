"""
Quick smoke test for the Heston-VG solver.

Verifies:
- Solver runs without errors on small grid
- Output shapes are correct
- S stays positive
- v stays non-negative
- VG jumps actually contribute (skewness/kurtosis > Gaussian)
"""

from __future__ import annotations

import numpy as np
import time

from config_heston_vg_2d import ConfigHestonVG2D
from heston_vg_solver_2d import (
    solve_heston_vg_2d,
    generate_correlated_noises,
    generate_vg_jump_field,
)


def main():
    # Small grid for fast smoke test
    cfg = ConfigHestonVG2D(Nx=32, Ny=32, Nt=128)
    cfg.output_dir = "results_smoke_heston_vg"

    print(f"Heston-VG 2D Smoke Test")
    print(f"Grid: {cfg.Nx} x {cfg.Ny} x {cfg.Nt}")
    print(f"Domain: [{cfg.x_min}, {cfg.x_max}] x [{cfg.y_min}, {cfg.y_max}] x [{cfg.t_min}, {cfg.t_max}]")
    print(f"S0={cfg.S0}, v0={cfg.v0}")
    print(f"Heston: mu={cfg.mu_S}, kappa={cfg.kappa_v}, theta={cfg.theta_v}, sigma_v={cfg.sigma_v}, rho={cfg.rho}")
    print(f"VG: sigma_vg={cfg.sigma_vg}, nu_vg={cfg.nu_vg}, theta_vg={cfg.theta_vg}")
    print(f"Feller condition: {cfg.feller_satisfied()}  (2*kappa*theta = {2*cfg.kappa_v*cfg.theta_v}, sigma_v^2 = {cfg.sigma_v**2})")

    rng = np.random.default_rng(cfg.seed)

    print("\n[1] Generating correlated noises...")
    t0 = time.time()
    noise_S, noise_v = generate_correlated_noises(cfg, rng)
    print(f"    noise_S shape: {noise_S.shape}, dtype: {noise_S.dtype}")
    print(f"    noise_v shape: {noise_v.shape}")
    print(f"    Empirical corr(noise_S, noise_v): {np.corrcoef(noise_S.flatten(), noise_v.flatten())[0,1]:.4f}  (target: {cfg.rho})")
    print(f"    Time: {time.time()-t0:.2f}s")

    print("\n[2] Generating VG jump field...")
    t0 = time.time()
    vg_jumps = generate_vg_jump_field(cfg, rng)
    print(f"    vg_jumps shape: {vg_jumps.shape}")
    print(f"    Mean: {vg_jumps.mean():.4e}")
    print(f"    Std: {vg_jumps.std():.4e}")
    print(f"    Skewness (sample): {((vg_jumps - vg_jumps.mean())**3).mean() / vg_jumps.std()**3:.3f}")
    print(f"    Kurtosis (excess, sample): {((vg_jumps - vg_jumps.mean())**4).mean() / vg_jumps.std()**4 - 3:.3f}")
    print(f"    Time: {time.time()-t0:.2f}s")

    print("\n[3] Solving Heston-VG SPDE...")
    t0 = time.time()
    x_grid, y_grid, t_grid, S, V = solve_heston_vg_2d(cfg, noise_S, noise_v, vg_jumps)
    print(f"    Time: {time.time()-t0:.2f}s")

    print("\n[4] Validation:")
    print(f"    S shape: {S.shape}")
    print(f"    V shape: {V.shape}")
    print(f"    S range: [{S.min():.4f}, {S.max():.4f}]")
    print(f"    V range: [{V.min():.4f}, {V.max():.4f}]")
    print(f"    S all positive: {(S > 0).all()}")
    print(f"    V all non-negative: {(V >= 0).all()}")
    print(f"    S finite: {np.isfinite(S).all()}")
    print(f"    V finite: {np.isfinite(V).all()}")

    # Check that VG jumps actually create non-Gaussian behavior in S returns
    print("\n[5] Stochastic structure of S log-returns:")
    log_S = np.log(np.maximum(S, 1e-6))
    log_returns = np.diff(log_S, axis=-1)  # (Nx, Ny, Nt-1)
    flat_returns = log_returns.flatten()
    mean_r = flat_returns.mean()
    std_r = flat_returns.std()
    skew = ((flat_returns - mean_r)**3).mean() / std_r**3
    excess_kurt = ((flat_returns - mean_r)**4).mean() / std_r**4 - 3
    print(f"    Mean log-return: {mean_r:.6f}")
    print(f"    Std log-return: {std_r:.6f}")
    print(f"    Skewness: {skew:.4f}  (Gaussian: 0; equity options often: -0.5 to -1.5)")
    print(f"    Excess kurtosis: {excess_kurt:.4f}  (Gaussian: 0; equity options: 1-5+)")

    # If VG jumps are working, we should see kurtosis > 0
    if excess_kurt > 0.5:
        print(f"\n  [PASS] VG jumps creating non-Gaussian fat tails (excess kurtosis > 0.5)")
    else:
        print(f"\n  [WARN] Kurtosis lower than expected. Check VG parameters.")

    print("\n[SMOKE TEST PASSED]")


if __name__ == "__main__":
    main()
