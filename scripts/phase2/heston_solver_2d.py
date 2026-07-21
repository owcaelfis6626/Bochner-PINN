"""
Classical finite-difference solver for the 2D *spatial* Heston SPDE.

Equations
---------
    ∂S/∂t = D_S Δ S + μ_S S          + √v · S · ξ_S(x, y, t)
    ∂v/∂t = D_v Δ v + κ (θ - v)      + σ_v √v · ξ_v(x, y, t)

with corr(ξ_S, ξ_v) = ρ at each (x, y, t) point (white in space).

Numerics
--------
* Spatial diffusion: Peaceman-Rachford ADI (re-uses helpers
  identical to ``spde_2d.classical_solver_2d``).
* Drift / mean-reversion: explicit forward Euler.
* Multiplicative noise: Itô-Euler  ΔW = ξ · dt
  (the noise field stored already has the correct dt scaling, so
   the increment is  noise · dt).
* Variance positivity: full-truncation scheme  v_+ ≡ max(v, 0)
  inside √v (the 2D analogue of Andersen / Lord-Koekkoek).

Returns the same 5-tuple shape as ``solve_classical`` for drop-in
compatibility with the existing PINN / FNO infrastructure:
    x_grid, y_grid, t_grid, S(Nx,Ny,Nt), V(Nx,Ny,Nt)
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import solve_banded

try:
    from .config_heston_2d import ConfigHeston2D
except ImportError:  # script-style run from inside the folder
    from config_heston_2d import ConfigHeston2D


# ─────────────────────────── ADI helpers ─────────────────────────────

def _tridiag_neumann(N: int, r: float):
    """Banded LHS (I - r/2·A) for Neumann BCs + RHS multiplier."""
    half_r = 0.5 * r
    diag = np.full(N, 1 + 2 * half_r)
    off = np.full(N - 1, -half_r)
    diag[0] = 1 + half_r
    diag[-1] = 1 + half_r

    lhs = np.zeros((3, N))
    lhs[0, 1:] = off
    lhs[1, :] = diag
    lhs[2, :-1] = off

    d_rhs = np.full(N, 1 - 2 * half_r)
    d_rhs[0] = 1 - half_r
    d_rhs[-1] = 1 - half_r
    o_rhs = half_r

    def rhs_mult(u):
        v = d_rhs * u
        v[:-1] += o_rhs * u[1:]
        v[1:] += o_rhs * u[:-1]
        return v

    return lhs, rhs_mult


def _adi_step(u: np.ndarray, D: float, dx: float, dy: float, dt: float,
              forcing: np.ndarray) -> np.ndarray:
    """Peaceman-Rachford ADI half-step pair on  u_t = D Δu + forcing."""
    Nx, Ny = u.shape
    rx = D * dt / dx ** 2
    ry = D * dt / dy ** 2
    lhs_x, rhs_x = _tridiag_neumann(Nx, rx)
    lhs_y, rhs_y = _tridiag_neumann(Ny, ry)

    u_half = np.empty_like(u)
    for j in range(Ny):
        col = u[:, j]
        if 0 < j < Ny - 1:
            yyc = ry * 0.5 * (u[:, j + 1] - 2 * u[:, j] + u[:, j - 1])
        elif j == 0:
            yyc = ry * 0.5 * (u[:, j + 1] - u[:, j])
        else:
            yyc = ry * 0.5 * (u[:, j - 1] - u[:, j])
        rhs_vec = rhs_x(col) + yyc + 0.5 * dt * forcing[:, j]
        u_half[:, j] = solve_banded((1, 1), lhs_x, rhs_vec)

    u_new = np.empty_like(u)
    for i in range(Nx):
        row = u_half[i, :]
        if 0 < i < Nx - 1:
            xxc = rx * 0.5 * (u_half[i + 1, :] - 2 * u_half[i, :]
                              + u_half[i - 1, :])
        elif i == 0:
            xxc = rx * 0.5 * (u_half[i + 1, :] - u_half[i, :])
        else:
            xxc = rx * 0.5 * (u_half[i - 1, :] - u_half[i, :])
        rhs_vec = rhs_y(row) + xxc + 0.5 * dt * forcing[i, :]
        u_new[i, :] = solve_banded((1, 1), lhs_y, rhs_vec)

    return u_new


# ─────────────────────────── ICs ─────────────────────────────────────

def _ic_S(X, Y, cfg: ConfigHeston2D):
    """Smooth bell around S0 with mild spatial perturbation."""
    Lx = cfg.x_max - cfg.x_min
    Ly = cfg.y_max - cfg.y_min
    bump = np.cos(np.pi * (X - cfg.x_min) / Lx) \
         * np.cos(np.pi * (Y - cfg.y_min) / Ly)
    return cfg.S0 * (1.0 + 0.10 * bump)


def _ic_v(X, Y, cfg: ConfigHeston2D):
    """Variance starts at v0 with small spatial heterogeneity."""
    Lx = cfg.x_max - cfg.x_min
    Ly = cfg.y_max - cfg.y_min
    bump = np.cos(2 * np.pi * (X - cfg.x_min) / Lx) \
         * np.cos(2 * np.pi * (Y - cfg.y_min) / Ly)
    return np.maximum(cfg.v0 + 0.005 * bump, 1e-6)


def ic_S_func(X, Y, cfg): return _ic_S(X, Y, cfg)
def ic_v_func(X, Y, cfg): return _ic_v(X, Y, cfg)


# ─────────────────────────── main solver ─────────────────────────────

def solve_heston_2d(
    cfg: ConfigHeston2D,
    noise_S: np.ndarray,
    noise_v: np.ndarray,
):
    """Solve the 2D spatial Heston SPDE.

    Parameters
    ----------
    cfg     : ConfigHeston2D
    noise_S : (Nx, Ny, Nt) array of noise driving S (already scaled
              by the noise generator, so multiply by dt for the
              Itô increment).
    noise_v : (Nx, Ny, Nt) array of noise driving v (correlated with
              noise_S at correlation ρ — caller is responsible for
              generating the correlated pair).

    Returns
    -------
    x_grid, y_grid, t_grid, S, V
    """
    Nx, Ny, Nt = cfg.Nx, cfg.Ny, cfg.Nt
    dx, dy, dt = cfg.dx, cfg.dy, cfg.dt

    x_grid = np.linspace(cfg.x_min, cfg.x_max, Nx)
    y_grid = np.linspace(cfg.y_min, cfg.y_max, Ny)
    t_grid = np.linspace(cfg.t_min, cfg.t_max, Nt)
    X, Yg = np.meshgrid(x_grid, y_grid, indexing="ij")

    S = np.zeros((Nx, Ny, Nt), dtype=np.float32)
    V = np.zeros((Nx, Ny, Nt), dtype=np.float32)

    S[:, :, 0] = _ic_S(X, Yg, cfg)
    V[:, :, 0] = _ic_v(X, Yg, cfg)

    for n in range(Nt - 1):
        S_n = S[:, :, n].astype(np.float64)
        v_n = V[:, :, n].astype(np.float64)

        # Full-truncation: use v⁺ inside √v, evolve unconstrained, then
        # clip at the end of the step (matches Lord-Koekkoek / Andersen).
        v_pos = np.maximum(v_n, 0.0)
        sqrt_v = np.sqrt(v_pos)

        # ── S step ──────────────────────────────────────────────────
        # forcing_S = μ S + √v · S · ξ_S
        forcing_S = cfg.mu_S * S_n + sqrt_v * S_n * noise_S[:, :, n]
        S_np1 = _adi_step(S_n, cfg.D_S, dx, dy, dt, forcing_S)

        # ── v step ──────────────────────────────────────────────────
        # forcing_v = κ (θ - v) + σ_v √v · ξ_v
        forcing_v = (cfg.kappa_v * (cfg.theta_v - v_n)
                     + cfg.sigma_v * sqrt_v * noise_v[:, :, n])
        v_np1 = _adi_step(v_n, cfg.D_v, dx, dy, dt, forcing_v)

        # Variance must remain non-negative (full-truncation cutoff)
        v_np1 = np.maximum(v_np1, 0.0)

        # Soft floor on S (keep prices positive in pathological scenarios)
        S_np1 = np.maximum(S_np1, 1e-6)

        S[:, :, n + 1] = S_np1.astype(np.float32)
        V[:, :, n + 1] = v_np1.astype(np.float32)

    return x_grid, y_grid, t_grid, S, V
