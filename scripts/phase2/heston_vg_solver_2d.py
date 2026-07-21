"""
Classical finite-difference solver for the 2D Heston-VG SPDE
(Heston stochastic volatility with Variance-Gamma jumps on the price).

Equations
---------
    ∂S/∂t = D_S ΔS + μ_S S + √v · S · ξ_S(x, y, t) + S · J_VG(x, y, t)
    ∂v/∂t = D_v Δv + κ (θ - v) + σ_v √v · ξ_v(x, y, t)

with corr(ξ_S, ξ_v) = ρ and J_VG a Variance-Gamma jump process
(infinite-activity Lévy with fat-tailed increments).

Numerics
--------
* Spatial diffusion: Peaceman-Rachford ADI (same helpers as Heston).
* Drift / mean-reversion: explicit forward Euler.
* Diffusive noise: Itô-Euler with full-truncation on √v.
* VG jumps: Explicit Itô-style addition  S_+ = S * (1 + J_VG · dt).
  - VG field generated via Gamma-subordinated Brownian motion
    (Madan-Seneta / Madan-Carr-Chang parameterization).
  - Each spatial-temporal point gets independent VG increments.
* Variance positivity: full-truncation v_+ ≡ max(v, 0).
* Price positivity: enforced via clipping after jump update.

Returns the same 5-tuple shape as ``solve_heston_2d`` for drop-in
compatibility:
    x_grid, y_grid, t_grid, S(Nx,Ny,Nt), V(Nx,Ny,Nt)

VG Jump Generator
-----------------
Per time step, per spatial point (i, j):
    G_ij ~ Gamma(shape=dt/nu_vg, scale=nu_vg)        # subordinator
    Z_ij ~ N(0, 1)
    J_ij = theta_vg * G_ij + sigma_vg * sqrt(G_ij) * Z_ij

This is the standard Madan-Seneta construction: VG = time-changed
Brownian motion via Gamma subordinator. Generates infinite-activity,
fat-tailed jumps suitable for equity returns.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import solve_banded

try:
    from .config_heston_vg_2d import ConfigHestonVG2D
except ImportError:
    from config_heston_vg_2d import ConfigHestonVG2D


# ─────────────────────────── ADI helpers ─────────────────────────────
# (Identical to heston_solver_2d.py; reproduced here for self-containment)

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

def _ic_S(X, Y, cfg: ConfigHestonVG2D):
    """Smooth bell around S0 with mild spatial perturbation."""
    Lx = cfg.x_max - cfg.x_min
    Ly = cfg.y_max - cfg.y_min
    bump = np.cos(np.pi * (X - cfg.x_min) / Lx) \
         * np.cos(np.pi * (Y - cfg.y_min) / Ly)
    return cfg.S0 * (1.0 + 0.10 * bump)


def _ic_v(X, Y, cfg: ConfigHestonVG2D):
    """Variance starts at v0 with small spatial heterogeneity."""
    Lx = cfg.x_max - cfg.x_min
    Ly = cfg.y_max - cfg.y_min
    bump = np.cos(2 * np.pi * (X - cfg.x_min) / Lx) \
         * np.cos(2 * np.pi * (Y - cfg.y_min) / Ly)
    return np.maximum(cfg.v0 + 0.005 * bump, 1e-6)


def ic_S_func(X, Y, cfg): return _ic_S(X, Y, cfg)
def ic_v_func(X, Y, cfg): return _ic_v(X, Y, cfg)


# ─────────────────────────── VG jump generator ───────────────────────

def generate_vg_jump_field(
    cfg: ConfigHestonVG2D,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate Variance-Gamma jump field on (Nx, Ny, Nt) grid.

    Per timestep, per (i, j):
        G_ij ~ Gamma(shape=dt/nu, scale=nu)        # Gamma subordinator
        Z_ij ~ N(0, 1)
        J_ij = theta * G_ij + sigma * sqrt(G_ij) * Z_ij

    Returns
    -------
    vg_field : (Nx, Ny, Nt) array
        VG increments (already scaled per timestep).
    """
    Nx, Ny, Nt = cfg.Nx, cfg.Ny, cfg.Nt
    dt = cfg.dt
    out = np.zeros((Nx, Ny, Nt), dtype=np.float32)

    for k in range(Nt):
        # Gamma subordinator: mean=dt, variance=nu*dt
        G = rng.gamma(shape=dt / cfg.nu_vg, scale=cfg.nu_vg, size=(Nx, Ny))
        # Subordinated Brownian: incr ~ N(theta*G, sigma²*G)
        Z = rng.normal(0.0, 1.0, size=(Nx, Ny))
        out[:, :, k] = (cfg.theta_vg * G
                        + cfg.sigma_vg * np.sqrt(np.maximum(G, 1e-12)) * Z)

    return out


# ─────────────────────────── main solver ─────────────────────────────

def solve_heston_vg_2d(
    cfg: ConfigHestonVG2D,
    noise_S: np.ndarray,
    noise_v: np.ndarray,
    vg_jumps: np.ndarray = None,
    rng: np.random.Generator = None,
):
    """Solve the 2D Heston-VG jump-diffusion SPDE.

    Parameters
    ----------
    cfg : ConfigHestonVG2D
    noise_S : (Nx, Ny, Nt) Gaussian noise driving S diffusion (correlated
              with noise_v at correlation ρ).
    noise_v : (Nx, Ny, Nt) Gaussian noise driving v.
    vg_jumps : (Nx, Ny, Nt) optional pre-generated VG jump field.
               If None, will be generated using rng.
    rng : np.random.Generator (only used if vg_jumps is None).

    Returns
    -------
    x_grid, y_grid, t_grid, S, V
    """
    Nx, Ny, Nt = cfg.Nx, cfg.Ny, cfg.Nt
    dx, dy, dt = cfg.dx, cfg.dy, cfg.dt

    # Generate VG jumps if not provided
    if vg_jumps is None:
        if rng is None:
            rng = np.random.default_rng(cfg.seed + 12345)  # different stream
        vg_jumps = generate_vg_jump_field(cfg, rng)

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

        # Full-truncation: use v⁺ inside √v
        v_pos = np.maximum(v_n, 0.0)
        sqrt_v = np.sqrt(v_pos)

        # ── S step: diffusion + drift + Gaussian noise ──────────────
        forcing_S = cfg.mu_S * S_n + sqrt_v * S_n * noise_S[:, :, n]
        S_np1 = _adi_step(S_n, cfg.D_S, dx, dy, dt, forcing_S)

        # ── VG jump step on S (multiplicative) ──────────────────────
        # S_+ = S_+ * (1 + J_VG · dt · jump_intensity)
        # Use a clipped multiplier to prevent extreme jumps from
        # blowing up the simulation (keeps within [0.01, 100]x).
        jump_mult = 1.0 + cfg.jump_intensity * vg_jumps[:, :, n] * dt
        jump_mult = np.clip(jump_mult, 0.01, 100.0)
        S_np1 = S_np1 * jump_mult

        # ── v step: diffusion + mean-reversion + Gaussian noise ─────
        forcing_v = (cfg.kappa_v * (cfg.theta_v - v_n)
                     + cfg.sigma_v * sqrt_v * noise_v[:, :, n])
        v_np1 = _adi_step(v_n, cfg.D_v, dx, dy, dt, forcing_v)

        # Variance must remain non-negative (full-truncation cutoff)
        v_np1 = np.maximum(v_np1, 0.0)

        # Soft floor on S (keep prices positive)
        S_np1 = np.maximum(S_np1, 1e-6)

        S[:, :, n + 1] = S_np1.astype(np.float32)
        V[:, :, n + 1] = v_np1.astype(np.float32)

    return x_grid, y_grid, t_grid, S, V


# ─────────────────────────── correlated noise helper ─────────────────

def generate_correlated_noises(
    cfg: ConfigHestonVG2D,
    rng: np.random.Generator,
):
    """Generate (noise_S, noise_v) correlated Gaussian fields.

    corr(noise_S, noise_v) = cfg.rho at each (x, y, t).
    Each field has variance 1/dt (so multiplied by dt gives unit variance
    per Itô increment).
    """
    Nx, Ny, Nt = cfg.Nx, cfg.Ny, cfg.Nt
    dt = cfg.dt

    Z1 = rng.normal(0, 1, size=(Nx, Ny, Nt)).astype(np.float32)
    Z2 = rng.normal(0, 1, size=(Nx, Ny, Nt)).astype(np.float32)

    noise_S = Z1 / np.sqrt(dt)
    noise_v = (cfg.rho * Z1 + np.sqrt(1 - cfg.rho ** 2) * Z2) / np.sqrt(dt)

    return noise_S, noise_v
