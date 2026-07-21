"""
Finite-difference solver for the 2D stochastic Reaction-Advection-Diffusion
SPDE.

    du/dt + a u_x + b u_y = D Laplacian(u) + alpha u (1 - u^2) + sigma xi

Numerics
--------
* Diffusion D Laplacian(u) : Peaceman-Rachford ADI (unconditionally stable).
* Linear advection         : explicit upwind based on sign(a), sign(b).
* Cubic reaction           : explicit forward Euler.
* Noise                    : Ito-Euler  sigma * xi * dt.

Boundary conditions: homogeneous Neumann (du/dn = 0).
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import solve_banded

try:
    from .config_rda_2d import ConfigRDA2D
except ImportError:
    from config_rda_2d import ConfigRDA2D


# ---------------------- ADI helpers (same as A-C / Burgers) ---------

def _tridiag_neumann(N: int, r: float):
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


# ---------------------- linear upwind advection ---------------------

def _linear_upwind(u: np.ndarray, a: float, b: float,
                   dx: float, dy: float) -> np.ndarray:
    """Compute a*u_x + b*u_y via 1st-order upwind given fixed (a, b).

    Neumann (zero-flux) at the boundaries.
    """
    # x-derivative
    if a >= 0:
        # backward
        u_x = np.zeros_like(u)
        u_x[1:, :] = (u[1:, :] - u[:-1, :]) / dx
    else:
        # forward
        u_x = np.zeros_like(u)
        u_x[:-1, :] = (u[1:, :] - u[:-1, :]) / dx

    # y-derivative
    if b >= 0:
        u_y = np.zeros_like(u)
        u_y[:, 1:] = (u[:, 1:] - u[:, :-1]) / dy
    else:
        u_y = np.zeros_like(u)
        u_y[:, :-1] = (u[:, 1:] - u[:, :-1]) / dy

    return a * u_x + b * u_y


# ---------------------- IC ------------------------------------------

def _ic_u(X, Y, cfg: ConfigRDA2D, rng: np.random.Generator | None = None):
    """Smooth localised cosine bump."""
    if rng is None:
        rng = np.random.default_rng(cfg.seed)
    base = cfg.ic_amplitude * np.cos(np.pi * X) * np.cos(np.pi * Y)
    perturb = 0.05 * rng.standard_normal(X.shape)
    return base + perturb


def ic_u_func(X, Y, cfg, rng=None):
    return _ic_u(X, Y, cfg, rng)


# ---------------------- main solver ---------------------------------

def solve_rda_2d(
    cfg: ConfigRDA2D,
    noise_u: np.ndarray,
    rng: np.random.Generator | None = None,
):
    """Solve the 2D stochastic RDA SPDE.

    Returns
    -------
    x_grid, y_grid, t_grid, U   with U.shape == (Nx, Ny, Nt)
    """
    Nx, Ny, Nt = cfg.Nx, cfg.Ny, cfg.Nt
    dx, dy, dt = cfg.dx, cfg.dy, cfg.dt

    if rng is None:
        rng = np.random.default_rng(cfg.seed)

    x_grid = np.linspace(cfg.x_min, cfg.x_max, Nx)
    y_grid = np.linspace(cfg.y_min, cfg.y_max, Ny)
    t_grid = np.linspace(cfg.t_min, cfg.t_max, Nt)
    X, Yg = np.meshgrid(x_grid, y_grid, indexing="ij")

    U = np.zeros((Nx, Ny, Nt), dtype=np.float32)
    U[:, :, 0] = _ic_u(X, Yg, cfg, rng)

    for n in range(Nt - 1):
        u_n = U[:, :, n].astype(np.float64)

        # advection (-(a u_x + b u_y) on RHS)
        adv = _linear_upwind(u_n, cfg.a_u, cfg.b_u, dx, dy)

        # cubic reaction alpha u (1 - u^2)
        reac = cfg.alpha_u * u_n * (1.0 - u_n ** 2)

        forcing = -adv + reac + noise_u[:, :, n]
        u_np1 = _adi_step(u_n, cfg.D_u, dx, dy, dt, forcing)

        u_np1 = np.clip(u_np1, -2.0, 2.0)
        U[:, :, n + 1] = u_np1.astype(np.float32)

    return x_grid, y_grid, t_grid, U
