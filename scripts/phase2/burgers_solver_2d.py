"""
Finite-difference solver for the 2D scalar stochastic Burgers SPDE.

    du/dt + u (du/dx + du/dy) = nu * Laplacian(u) + sigma * xi(x, y, t)

Numerics
--------
* Diffusion nu * Laplacian(u) : Peaceman-Rachford ADI (unconditionally
  stable half-step pair, identical helper to the Heston/A-C solvers).
* Advection u (u_x + u_y)     : explicit upwind (1st order).  Upwind is
  chosen for shock-robustness; for moderate viscosity this gives a
  stable, slightly-diffusive solution that is appropriate as PINN
  ground-truth.
* Noise                       : Ito-Euler  sigma * xi * dt.  The noise
  grid already contains the 1/sqrt(dx dy dt) scaling so the increment
  is dimensionally correct.

Boundary conditions: homogeneous Neumann (du/dn = 0).

Returns ``x_grid, y_grid, t_grid, U`` where U has shape (Nx, Ny, Nt).
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import solve_banded

try:
    from .config_burgers_2d import ConfigBurgers2D
except ImportError:
    from config_burgers_2d import ConfigBurgers2D


# ---------------------- ADI helpers ---------------------------------

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


# ---------------------- upwind advection ----------------------------

def _upwind_advection(u: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Compute u (u_x + u_y) via 1st-order upwind.

    For the scalar Burgers form du/dt + u (du/dx + du/dy) = ..., we use
    upwind based on sign(u):
        if u >= 0: u_x backward, u_y backward
        if u <  0: u_x forward,  u_y forward
    With Neumann BC (zero-flux) at the boundaries.
    """
    u_x = np.zeros_like(u)
    u_y = np.zeros_like(u)

    # x-derivative (axis 0)
    pos = (u >= 0)
    # backward difference for pos, forward for neg
    fwd_x = np.zeros_like(u)
    bwd_x = np.zeros_like(u)
    bwd_x[1:, :] = (u[1:, :] - u[:-1, :]) / dx
    bwd_x[0, :]  = 0.0  # Neumann
    fwd_x[:-1, :] = (u[1:, :] - u[:-1, :]) / dx
    fwd_x[-1, :]  = 0.0
    u_x = np.where(pos, bwd_x, fwd_x)

    # y-derivative (axis 1)
    fwd_y = np.zeros_like(u)
    bwd_y = np.zeros_like(u)
    bwd_y[:, 1:] = (u[:, 1:] - u[:, :-1]) / dy
    bwd_y[:, 0]  = 0.0
    fwd_y[:, :-1] = (u[:, 1:] - u[:, :-1]) / dy
    fwd_y[:, -1]  = 0.0
    u_y = np.where(pos, bwd_y, fwd_y)

    return u * (u_x + u_y)


# ---------------------- IC ------------------------------------------

def _ic_u(X, Y, cfg: ConfigBurgers2D, rng: np.random.Generator | None = None):
    """Smooth sinusoidal IC with a small perturbation."""
    if rng is None:
        rng = np.random.default_rng(cfg.seed)
    base = cfg.ic_amplitude * np.sin(np.pi * X) * np.sin(np.pi * Y)
    perturb = 0.05 * rng.standard_normal(X.shape)
    return base + perturb


def ic_u_func(X, Y, cfg, rng=None):
    return _ic_u(X, Y, cfg, rng)


# ---------------------- main solver ---------------------------------

def solve_burgers_2d(
    cfg: ConfigBurgers2D,
    noise_u: np.ndarray,
    rng: np.random.Generator | None = None,
):
    """Solve the 2D scalar stochastic Burgers SPDE.

    Parameters
    ----------
    cfg      : ConfigBurgers2D
    noise_u  : (Nx, Ny, Nt) noise field for the u-equation
    rng      : optional generator for the IC perturbation

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

        # Nonlinear advection - u (u_x + u_y) becomes -advect on RHS
        advect = _upwind_advection(u_n, dx, dy)
        forcing = -advect + noise_u[:, :, n]

        u_np1 = _adi_step(u_n, cfg.nu_u, dx, dy, dt, forcing)

        # Soft clip to a reasonable range; viscous Burgers should keep
        # |u| comparable to the IC amplitude unless the noise is huge.
        u_np1 = np.clip(u_np1, -3.0, 3.0)

        U[:, :, n + 1] = u_np1.astype(np.float32)

    return x_grid, y_grid, t_grid, U
