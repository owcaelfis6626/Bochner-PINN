"""
Classical finite-difference solver for the 2D stochastic
Allen-Cahn SPDE.

    ∂u/∂t = D Δ u + (u - u^3) / ε² + σ · ξ(x, y, t)

with space-time white noise ξ.

Numerics
--------
* Diffusion D Δu  : Peaceman-Rachford ADI (unconditionally stable
  half-step pair, identical helper to the Heston / hierarchical
  solvers).
* Reaction (u - u³)/ε² : explicit forward Euler.
* Noise            : Itô-Euler  σ · ξ · dt (the noise grid already
                     contains the correct dt scaling).

Boundary conditions: homogeneous Neumann (matches the deterministic
A-C convention).

Returns ``x_grid, y_grid, t_grid, U`` where U has shape (Nx, Ny, Nt)
so the trainers / visualisers can treat it as a single-channel field.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import solve_banded

try:
    from .config_allen_cahn_2d import ConfigAllenCahn2D
except ImportError:
    from config_allen_cahn_2d import ConfigAllenCahn2D


# ─────────────────────────── ADI helpers ─────────────────────────────

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


# ─────────────────────────── ICs ─────────────────────────────────────

def _ic_u(X, Y, cfg: ConfigAllenCahn2D, rng: np.random.Generator | None = None):
    """Initial phase field u(x, y, 0).

    Three options driven by ``cfg.ic_kind``:
        * "random_blob": small-amplitude smooth random patch
        * "interface" : tanh-shaped vertical interface
        * "uniform"   : flat field at +ic_amplitude (perturbed by IC noise)
    """
    if cfg.ic_kind == "interface":
        return np.tanh(X / (np.sqrt(2.0) * cfg.epsilon))

    if cfg.ic_kind == "uniform":
        return np.full_like(X, cfg.ic_amplitude)

    # default: random_blob
    if rng is None:
        rng = np.random.default_rng(cfg.seed)
    base = cfg.ic_amplitude * np.cos(np.pi * X) * np.cos(np.pi * Y)
    perturb = 0.05 * rng.standard_normal(X.shape)
    return base + perturb


def ic_u_func(X, Y, cfg, rng=None): return _ic_u(X, Y, cfg, rng)


# ─────────────────────────── main solver ─────────────────────────────

def solve_allen_cahn_2d(
    cfg: ConfigAllenCahn2D,
    noise_u: np.ndarray,
    rng: np.random.Generator | None = None,
):
    """Solve the 2D stochastic Allen-Cahn SPDE.

    Parameters
    ----------
    cfg      : ConfigAllenCahn2D
    noise_u  : (Nx, Ny, Nt) noise field for the u-equation.
    rng      : (optional) random generator used for the IC if
               ``cfg.ic_kind == "random_blob"``.

    Returns
    -------
    x_grid, y_grid, t_grid, U   with U.shape == (Nx, Ny, Nt)
    """
    Nx, Ny, Nt = cfg.Nx, cfg.Ny, cfg.Nt
    dx, dy, dt = cfg.dx, cfg.dy, cfg.dt
    eps2 = cfg.epsilon ** 2

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

        # Reaction term (u - u^3)/ε²  +  multiplicative-style additive noise
        reaction = (u_n - u_n ** 3) / eps2
        forcing = reaction + noise_u[:, :, n]

        u_np1 = _adi_step(u_n, cfg.D_u, dx, dy, dt, forcing)

        # Soft clip to a reasonable range; A-C dynamics keep |u| ~ 1 but
        # the noise can push transient excursions further.  The 1.5
        # ceiling is large enough not to interfere with normal physics
        # but tight enough to catch numerical blow-up on coarse grids.
        u_np1 = np.clip(u_np1, -1.5, 1.5)

        U[:, :, n + 1] = u_np1.astype(np.float32)

    return x_grid, y_grid, t_grid, U
