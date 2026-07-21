"""
Pseudo-spectral solver for the 2D stochastic fractional diffusion SPDE.

    du/dt = -nu * (-Laplacian)^(alpha/2) u + sigma * xi(x, y, t)

The fractional Laplacian is local in Fourier space:
    F[(-Laplacian)^(alpha/2) u](k) = |k|^alpha F[u](k)

Time-stepping uses the exponential integrator (exact for the linear
part since the operator is diagonal in Fourier), giving unconditional
stability:
    u_hat(t + dt) = exp(-nu |k|^alpha dt) * (u_hat(t) + dt * xi_hat(t))

Boundary conditions: periodic (intrinsic to the FFT-based fractional
Laplacian).
"""

from __future__ import annotations

import numpy as np

try:
    from .config_fracdiff_2d import ConfigFracDiff2D
except ImportError:
    from config_fracdiff_2d import ConfigFracDiff2D


def _build_kfrac(cfg: ConfigFracDiff2D):
    """Build |k|^alpha on the 2D FFT grid."""
    kx = 2.0 * np.pi * np.fft.fftfreq(cfg.Nx, d=cfg.dx)
    ky = 2.0 * np.pi * np.fft.fftfreq(cfg.Ny, d=cfg.dy)
    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    Kmag = np.sqrt(KX ** 2 + KY ** 2)
    return Kmag ** cfg.alpha   # shape (Nx, Ny)


def _ic_u(X, Y, cfg: ConfigFracDiff2D, rng: np.random.Generator | None = None):
    """Smooth periodic IC: sum of low-frequency sinusoids."""
    if rng is None:
        rng = np.random.default_rng(cfg.seed)
    Lx, Ly = cfg.Lx, cfg.Ly
    base = (cfg.ic_amplitude
            * (np.sin(2 * np.pi * (X - cfg.x_min) / Lx)
               * np.sin(2 * np.pi * (Y - cfg.y_min) / Ly)
               + 0.5 * np.cos(4 * np.pi * (X - cfg.x_min) / Lx)
               * np.cos(2 * np.pi * (Y - cfg.y_min) / Ly)))
    perturb = 0.05 * rng.standard_normal(X.shape)
    return base + perturb


def ic_u_func(X, Y, cfg, rng=None):
    return _ic_u(X, Y, cfg, rng)


def solve_fracdiff_2d(
    cfg: ConfigFracDiff2D,
    noise_u: np.ndarray,
    rng: np.random.Generator | None = None,
):
    """Solve the 2D stochastic fractional diffusion SPDE.

    Returns
    -------
    x_grid, y_grid, t_grid, U   with U.shape == (Nx, Ny, Nt)
    """
    Nx, Ny, Nt = cfg.Nx, cfg.Ny, cfg.Nt
    dt = cfg.dt

    if rng is None:
        rng = np.random.default_rng(cfg.seed)

    # periodic grid (last point identified with first)
    x_grid = np.linspace(cfg.x_min, cfg.x_max, Nx, endpoint=False)
    y_grid = np.linspace(cfg.y_min, cfg.y_max, Ny, endpoint=False)
    t_grid = np.linspace(cfg.t_min, cfg.t_max, Nt)
    X, Yg = np.meshgrid(x_grid, y_grid, indexing="ij")

    Kfrac = _build_kfrac(cfg)
    decay = np.exp(-cfg.nu_u * Kfrac * dt)   # shape (Nx, Ny)

    U = np.zeros((Nx, Ny, Nt), dtype=np.float32)
    U[:, :, 0] = _ic_u(X, Yg, cfg, rng)

    u = U[:, :, 0].astype(np.float64).copy()
    for n in range(Nt - 1):
        # u_hat after dt: exp(-nu |k|^alpha dt) * (u_hat + dt * xi_hat)
        rhs = u + dt * noise_u[:, :, n]
        rhs_hat = np.fft.fft2(rhs)
        u_hat_new = decay * rhs_hat
        u = np.real(np.fft.ifft2(u_hat_new))
        # Soft clip
        u = np.clip(u, -3.0, 3.0)
        U[:, :, n + 1] = u.astype(np.float32)

    return x_grid, y_grid, t_grid, U
