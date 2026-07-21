"""
Shared noise generation for the additional 2D SPDE problems
(Heston and Allen-Cahn).

Mirrors the conventions used in ``spde_2d.noise_2d``:
    * white-in-space-and-time:
          <ξ_{ij}^n  ξ_{kl}^m> = δ δ δ / (dx dy dt)
    * fBm-in-time, white-in-space (Hurst H ∈ (0, 1))

The Heston solver also needs *correlated pairs* of noises, so we
expose ``generate_correlated_white_pair`` returning a 2-tuple with
prescribed correlation ρ.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import cholesky


# ─────────────────────────── primitives ──────────────────────────────

def white_noise_grid(Nx: int, Ny: int, Nt: int, sigma: float,
                     dx: float, dy: float, dt: float,
                     rng: np.random.Generator) -> np.ndarray:
    raw = rng.standard_normal((Nx, Ny, Nt))
    return sigma * raw / np.sqrt(dx * dy * dt)


def _fbm_covariance(n: int, H: float) -> np.ndarray:
    idx = np.arange(n)
    d = np.abs(idx[:, None] - idx[None, :]).astype(float)
    return 0.5 * (np.abs(d + 1) ** (2 * H) - 2 * np.abs(d) ** (2 * H)
                  + np.abs(d - 1) ** (2 * H))


def fbm_noise_grid(Nx: int, Ny: int, Nt: int, sigma: float,
                   dx: float, dy: float, dt: float, H: float,
                   rng: np.random.Generator) -> np.ndarray:
    """fBm in time, independent across (x, y)."""
    C = _fbm_covariance(Nt, H) + 1e-10 * np.eye(Nt)
    L = cholesky(C, lower=True)
    Z = rng.standard_normal((Nx * Ny, Nt))
    inc = (Z @ L.T) * (dt ** H)
    return (sigma * inc / np.sqrt(dx * dy)).reshape(Nx, Ny, Nt)


# ─────────────────────────── public API ──────────────────────────────

def generate_white(Nx: int, Ny: int, Nt: int, sigma: float,
                   dx: float, dy: float, dt: float,
                   rng: np.random.Generator) -> np.ndarray:
    return white_noise_grid(Nx, Ny, Nt, sigma, dx, dy, dt, rng)


def generate_correlated_white_pair(
    Nx: int, Ny: int, Nt: int,
    sigma1: float, sigma2: float, rho: float,
    dx: float, dy: float, dt: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Two white-noise fields with corr = rho.

    Implementation:
        ξ₁ = Z₁
        ξ₂ = ρ Z₁ + √(1-ρ²) Z₂
    where Z₁, Z₂ are independent standard white noises.
    """
    if not -1.0 <= rho <= 1.0:
        raise ValueError(f"rho must be in [-1, 1], got {rho}")

    Z1 = rng.standard_normal((Nx, Ny, Nt))
    Z2 = rng.standard_normal((Nx, Ny, Nt))
    xi1 = Z1
    xi2 = rho * Z1 + np.sqrt(1.0 - rho ** 2) * Z2

    scale = 1.0 / np.sqrt(dx * dy * dt)
    return sigma1 * xi1 * scale, sigma2 * xi2 * scale


def generate_fbm(Nx: int, Ny: int, Nt: int, sigma: float, hurst: float,
                 dx: float, dy: float, dt: float,
                 rng: np.random.Generator) -> np.ndarray:
    return fbm_noise_grid(Nx, Ny, Nt, sigma, dx, dy, dt, hurst, rng)
