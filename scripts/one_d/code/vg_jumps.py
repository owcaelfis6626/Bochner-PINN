"""
Variance-Gamma and Gamma jump field generators for the SPDE hierarchy.

VARIANCE-GAMMA PROCESS
----------------------
A Lévy process with infinite activity (infinitely many small jumps in any
finite interval) but finite variation. Constructed as time-changed Brownian
motion with a Gamma subordinator:
    X_t = theta * G_t + sigma * B_{G_t}
where G_t is a Gamma process with rate 1/nu and unit mean.

Lévy measure (Madan-Seneta parameterization):
    nu_VG(dx) = (1/(nu*|x|)) * exp(-A*|x|) dx
with A = sqrt(theta^2 + 2*sigma^2/nu) / sigma^2.

Why VG > Compound Poisson for our use case:
  - Continuous distribution of jump sizes (no point masses)
  - Infinite activity → smooth empirical distribution
  - Easier for PINN to fit (less discontinuous target)
  - Still has fat tails → high-frequency content in PSD

GAMMA PROCESS
-------------
Pure jump Lévy process with strictly positive jumps. Used here as a structural
jump component for the volatility level y (which is always non-negative).

Lévy measure:
    nu_Gamma(dx) = (alpha/x) * exp(-beta*x) dx,  x > 0

Each time-step increment G ~ Gamma(alpha*dt, beta).
"""

import numpy as np
from config import Config


def generate_vg_jump_field(
    cfg: Config,
    sigma: float = 0.15,
    nu: float = 0.10,
    theta: float = 0.0,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """Generate Variance-Gamma jump field on (Nx, Nt) grid.

    Parameters
    ----------
    cfg : Config
    sigma : float
        Diffusion coefficient of the subordinated Brownian motion.
    nu : float
        Variance rate of the Gamma subordinator. Higher nu -> larger jumps.
    theta : float
        Drift of the subordinated Brownian. Theta=0 gives symmetric VG.
    rng : np.random.Generator

    Returns
    -------
    vg_field : (Nx, Nt) array
        VG increments per time step. Spatially independent.
    """
    if rng is None:
        rng = np.random.default_rng()

    Nx, Nt = cfg.Nx, cfg.Nt
    dt = cfg.dt
    vg_field = np.zeros((Nx, Nt))

    for k in range(Nt):
        # Gamma subordinator increment: shape=dt/nu, scale=nu => mean=dt, var=nu*dt
        G = rng.gamma(shape=dt / nu, scale=nu, size=Nx)
        # Subordinated Brownian: increment ~ N(theta*G, sigma^2 * G)
        N = rng.normal(0.0, 1.0, size=Nx)
        vg_field[:, k] = theta * G + sigma * np.sqrt(G) * N

    return vg_field


def generate_gamma_jump_field(
    cfg: Config,
    alpha_rate: float = 2.0,
    beta_scale: float = 0.05,
    demean: bool = True,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """Generate Gamma process increments on (Nx, Nt) grid.

    Pure-jump Lévy process with positive increments. Each time-step:
        G_k ~ Gamma(shape=alpha_rate*dt, scale=beta_scale)

    By default the process is de-meaned so it does not drift the
    field systematically. The structural value comes from path roughness,
    not the mean.

    Parameters
    ----------
    cfg : Config
    alpha_rate : float
        Rate of the Gamma process (mean = alpha_rate * beta_scale per unit time).
    beta_scale : float
        Scale of individual jumps.
    demean : bool
        If True, subtract the per-step mean so the process is mean-zero.

    Returns
    -------
    gamma_field : (Nx, Nt) array
    """
    if rng is None:
        rng = np.random.default_rng()

    Nx, Nt = cfg.Nx, cfg.Nt
    dt = cfg.dt
    gamma_field = np.zeros((Nx, Nt))

    for k in range(Nt):
        G = rng.gamma(shape=alpha_rate * dt, scale=beta_scale, size=Nx)
        if demean:
            G = G - alpha_rate * dt * beta_scale
        gamma_field[:, k] = G

    return gamma_field
