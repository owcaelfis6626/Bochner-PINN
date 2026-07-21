"""
Generate Poisson jump field for z in the SPDE hierarchy.

A Poisson jump process at rate λ:
  - At each time step dt, probability λ·dt of a jump occurring
  - Jump size J ~ N(0, σ²)
  - Spatially: independent across x

This module pre-computes the jump field on the full (Nx, Nt) grid
so it can be interpolated to arbitrary (x,t) points just like the
noise fields.
"""

import numpy as np
from config import Config


def generate_poisson_jump_field(
    cfg: Config,
    jump_rate: float = 0.5,        # λ: jumps per unit time
    jump_std: float = 0.10,         # σ_J: jump size std dev
    rng: np.random.Generator = None,
) -> np.ndarray:
    """
    Generate Poisson jump field on (Nx, Nt) grid.

    Parameters
    ----------
    cfg : Config
        Configuration with Nx, Nt, dt
    jump_rate : float
        λ — average number of jumps per unit time interval [0, 1]
    jump_std : float
        σ_J — standard deviation of jump sizes
    rng : np.random.Generator
        Random number generator (uses default if None)

    Returns
    -------
    jump_field : (Nx, Nt) array
        Precomputed jump amplitudes. Zero if no jump, N(0, σ_J²) if jump.
    """
    if rng is None:
        rng = np.random.default_rng()

    Nx, Nt = cfg.Nx, cfg.Nt
    dt = cfg.dt

    jump_field = np.zeros((Nx, Nt))

    # Probability that a jump fires in this time step
    jump_prob = jump_rate * dt

    # At each (x, t_k), independently decide if jump fires
    # If yes, draw jump size from N(0, σ_J²)
    for k in range(Nt):
        # Bernoulli: does a jump happen at this time step?
        jump_happens = rng.uniform(0, 1, Nx) < jump_prob

        # For locations where jump happens, draw jump magnitude
        n_jumps = np.sum(jump_happens)
        if n_jumps > 0:
            jump_field[jump_happens, k] = rng.normal(0, jump_std, n_jumps)

    return jump_field
