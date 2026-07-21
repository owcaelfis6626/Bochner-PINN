"""
2D Heston model configuration.

Domain:  (x, y, t) on Nx x Ny x Nt grid (mirrors spde_2d.config_2d).
Fields:  S(x, y, t)   — asset / underlying price field
         v(x, y, t)   — instantaneous variance (volatility^2) field

Continuum SDEs (per spatial point) are augmented with diffusion in
(x, y) so the system is a genuine SPDE rather than an ensemble of
uncoupled point-wise SDEs:

    ∂S/∂t = D_S Δ S + μ S            + √v · S · ξ_S(x, y, t)
    ∂v/∂t = D_v Δ v + κ (θ - v)      + σ_v √v · ξ_v(x, y, t)

Noises ξ_S and ξ_v are correlated in time (corr ρ) but white in space.
Feller condition  2 κ θ ≥ σ_v²  is enforced via the default values.
"""

from dataclasses import dataclass


@dataclass
class ConfigHeston2D:
    # ── Domain ────────────────────────────────────────────────────────
    x_min: float = -3.0
    x_max: float = 3.0
    y_min: float = -3.0
    y_max: float = 3.0
    t_min: float = 0.0
    t_max: float = 1.0

    # ── Grid ──────────────────────────────────────────────────────────
    Nx: int = 64
    Ny: int = 64
    Nt: int = 512

    # ── S field (price) ──────────────────────────────────────────────
    D_S: float = 0.02
    mu_S: float = 0.05            # drift
    noise_type_S: str = "white"
    hurst_S: float = 0.7

    # ── v field (variance) ───────────────────────────────────────────
    D_v: float = 0.01
    kappa_v: float = 2.0          # mean-reversion speed
    theta_v: float = 0.04         # long-run variance (≈20% vol²)
    sigma_v: float = 0.30         # vol-of-vol
    noise_type_v: str = "white"
    hurst_v: float = 0.7

    # ── Coupling ─────────────────────────────────────────────────────
    rho: float = -0.7             # leverage effect (negative for equities)

    # ── Initial conditions ───────────────────────────────────────────
    S0: float = 1.0
    v0: float = 0.04

    # ── Misc ─────────────────────────────────────────────────────────
    seed: int = 42
    output_dir: str = "results_heston_2d"

    # ── Derived ──────────────────────────────────────────────────────
    @property
    def dx(self) -> float:
        return (self.x_max - self.x_min) / (self.Nx - 1)

    @property
    def dy(self) -> float:
        return (self.y_max - self.y_min) / (self.Ny - 1)

    @property
    def dt(self) -> float:
        return (self.t_max - self.t_min) / (self.Nt - 1)

    def feller_satisfied(self) -> bool:
        """Return True if 2 κ θ ≥ σ_v² (variance stays positive)."""
        return 2.0 * self.kappa_v * self.theta_v >= self.sigma_v ** 2
