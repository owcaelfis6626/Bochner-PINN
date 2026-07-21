"""
2D Heston-VG (Variance-Gamma jump-diffusion) model configuration.

Domain:  (x, y, t) on Nx x Ny x Nt grid.
Fields:  S(x, y, t)   — asset / underlying price field with VG jumps
         v(x, y, t)   — instantaneous variance (volatility^2) field

Continuum SDE-PDEs (per spatial point) augmented with diffusion in (x, y):

    ∂S/∂t = D_S ΔS + μ_S S + √v · S · ξ_S(x, y, t) + S · J_VG(x, y, t)
    ∂v/∂t = D_v Δv + κ (θ - v) + σ_v √v · ξ_v(x, y, t)

Where:
- ξ_S, ξ_v: Gaussian noises, correlated at (x, y, t) with corr ρ.
- J_VG: Variance-Gamma jump field (Lévy process with infinite activity).
- VG jumps enter MULTIPLICATIVELY on S: S → S · (1 + J_VG · dt).

VG parameters (Madan-Carr-Chang notation):
- sigma_vg: diffusion of subordinated Brownian motion
- nu_vg: variance rate of Gamma subordinator (higher = larger jumps)
- theta_vg: drift of subordinated Brownian (negative = downward bias)

Why VG > pure Heston:
- Captures negative skewness in equity options
- Generates excess kurtosis (fat tails)
- Models tail risk that pure-diffusion misses
- Real markets have jumps; this matches data better

Feller condition  2 κ θ ≥ σ_v²  is enforced via defaults.
"""

from dataclasses import dataclass


@dataclass
class ConfigHestonVG2D:
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

    # ── S field (price) - diffusive part ─────────────────────────────
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

    # ── Brownian Coupling ────────────────────────────────────────────
    rho: float = -0.7             # leverage effect (negative for equities)

    # ── VG Jump Component ────────────────────────────────────────────
    sigma_vg: float = 0.12        # diffusion of subordinated Brownian
    nu_vg: float = 0.20           # variance rate of Gamma subordinator
    theta_vg: float = -0.05       # drift (negative = downward jump bias)
    jump_intensity: float = 1.0   # scaling factor for jump contribution

    # ── Initial conditions ───────────────────────────────────────────
    S0: float = 1.0
    v0: float = 0.04

    # ── Misc ─────────────────────────────────────────────────────────
    seed: int = 42
    output_dir: str = "results_heston_vg_2d"

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

    def vg_levy_A(self) -> float:
        """Compute A parameter of VG Lévy density (for diagnostics)."""
        import math
        return (math.sqrt(self.theta_vg ** 2 + 2 * self.sigma_vg ** 2 / self.nu_vg)
                / self.sigma_vg ** 2)
