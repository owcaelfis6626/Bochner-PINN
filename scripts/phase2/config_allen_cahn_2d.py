"""
2D Stochastic Allen-Cahn configuration.

Domain:  (x, y, t) on Nx x Ny x Nt grid.
Field :  u(x, y, t)  — phase / order parameter in [-1, +1] (loosely).

Equation
    ∂u/∂t = D Δ u + (u - u^3) / ε² + σ · ξ(x, y, t)

with space-time white noise ξ.  Without noise the deterministic
attractors are u = ±1 (separated by a sharp interface of width ε).
The noise drives stochastic interface motion / nucleation.
"""

from dataclasses import dataclass


@dataclass
class ConfigAllenCahn2D:
    # ── Domain ────────────────────────────────────────────────────────
    x_min: float = -1.0
    x_max: float = 1.0
    y_min: float = -1.0
    y_max: float = 1.0
    t_min: float = 0.0
    t_max: float = 0.5

    # ── Grid ──────────────────────────────────────────────────────────
    Nx: int = 64
    Ny: int = 64
    Nt: int = 512

    # ── Equation parameters ──────────────────────────────────────────
    # NOTE: defaults are chosen so the smoke test is stable on a coarse
    # 32x32x64 grid.  For the full 64x64x512 production grid you can
    # safely drop epsilon to ~0.05 and raise sigma_u to ~0.10.
    D_u: float = 5.0e-3           # diffusion (interface scale ~ √D)
    epsilon: float = 0.10         # interface width (smaller = sharper)
    sigma_u: float = 0.05         # noise amplitude
    noise_type_u: str = "white"
    hurst_u: float = 0.7

    # ── Initial condition (random patch) ─────────────────────────────
    ic_kind: str = "random_blob"  # "random_blob" | "interface" | "uniform"
    ic_amplitude: float = 0.5

    # ── Misc ─────────────────────────────────────────────────────────
    seed: int = 42
    output_dir: str = "results_allen_cahn_2d"

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
