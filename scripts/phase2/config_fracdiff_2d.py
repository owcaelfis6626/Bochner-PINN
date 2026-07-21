"""
2D Stochastic Fractional Diffusion configuration.

Field:  u(x, y, t)
Equation:
    du/dt = -nu * (-Laplacian)^(alpha/2) u + sigma * xi(x, y, t)

with fractional order alpha in (0, 2).  The fractional Laplacian
(-Laplacian)^(alpha/2) is defined spectrally:
    F[(-Laplacian)^(alpha/2) u](k) = |k|^alpha * F[u](k)

For alpha = 2 this reduces to the standard heat equation; for alpha < 2
the operator is *non-local* and yields anomalous (super-)diffusion with
power-law spreading.  Alpha = 1.5 gives a clear, distinguishable
non-local behaviour while still being smooth enough that PINNs can
match it without instability.

Boundary handling: periodic (the fractional Laplacian is most naturally
defined via Fourier on a periodic domain).  This is the *only* problem
in spde_problems_2d/ that uses periodic BCs — a deliberate distinction.
"""

from dataclasses import dataclass


@dataclass
class ConfigFracDiff2D:
    # -- Domain --------------------------------------------------------
    x_min: float = 0.0
    x_max: float = 2.0
    y_min: float = 0.0
    y_max: float = 2.0
    t_min: float = 0.0
    t_max: float = 0.5

    # -- Grid ----------------------------------------------------------
    Nx: int = 64
    Ny: int = 64
    Nt: int = 512

    # -- Equation parameters ------------------------------------------
    nu_u:    float = 5.0e-3      # diffusion coefficient
    alpha:   float = 1.5         # fractional order (0 < alpha < 2)
    sigma_u: float = 0.05        # noise amplitude
    noise_type_u: str = "white"
    hurst_u: float = 0.7

    # -- Initial condition --------------------------------------------
    ic_amplitude: float = 0.5

    # -- Misc ----------------------------------------------------------
    seed: int = 42
    output_dir: str = "results_fracdiff_2d"

    # -- Derived (note: periodic dx uses Lx / Nx, not Lx / (Nx-1)) -----
    @property
    def Lx(self) -> float:
        return self.x_max - self.x_min

    @property
    def Ly(self) -> float:
        return self.y_max - self.y_min

    @property
    def dx(self) -> float:
        return self.Lx / self.Nx          # periodic spacing

    @property
    def dy(self) -> float:
        return self.Ly / self.Ny

    @property
    def dt(self) -> float:
        return (self.t_max - self.t_min) / (self.Nt - 1)
