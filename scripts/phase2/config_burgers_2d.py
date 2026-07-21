"""
2D Stochastic Burgers configuration (scalar form).

Field:  u(x, y, t)
Equation:
    du/dt + u (du/dx + du/dy) = nu * Laplacian(u) + sigma * xi(x, y, t)

This is the *scalar* form of 2D Burgers (a single advected scalar that
carries itself).  It exhibits genuinely nonlinear advection coupled with
viscous diffusion and additive space-time white noise — a problem class
distinct from both Heston (multiplicative state-dependent noise) and
Allen-Cahn (cubic algebraic nonlinearity).

The combination of nonlinear self-advection + viscosity makes Burgers a
classical stress-test for PINNs: shocks may form when nu is small, and
the spectral content shifts toward higher wavenumbers.
"""

from dataclasses import dataclass


@dataclass
class ConfigBurgers2D:
    # -- Domain --------------------------------------------------------
    x_min: float = -1.0
    x_max: float =  1.0
    y_min: float = -1.0
    y_max: float =  1.0
    t_min: float =  0.0
    t_max: float =  0.5

    # -- Grid ----------------------------------------------------------
    Nx: int = 64
    Ny: int = 64
    Nt: int = 512

    # -- Equation parameters ------------------------------------------
    nu_u:    float = 1.0e-2     # viscosity (large enough to keep shocks smooth)
    sigma_u: float = 0.05       # noise amplitude
    noise_type_u: str = "white"
    hurst_u: float = 0.7

    # -- Initial condition --------------------------------------------
    # smooth periodic-like sinusoidal blob
    ic_amplitude: float = 0.6

    # -- Misc ----------------------------------------------------------
    seed: int = 42
    output_dir: str = "results_burgers_2d"

    # -- Derived -------------------------------------------------------
    @property
    def dx(self) -> float:
        return (self.x_max - self.x_min) / (self.Nx - 1)

    @property
    def dy(self) -> float:
        return (self.y_max - self.y_min) / (self.Ny - 1)

    @property
    def dt(self) -> float:
        return (self.t_max - self.t_min) / (self.Nt - 1)
