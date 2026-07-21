"""
2D Stochastic Reaction-Advection-Diffusion (RDA) configuration.

Field:  u(x, y, t)
Equation:
    du/dt + a * du/dx + b * du/dy = D * Laplacian(u)
                                  + alpha * u * (1 - u^2)
                                  + sigma * xi(x, y, t)

This combines linear advection (constant velocity (a, b)), small
diffusion (D), a Fisher-KPP-like cubic reaction (alpha * u * (1 - u^2))
and additive space-time white noise.  The advection direction breaks
the symmetry of standard reaction-diffusion problems and provides yet
another distinct nonlinear morphology for testing the generalisation
of Spectral-PINN.

Distinct from:
  * Heston      : multiplicative state-dependent noise.
  * Allen-Cahn  : pure diffusion + cubic reaction (no advection).
  * Burgers     : nonlinear self-advection (no reaction).
  * RDA         : *linear* advection + cubic reaction (drift + bistable).
"""

from dataclasses import dataclass


@dataclass
class ConfigRDA2D:
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
    a_u: float = 0.5             # advection x-velocity
    b_u: float = 0.3             # advection y-velocity
    D_u: float = 5.0e-3          # diffusion (small)
    alpha_u: float = 1.0         # reaction strength
    sigma_u: float = 0.05        # noise amplitude
    noise_type_u: str = "white"
    hurst_u: float = 0.7

    # -- Initial condition --------------------------------------------
    ic_amplitude: float = 0.5

    # -- Misc ----------------------------------------------------------
    seed: int = 42
    output_dir: str = "results_rda_2d"

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
