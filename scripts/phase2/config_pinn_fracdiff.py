"""
PINN configs for the 2D Stochastic Fractional Diffusion SPDE.
"""

from dataclasses import dataclass

try:
    from .config_fracdiff_2d import ConfigFracDiff2D
except ImportError:
    from config_fracdiff_2d import ConfigFracDiff2D


@dataclass
class ConfigPINNFracDiff(ConfigFracDiff2D):
    # -- Network ------------------------------------------------------
    hidden_layers: int = 4
    hidden_dim: int = 96
    num_fourier_features: int = 48
    fourier_scale: float = 2.0

    # -- Training -----------------------------------------------------
    num_epochs: int = 5000
    warmup_epochs: int = 800
    learning_rate: float = 1e-3
    lr_min: float = 1e-6

    # NOTE: PDE residual for fractional diffusion is computed on a
    # full-grid sub-sample (n_pde_t random time slices) rather than at
    # collocation points, so batch_coll is replaced by n_pde_t below.
    n_pde_t: int = 8                  # number of t-slices per PDE-residual call
    batch_ic:   int = 512
    batch_bc:   int = 512
    batch_data: int = 1024
    grad_clip:  float = 5.0

    # -- Loss weights -------------------------------------------------
    w_pde_u: float = 1.0
    w_ic:    float = 50.0
    w_bc:    float = 5.0           # weak periodic BC penalty
    w_data:  float = 5.0

    # -- Misc ---------------------------------------------------------
    seed: int = 42
    log_every: int = 100
    output_dir: str = "results_pinn_fracdiff"


@dataclass
class ConfigPINNFracDiffSpectral(ConfigPINNFracDiff):
    spec_Nx: int = 24
    spec_Ny: int = 24
    spec_Nt: int = 96

    w_spec_xy: float = 5.0
    w_spec_t:  float = 5.0
    w_var:     float = 50.0
    spec_every: int = 1

    output_dir: str = "results_pinn_fracdiff_spectral"
