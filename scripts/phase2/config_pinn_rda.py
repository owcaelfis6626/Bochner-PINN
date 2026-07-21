"""
PINN configs for the 2D Reaction-Advection-Diffusion SPDE.

* ConfigPINNRDA          - vanilla PINN
* ConfigPINNRDASpectral  - + log-PSD / variance losses on a sub-grid
"""

from dataclasses import dataclass

try:
    from .config_rda_2d import ConfigRDA2D
except ImportError:
    from config_rda_2d import ConfigRDA2D


@dataclass
class ConfigPINNRDA(ConfigRDA2D):
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
    batch_coll: int = 4096
    batch_ic:   int = 512
    batch_bc:   int = 512
    batch_data: int = 1024
    grad_clip:  float = 5.0

    # -- Loss weights -------------------------------------------------
    w_pde_u: float = 1.0
    w_ic:    float = 50.0
    w_bc:    float = 5.0
    w_data:  float = 5.0

    # -- Misc ---------------------------------------------------------
    seed: int = 42
    log_every: int = 100
    output_dir: str = "results_pinn_rda"


@dataclass
class ConfigPINNRDASpectral(ConfigPINNRDA):
    # -- Spectral sub-grid -------------------------------------------
    spec_Nx: int = 24
    spec_Ny: int = 24
    spec_Nt: int = 96

    # -- Spectral loss weights ---------------------------------------
    w_spec_xy: float = 5.0
    w_spec_t:  float = 5.0
    w_var:     float = 50.0
    spec_every: int = 1

    output_dir: str = "results_pinn_rda_spectral"
