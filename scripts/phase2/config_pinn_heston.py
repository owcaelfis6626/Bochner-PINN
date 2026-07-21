"""
PINN configs for the 2D Heston SPDE problem.

* ConfigPINNHeston         — vanilla PINN (PDE + IC + BC + data + positivity)
* ConfigPINNHestonSpectral — same + log-PSD / variance losses on a sub-grid

Both inherit the *physics* parameters from ConfigHeston2D so the FD solver
and the network see exactly the same equation.

Sized for a local single-GPU 48-hour budget:
    * 4 seeds x ~5000 epochs x 4 trainings (Heston-V/S, AC-V/S)
    * spectral sub-grid 24x24x96 (smaller than main work's 32x32x128)
"""

from dataclasses import dataclass

try:
    from .config_heston_2d import ConfigHeston2D
except ImportError:
    from config_heston_2d import ConfigHeston2D


@dataclass
class ConfigPINNHeston(ConfigHeston2D):
    # ── Network ─────────────────────────────────────────────────────────
    hidden_layers: int = 4
    hidden_dim: int = 96
    num_fourier_features: int = 48
    fourier_scale: float = 2.0

    # ── Training ────────────────────────────────────────────────────────
    num_epochs: int = 5000
    warmup_epochs: int = 800
    learning_rate: float = 1e-3
    lr_min: float = 1e-6
    batch_coll: int = 4096
    batch_ic:   int = 512
    batch_bc:   int = 512
    batch_data: int = 1024
    grad_clip:  float = 5.0

    # ── Loss weights ────────────────────────────────────────────────────
    w_pde_S: float = 1.0
    w_pde_v: float = 1.0
    w_ic:    float = 50.0
    w_bc:    float = 10.0
    w_data:  float = 5.0
    w_pos_v: float = 1.0   # v >= 0
    w_pos_S: float = 0.5   # S >= 0

    # ── Misc ────────────────────────────────────────────────────────────
    seed: int = 42
    log_every: int = 100
    output_dir: str = "results_pinn_heston"


@dataclass
class ConfigPINNHestonSpectral(ConfigPINNHeston):
    # ── Spectral sub-grid ───────────────────────────────────────────────
    spec_Nx: int = 24
    spec_Ny: int = 24
    spec_Nt: int = 96

    # ── Spectral loss weights ───────────────────────────────────────────
    w_spec_xy: float = 5.0
    w_spec_t:  float = 5.0
    w_var:     float = 50.0
    spec_every: int = 1

    output_dir: str = "results_pinn_heston_spectral"
