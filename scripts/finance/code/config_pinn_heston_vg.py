"""
PINN configs for the 2D Heston-VG (Variance-Gamma jump-diffusion) SPDE.

* ConfigPINNHestonVG         — vanilla PINN
                               (PDE + IC + BC + data + positivity)
* ConfigPINNHestonVGSpectral — same + log-PSD / variance losses

Both inherit physics parameters from ConfigHestonVG2D so the FD solver
and the network see exactly the same equation, including the VG jumps.

Architectural choices specific to Heston-VG:
- Wider network (hidden_dim=128) because VG jumps make the target field
  rougher (fat-tailed return distribution).
- More Fourier features (64) to capture high-frequency content from jumps.
- Higher fourier_scale (3.0) for jump-induced discontinuities.
- Stronger PDE weight on S (jumps are in S equation, not v).

Sized for local single-GPU 24-48h budget:
    * 4 seeds x 5000 epochs x 2 modes (vanilla + spectral) = 8 trainings
    * Spectral sub-grid 24x24x96
"""

from dataclasses import dataclass

try:
    from .config_heston_vg_2d import ConfigHestonVG2D
except ImportError:
    from config_heston_vg_2d import ConfigHestonVG2D


@dataclass
class ConfigPINNHestonVG(ConfigHestonVG2D):
    # ── Network ─────────────────────────────────────────────────────────
    # Wider + more features than vanilla Heston because VG jumps add
    # non-Gaussian roughness that smooth networks struggle to fit.
    hidden_layers: int = 4
    hidden_dim: int = 128
    num_fourier_features: int = 64
    fourier_scale: float = 3.0

    # ── Training ────────────────────────────────────────────────────────
    num_epochs: int = 5000
    warmup_epochs: int = 1000     # longer warmup for jump-diffusion
    learning_rate: float = 1e-3
    lr_min: float = 1e-6
    batch_coll: int = 4096
    batch_ic:   int = 512
    batch_bc:   int = 512
    batch_data: int = 1024
    grad_clip:  float = 5.0

    # ── Loss weights ────────────────────────────────────────────────────
    w_pde_S: float = 1.0          # full S equation (diffusion + drift + noise + jump)
    w_pde_v: float = 1.0          # v equation (no jump)
    w_ic:    float = 50.0
    w_bc:    float = 10.0
    w_data:  float = 5.0
    w_pos_v: float = 1.0          # v >= 0
    w_pos_S: float = 0.5          # S > 0

    # ── Misc ────────────────────────────────────────────────────────────
    seed: int = 42
    log_every: int = 100
    output_dir: str = "results_pinn_heston_vg"


@dataclass
class ConfigPINNHestonVGSpectral(ConfigPINNHestonVG):
    # ── Spectral sub-grid ───────────────────────────────────────────────
    spec_Nx: int = 24
    spec_Ny: int = 24
    spec_Nt: int = 96

    # ── Spectral loss weights ───────────────────────────────────────────
    # Slightly lower weights than other Heston runs because VG jumps
    # already add high-frequency content; aggressive spectral matching
    # might be redundant or unstable.
    w_spec_xy: float = 3.0
    w_spec_t:  float = 5.0
    w_var:     float = 50.0       # variance ratio is the key UQ metric
    spec_every: int = 1

    output_dir: str = "results_pinn_heston_vg_spectral"
