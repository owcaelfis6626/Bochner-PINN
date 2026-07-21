"""
Configuration for the FNO-based SPDE operator.

The FNO learns the map
    G : (noise_z, noise_y, noise_p, jump_z, jump_y) -> (Z, Y, P)
on a fixed (Nx, Nt) grid. We reuse the physical parameters from the
PINN side for an apples-to-apples comparison.

Architecture summary
--------------------
2D FNO (x, t):
  - Lifting:  C_in -> width      via 1x1 conv
  - L spectral-conv layers:      width -> width
  - Projection: width -> 128 -> C_out
Each spectral-conv layer keeps `modes_x * modes_t` low-frequency Fourier
coefficients and applies a learned complex linear map; the rest are zeroed.

Why this maps well to our problem
---------------------------------
- Spectral metric is the *target*, FNO is *spectral by construction*.
- One trained model evaluates an entire ensemble of realizations cheaply.
- Removes the seed-dependent bimodality observed in PINN training: the
  FNO sees many realizations during training and amortizes over noise.
"""

from dataclasses import dataclass, field
from typing import Tuple

# Reuse physical Config from the parent module
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config as PhysicsConfig  # noqa: E402


@dataclass
class FNOConfig:
    # ── Grid (must match the physics solver) ─────────────────────────
    Nx: int = 128
    Nt: int = 512

    # ── Channels ─────────────────────────────────────────────────────
    # Inputs:  noise_z, noise_y, noise_p, jump_z, jump_y, IC_z, IC_y, IC_p
    # (8 channels)
    in_channels: int = 8
    # Outputs: Z, Y, P
    out_channels: int = 3

    # ── FNO architecture ─────────────────────────────────────────────
    # Sized for 24 GB GPU (RTX 4090). Shrink for smaller cards.
    modes_x: int = 16          # Fourier modes kept along x
    modes_t: int = 24          # Fourier modes kept along t
    width: int = 32            # hidden channel width
    n_layers: int = 4          # number of spectral-conv layers
    activation: str = "gelu"   # relu | gelu | silu

    # ── Dataset ──────────────────────────────────────────────────────
    n_train: int = 256
    n_val:   int = 64
    n_test:  int = 256         # held-out for distributional metrics
    base_seed_train: int = 1000
    base_seed_val:   int = 5000
    base_seed_test:  int = 9000

    # Whether to apply VG jumps on z and Gamma jumps on y
    use_jumps: bool = True
    vg_sigma: float = 0.15
    vg_nu:    float = 0.10
    vg_theta: float = 0.0
    gamma_alpha: float = 2.0
    gamma_beta:  float = 0.05

    # ── Training ─────────────────────────────────────────────────────
    epochs: int = 200
    batch_size: int = 16       # 24 GB GPU
    lr: float = 1e-3
    lr_min: float = 1e-5
    weight_decay: float = 1e-5
    grad_clip: float = 1.0

    # ── Loss weights ─────────────────────────────────────────────────
    w_mse:      float = 1.0    # pointwise MSE
    w_spec_x:   float = 1.0    # spatial PSD (log-MSE)
    w_spec_t:   float = 1.0    # temporal PSD (log-MSE)
    w_mean:     float = 1.0
    w_var:      float = 100.0

    # ── Performance toggles ─────────────────────────────────────────
    # torch.compile is the big win — fuses ops, eliminates host overhead.
    # AMP bf16 is OFF because rfft2 doesn't support bf16 on the version
    # of PyTorch on the pod (errors out in the SpectralConv2d forward).
    use_compile: bool = True
    use_amp:     bool = False

    # ── I/O ──────────────────────────────────────────────────────────
    output_dir: str = "results_fno"
    data_dir:   str = "data_fno"     # cached datasets
    log_every:  int = 10
    seed:       int = 0

    def physics(self) -> PhysicsConfig:
        """Return a PhysicsConfig matching this FNO grid."""
        cfg = PhysicsConfig()
        cfg.Nx = self.Nx
        cfg.Nt = self.Nt
        return cfg
