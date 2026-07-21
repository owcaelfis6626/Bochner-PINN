"""
Supervised + spectral loss for FNO training.

Two terms (per-channel, then averaged):
  - MSE on the predicted field (pointwise)
  - log-MSE on spatial PSD (FFT along x, average over t)
  - log-MSE on temporal PSD (FFT along t, average over x)

The spectral terms keep the FNO honest about *distributional* behaviour
even when the seed-specific path differs slightly. This mirrors the
spectral_loss used by the PINN side, so we report the same metrics.
"""

import torch
import torch.nn.functional as F


def _spatial_psd(field: torch.Tensor) -> torch.Tensor:
    """field: (B, C, Nx, Nt) -> (B, C, kx)."""
    Fx = torch.fft.rfft(field, dim=-2)
    return (Fx.abs() ** 2).mean(dim=-1)


def _temporal_psd(field: torch.Tensor) -> torch.Tensor:
    """field: (B, C, Nx, Nt) -> (B, C, w)."""
    Ft = torch.fft.rfft(field, dim=-1)
    return (Ft.abs() ** 2).mean(dim=-2)


def fno_loss(pred: torch.Tensor, target: torch.Tensor,
             w_mse: float = 1.0,
             w_spec_x: float = 1.0,
             w_spec_t: float = 1.0,
             w_mean: float = 1.0,
             w_var: float = 100.0,
             eps: float = 1e-10):
    """Compute composite loss + a dict of components.

    Both `pred` and `target` are (B, C, Nx, Nt).
    """
    L_mse = F.mse_loss(pred, target)

    # Per-channel mean / variance
    mu_p = pred.mean(dim=(-2, -1))
    mu_t = target.mean(dim=(-2, -1))
    var_p = pred.var(dim=(-2, -1))
    var_t = target.var(dim=(-2, -1))
    L_mean = ((mu_p - mu_t) ** 2).mean()
    L_var = ((var_p - var_t) ** 2).mean()

    sp_p = _spatial_psd(pred)
    sp_t = _spatial_psd(target)
    L_spec_x = F.mse_loss(torch.log(sp_p + eps), torch.log(sp_t + eps))

    tp_p = _temporal_psd(pred)
    tp_t = _temporal_psd(target)
    L_spec_t = F.mse_loss(torch.log(tp_p + eps), torch.log(tp_t + eps))

    total = (w_mse * L_mse
             + w_mean * L_mean
             + w_var * L_var
             + w_spec_x * L_spec_x
             + w_spec_t * L_spec_t)

    # Return component tensors (detached) so the caller can accumulate on
    # GPU and .item() once per epoch. .item() per-batch was forcing
    # GPU->CPU syncs that left the 5090 idle most of the time.
    return total, {
        "mse":    L_mse.detach(),
        "mean":   L_mean.detach(),
        "var":    L_var.detach(),
        "spec_x": L_spec_x.detach(),
        "spec_t": L_spec_t.detach(),
    }
