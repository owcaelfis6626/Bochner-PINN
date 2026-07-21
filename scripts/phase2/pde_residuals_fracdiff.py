"""
PDE residual for the 2D Stochastic Fractional Diffusion SPDE.

  R = u_t + nu * (-Laplacian)^(alpha/2) u - xi

The fractional Laplacian is non-local — it cannot be evaluated at
isolated collocation points the way local derivatives can.  Instead we
compute the residual on a *fixed regular sub-grid* in Fourier space
each epoch:

    1. Evaluate PINN at all (Nx, Ny) grid points at a small set of
       random time slices.
    2. Compute u_t at those points via autograd.
    3. Compute (-Lap)^(a/2) u in Fourier space:
           u_hat = FFT2(u);   apply |k|^a;   IFFT2.
       (torch.fft is autograd-aware so backprop flows through.)
    4. Sample xi at those points.
    5. R = u_t + nu * frac_lap_u - xi.

This is the standard approach for fractional-PINN training.

Note: the FFT assumes periodic BCs, which match the FD ground-truth
solver.
"""

from __future__ import annotations

import torch

from noise_interp_3d import NoiseInterpolator3D


def _grad(y, x):
    return torch.autograd.grad(
        y, x, grad_outputs=torch.ones_like(y),
        create_graph=True, retain_graph=True,
    )[0]


def build_kfrac_torch(cfg, device):
    """Pre-compute |k|^alpha on the 2D FFT grid as a torch tensor.

    Call this once outside the training loop.
    """
    import numpy as np
    kx = 2.0 * torch.pi * torch.fft.fftfreq(cfg.Nx, d=cfg.dx, device=device)
    ky = 2.0 * torch.pi * torch.fft.fftfreq(cfg.Ny, d=cfg.dy, device=device)
    KX, KY = torch.meshgrid(kx, ky, indexing="ij")
    Kmag = torch.sqrt(KX ** 2 + KY ** 2)
    return Kmag ** cfg.alpha   # (Nx, Ny)


def fractional_laplacian_2d(u_field: torch.Tensor,
                            kfrac: torch.Tensor) -> torch.Tensor:
    """Compute (-Lap)^(alpha/2) u_field via FFT.

    u_field : (Nx, Ny, n_t) — real
    kfrac   : (Nx, Ny)      — pre-computed |k|^alpha
    Returns : (Nx, Ny, n_t)
    """
    u_hat = torch.fft.fft2(u_field, dim=(-3, -2))
    # broadcast kfrac to (Nx, Ny, 1)
    out_hat = kfrac.unsqueeze(-1) * u_hat
    out = torch.fft.ifft2(out_hat, dim=(-3, -2)).real
    return out


def compute_residual_fracdiff(
    system,
    x_full: torch.Tensor, y_full: torch.Tensor,  # (Nx*Ny, 1) flattened grid
    t_slice: torch.Tensor,                        # (n_t,) sampled time values
    noise_u: NoiseInterpolator3D,
    cfg,
    kfrac: torch.Tensor,
):
    """Compute fractional-diffusion residual on a fixed sub-grid.

    Strategy
    --------
    For each of n_t sampled times, we evaluate PINN at all (Nx*Ny) grid
    points and form a (Nx, Ny, n_t) tensor.  We then compute u_t via
    autograd and the fractional Laplacian via FFT.

    Returns
    -------
    R : (Nx, Ny, n_t) tensor
    """
    Nx, Ny = cfg.Nx, cfg.Ny
    n_t = t_slice.numel()

    # Build (Nx*Ny*n_t, 1) tensors of (x, y, t).
    # We want require_grad on t for autograd through u_t.
    XY_xy = torch.cat([x_full, y_full], dim=1)        # (Nx*Ny, 2)
    XY = XY_xy.unsqueeze(1).expand(-1, n_t, -1).reshape(-1, 2)
    T = t_slice.view(1, -1).expand(Nx * Ny, -1).reshape(-1, 1)
    T = T.detach().clone().requires_grad_(True)

    x = XY[:, 0:1]
    y = XY[:, 1:2]
    # x and y do NOT need grad here — fractional Laplacian is computed
    # via FFT, and we only need d/dt via autograd.
    u = system.net_u(x, y, T)              # (Nx*Ny*n_t, 1)
    u_t = _grad(u, T)                      # (Nx*Ny*n_t, 1)

    u_field  = u.view(Nx, Ny, n_t)
    u_t_fld  = u_t.view(Nx, Ny, n_t)

    # Fractional Laplacian via FFT
    fl = fractional_laplacian_2d(u_field, kfrac)

    # Noise
    xi = noise_u(x, y, T).view(Nx, Ny, n_t)

    R = u_t_fld + cfg.nu_u * fl - xi
    return R


# -- IC / BC ----------------------------------------------------------

def ic_loss(net, x, y, t0, target):
    return torch.mean((net(x, y, t0) - target) ** 2)


def bc_loss_periodic(net, x_lo, x_hi, y_in, t):
    """Periodic BC penalty: u(x_min, y, t) == u(x_max, y, t)."""
    a = net(x_lo, y_in, t)
    b = net(x_hi, y_in, t)
    return torch.mean((a - b) ** 2)
