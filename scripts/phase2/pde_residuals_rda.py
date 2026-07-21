"""
PDE residual for the 2D Reaction-Advection-Diffusion SPDE.

  R = u_t + a u_x + b u_y - D (u_xx + u_yy) - alpha u (1 - u^2) - xi
"""

import torch

from noise_interp_3d import NoiseInterpolator3D


def _grad(y, x):
    return torch.autograd.grad(
        y, x, grad_outputs=torch.ones_like(y),
        create_graph=True, retain_graph=True,
    )[0]


def compute_residual_rda(
    system,
    x: torch.Tensor, y: torch.Tensor, t: torch.Tensor,
    noise_u: NoiseInterpolator3D,
    cfg,
):
    u = system.net_u(x, y, t)

    u_t  = _grad(u, t)
    u_x  = _grad(u, x); u_xx = _grad(u_x, x)
    u_y  = _grad(u, y); u_yy = _grad(u_y, y)

    xi = noise_u(x, y, t)

    R = (u_t
         + cfg.a_u * u_x + cfg.b_u * u_y
         - cfg.D_u * (u_xx + u_yy)
         - cfg.alpha_u * u * (1.0 - u ** 2)
         - xi)
    return R


# -- IC / BC ----------------------------------------------------------

def ic_loss(net, x, y, t0, target):
    return torch.mean((net(x, y, t0) - target) ** 2)


def bc_loss_neumann(net, x, y, t, axis: str):
    u = net(x, y, t)
    u_d = _grad(u, x if axis == "x" else y)
    return torch.mean(u_d ** 2)
