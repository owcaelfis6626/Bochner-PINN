"""
PDE residuals for the 2D spatial Heston SPDE.

  R_S = S_t - D_S (S_xx + S_yy) - mu_S S - sqrt(max(v,0)) * S * xi_S
  R_v = v_t - D_v (v_xx + v_yy) - kappa (theta - v) - sigma_v sqrt(max(v,0)) * xi_v
"""

import torch

from noise_interp_3d import NoiseInterpolator3D


def _grad(y, x):
    return torch.autograd.grad(
        y, x, grad_outputs=torch.ones_like(y),
        create_graph=True, retain_graph=True,
    )[0]


def compute_residuals_heston(
    system,
    x: torch.Tensor, y: torch.Tensor, t: torch.Tensor,
    noise_S: NoiseInterpolator3D,
    noise_v: NoiseInterpolator3D,
    cfg,
):
    S = system.net_S(x, y, t)
    V = system.net_v(x, y, t)

    # S derivatives
    S_t = _grad(S, t)
    S_x = _grad(S, x); S_xx = _grad(S_x, x)
    S_y = _grad(S, y); S_yy = _grad(S_y, y)

    # v derivatives
    V_t = _grad(V, t)
    V_x = _grad(V, x); V_xx = _grad(V_x, x)
    V_y = _grad(V, y); V_yy = _grad(V_y, y)

    sqrt_v = torch.sqrt(torch.clamp(V, min=1e-8))

    xi_S = noise_S(x, y, t)
    xi_v = noise_v(x, y, t)

    R_S = (S_t - cfg.D_S * (S_xx + S_yy)
           - cfg.mu_S * S
           - sqrt_v * S * xi_S)

    R_v = (V_t - cfg.D_v * (V_xx + V_yy)
           - cfg.kappa_v * (cfg.theta_v - V)
           - cfg.sigma_v * sqrt_v * xi_v)

    return R_S, R_v


# ── IC / BC / penalties ──────────────────────────────────────────────

def ic_loss(net, x, y, t0, target):
    return torch.mean((net(x, y, t0) - target) ** 2)


def bc_loss_neumann(net, x, y, t, axis: str):
    u = net(x, y, t)
    u_d = _grad(u, x if axis == "x" else y)
    return torch.mean(u_d ** 2)


def positivity_penalty_v(net_v, x, y, t):
    """Penalise v < 0."""
    return torch.mean(torch.relu(-net_v(x, y, t)) ** 2)


def positivity_penalty_S(net_S, x, y, t):
    """Penalise S < 0 (asset price can't go negative)."""
    return torch.mean(torch.relu(-net_S(x, y, t)) ** 2)
