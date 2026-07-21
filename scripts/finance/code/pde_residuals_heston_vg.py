"""
PDE residuals for the 2D Heston-VG SPDE (Heston with Variance-Gamma jumps).

System:
  R_S = S_t - D_S (S_xx + S_yy) - mu_S S
        - sqrt(max(v,0)) * S * xi_S(x,y,t)
        - jump_intensity * S * J_VG(x,y,t)

  R_v = v_t - D_v (v_xx + v_yy) - kappa (theta - v)
        - sigma_v sqrt(max(v,0)) * xi_v(x,y,t)

Notes
-----
The VG jump term is the new piece. From the PINN's perspective it is
just another forcing field (interpolated from the FD-generated VG jumps).

Why this is OK:
- VG jumps in the FD solver are sample paths of the Lévy process.
- PINN must match THIS specific sample path (via L_data) so the
  same path appears in the residual.
- The PINN can't learn the VG distribution itself (would need many
  samples), but it can learn THIS realization conditioned on the
  jump field.
- The spectral loss tries to match second-moment structure (variance,
  PSD) which CAN be learned from one sample path.

Why VG is harder than Gaussian noise:
- Gaussian: smooth in distribution, network can fit easily
- VG: fat-tailed, sparse large jumps, harder gradient signal
- Solution: stronger network (wider, more Fourier features),
  longer warmup, careful loss balancing.
"""

import torch

from noise_interp_3d import NoiseInterpolator3D


def _grad(y, x):
    return torch.autograd.grad(
        y, x, grad_outputs=torch.ones_like(y),
        create_graph=True, retain_graph=True,
    )[0]


def compute_residuals_heston_vg(
    system,
    x: torch.Tensor, y: torch.Tensor, t: torch.Tensor,
    noise_S: NoiseInterpolator3D,
    noise_v: NoiseInterpolator3D,
    vg_jumps: NoiseInterpolator3D,
    cfg,
):
    """Compute PDE residuals for Heston-VG.

    Parameters
    ----------
    system : PINN system with .net_S(x,y,t), .net_v(x,y,t)
    x, y, t : collocation point tensors (with requires_grad=True)
    noise_S : NoiseInterpolator3D for the Brownian noise driving S
    noise_v : NoiseInterpolator3D for the Brownian noise driving v
    vg_jumps : NoiseInterpolator3D for the VG jump field
    cfg : ConfigHestonVG2D (or PINN config inheriting from it)

    Returns
    -------
    R_S : (N,) residual of S equation
    R_v : (N,) residual of v equation
    """
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

    # Noise fields at collocation points
    xi_S = noise_S(x, y, t)
    xi_v = noise_v(x, y, t)

    # VG jump field at collocation points
    j_vg = vg_jumps(x, y, t)

    # ── R_S: full Heston-VG equation for S ─────────────────────────
    R_S = (S_t
           - cfg.D_S * (S_xx + S_yy)
           - cfg.mu_S * S
           - sqrt_v * S * xi_S
           - cfg.jump_intensity * S * j_vg)

    # ── R_v: standard Heston for v (no jumps in v) ─────────────────
    R_v = (V_t
           - cfg.D_v * (V_xx + V_yy)
           - cfg.kappa_v * (cfg.theta_v - V)
           - cfg.sigma_v * sqrt_v * xi_v)

    return R_S, R_v


# ── IC / BC / penalties (identical pattern to Heston) ────────────────

def ic_loss(net, x, y, t0, target):
    """Initial condition loss: ||net(x,y,t=0) - target||^2"""
    return torch.mean((net(x, y, t0) - target) ** 2)


def bc_loss_neumann(net, x, y, t, axis: str):
    """Neumann BC: ||du/dn||^2 on boundary."""
    u = net(x, y, t)
    u_d = _grad(u, x if axis == "x" else y)
    return torch.mean(u_d ** 2)


def positivity_penalty_v(net_v, x, y, t):
    """Penalise v < 0 (variance must be non-negative)."""
    return torch.mean(torch.relu(-net_v(x, y, t)) ** 2)


def positivity_penalty_S(net_S, x, y, t):
    """Penalise S < 0 (asset price must be positive)."""
    return torch.mean(torch.relu(-net_S(x, y, t)) ** 2)


# ── Diagnostic helpers for jump-aware PINN ───────────────────────────

def jump_signature_loss(net_S, x, y, t, vg_jumps_interp, cfg):
    """Optional: penalise the deviation of net_S log-returns from the
    VG jump field's higher moments.

    This is an EXPERIMENTAL loss that might help the network learn
    the jump structure. Use with caution: it requires the network
    to predict at multiple time slices to get returns.

    Currently unused in main trainer; provided for future experiments.
    """
    # Get S at t and t+dt for finite-difference returns
    dt = cfg.dt
    t_next = t + dt
    S_t = net_S(x, y, t)
    S_tp = net_S(x, y, t_next)

    # Log-returns
    log_ret = torch.log(torch.clamp(S_tp, min=1e-6)) - torch.log(torch.clamp(S_t, min=1e-6))

    # VG jump increment
    j_vg = vg_jumps_interp(x, y, t)

    # If jumps drive returns, log_ret ≈ (mu - 0.5*v)*dt + sqrt(v)*xi*dt + j_vg*dt
    # For the spectral statistics, just check that returns have the
    # same kurtosis as the jumps when they are large.
    # Penalise deviation in 4th moment:
    ret_4 = (log_ret ** 4).mean()
    j_4 = ((j_vg * dt) ** 4).mean()

    return torch.abs(ret_4 - j_4)
