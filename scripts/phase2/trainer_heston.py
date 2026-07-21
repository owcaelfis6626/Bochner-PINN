"""
Unified Heston PINN trainer (vanilla / spectral mode flag).

Mirrors the trainer pattern of the main spde_2d package:
    * ConfigPINNHeston          -> mode='vanilla'
    * ConfigPINNHestonSpectral  -> mode='spectral'

Spectral mode adds three losses on a fixed sub-grid:
    L_spec_xy : MSE of  log10(spatial-PSD)   PINN vs FD
    L_spec_t  : MSE of  log10(temporal-PSD)  PINN vs FD
    L_var     : MSE of  Var(PINN) / Var(FD)  per channel
"""

from __future__ import annotations

import os
import time
import numpy as np
import torch

try:
    from .config_pinn_heston import ConfigPINNHeston, ConfigPINNHestonSpectral
    from .pinn_models import HestonPINNSystem
    from .pde_residuals_heston import (
        compute_residuals_heston,
        ic_loss, bc_loss_neumann,
        positivity_penalty_v, positivity_penalty_S,
    )
    from .noise_interp_3d import NoiseInterpolator3D
    from .noise_2d_shared import generate_correlated_white_pair
    from .heston_solver_2d import solve_heston_2d, ic_S_func, ic_v_func
except ImportError:
    from config_pinn_heston import ConfigPINNHeston, ConfigPINNHestonSpectral
    from pinn_models import HestonPINNSystem
    from pde_residuals_heston import (
        compute_residuals_heston,
        ic_loss, bc_loss_neumann,
        positivity_penalty_v, positivity_penalty_S,
    )
    from noise_interp_3d import NoiseInterpolator3D
    from noise_2d_shared import generate_correlated_white_pair
    from heston_solver_2d import solve_heston_2d, ic_S_func, ic_v_func


# ─── Spectral helpers (same definitions as main spde_2d package) ──────

def _spatial_psd_xy(field: torch.Tensor) -> torch.Tensor:
    Fx = torch.fft.rfft(field, dim=-3).abs() ** 2
    Fy = torch.fft.rfft(field, dim=-2).abs() ** 2
    psd_x = Fx.mean(dim=(-2, -1))
    psd_y = Fy.mean(dim=(-3, -1))
    K = min(psd_x.shape[-1], psd_y.shape[-1])
    return 0.5 * (psd_x[..., :K] + psd_y[..., :K])


def _temporal_psd(field: torch.Tensor) -> torch.Tensor:
    Ft = torch.fft.rfft(field, dim=-1).abs() ** 2
    return Ft.mean(dim=(-3, -2))


def _build_spectral_subgrid(cfg, grids, device):
    x_grid, y_grid, t_grid = grids
    Nx, Ny, Nt = len(x_grid), len(y_grid), len(t_grid)
    ix = torch.linspace(0, Nx - 1, cfg.spec_Nx, device=device).round().long()
    iy = torch.linspace(0, Ny - 1, cfg.spec_Ny, device=device).round().long()
    it = torch.linspace(0, Nt - 1, cfg.spec_Nt, device=device).round().long()
    xg = torch.tensor(x_grid, dtype=torch.float32, device=device)[ix]
    yg = torch.tensor(y_grid, dtype=torch.float32, device=device)[iy]
    tg = torch.tensor(t_grid, dtype=torch.float32, device=device)[it]
    XX, YY, TT = torch.meshgrid(xg, yg, tg, indexing="ij")
    return (XX.reshape(-1, 1).contiguous(),
            YY.reshape(-1, 1).contiguous(),
            TT.reshape(-1, 1).contiguous(),
            ix, iy, it)


def _spectral_targets(fields, ix, iy, it, channels):
    targets = {}
    with torch.no_grad():
        for name, F in zip(channels, fields):
            sub = F[ix][:, iy][:, :, it]
            sp = _spatial_psd_xy(sub.unsqueeze(0)).squeeze(0)
            tp = _temporal_psd(sub.unsqueeze(0)).squeeze(0)
            targets[name] = {
                "log_sp": torch.log10(sp.clamp(min=1e-12)),
                "log_tp": torch.log10(tp.clamp(min=1e-12)),
                "var":    sub.var(),
            }
    return targets


def _spectral_loss(preds, targets, channels, eps=1e-12):
    device = preds[0].device
    L_xy = torch.zeros((), device=device)
    L_t  = torch.zeros((), device=device)
    L_v  = torch.zeros((), device=device)
    for name, F in zip(channels, preds):
        sp = _spatial_psd_xy(F.unsqueeze(0)).squeeze(0)
        tp = _temporal_psd(F.unsqueeze(0)).squeeze(0)
        log_sp = torch.log10(sp.clamp(min=eps))
        log_tp = torch.log10(tp.clamp(min=eps))
        L_xy = L_xy + ((log_sp - targets[name]["log_sp"]) ** 2).mean()
        L_t  = L_t  + ((log_tp - targets[name]["log_tp"]) ** 2).mean()
        var_p = F.var()
        var_t = targets[name]["var"]
        L_v = L_v + ((var_p - var_t) / (var_t + eps)) ** 2
    return L_xy, L_t, L_v


# ─── Realisation builder ──────────────────────────────────────────────

def _build_realisation(cfg, seed, device):
    rng = np.random.default_rng(seed)

    nS, nv = generate_correlated_white_pair(
        cfg.Nx, cfg.Ny, cfg.Nt,
        sigma1=1.0, sigma2=1.0, rho=cfg.rho,
        dx=cfg.dx, dy=cfg.dy, dt=cfg.dt, rng=rng,
    )

    x_grid, y_grid, t_grid, S_fd, V_fd = solve_heston_2d(cfg, nS, nv)

    interp = {
        "S": NoiseInterpolator3D(nS, x_grid, y_grid, t_grid, device),
        "v": NoiseInterpolator3D(nv, x_grid, y_grid, t_grid, device),
    }
    grids = (x_grid, y_grid, t_grid)
    fields = (
        torch.tensor(S_fd, dtype=torch.float32, device=device),
        torch.tensor(V_fd, dtype=torch.float32, device=device),
    )
    return interp, grids, fields


# ─── Sampling helpers ─────────────────────────────────────────────────

def _sample_collocation(cfg, n, device):
    x = torch.rand(n, 1, device=device) * (cfg.x_max - cfg.x_min) + cfg.x_min
    y = torch.rand(n, 1, device=device) * (cfg.y_max - cfg.y_min) + cfg.y_min
    t = torch.rand(n, 1, device=device) * (cfg.t_max - cfg.t_min) + cfg.t_min
    x.requires_grad_(True); y.requires_grad_(True); t.requires_grad_(True)
    return x, y, t


def _sample_ic(cfg, n, device):
    x = torch.rand(n, 1, device=device) * (cfg.x_max - cfg.x_min) + cfg.x_min
    y = torch.rand(n, 1, device=device) * (cfg.y_max - cfg.y_min) + cfg.y_min
    t = torch.full((n, 1), cfg.t_min, device=device)
    return x, y, t


def _sample_bc(cfg, n, device):
    t = torch.rand(n, 1, device=device) * (cfg.t_max - cfg.t_min) + cfg.t_min
    y_in = torch.rand(n, 1, device=device) * (cfg.y_max - cfg.y_min) + cfg.y_min
    x_in = torch.rand(n, 1, device=device) * (cfg.x_max - cfg.x_min) + cfg.x_min
    bnds = {}
    for tag, fixed_axis, fixed_val in [
        ("xmin", "x", cfg.x_min), ("xmax", "x", cfg.x_max),
        ("ymin", "y", cfg.y_min), ("ymax", "y", cfg.y_max),
    ]:
        if fixed_axis == "x":
            xb = torch.full((n, 1), fixed_val, device=device); yb = y_in.clone()
        else:
            xb = x_in.clone(); yb = torch.full((n, 1), fixed_val, device=device)
        tb = t.clone()
        xb.requires_grad_(True); yb.requires_grad_(True)
        bnds[tag] = (xb, yb, tb)
    return bnds


# ─── Training ─────────────────────────────────────────────────────────

def train_one(cfg, seed: int, device: torch.device,
              mode: str = "vanilla", save_path: str | None = None):
    """mode in {"vanilla", "spectral"}"""
    assert mode in ("vanilla", "spectral")
    torch.manual_seed(seed); np.random.seed(seed)

    print(f"  [heston-{mode}] building realisation seed={seed}", flush=True)
    interp, grids, fields = _build_realisation(cfg, seed, device)
    S_fd, V_fd = fields
    Nx, Ny, Nt = S_fd.shape

    system = HestonPINNSystem(cfg).to(device)
    print(f"  [heston-{mode}] params="
          f"{sum(p.numel() for p in system.parameters()):,}", flush=True)

    opt = torch.optim.Adam(system.parameters(), lr=cfg.learning_rate)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=cfg.num_epochs, eta_min=cfg.lr_min)

    x_grid_t = torch.tensor(grids[0], dtype=torch.float32, device=device)
    y_grid_t = torch.tensor(grids[1], dtype=torch.float32, device=device)

    if mode == "spectral":
        xs, ys, ts, ix_s, iy_s, it_s = _build_spectral_subgrid(cfg, grids, device)
        spec_targets = _spectral_targets(fields, ix_s, iy_s, it_s, ("S", "v"))
        print(f"  [heston-spectral] sub-grid: "
              f"{cfg.spec_Nx}x{cfg.spec_Ny}x{cfg.spec_Nt} = "
              f"{xs.numel():,} pts", flush=True)

    t0 = time.time()
    for ep in range(1, cfg.num_epochs + 1):
        system.train(); opt.zero_grad(set_to_none=True)

        # 1) PDE residual
        xc, yc, tc = _sample_collocation(cfg, cfg.batch_coll, device)
        if ep <= cfg.warmup_epochs:
            L_pde = torch.zeros((), device=device)
        else:
            R_S, R_v = compute_residuals_heston(
                system, xc, yc, tc, interp["S"], interp["v"], cfg)
            L_pde = (cfg.w_pde_S * (R_S ** 2).mean()
                     + cfg.w_pde_v * (R_v ** 2).mean())

        # 2) IC loss
        xi, yi, ti = _sample_ic(cfg, cfg.batch_ic, device)
        x_np = xi.detach().cpu().numpy().flatten()
        y_np = yi.detach().cpu().numpy().flatten()
        ic_S_t = torch.tensor(ic_S_func(x_np, y_np, cfg).reshape(-1, 1),
                              dtype=torch.float32, device=device)
        ic_v_t = torch.tensor(ic_v_func(x_np, y_np, cfg).reshape(-1, 1),
                              dtype=torch.float32, device=device)
        L_ic = (ic_loss(system.net_S, xi, yi, ti, ic_S_t)
                + ic_loss(system.net_v, xi, yi, ti, ic_v_t))

        # 3) BC losses (Neumann on both)
        bnds = _sample_bc(cfg, cfg.batch_bc, device)
        L_bc = torch.zeros((), device=device)
        for tag, (xb, yb, tb) in bnds.items():
            ax = "x" if tag.startswith("x") else "y"
            L_bc = L_bc + bc_loss_neumann(system.net_S, xb, yb, tb, ax)
            L_bc = L_bc + bc_loss_neumann(system.net_v, xb, yb, tb, ax)

        # 4) Pointwise data loss
        n_d = cfg.batch_data
        ix = torch.randint(0, Nx, (n_d,), device=device)
        iy = torch.randint(0, Ny, (n_d,), device=device)
        it = torch.randint(0, Nt, (n_d,), device=device)
        xd = x_grid_t[ix].view(-1, 1)
        yd = y_grid_t[iy].view(-1, 1)
        td = (it.float() / max(Nt - 1, 1)).view(-1, 1) * (cfg.t_max - cfg.t_min) + cfg.t_min
        with torch.no_grad():
            St = S_fd[ix, iy, it].view(-1, 1)
            Vt = V_fd[ix, iy, it].view(-1, 1)
        Sp_d = system.net_S(xd, yd, td)
        Vp_d = system.net_v(xd, yd, td)
        L_data = ((Sp_d - St) ** 2 + (Vp_d - Vt) ** 2).mean()

        # 5) Positivity penalties
        L_posv = positivity_penalty_v(system.net_v,
                                      xc.detach(), yc.detach(), tc.detach())
        L_posS = positivity_penalty_S(system.net_S,
                                      xc.detach(), yc.detach(), tc.detach())

        # 6) Spectral losses (spectral mode only)
        if mode == "spectral" and ep % cfg.spec_every == 0:
            Sp_s = system.net_S(xs, ys, ts).reshape(
                cfg.spec_Nx, cfg.spec_Ny, cfg.spec_Nt)
            Vp_s = system.net_v(xs, ys, ts).reshape(
                cfg.spec_Nx, cfg.spec_Ny, cfg.spec_Nt)
            L_spec_xy, L_spec_t, L_var = _spectral_loss(
                (Sp_s, Vp_s), spec_targets, ("S", "v"))
        else:
            L_spec_xy = torch.zeros((), device=device)
            L_spec_t  = torch.zeros((), device=device)
            L_var     = torch.zeros((), device=device)

        loss = (L_pde
                + cfg.w_ic   * L_ic
                + cfg.w_bc   * L_bc
                + cfg.w_data * L_data
                + cfg.w_pos_v * L_posv
                + cfg.w_pos_S * L_posS)
        if mode == "spectral":
            loss = (loss
                    + cfg.w_spec_xy * L_spec_xy
                    + cfg.w_spec_t  * L_spec_t
                    + cfg.w_var     * L_var)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(system.parameters(), cfg.grad_clip)
        opt.step(); sched.step()

        if ep % cfg.log_every == 0 or ep == 1:
            elapsed = time.time() - t0
            extra = ""
            if mode == "spectral":
                extra = (f"  sxy={L_spec_xy.item():.2e}  "
                         f"st={L_spec_t.item():.2e}  "
                         f"var={L_var.item():.2e}")
            print(f"  [ep {ep:5d}/{cfg.num_epochs}] "
                  f"loss={loss.item():.3e}  pde={L_pde.item():.2e}  "
                  f"ic={L_ic.item():.2e}  data={L_data.item():.2e}"
                  f"{extra}  t={elapsed/60:.1f}m", flush=True)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save({
            "model": system.state_dict(),
            "cfg":   cfg.__dict__,
            "seed":  seed,
            "mode":  mode,
        }, save_path)
        print(f"  [heston-{mode}] saved -> {save_path}", flush=True)
    return system, fields, grids
