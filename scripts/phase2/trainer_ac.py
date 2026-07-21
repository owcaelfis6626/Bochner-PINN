"""
Unified Allen-Cahn PINN trainer (vanilla / spectral mode flag).

Single scalar field u(x, y, t).  Same loss skeleton as the Heston trainer
but only one channel and no positivity penalties.
"""

from __future__ import annotations

import os
import time
import numpy as np
import torch

try:
    from .config_pinn_ac import ConfigPINNAC, ConfigPINNACSpectral
    from .pinn_models import ACPINNSystem
    from .pde_residuals_ac import compute_residual_ac, ic_loss, bc_loss_neumann
    from .noise_interp_3d import NoiseInterpolator3D
    from .noise_2d_shared import generate_white, generate_fbm
    from .allen_cahn_solver_2d import solve_allen_cahn_2d, ic_u_func
    from .trainer_heston import (
        _spatial_psd_xy, _temporal_psd,
        _build_spectral_subgrid, _spectral_targets, _spectral_loss,
        _sample_collocation, _sample_ic, _sample_bc,
    )
except ImportError:
    from config_pinn_ac import ConfigPINNAC, ConfigPINNACSpectral
    from pinn_models import ACPINNSystem
    from pde_residuals_ac import compute_residual_ac, ic_loss, bc_loss_neumann
    from noise_interp_3d import NoiseInterpolator3D
    from noise_2d_shared import generate_white, generate_fbm
    from allen_cahn_solver_2d import solve_allen_cahn_2d, ic_u_func
    from trainer_heston import (
        _spatial_psd_xy, _temporal_psd,
        _build_spectral_subgrid, _spectral_targets, _spectral_loss,
        _sample_collocation, _sample_ic, _sample_bc,
    )


# ─── Realisation builder ──────────────────────────────────────────────

def _build_realisation(cfg, seed, device):
    rng = np.random.default_rng(seed)
    if cfg.noise_type_u == "fbm":
        nu = generate_fbm(cfg.Nx, cfg.Ny, cfg.Nt,
                          sigma=cfg.sigma_u, hurst=cfg.hurst_u,
                          dx=cfg.dx, dy=cfg.dy, dt=cfg.dt, rng=rng)
    else:
        nu = generate_white(cfg.Nx, cfg.Ny, cfg.Nt, sigma=cfg.sigma_u,
                            dx=cfg.dx, dy=cfg.dy, dt=cfg.dt, rng=rng)

    x_grid, y_grid, t_grid, U = solve_allen_cahn_2d(cfg, nu, rng=rng)

    interp = NoiseInterpolator3D(nu, x_grid, y_grid, t_grid, device)
    grids = (x_grid, y_grid, t_grid)
    field = torch.tensor(U, dtype=torch.float32, device=device)
    return interp, grids, field


# ─── Training ─────────────────────────────────────────────────────────

def train_one(cfg, seed: int, device: torch.device,
              mode: str = "vanilla", save_path: str | None = None):
    assert mode in ("vanilla", "spectral")
    torch.manual_seed(seed); np.random.seed(seed)

    print(f"  [ac-{mode}] building realisation seed={seed}", flush=True)
    interp, grids, U_fd = _build_realisation(cfg, seed, device)
    Nx, Ny, Nt = U_fd.shape

    system = ACPINNSystem(cfg).to(device)
    print(f"  [ac-{mode}] params="
          f"{sum(p.numel() for p in system.parameters()):,}", flush=True)

    opt = torch.optim.Adam(system.parameters(), lr=cfg.learning_rate)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=cfg.num_epochs, eta_min=cfg.lr_min)

    x_grid_t = torch.tensor(grids[0], dtype=torch.float32, device=device)
    y_grid_t = torch.tensor(grids[1], dtype=torch.float32, device=device)

    if mode == "spectral":
        xs, ys, ts, ix_s, iy_s, it_s = _build_spectral_subgrid(cfg, grids, device)
        spec_targets = _spectral_targets((U_fd,), ix_s, iy_s, it_s, ("u",))
        print(f"  [ac-spectral] sub-grid: "
              f"{cfg.spec_Nx}x{cfg.spec_Ny}x{cfg.spec_Nt} = "
              f"{xs.numel():,} pts", flush=True)

    # Pre-compute IC target grid (for the random_blob IC the IC is
    # realisation-dependent so we just use the FD IC at t = t_min).
    U0_np = U_fd[:, :, 0].detach().cpu().numpy()

    def _ic_target_at(x_np, y_np):
        # Bilinear interp of U0 at queried (x, y).
        ix = np.clip((x_np - cfg.x_min) / (cfg.x_max - cfg.x_min)
                    * (Nx - 1), 0, Nx - 1 - 1e-5)
        iy = np.clip((y_np - cfg.y_min) / (cfg.y_max - cfg.y_min)
                    * (Ny - 1), 0, Ny - 1 - 1e-5)
        ix0 = ix.astype(int); iy0 = iy.astype(int)
        ix1 = np.minimum(ix0 + 1, Nx - 1); iy1 = np.minimum(iy0 + 1, Ny - 1)
        fx = ix - ix0; fy = iy - iy0
        v00 = U0_np[ix0, iy0]; v10 = U0_np[ix1, iy0]
        v01 = U0_np[ix0, iy1]; v11 = U0_np[ix1, iy1]
        c0 = v00 * (1 - fx) + v10 * fx
        c1 = v01 * (1 - fx) + v11 * fx
        return c0 * (1 - fy) + c1 * fy

    t0 = time.time()
    for ep in range(1, cfg.num_epochs + 1):
        system.train(); opt.zero_grad(set_to_none=True)

        # 1) PDE residual
        xc, yc, tc = _sample_collocation(cfg, cfg.batch_coll, device)
        if ep <= cfg.warmup_epochs:
            L_pde = torch.zeros((), device=device)
        else:
            R = compute_residual_ac(system, xc, yc, tc, interp, cfg)
            L_pde = cfg.w_pde_u * (R ** 2).mean()

        # 2) IC loss — use FD initial slice
        xi, yi, ti = _sample_ic(cfg, cfg.batch_ic, device)
        x_np = xi.detach().cpu().numpy().flatten()
        y_np = yi.detach().cpu().numpy().flatten()
        ic_u_t = torch.tensor(_ic_target_at(x_np, y_np).reshape(-1, 1),
                              dtype=torch.float32, device=device)
        L_ic = ic_loss(system.net_u, xi, yi, ti, ic_u_t)

        # 3) Neumann BCs on all 4 walls
        bnds = _sample_bc(cfg, cfg.batch_bc, device)
        L_bc = torch.zeros((), device=device)
        for tag, (xb, yb, tb) in bnds.items():
            ax = "x" if tag.startswith("x") else "y"
            L_bc = L_bc + bc_loss_neumann(system.net_u, xb, yb, tb, ax)

        # 4) Pointwise data loss
        n_d = cfg.batch_data
        ix = torch.randint(0, Nx, (n_d,), device=device)
        iy = torch.randint(0, Ny, (n_d,), device=device)
        it = torch.randint(0, Nt, (n_d,), device=device)
        xd = x_grid_t[ix].view(-1, 1)
        yd = y_grid_t[iy].view(-1, 1)
        td = (it.float() / max(Nt - 1, 1)).view(-1, 1) * (cfg.t_max - cfg.t_min) + cfg.t_min
        with torch.no_grad():
            ut = U_fd[ix, iy, it].view(-1, 1)
        up_d = system.net_u(xd, yd, td)
        L_data = ((up_d - ut) ** 2).mean()

        # 5) Spectral losses (spectral mode only)
        if mode == "spectral" and ep % cfg.spec_every == 0:
            up_s = system.net_u(xs, ys, ts).reshape(
                cfg.spec_Nx, cfg.spec_Ny, cfg.spec_Nt)
            L_spec_xy, L_spec_t, L_var = _spectral_loss(
                (up_s,), spec_targets, ("u",))
        else:
            L_spec_xy = torch.zeros((), device=device)
            L_spec_t  = torch.zeros((), device=device)
            L_var     = torch.zeros((), device=device)

        loss = (L_pde
                + cfg.w_ic   * L_ic
                + cfg.w_bc   * L_bc
                + cfg.w_data * L_data)
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
        print(f"  [ac-{mode}] saved -> {save_path}", flush=True)
    return system, U_fd, grids
