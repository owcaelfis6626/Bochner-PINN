"""
Dataset generation for the FNO operator.

For each seed we run the classical FD solver once and produce one
training tuple:

    inputs (8, Nx, Nt):
        ch 0: noise_z       (white)
        ch 1: noise_y       (white)
        ch 2: noise_p       (white)
        ch 3: jump_z        (VG)            — zero if use_jumps = False
        ch 4: jump_y        (Gamma)         — zero if use_jumps = False
        ch 5: IC_z(x) broadcast over t
        ch 6: IC_y(x) broadcast over t
        ch 7: IC_p(x) broadcast over t

    outputs (3, Nx, Nt):
        ch 0: Z(x, t)
        ch 1: Y(x, t)
        ch 2: P(x, t)

Datasets are cached as torch tensors in `data_dir` so we don't re-solve
on every training run.
"""

import os
import sys
import time
import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from classical_solver import solve_classical, ic_z_func, ic_y_func, ic_p_func  # noqa
from noise import generate_noise_field                                         # noqa
from vg_jumps import generate_vg_jump_field, generate_gamma_jump_field         # noqa
from config_fno import FNOConfig


def _build_one_sample(physics_cfg, fno_cfg: FNOConfig, seed: int):
    """Run FD once and stack into (in, out) tensors."""
    rng = np.random.default_rng(seed)

    nz = generate_noise_field(physics_cfg, "z", rng)
    ny = generate_noise_field(physics_cfg, "y", rng)
    np_field = generate_noise_field(physics_cfg, "p", rng)

    if fno_cfg.use_jumps:
        jz = generate_vg_jump_field(
            physics_cfg, sigma=fno_cfg.vg_sigma, nu=fno_cfg.vg_nu,
            theta=fno_cfg.vg_theta, rng=rng,
        )
        jy = generate_gamma_jump_field(
            physics_cfg, alpha_rate=fno_cfg.gamma_alpha,
            beta_scale=fno_cfg.gamma_beta, rng=rng,
        )
    else:
        jz = np.zeros((physics_cfg.Nx, physics_cfg.Nt))
        jy = np.zeros((physics_cfg.Nx, physics_cfg.Nt))

    x_grid, t_grid, Z, Y, P = solve_classical(
        physics_cfg, nz, ny, np_field,
        jump_z=jz if fno_cfg.use_jumps else None,
        jump_y=jy if fno_cfg.use_jumps else None,
    )

    # IC channels: broadcast initial profile across t
    ic_z = ic_z_func(x_grid, physics_cfg)[:, None].repeat(physics_cfg.Nt, axis=1)
    ic_y = ic_y_func(x_grid, physics_cfg)[:, None].repeat(physics_cfg.Nt, axis=1)
    ic_p = ic_p_func(x_grid, physics_cfg)[:, None].repeat(physics_cfg.Nt, axis=1)

    inputs = np.stack([nz, ny, np_field, jz, jy, ic_z, ic_y, ic_p], axis=0)
    outputs = np.stack([Z, Y, P], axis=0)
    return inputs.astype(np.float32), outputs.astype(np.float32)


def build_split(fno_cfg: FNOConfig, n: int, base_seed: int,
                tag: str, force: bool = False):
    """Build (or load cached) split. Returns (X, Y) tensors of shapes
    (n, C_in, Nx, Nt) and (n, C_out, Nx, Nt)."""
    os.makedirs(fno_cfg.data_dir, exist_ok=True)
    cache = os.path.join(fno_cfg.data_dir,
                         f"{tag}_n{n}_seed{base_seed}_jumps{int(fno_cfg.use_jumps)}.pt")
    if os.path.exists(cache) and not force:
        blob = torch.load(cache, map_location="cpu", weights_only=True)
        return blob["x"], blob["y"]

    physics_cfg = fno_cfg.physics()
    print(f"  [data] generating {tag}: n={n}, base_seed={base_seed}")
    t0 = time.time()
    Xs, Ys = [], []
    for i in range(n):
        x_np, y_np = _build_one_sample(physics_cfg, fno_cfg, base_seed + i)
        Xs.append(x_np)
        Ys.append(y_np)
        if (i + 1) % max(1, n // 10) == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (n - i - 1)
            print(f"    {i+1}/{n}   elapsed={elapsed:.1f}s   eta={eta:.1f}s")

    X = torch.tensor(np.stack(Xs), dtype=torch.float32)
    Y = torch.tensor(np.stack(Ys), dtype=torch.float32)
    torch.save({"x": X, "y": Y}, cache)
    print(f"  [data] cached -> {cache}   ({time.time()-t0:.1f}s)")
    return X, Y


def normalize_per_channel(X: torch.Tensor, stats=None):
    """Zero-mean, unit-std per channel using stats from `stats` dict
    (if None, computed from X). Returns (X_norm, stats)."""
    if stats is None:
        mean = X.mean(dim=(0, 2, 3), keepdim=True)
        std = X.std(dim=(0, 2, 3), keepdim=True).clamp_min(1e-6)
        stats = {"mean": mean, "std": std}
    X_norm = (X - stats["mean"]) / stats["std"]
    return X_norm, stats


def denormalize_per_channel(X: torch.Tensor, stats: dict) -> torch.Tensor:
    return X * stats["std"].to(X.device) + stats["mean"].to(X.device)
