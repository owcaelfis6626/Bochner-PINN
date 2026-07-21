"""Trilinear interpolation of a (Nx, Ny, Nt) noise field at arbitrary
(x, y, t) query points (no gradient tracking).

Self-contained copy of ``spde_2d.noise_interp_2d`` so this folder is
independent of the main package.
"""

import numpy as np
import torch


class NoiseInterpolator3D:
    def __init__(self, noise_grid: np.ndarray,
                 x_grid: np.ndarray, y_grid: np.ndarray, t_grid: np.ndarray,
                 device: torch.device):
        self.device = device
        self.noise = torch.tensor(noise_grid, dtype=torch.float32,
                                  device=device)
        self.x0, self.x1 = float(x_grid[0]), float(x_grid[-1])
        self.y0, self.y1 = float(y_grid[0]), float(y_grid[-1])
        self.t0, self.t1 = float(t_grid[0]), float(t_grid[-1])
        self.Nx, self.Ny, self.Nt = noise_grid.shape

    @torch.no_grad()
    def __call__(self, x: torch.Tensor, y: torch.Tensor,
                 t: torch.Tensor) -> torch.Tensor:
        x_ = x.detach().view(-1)
        y_ = y.detach().view(-1)
        t_ = t.detach().view(-1)

        ix = ((x_ - self.x0) / (self.x1 - self.x0) * (self.Nx - 1)) \
            .clamp(0, self.Nx - 1 - 1e-5)
        iy = ((y_ - self.y0) / (self.y1 - self.y0) * (self.Ny - 1)) \
            .clamp(0, self.Ny - 1 - 1e-5)
        it = ((t_ - self.t0) / (self.t1 - self.t0) * (self.Nt - 1)) \
            .clamp(0, self.Nt - 1 - 1e-5)

        ix0 = ix.long(); iy0 = iy.long(); it0 = it.long()
        ix1 = (ix0 + 1).clamp(max=self.Nx - 1)
        iy1 = (iy0 + 1).clamp(max=self.Ny - 1)
        it1 = (it0 + 1).clamp(max=self.Nt - 1)

        fx = ix - ix0.float()
        fy = iy - iy0.float()
        ft = it - it0.float()

        v000 = self.noise[ix0, iy0, it0]
        v001 = self.noise[ix0, iy0, it1]
        v010 = self.noise[ix0, iy1, it0]
        v011 = self.noise[ix0, iy1, it1]
        v100 = self.noise[ix1, iy0, it0]
        v101 = self.noise[ix1, iy0, it1]
        v110 = self.noise[ix1, iy1, it0]
        v111 = self.noise[ix1, iy1, it1]

        c00 = v000 * (1 - fx) + v100 * fx
        c01 = v001 * (1 - fx) + v101 * fx
        c10 = v010 * (1 - fx) + v110 * fx
        c11 = v011 * (1 - fx) + v111 * fx
        c0 = c00 * (1 - fy) + c10 * fy
        c1 = c01 * (1 - fy) + c11 * fy
        v = c0 * (1 - ft) + c1 * ft
        return v.unsqueeze(-1)
